from tvm import te
from tvm.topi.utils import get_stages_and_cfgs

def schedule_depth_conv_fused_nhwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, param_dict, _, bn_relu, _ = get_stages_and_cfgs(s, outs)

    # Searchable parameters
    # --------------------
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 4
    step_num_w = 4
    reduce_split = 16
    intermediate_reuse = 4 # How many 32x32 blocks of 1x1 filter reuse the intermediate data
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_y = output_step_tile_size_h
    num_thread_z = output_step_tile_size_w
    num_vthread_z = step_num_h * step_num_w
    num_vthread_y = 1
    num_vthread_x = 32

    # ######## Input data, weights, BN, etc
    s[stage_dict['FusedConv2D_PaddedInput_0']].compute_inline()
    PaddedSharedInput = s.cache_read(stage_dict['FusedConv2D_PaddedInput_0'], 'shared', [stage_dict['Output_0']])
    FL_1 = s.cache_read(param_dict['Filter_0'], 'local', [stage_dict['Output_0']])
    FS_2 = s.cache_read(param_dict['Filter_1'], 'shared', [stage_dict['Output_1']])
    s[layer_output_dict['Layer_0']].set_scope('shared')

    if bn_relu[0]:
        s[stage_dict['Output_0_BiasAdd']].compute_inline()
        s[stage_dict['Output_0']].set_scope('local')
        BiasL_1 = s.cache_read(param_dict['Bias_0'], 'local', [stage_dict['Output_0_BiasAdd']])
        DepthwiseLocalAccumulator = stage_dict['Output_0']
    else:
        DepthwiseLocalAccumulator = s.cache_write(layer_output_dict['Layer_0'], 'local')

    if bn_relu[1]:
        s[stage_dict['Output_1_BiasAdd']].compute_inline()
        s[stage_dict['Output_1']].set_scope('local')
        BiasL_2 = s.cache_read(param_dict['Bias_1'], 'local', [stage_dict['Output_1_BiasAdd']])
        OL = stage_dict['Output_1']
    else:
        OL = s.cache_write(layer_output_dict['Layer_1'], 'local')

    # ######## Blocks and threads
    block_x = te.thread_axis('blockIdx.x')
    thread_x = te.thread_axis((0, num_thread_x), 'threadIdx.x')
    thread_y = te.thread_axis((0, num_thread_y), 'threadIdx.y')
    thread_z = te.thread_axis((0, num_thread_z), 'threadIdx.z')

    # ######## Vthreads
    vthread_x = te.thread_axis((0, num_vthread_x), 'vthread', name='vthread_x')
    vthread_y = te.thread_axis((0, num_vthread_y), 'vthread', name='vthread_y')
    vthread_z = te.thread_axis((0, num_vthread_z), 'vthread', name='vthread_z')

    # ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    c_outer, thx = s[layer_output_dict['Layer_1']].split(c, factor=num_thread_x)
    recompute, reuse = s[layer_output_dict['Layer_1']].split(c_outer, factor=intermediate_reuse)
    ho, wo, h_tile, w_tile = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    thy, h_tile = s[layer_output_dict['Layer_1']].split(h_tile, nparts=num_thread_y)
    thz, h_tile = s[layer_output_dict['Layer_1']].split(h_tile, nparts=num_thread_z)
    vthy, w_tile = s[layer_output_dict['Layer_1']].split(w_tile, nparts=num_vthread_y)
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, recompute, reuse, vthy, thz, thy, thx, h_tile, w_tile)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, recompute)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vthy, vthread_y)
    s[layer_output_dict['Layer_1']].bind(reuse, vthread_x)
    s[layer_output_dict['Layer_1']].bind(thz, thread_z)
    s[layer_output_dict['Layer_1']].bind(thy, thread_y)
    s[layer_output_dict['Layer_1']].bind(thx, thread_x)

    # ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], thx)
    xocc, xicc = s[OL].split(s[OL].op.reduce_axis[0], factor=num_thread_x)
    xoicc, xiicc = s[OL].split(xicc, factor=reduce_split)
    n, h, w, oc = s[OL].op.axis
    s[OL].reorder(n, xocc, xoicc, h, w, oc, xiicc)

    if bn_relu[1]:
        s[BiasL_2].compute_at(s[layer_output_dict['Layer_1']], thx)

    # ######## Shared 1by1 filter
    s[FS_2].compute_at(s[OL], xoicc)
    h1, w1, i1, o1 = s[FS_2].op.axis
    io = s[FS_2].fuse(i1, o1)
    io, iox = s[FS_2].split(io, factor=num_thread_x * 4)
    ioy, io = s[FS_2].split(io, nparts=num_thread_y)
    iox, io4 = s[FS_2].split(iox, factor=4)
    s[FS_2].reorder(h1, w1, io, ioy, iox, io4)
    s[FS_2].bind(iox, thread_x)
    s[FS_2].bind(ioy, thread_y)
    s[FS_2].vectorize(io4)

    # ######## Intermediate output in shared memory
    s[layer_output_dict['Layer_0']].compute_at(s[OL], xocc)
    n, h, w, c = s[layer_output_dict['Layer_0']].op.axis
    inter_co, inter_ci = s[layer_output_dict['Layer_0']].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[layer_output_dict['Layer_0']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    h_step, w_step, h_step_tile, w_step_tile = s[layer_output_dict['Layer_0']].tile(h_tile, w_tile, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[layer_output_dict['Layer_0']].reorder(n, ho, wo, inter_co, h_step, w_step, h_step_tile, w_step_tile, inter_ci)
    vthz = s[layer_output_dict['Layer_0']].fuse(h_step, w_step)
    s[layer_output_dict['Layer_0']].bind(h_step_tile, thread_z)
    s[layer_output_dict['Layer_0']].bind(w_step_tile, thread_y)
    s[layer_output_dict['Layer_0']].bind(inter_ci, thread_x)
    s[layer_output_dict['Layer_0']].bind(vthz, vthread_z)

    ######## Intermediate output local accumulator
    s[DepthwiseLocalAccumulator].compute_at(s[layer_output_dict['Layer_0']], inter_ci)
    ry, rx = s[DepthwiseLocalAccumulator].op.reduce_axis
    n, h, w, c = s[DepthwiseLocalAccumulator].op.axis
    s[DepthwiseLocalAccumulator].reorder(n, c, ry, rx, h, w)

    if bn_relu[0]:
        s[BiasL_1].compute_at(s[layer_output_dict['Layer_0']], inter_ci)

    # ######## Depthwise filter
    s[FL_1].compute_at(s[layer_output_dict['Layer_0']], inter_co)
    # h, w, i, o = s[FL_1].op.axis
    # io = s[FL_1].fuse(i, o)
    # s[FL_1].bind(io, thread_x)

    # ######## Shared Input
    s[PaddedSharedInput].compute_at(s[layer_output_dict['Layer_0']], inter_co)
    n, h, w, c = s[PaddedSharedInput].op.axis
    co, ci = s[PaddedSharedInput].split(c, factor=num_thread_x)
    ho, wo, h_tile, w_tile = s[PaddedSharedInput].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    s[PaddedSharedInput].reorder(co, n, ho, wo, h_tile, w_tile, ci)
    s[PaddedSharedInput].bind(h_tile, thread_z)
    s[PaddedSharedInput].bind(w_tile, thread_y)
    s[PaddedSharedInput].bind(ci, thread_x)

    return s
