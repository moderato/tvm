from tvm import te
from tvm.topi.utils import get_stages_and_cfgs

def schedule_conv_conv_fused_nhwc(outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, param_dict, _, bn_relu, hasPaddedInput = get_stages_and_cfgs(s, outs)

    ######## Searchable parameters
    # --------------------
    thread_step = 1
    reduce_split1 = 4
    reduce_split2 = 4
    intermediate_reuse = 2
    num_thread_z = num_thread_y = 2
    num_vthread_z_2 = num_vthread_y_2 = 2
    num_thread_x = 32
    # --------------------
    thread_tile = num_thread_y * num_vthread_y_2
    blk_tile = thread_step * thread_tile
    # --------------------
    # (4x4x32, blk tile) -> (4x4, thread tile) -> (2x2, vthread tile)
    # (6x6x32, blk tile) -> (6x6, thread tile) -> (2x2, vthread tile)

    ######## Input data, weights, BN, etc
    # End
    if bn_relu[1]:
        s[stage_dict['Output_1']].set_scope('local')
        BiasL_2 = s.cache_read(param_dict['Bias_1'], 'local', [stage_dict['Output_1_BiasAdd']])
        OL = stage_dict['Output_1']
        if bn_relu[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    else:
        OL = s.cache_write(layer_output_dict['Layer_1'], 'local')

    # Intermediate
    if bn_relu[0]: 
        s[stage_dict['Output_0_BiasAdd']].compute_inline()
        BiasL_1 = s.cache_read(param_dict['Bias_0'], 'local', [stage_dict['Output_0_BiasAdd']])

    s[stage_dict['Output_0']].set_scope('local') # Accumulate in local
    if hasPaddedInput[1]:
        if bn_relu[0]:
            s[layer_output_dict['Layer_0']].compute_inline()
        s[stage_dict['FusedConv2D_PaddedInput_1']].set_scope('shared')
        stage_dict['SharedInput_1'] = stage_dict['FusedConv2D_PaddedInput_1'] # For disambiguity: 'FusedConv2D_PaddedInput_1' won't be used in scheduling
    else:
        if bn_relu[0]:
            s[layer_output_dict['Layer_0']].set_scope('shared') # Results of ReLU go to shared
            stage_dict['SharedInput_1'] = layer_output_dict['Layer_0']
        else:
            stage_dict['SharedInput_1'] = s.cache_read(stage_dict['Output_0'], 'shared', [OL]) # Move conv result from local to shared
    FS_2 = s.cache_read(param_dict['Filter_1'], 'shared', [OL])

    # Beginning
    if hasPaddedInput[0]:
        s[stage_dict['FusedConv2D_PaddedInput_0']].set_scope('shared')
    FS_1 = s.cache_read(param_dict['Filter_0'], 'shared', [stage_dict['Output_0']])

    # from pprint import pprint
    # pprint(stage_dict)
    # print('&&&')
    # print(layer_output_dict)
    # print('&&&')
    # pprint(param_dict)

    ######## Blocks and threads
    block_x = te.thread_axis('blockIdx.x')
    thread_x = te.thread_axis((0, num_thread_x), 'threadIdx.x')
    thread_y = te.thread_axis((0, num_thread_y), 'threadIdx.y')
    thread_z = te.thread_axis((0, num_thread_z), 'threadIdx.z')

    ######## Vthreads
    vthread_y_2 = te.thread_axis('vthread', name='vthread_y_2')
    vthread_z_2 = te.thread_axis('vthread', name='vthread_z_2')
    # --
    vthread_x_1 = te.thread_axis('vthread', name='vthread_x_1')
    vthread_y_1 = te.thread_axis('vthread', name='vthread_y_1')
    vthread_z_1 = te.thread_axis('vthread', name='vthread_z_1')
    # --
    vthread_y_0 = te.thread_axis('vthread', name='vthread_y_0')
    vthread_z_0 = te.thread_axis('vthread', name='vthread_z_0')

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    # c
    c, thx = s[layer_output_dict['Layer_1']].split(c, factor=num_thread_x)
    oc, ic = s[layer_output_dict['Layer_1']].split(c, factor=intermediate_reuse)
    # h, w
    ho, wo, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=blk_tile, y_factor=blk_tile)
    h, w, thz, thy = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=num_thread_y, y_factor=num_thread_z)
    h, w, vthz, vthy = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=num_vthread_y_2, y_factor=num_vthread_z_2)
    # reorder and bind
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, oc,   ic, h, w,   vthz, vthy, thz, thy, thx)
    if bn_relu[1]:
        s[BiasL_2].compute_at(s[layer_output_dict['Layer_1']], thx)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, oc)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vthz, vthread_z_2)
    s[layer_output_dict['Layer_1']].bind(vthy, vthread_y_2)
    s[layer_output_dict['Layer_1']].bind(thz, thread_z)
    s[layer_output_dict['Layer_1']].bind(thy, thread_y)
    s[layer_output_dict['Layer_1']].bind(thx, thread_x)

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], thx) # thx <----- Must be so! Same below.
    rc, ry, rx = s[OL].op.reduce_axis
    n, h, w, c = s[OL].op.axis
    orc, irc = s[OL].split(rc, factor=num_thread_x)
    oirc, iirc = s[OL].split(irc, factor=reduce_split2)
    s[OL].reorder(n, orc, oirc, h, w, iirc, ry, rx, c)

    ######## Filter 2
    s[FS_2].compute_at(s[OL], orc)
    h, w, i, o = s[FS_2].op.axis
    oo, io = s[FS_2].split(o, nparts=num_thread_x)
    s[FS_2].bind(oo, thread_x)
    s[FS_2].vectorize(io)
    oi, ii = s[FS_2].split(i, factor=num_thread_y)
    _, oi = s[FS_2].split(oi, factor=num_thread_z)
    s[FS_2].bind(ii, thread_y)
    s[FS_2].bind(oi, thread_z)

    # ######## Intermediate output in shared memory
    s[stage_dict['SharedInput_1']].compute_at(s[layer_output_dict['Layer_1']], thx)
    n, h, w, c = s[stage_dict['SharedInput_1']].op.axis
    vthx, thx = s[stage_dict['SharedInput_1']].split(c, factor=num_thread_x)
    vthz, vthy, thz, thy = s[stage_dict['SharedInput_1']].tile(h, w, x_factor=num_thread_y, y_factor=num_thread_z)
    s[stage_dict['SharedInput_1']].reorder(n, vthz, vthy, vthx, thz, thy, thx)
    if bn_relu[0]:
        s[BiasL_1].compute_at(s[stage_dict['SharedInput_1']], thx)
    s[stage_dict['SharedInput_1']].bind(vthz, vthread_z_1)
    s[stage_dict['SharedInput_1']].bind(vthy, vthread_y_1)
    s[stage_dict['SharedInput_1']].bind(vthx, vthread_x_1)
    s[stage_dict['SharedInput_1']].bind(thz, thread_z)
    s[stage_dict['SharedInput_1']].bind(thy, thread_y)
    s[stage_dict['SharedInput_1']].bind(thx, thread_x)

    ####### Intermediate output local accumulator
    s[stage_dict['Output_0']].compute_at(s[stage_dict['SharedInput_1']], thx)
    rc, ry, rx = s[stage_dict['Output_0']].op.reduce_axis
    orc, irc = s[stage_dict['Output_0']].split(rc, factor=num_thread_x)
    oirc, iirc = s[stage_dict['Output_0']].split(irc, factor=reduce_split1)
    n, h, w, c = s[stage_dict['Output_0']].op.axis
    _, thx = s[stage_dict['Output_0']].split(c, factor=num_thread_x)
    s[stage_dict['Output_0']].reorder(n, orc, oirc, h, w, iirc, ry, rx, thx)
    s[stage_dict['Output_0']].bind(thx, thread_x)

    ######## Filter 1
    s[FS_1].compute_at(s[stage_dict['Output_0']], oirc)
    h, w, i, o = s[FS_1].op.axis
    oo, io = s[FS_1].split(o, nparts=num_thread_x)
    s[FS_1].bind(oo, thread_x)
    s[FS_1].vectorize(io)
    oi, ii = s[FS_1].split(i, factor=num_thread_y)
    _, oi = s[FS_1].split(oi, factor=num_thread_z)
    s[FS_1].bind(ii, thread_y)
    s[FS_1].bind(oi, thread_z)

    ####### Shared Input
    s[stage_dict['FusedConv2D_PaddedInput_0']].compute_at(s[stage_dict['Output_0']], orc)
    n, h, w, c = s[stage_dict['FusedConv2D_PaddedInput_0']].op.axis
    oc, thx = s[stage_dict['FusedConv2D_PaddedInput_0']].split(c, factor=num_thread_x)
    vthz, vthy, thz, thy = s[stage_dict['FusedConv2D_PaddedInput_0']].tile(h, w, x_factor=num_thread_z, y_factor=num_thread_y)
    s[stage_dict['FusedConv2D_PaddedInput_0']].reorder(n, oc, vthz, vthy, thz, thy, thx)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(vthz, vthread_z_0)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(vthy, vthread_y_0)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(thz, thread_z)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(thy, thread_y)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(thx, thread_x)

    return s
