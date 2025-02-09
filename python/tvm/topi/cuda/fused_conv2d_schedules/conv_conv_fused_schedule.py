from tvm import te
from tvm.topi.utils import get_stages_and_cfgs

def schedule_conv_conv_fused_nhwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, param_dict, _, bn_relu, hasPaddedInput = get_stages_and_cfgs(s, outs)

    ######## Searchable parameters
    # --------------------
    thread_step = 8
    reduce_split1 = 1
    reduce_split2 = 32
    oc_split = 32
    num_thread_z = num_thread_y = 2
    num_thread_x = 32
    num_vthread_z_2 = num_vthread_y_2 =2
    num_vthread_x_2 = 1

    ######## Input data, weights, BN, etc
    # End
    if bn_relu[1]:
        s[stage_dict['Output_1']].set_scope('local')
        BiasL_2 = s.cache_read(param_dict['Bias_1'], 'local', [stage_dict['Output_1_BiasAdd']])
        OL = stage_dict['Output_1']
        if bn_relu[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    else:
        n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
        rc, _, _ = s[layer_output_dict['Layer_1']].op.reduce_axis
        rco, _ = s[layer_output_dict['Layer_1']].split(rc, reduce_split1 * reduce_split2)
        OL = s.rfactor(layer_output_dict['Layer_1'], rco)
        s[OL].set_scope('local')

    # Intermediate
    if bn_relu[0]: 
        s[stage_dict['Output_0_BiasAdd']].compute_inline()
        BiasL_1 = s.cache_read(param_dict['Bias_0'], 'local', [stage_dict['Output_0_BiasAdd']])

    s[stage_dict['Output_0']].set_scope('local') # Accumulate in local
    if hasPaddedInput[1]: # Skip this for now
        if bn_relu[0]:
            s[layer_output_dict['Layer_0']].compute_inline()
        s[stage_dict['FusedConv2D_PaddedInput_1']].set_scope('shared')
        stage_dict['Intermediate'] = stage_dict['FusedConv2D_PaddedInput_1'] # For disambiguity: 'FusedConv2D_PaddedInput_1' won't be used in scheduling
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

    ######## Blocks and threads
    block_x = te.thread_axis('blockIdx.x')
    thread_z = te.thread_axis((0, num_thread_z), 'threadIdx.z')
    thread_y = te.thread_axis((0, num_thread_y), 'threadIdx.y')
    thread_x = te.thread_axis((0, num_thread_x), 'threadIdx.x')

    ######## Vthreads
    # Output
    vthread_z_2 = te.thread_axis('vthread', name='vthread_z_2')
    vthread_y_2 = te.thread_axis('vthread', name='vthread_y_2')
    vthread_x_2 = te.thread_axis('vthread', name='vthread_x_2')
    # Intermediate output
    vthread_z_1 = te.thread_axis('vthread', name='vthread_z_1')
    vthread_y_1 = te.thread_axis('vthread', name='vthread_y_1')
    vthread_x_1 = te.thread_axis('vthread', name='vthread_x_1')
    # # Padded input
    # vthread_z_0 = te.thread_axis('vthread', name='vthread_z_0')
    # vthread_y_0 = te.thread_axis('vthread', name='vthread_y_0')
    # vthread_x_0 = te.thread_axis('vthread', name='vthread_x_0')

    ######## Global output
    n, h, w, c = s[layer_output_dict['Layer_1']].op.axis
    rco, = s[layer_output_dict['Layer_1']].op.reduce_axis
    # h, w
    ho, h = s[layer_output_dict['Layer_1']].split(h, factor=thread_step)
    vh, h = s[layer_output_dict['Layer_1']].split(h, nparts=num_vthread_z_2)
    th, h = s[layer_output_dict['Layer_1']].split(h, nparts=num_thread_z)

    wo, w = s[layer_output_dict['Layer_1']].split(w, factor=thread_step)
    vw, w = s[layer_output_dict['Layer_1']].split(w, nparts=num_vthread_y_2)
    tw, w = s[layer_output_dict['Layer_1']].split(w, nparts=num_thread_y)
    # c
    oc, c = s[layer_output_dict['Layer_1']].split(c, factor=oc_split)
    vc, c = s[layer_output_dict['Layer_1']].split(c, nparts=num_vthread_x_2)
    tc, c = s[layer_output_dict['Layer_1']].split(c, nparts=num_thread_x)
    # reorder and bind
    s[layer_output_dict['Layer_1']].reorder(n, ho, wo, oc,   rco,   vh, vw, vc,   th, tw, tc,   h, w, c) # reduce axis at the outer most
    if bn_relu[1]:
        s[BiasL_2].compute_at(s[layer_output_dict['Layer_1']], tc)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, ho, wo, oc)
    s[layer_output_dict['Layer_1']].bind(fused_blx, block_x)
    s[layer_output_dict['Layer_1']].bind(vh, vthread_z_2)
    s[layer_output_dict['Layer_1']].bind(vw, vthread_y_2)
    s[layer_output_dict['Layer_1']].bind(vc, vthread_x_2)
    s[layer_output_dict['Layer_1']].bind(th, thread_z)
    s[layer_output_dict['Layer_1']].bind(tw, thread_y)
    s[layer_output_dict['Layer_1']].bind(tc, thread_x)

    ######## Local output
    s[OL].compute_at(s[layer_output_dict['Layer_1']], tc)
    _, n, h, w, c = s[OL].op.axis
    ry, rx, rc = s[OL].op.reduce_axis
    orc, irc = s[OL].split(rc, factor=reduce_split2)
    s[OL].reorder(n, orc, h, w, c, ry, rx, irc)

    ######## Filter 2
    s[FS_2].compute_at(s[OL], orc)
    h, w, i, o = s[FS_2].op.axis
    fused = s[FS_2].fuse(h, w, i, o)
    tz, fused = s[FS_2].split(fused, nparts=num_thread_z)
    ty, fused = s[FS_2].split(fused, nparts=num_thread_y)
    tx, fused = s[FS_2].split(fused, nparts=num_thread_x)
    s[FS_2].bind(tz, thread_z)
    s[FS_2].bind(ty, thread_y)
    s[FS_2].bind(tx, thread_x)

    ######## Intermediate output in shared memory
    s[stage_dict['SharedInput_1']].compute_at(s[layer_output_dict['Layer_1']], tc)
    n, h, w, c = s[stage_dict['SharedInput_1']].op.axis
    vc, tc = s[stage_dict['SharedInput_1']].split(c, factor=num_thread_x)
    vw, tw = s[stage_dict['SharedInput_1']].split(w, factor=num_thread_y)
    vh, th = s[stage_dict['SharedInput_1']].split(h, factor=num_thread_z)
    s[stage_dict['SharedInput_1']].reorder(n, vh, vw, vc, th, tw, tc)
    if bn_relu[0]:
        s[BiasL_1].compute_at(s[stage_dict['SharedInput_1']], tc)
    s[stage_dict['SharedInput_1']].bind(vh, vthread_z_1)
    s[stage_dict['SharedInput_1']].bind(vw, vthread_y_1)
    s[stage_dict['SharedInput_1']].bind(vc, vthread_x_1)
    s[stage_dict['SharedInput_1']].bind(th, thread_z)
    s[stage_dict['SharedInput_1']].bind(tw, thread_y)
    s[stage_dict['SharedInput_1']].bind(tc, thread_x)

    ####### Intermediate output local accumulator
    s[stage_dict['Output_0']].compute_at(s[stage_dict['SharedInput_1']], tc)
    rc, ry, rx = s[stage_dict['Output_0']].op.reduce_axis
    orc, irc = s[stage_dict['Output_0']].split(rc, factor=num_thread_x)
    oirc, iirc = s[stage_dict['Output_0']].split(irc, factor=reduce_split1)
    n, h, w, c = s[stage_dict['Output_0']].op.axis
    c, tc = s[stage_dict['Output_0']].split(c, factor=num_thread_x)
    w, tw = s[stage_dict['Output_0']].split(w, factor=num_thread_y)
    h, th = s[stage_dict['Output_0']].split(h, factor=num_thread_z)
    s[stage_dict['Output_0']].reorder(n, orc, oirc, h, w, c, th, tw, tc, ry, rx, iirc)
    s[stage_dict['Output_0']].bind(tc, thread_x)
    s[stage_dict['Output_0']].bind(tw, thread_y)
    s[stage_dict['Output_0']].bind(th, thread_z)

    # ######## Filter 1
    s[FS_1].compute_at(s[stage_dict['Output_0']], orc)
    h, w, i, o = s[FS_1].op.axis
    fused = s[FS_1].fuse(h, w, i, o)
    tz, fused = s[FS_1].split(fused, nparts=num_thread_z)
    ty, fused = s[FS_1].split(fused, nparts=num_thread_y)
    tx, fused = s[FS_1].split(fused, nparts=num_thread_x)
    s[FS_1].bind(tz, thread_z)
    s[FS_1].bind(ty, thread_y)
    s[FS_1].bind(tx, thread_x)

    ####### Shared Input
    s[stage_dict['FusedConv2D_PaddedInput_0']].compute_at(s[stage_dict['Output_0']], orc)
    n, h, w, c = s[stage_dict['FusedConv2D_PaddedInput_0']].op.axis
    # vc, tc = s[stage_dict['FusedConv2D_PaddedInput_0']].split(c, factor=num_thread_x)
    # vh, vw, th, tw = s[stage_dict['FusedConv2D_PaddedInput_0']].tile(h, w, x_factor=num_thread_z, y_factor=num_thread_y)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].reorder(n, vh, vw, vc, th, tw, tc)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].bind(vh, vthread_z_0)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].bind(vw, vthread_y_0)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].bind(vc, vthread_x_0)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].bind(th, thread_z)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].bind(tw, thread_y)
    # s[stage_dict['FusedConv2D_PaddedInput_0']].bind(tc, thread_x)
    fused = s[stage_dict['FusedConv2D_PaddedInput_0']].fuse(n, h, w, c)
    tz, fused = s[stage_dict['FusedConv2D_PaddedInput_0']].split(fused, nparts=num_thread_z)
    ty, fused = s[stage_dict['FusedConv2D_PaddedInput_0']].split(fused, nparts=num_thread_y)
    tx, fused = s[stage_dict['FusedConv2D_PaddedInput_0']].split(fused, nparts=num_thread_x)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(tz, thread_z)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(ty, thread_y)
    s[stage_dict['FusedConv2D_PaddedInput_0']].bind(tx, thread_x)

    return s
