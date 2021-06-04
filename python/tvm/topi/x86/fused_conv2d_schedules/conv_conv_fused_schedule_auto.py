from tvm import te
from tvm.topi.utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm

def schedule_conv_conv_fused_nchwc_auto_search(cfg, outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, _, hasPaddedInput = get_stages_and_cfgs(s, outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']
    axis = ['oc', 'ic', 'h', 'w', 'root'][cfg['bind_axis'].val]

    ######## Final output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i_1 = cfg['split_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_1']].op.reduce_axis
    ic_chunk_o_1, ic_chunk_i = cfg['split_0_c'].apply(s, layer_output_dict['Layer_1'], ic_chunk)
    ht, ho_1, h = cfg['split_h'].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo_1, w = cfg['split_w'].apply(s, layer_output_dict['Layer_1'], w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    cfg.define_reorder('reorder_1_outer', [oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1], policy='candidate',
                        candidate=[[oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1], [oc_chunk_i_1, ho_1, ic_chunk_o_1, wo_1], [oc_chunk_i_1, ho_1, wo_1, ic_chunk_o_1],
                                    [ho_1, oc_chunk_i_1, ic_chunk_o_1, wo_1], [ho_1, oc_chunk_i_1, wo_1, ic_chunk_o_1], [ho_1, wo_1, oc_chunk_i_1, ic_chunk_o_1],
                                    [ic_chunk_o_1, oc_chunk_i_1, ho_1, wo_1], [ic_chunk_o_1, ho_1, oc_chunk_i_1, wo_1], [ic_chunk_o_1, ho_1, wo_1, oc_chunk_i_1],
                                    [ho_1, ic_chunk_o_1, oc_chunk_i_1, wo_1], [ho_1, ic_chunk_o_1, wo_1, oc_chunk_i_1], [ho_1, wo_1, ic_chunk_o_1, oc_chunk_i_1]])
    cfg['reorder_1_outer'].apply(s, layer_output_dict['Layer_1'], [oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_1'].H == 1 and filters_cfg['Layer_1'].W == 1 and \
            filters_cfg['Layer_1'].stride_h == 1 and filters_cfg['Layer_1'].stride_w == 1)) and \
        (cfg['split_h'].size[-2] > 1 and cfg['split_w'].size[-1] == outputs_cfg['Layer_1'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> rci
                                                oc.dom.extent,                      # n of brgemm   -> ki
                                                cfg['split_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_1'].W,           #               -> rx
                                                filters_cfg['Layer_1'].H,           #               -> ry
                                                cfg['split_0_c'].size[-1],    #               -> rco_i TODO: Not necessary so, can be further split, e.g. calculate 4 in the
                                                                                    #                               first stage and calculate 2 twice in the second stage

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[layer_output_dict['Layer_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    if axis == 'w':
        bind_axis = wo_1
    elif axis == 'h':
        bind_axis = ho_1
    elif axis == 'oc':
        bind_axis = oc_chunk_i_1
    elif axis == 'ic':
        bind_axis = ic_chunk_o_1
    else:
        bind_axis = fused_blx
    if hasPaddedInput[1]:
        s[stage_dict['PaddedInput_1']].compute_at(s[layer_output_dict['Layer_1']], bind_axis)
    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], bind_axis)

    ######## Intermediate output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    ho, h = s[stage_dict['Output_0']].split(h, factor=cfg['split_h'].size[-1])
    wo, w = s[stage_dict['Output_0']].split(w, factor=cfg['split_w'].size[-1])
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_0_rc'].apply(s, layer_output_dict['Layer_0'], ic_chunk)
    s[layer_output_dict['Layer_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)

    cfg.define_reorder('reorder_0_outer', [oc_chunk, ic_chunk_o, ho, wo], policy='candidate',
                        candidate=[[oc_chunk, ic_chunk_o, ho, wo], [oc_chunk, ho, ic_chunk_o, wo], [oc_chunk, ho, wo, ic_chunk_o],
                                    [ho, oc_chunk, ic_chunk_o, wo], [ho, oc_chunk, wo, ic_chunk_o], [ho, wo, oc_chunk, ic_chunk_o],
                                    [ic_chunk_o, oc_chunk, ho, wo], [ic_chunk_o, ho, oc_chunk, wo], [ic_chunk_o, ho, wo, oc_chunk],
                                    [ho, ic_chunk_o, oc_chunk, wo], [ho, ic_chunk_o, wo, oc_chunk], [ho, wo, ic_chunk_o, oc_chunk]])
    cfg['reorder_0_outer'].apply(s, layer_output_dict['Layer_0'], [oc_chunk, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (cfg['split_h'].size[-2] > 1 and cfg['split_w'].size[-1] == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> ic
                                                oc.dom.extent,                      # n of brgemm   -> oc
                                                cfg['split_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_0'].W,           #               -> rx
                                                filters_cfg['Layer_0'].H,           #               -> ry
                                                cfg['split_0_rc'].size[-1],   #              -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)
    s[layer_output_dict['Layer_0']].tensorize(tensorize_axis, libxsmm_tensorize)
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_at(s[layer_output_dict['Layer_1']], bind_axis)

    s = s.normalize()

    return s

def schedule_conv_conv_fused_nchwc_auto_inference(cfg, outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, post_ops, hasPaddedInput = get_stages_and_cfgs(s, outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']
    axis = ['oc', 'ic', 'h', 'w', 'root'][cfg['bind_axis'].val]

    ######## Final output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i_1 = cfg['split_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=cfg['split_h'].size[-2] * cfg['split_h'].size[-1], y_factor=cfg['split_w'].size[-2] * cfg['split_w'].size[-1])
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i_1, h, w, oc) # Temporary
    s[layer_output_dict['Layer_1']].vectorize(oc)

    # Example: [2, 1, 3, 0] => ['h', 'ic', 'w', 'oc']
    # => Split 'h', and 'w' and 'oc' follow
    # => ['ho', (ic), 'h', 'w', 'oc'], compute at 'ho'

    # Consumer of the previous stage
    prev_consumer = layer_output_dict['Layer_1']

    # Get the axis labels and find where is the reduce axis
    perm = cfg['reorder_1_outer'].perm
    axis_labels = [['oc', 'ic', 'h', 'w'][i] for i in perm]
    ic_idx = axis_labels.index('ic')

    # reorder the axes
    axes = []
    for label in axis_labels[0:ic_idx]:
        if label == 'h':
            ho_1, h = s[layer_output_dict['Layer_1']].split(h, cfg['split_h'].size[-1])
            axes.append(ho_1)
        if label == 'w':
            wo_1, w = s[layer_output_dict['Layer_1']].split(w, cfg['split_w'].size[-1])
            axes.append(wo_1)
        if label == 'oc':
            axes.append(oc_chunk_i_1)
    relu_compute_axis = wt if len(axes) == 0 else axes[-1] # compute relu at the axis right before the reduce axis
    if ic_idx < axis_labels.index('oc'):
        axes.append(oc_chunk_i_1)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, *axes, h, w, oc)

    # If has post ops
    if post_ops[1]:
        s[stage_dict['Output_1']].compute_at(s[layer_output_dict['Layer_1']], relu_compute_axis)
        _, oc_chunk_i_1, h, w, oc = s[stage_dict['Output_1']].op.axis
        if post_ops[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    ic_chunk, ry, rx, ic = s[stage_dict['Output_1']].op.reduce_axis
    ic_chunk_o_1, ic_chunk_i = cfg['split_0_c'].apply(s, stage_dict['Output_1'], ic_chunk)
    
    # Split h and w if they're not yet split
    axes = []
    for label in axis_labels[ic_idx:]: # ic should be the first axis here
        if label == axis:
            prev_consumer = stage_dict['Output_1']
        if label == 'h':
            ho_1, h = s[stage_dict['Output_1']].split(h, cfg['split_h'].size[-1])
            axes.append(ho_1)
        if label == 'w':
            wo_1, w = s[stage_dict['Output_1']].split(w, cfg['split_w'].size[-1])
            axes.append(wo_1)
        if label == 'oc':
            axes.append(oc_chunk_i_1)
        if label == 'ic':
            axes.append(ic_chunk_o_1)

    s[stage_dict['Output_1']].reorder(*axes, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_1'].H == 1 and filters_cfg['Layer_1'].W == 1 and \
            filters_cfg['Layer_1'].stride_h == 1 and filters_cfg['Layer_1'].stride_w == 1)) and \
        (cfg['split_h'].size[-2] > 1 and cfg['split_w'].size[-1] == outputs_cfg['Layer_1'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> rci
                                                oc.dom.extent,                      # n of brgemm   -> ki
                                                cfg['split_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_1'].W,           #               -> rx
                                                filters_cfg['Layer_1'].H,           #               -> ry
                                                cfg['split_0_c'].size[-1],    #               -> rco_i TODO: Not necessary so, can be further split, e.g. calculate 4 in the
                                                                                    #                               first stage and calculate 2 twice in the second stage

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[stage_dict['Output_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    if axis == 'w':
        bind_axis = wo_1
    elif axis == 'h':
        bind_axis = ho_1
    elif axis == 'oc':
        bind_axis = oc_chunk_i_1
    elif axis == 'ic':
        bind_axis = ic_chunk_o_1
    else:
        bind_axis = fused_blx
    if hasPaddedInput[1]:
        s[stage_dict['PaddedInput_1']].compute_at(s[prev_consumer], bind_axis)

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[prev_consumer], bind_axis)
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    s[layer_output_dict['Layer_0']].vectorize(oc)
    if post_ops[0]:
        s[stage_dict['Output_0']].compute_at(s[prev_consumer], bind_axis)
        _, oc_chunk, h, w, oc = s[stage_dict['Output_0']].op.axis
        if post_ops[0] != 'bias':
            s[stage_dict['Output_0_BiasAdd']].compute_inline()

    ho, h = s[stage_dict['Output_0']].split(h, factor=cfg['split_h'].size[-1])
    wo, w = s[stage_dict['Output_0']].split(w, factor=cfg['split_w'].size[-1])
    ic_chunk, ry, rx, ic = s[stage_dict['Output_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_0_rc'].apply(s, stage_dict['Output_0'], ic_chunk)
    s[stage_dict['Output_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    cfg['reorder_0_outer'].apply(s, stage_dict['Output_0'], [oc_chunk, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (cfg['split_h'].size[-2] > 1 and cfg['split_w'].size[-1] == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> ic
                                                oc.dom.extent,                      # n of brgemm   -> oc
                                                cfg['split_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_0'].W,           #               -> rx
                                                filters_cfg['Layer_0'].H,           #               -> ry
                                                cfg['split_0_rc'].size[-1],   #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)
    s[stage_dict['Output_0']].tensorize(tensorize_axis, libxsmm_tensorize)
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_at(s[prev_consumer], bind_axis)

    s = s.normalize()

    return s
