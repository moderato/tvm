from tvm import te
from tvm.topi.utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm
from .schedule_utils import get_layer_cfg

# Currently, schedule with relu is not able to get the best search result. Use the non-relu schedule to search and apply the result to the relu schedule for inference.
# Separate search and inference, and give a little twist to the inference schedule.
def schedule_conv_depth_fused_nchwc_auto_search(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, _, hasPaddedInput = get_stages_and_cfgs(s, outs)
    inputs_cfg, filters_cfg, outputs_cfg = get_layer_cfg()
    axis = ['oc', 'h', 'w', 'root'][cfg['bind_axis'].val]

    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = cfg['split_1_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ht, ho, h = cfg['split_h'].apply(s, layer_output_dict['Layer_1'], h)
    wt, wo, w = cfg['split_w'].apply(s, layer_output_dict['Layer_1'], w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, ho, wo, h, w, oc)
    s[layer_output_dict['Layer_1']].vectorize(oc)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    cfg.define_reorder('reorder_outer', [oc_chunk_i, ho, wo], policy='candidate',
                        candidate=[[oc_chunk_i, ho, wo], [ho, oc_chunk_i, wo], [oc_chunk_i, ho, wo]])
    cfg['reorder_outer'].apply(s, layer_output_dict['Layer_1'], [oc_chunk_i, ho, wo])

    if axis == 'w':
        bind_axis = wo
    elif axis == 'h':
        bind_axis = ho
    elif axis == 'oc':
        bind_axis = oc_chunk_i
    else:
        bind_axis = fused_blx
    if hasPaddedInput[1]:
        s[stage_dict['FusedConv2D_PaddedInput_1']].compute_at(s[layer_output_dict['Layer_1']], bind_axis)
    s[layer_output_dict['Layer_0']].compute_at(s[layer_output_dict['Layer_1']], bind_axis)

    ######## Intermediate output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    ho, h = s[stage_dict['Output_0']].split(h, factor=cfg['split_h'].size[-1])
    wo, w = s[stage_dict['Output_0']].split(w, factor=cfg['split_w'].size[-1])
    ic_chunk, ry, rx, ic = s[layer_output_dict['Layer_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_0_rc'].apply(s, layer_output_dict['Layer_0'], ic_chunk)
    s[layer_output_dict['Layer_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)

    cfg.define_reorder('reorder_0_outer', [oc_chunk, ho, wo], policy='candidate',
                        candidate=[[oc_chunk, ho, wo], [ho, oc_chunk, wo], [ho, wo, oc_chunk]])
    cfg['reorder_0_outer'].apply(s, layer_output_dict['Layer_0'], [oc_chunk, ho, wo])

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
        s[stage_dict['FusedConv2D_PaddedInput_0']].compute_at(s[layer_output_dict['Layer_1']], bind_axis)

    s = s.normalize()

    return s


def schedule_conv_depth_fused_nchwc_auto_inference(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, post_ops, hasPaddedInput = get_stages_and_cfgs(s, outs)
    inputs_cfg, filters_cfg, outputs_cfg = get_layer_cfg()
    axis = ['oc', 'h', 'w'][cfg['bind_axis'].val]

    ######## Final output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = cfg['split_0_c'].apply(s, layer_output_dict['Layer_1'], oc_chunk)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=cfg['split_h'].size[-2] * cfg['split_h'].size[-1], y_factor=cfg['split_w'].size[-2] * cfg['split_w'].size[-1])
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, h, w, oc) # Temporary
    s[layer_output_dict['Layer_1']].vectorize(oc)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    # Example: [2, 1, 3, 0] => ['h', 'ic', 'w', 'oc']
    # => Split 'h', and 'w' and 'oc' follow
    # => ['ho', (ic), 'h', 'w', 'oc'], compute at 'ho'

    # Consumer of the previous stage
    prev_consumer = layer_output_dict['Layer_1']

    # Get the axis labels and find where is the reduce axis
    perm = cfg['reorder_outer'].perm
    axis_labels = [['oc', 'h', 'w'][i] for i in perm]

    # If has post ops
    if post_ops[1]:
        s[stage_dict['Output_1']].compute_at(s[layer_output_dict['Layer_1']], wt)
        _, oc_chunk_i, h, w, oc = s[stage_dict['Output_1']].op.axis
        if post_ops[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    
    # Split h and w if they're not yet split
    axes = []
    axes_2 = []
    for label in axis_labels:
        if label == axis:
            prev_consumer = stage_dict['Output_1']
        if label == 'h':
            ho, h = s[stage_dict['Output_1']].split(h, cfg['split_h'].size[-1])
            axes.append(ho)
            axes_2.append(h)
        if label == 'w':
            wo, w = s[stage_dict['Output_1']].split(w, cfg['split_w'].size[-1])
            axes.append(wo)
            axes_2.append(w)
        if label == 'oc':
            axes.append(oc_chunk_i)

    s[stage_dict['Output_1']].reorder(*axes, *axes_2, h, w, oc)
    s[stage_dict['Output_1']].vectorize(oc)

    if axis == 'w':
        bind_axis = wo
    elif axis == 'h':
        bind_axis = ho
    elif axis == 'oc':
        bind_axis = oc_chunk_i
    else:
        bind_axis = fused_blx

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[prev_consumer], bind_axis)
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    s[layer_output_dict['Layer_0']].vectorize(oc)
    if post_ops[0]:
        s[stage_dict['Output_0']].compute_at(s[prev_consumer], bind_axis)
        _, oc_chunk, h, w, oc = s[stage_dict['Output_0']].op.axis
        if post_ops[0] != 'bias':
            s[stage_dict['Output_0_BiasAdd']].compute_inline()

    ho, h = s[stage_dict['Output_0']].split(h, factor=cfg['split_1_h'].size[-1])
    wo, w = s[stage_dict['Output_0']].split(w, factor=cfg['split_1_w'].size[-1])
    ic_chunk, ry, rx, ic = s[stage_dict['Output_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = cfg['split_0_rc'].apply(s, stage_dict['Output_0'], ic_chunk)
    s[stage_dict['Output_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)
    cfg['reorder_0_outer'].apply(s, stage_dict['Output_0'], [oc_chunk, ic_chunk_o, ho, wo])

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (cfg['split_1_h'].size[-2] > 1 and cfg['split_1_w'].size[-1] == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
        # print('small: bind to h')
        tensorize_axis = h
        block_output_height = cfg['split_1_h'].size[-1]
    else:
        # print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,                      # k of brgemm   -> ic
                                                oc.dom.extent,                      # n of brgemm   -> oc
                                                cfg['split_1_w'].size[-1],    # m of brgemm   -> wi
                                                filters_cfg['Layer_0'].W,           #               -> rx
                                                filters_cfg['Layer_0'].H,           #               -> ry
                                                cfg['split_0_rc'].size[-1],   #               -> rco_i

                                                block_output_height,                #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)
    s[stage_dict['Output_0']].tensorize(tensorize_axis, libxsmm_tensorize)
    if hasPaddedInput[0]:
        s[stage_dict['FusedConv2D_PaddedInput_0']].compute_at(s[prev_consumer], bind_axis)

    return s
