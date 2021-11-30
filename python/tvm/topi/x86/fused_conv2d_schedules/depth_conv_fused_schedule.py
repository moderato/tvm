from tvm import te
from tvm.topi.utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm
from .schedule_utils import get_layer_cfg

def schedule_depth_conv_fused_nchwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, post_ops, hasPaddedInput = get_stages_and_cfgs(s, outs)
    inputs_cfg, filters_cfg, outputs_cfg = get_layer_cfg()

    # Searchable parameters
    # --------------------
    output_step_tile_size_h = 7
    output_step_tile_size_w = 56
    step_num_h = 2
    step_num_w = 1
    reduce_split = 4
    oc_chunk_split = 4
    # ---
    # output_step_tile_size_h = 2
    # output_step_tile_size_w = 2
    # step_num_h = 2
    # step_num_w = 2
    # reduce_split = 2
    # oc_chunk_split = 1
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    # --------------------

    ######## Final output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i = s[layer_output_dict['Layer_1']].split(oc_chunk, factor=oc_chunk_split)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i, h, w, oc)
    if post_ops[1]:
        s[stage_dict['Output_1']].compute_at(s[layer_output_dict['Layer_1']], wt)
        _, oc_chunk_i, h, w, oc = s[stage_dict['Output_1']].op.axis
        if post_ops[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    ho, wo, h, w = s[stage_dict['Output_1']].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w) ####
    ic_chunk, ry, rx, ic = s[stage_dict['Output_1']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = s[stage_dict['Output_1']].split(ic_chunk, factor=reduce_split)
    s[stage_dict['Output_1']].reorder(ho, ic_chunk_o, wo, oc_chunk_i, h, ic_chunk_i, ry, rx, w, oc, ic)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)

    # Temporary skip the case of 1x1 stride > 1
    if (((filters_cfg['Layer_1'].H == 1 and filters_cfg['Layer_1'].W == 1 and \
            filters_cfg['Layer_1'].stride_h == 1 and filters_cfg['Layer_1'].stride_w == 1)) and \
        (step_num_h > 1 and output_step_tile_size_w == outputs_cfg['Layer_1'].W)): # HM > 1 & WI = OW (small W)
        print('small: bind to h')
        tensorize_axis = h
        block_output_height = output_step_tile_size_h
    else:
        print('big: bind to ic_chunk_i')
        tensorize_axis = ic_chunk_i
        block_output_height = 1

    libxsmm_tensorize = intrin_libxsmm_brgemm(
                                                ic.dom.extent,              # k of brgemm   -> ic
                                                oc.dom.extent,              # n of brgemm   -> oc
                                                output_step_tile_size_w,    # m of brgemm   -> w
                                                filters_cfg['Layer_1'].W,   #               -> rx
                                                filters_cfg['Layer_1'].H,   #               -> ry
                                                reduce_split,               #               -> ic_chunk_i

                                                block_output_height,        #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)

    s[stage_dict['Output_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    ######## Intermediate output
    s[layer_output_dict['Layer_0']].compute_at(s[stage_dict['Output_1']], wo)
    _, _, h, w, c_vec = s[layer_output_dict['Layer_0']].op.axis
    if post_ops[0]:
        s[layer_output_dict['Layer_0']].vectorize(c_vec)
        s[stage_dict['Output_0']].compute_at(s[stage_dict['Output_1']], wo)
        if post_ops[0] != 'bias':
            s[stage_dict['Output_0_BiasAdd']].compute_inline()
    n, c_chunk, h, w, c_vec = s[stage_dict['Output_0']].op.axis
    ry, rx = s[stage_dict['Output_0']].op.reduce_axis
    s[stage_dict['Output_0']].reorder(n, c_chunk, h, w, ry, rx, c_vec)
    s[stage_dict['Output_0']].vectorize(c_vec)

    ######## PaddedInput 0
    if hasPaddedInput[0]:
        s[stage_dict['FusedConv2D_PaddedInput_0']].compute_inline()

    s = s.normalize()

    return s