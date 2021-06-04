from tvm import te
from tvm.topi.utils import get_stages_and_cfgs
from .libxsmm_intrin import intrin_libxsmm_brgemm

def schedule_conv_conv_fused_nchwc(outs, *args, **kwargs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    stage_dict, layer_output_dict, _, _, post_ops, hasPaddedInput = get_stages_and_cfgs(s, outs)
    inputs_cfg = kwargs['inputs_cfg']
    filters_cfg = kwargs['filters_cfg']
    outputs_cfg = kwargs['outputs_cfg']

    ######## Searchable parameters
    # -------------------- res_3x
    output_step_tile_size_h = 1
    output_step_tile_size_w = 28
    step_num_h = 28
    step_num_w = 2
    output_tile_0_h = 14
    output_tile_0_w = 2
    c_split2 = 4
    reduce_split1 = 2
    reduce_split2 = 2
    # -------------------- conv3_conv3_test_tiny
    # output_step_tile_size_h = 1
    # output_step_tile_size_w = 8
    # step_num_h = 2
    # step_num_w = 2
    # output_tile_0_h = 2
    # output_tile_0_w = 1
    # c_split2 = 4
    # reduce_split1 = 2
    # reduce_split2 = 2
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    # --------------------

    ######## Global output
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_1']].op.axis
    oc_chunk_o, oc_chunk_i_1 = s[layer_output_dict['Layer_1']].split(oc_chunk, factor=c_split2)
    ht, wt, h, w = s[layer_output_dict['Layer_1']].tile(h, w, x_factor=output_tile_size_h, y_factor=output_tile_size_w)
    s[layer_output_dict['Layer_1']].reorder(n, oc_chunk_o, ht, wt, oc_chunk_i_1, h, w, oc)
    fused_blx = s[layer_output_dict['Layer_1']].fuse(n, oc_chunk_o, ht, wt)
    s[layer_output_dict['Layer_1']].parallel(fused_blx)
    if post_ops[1]:
        s[layer_output_dict['Layer_1']].vectorize(oc)
        s[stage_dict['Output_1']].compute_at(s[layer_output_dict['Layer_1']], fused_blx)
        _, oc_chunk_i_1, h, w, oc = s[stage_dict['Output_1']].op.axis
        if post_ops[1] != 'bias':
            s[stage_dict['Output_1_BiasAdd']].compute_inline()
    ho_1, wo_1, h, w = s[stage_dict['Output_1']].tile(h, w, x_factor=output_step_tile_size_h, y_factor=output_step_tile_size_w)
    ic_chunk, ry, rx, ic = s[stage_dict['Output_1']].op.reduce_axis
    ic_chunk_o_1, ic_chunk_i = s[stage_dict['Output_1']].split(ic_chunk, factor=reduce_split2)
    s[stage_dict['Output_1']].reorder(oc_chunk_i_1, ic_chunk_o_1, ho_1, wo_1, h, ic_chunk_i, ry, rx, w, oc, ic)

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
                                                reduce_split2,              #               -> ic_chunk_i

                                                block_output_height,        #               -> hi

                                                filters_cfg['Layer_1'].stride_h,
                                                filters_cfg['Layer_1'].stride_w,

                                                inputs_cfg['Layer_1'].C)
    s[stage_dict['Output_1']].tensorize(tensorize_axis, libxsmm_tensorize)

    ######## Intermediate output
    if hasPaddedInput[1]:
        s[stage_dict['PaddedInput_1']].compute_at(s[stage_dict['Output_1']], fused_blx)
    s[layer_output_dict['Layer_0']].compute_at(s[stage_dict['Output_1']], fused_blx)
    if hasPaddedInput[0]:
        s[stage_dict['PaddedInput_0']].compute_at(s[stage_dict['Output_1']], fused_blx)
    n, oc_chunk, h, w, oc = s[layer_output_dict['Layer_0']].op.axis
    if post_ops[0]:
        s[layer_output_dict['Layer_0']].vectorize(oc)
        s[stage_dict['Output_0']].compute_at(s[stage_dict['Output_1']], wo_1)
        _, oc_chunk, h, w, oc = s[stage_dict['Output_0']].op.axis
        if post_ops[0] != 'bias':
            s[stage_dict['Output_0_BiasAdd']].compute_inline()
    ho, wo, h, w = s[stage_dict['Output_0']].tile(h, w, x_factor=output_tile_0_h, y_factor=output_tile_0_w)
    ic_chunk, ry, rx, ic = s[stage_dict['Output_0']].op.reduce_axis
    ic_chunk_o, ic_chunk_i = s[stage_dict['Output_0']].split(ic_chunk, factor=reduce_split1)
    s[stage_dict['Output_0']].reorder(oc_chunk, ic_chunk_o, ho, wo, h, ic_chunk_i, ry, rx, w, oc, ic)

    # TODO: Deal with this. Currently assuming the first layer is never 1x1
    if (((filters_cfg['Layer_0'].H == 1 and filters_cfg['Layer_0'].W == 1 and \
            filters_cfg['Layer_0'].stride_h == 1 and filters_cfg['Layer_0'].stride_w == 1)) and \
        (step_num_h > 1 and output_step_tile_size_w == outputs_cfg['Layer_0'].W)): # HM > 1 & WI = OW (small W)
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
                                                output_tile_0_w,            # m of brgemm   -> w
                                                filters_cfg['Layer_0'].W,   #               -> rx
                                                filters_cfg['Layer_0'].H,   #               -> ry
                                                reduce_split1,              #               -> ic_chunk_i

                                                block_output_height,        #               -> hi

                                                filters_cfg['Layer_0'].stride_h,
                                                filters_cfg['Layer_0'].stride_w,

                                                inputs_cfg['Layer_0'].C)
    s[stage_dict['Output_0']].tensorize(tensorize_axis, libxsmm_tensorize)

    s = s.normalize()

    return s
