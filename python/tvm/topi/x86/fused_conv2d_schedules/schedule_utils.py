import tvm

def cpu_schedules(name, is_autotvm=True, tuning=False):
    # TODO: Don't use workload name to select the schedule
    if is_autotvm:
        if name == 'depth_conv':
            if tuning:
                from .depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nchwc_auto_search as f
            else:
                from .depth_conv_fused_schedule_auto import schedule_depth_conv_fused_nchwc_auto_inference as f
        elif name == 'conv_conv':
            if tuning:
                from .conv_conv_fused_schedule_auto import schedule_conv_conv_fused_nchwc_auto_search as f
            else:
                from .conv_conv_fused_schedule_auto import schedule_conv_conv_fused_nchwc_auto_inference as f
        elif name == 'conv_depth':
            if tuning:
                from .conv_depth_fused_schedule_auto import schedule_conv_depth_fused_nchwc_auto_search as f
            else:
                from .conv_depth_fused_schedule_auto import schedule_conv_depth_fused_nchwc_auto_inference as f
        else: # resnet block, etc
            from .block_fused_schedule_auto import schedule_block_fused_nhwc_auto as f
    else:
        if name == 'depth_conv':
            from .depth_conv_fused_schedule import schedule_depth_conv_fused_nchwc as f
        elif name == 'conv_conv':
            from .conv_conv_fused_schedule import schedule_conv_conv_fused_nchwc as f
        elif name == 'conv_depth':
            from .conv_depth_fused_schedule import schedule_conv_depth_fused_nchwc as f 
        else: # resnet block, etc
            from .block_fused_schedule import schedule_block_fused_nhwc as f
    return f


def get_layer_cfg():
    assert tvm.topi.FUSION_COMPOSER is not None
    inputs_cfg = {}
    filters_cfg = {}
    outputs_cfg = {}
    for l in range(tvm.topi.FUSION_COMPOSER.layer_num):
        inputs_cfg['Layer_{}'.format(l)] = tvm.topi.FUSION_COMPOSER.get_input_cfg(l)
        filters_cfg['Layer_{}'.format(l)] = tvm.topi.FUSION_COMPOSER.get_filter_cfg(l)
        outputs_cfg['Layer_{}'.format(l)] = tvm.topi.FUSION_COMPOSER.get_output_cfg(l)

    return inputs_cfg, filters_cfg, outputs_cfg
