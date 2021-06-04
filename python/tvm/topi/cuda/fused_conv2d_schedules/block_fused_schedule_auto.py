import tvm

def schedule_block_fused_nhwc_auto(outs, stages, params, device="cuda", bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    return s
