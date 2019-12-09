# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Schedule for depthwise_conv2d with auto fusion"""
import tvm
from tvm import autotvm
from tvm.contrib import cudnn
from ..util import get_const_tuple, traverse_inline
from .. import tag
from .. import generic, nn

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
autotvm.register_topi_compute(nn.depthwise_conv2d_nchw, ['cuda', 'gpu'], 'direct')
                              # nn.depthwise_conv2d_nchw.fdefault)

def depthwise_conv2d_cuda(cfg, data, kernel, strides, padding, dilation, layout='NCHW', out_dtype='float32', algo=-1):
    """Depthwise Conv2D operator for cuda backend.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, ic_chunk, in_height, in_width, ic_block]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block, in_channel_block]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    target = tvm.target.current_target()

    if "cudnn" in target.libs:
        # group_count = in_channel for depthwise
        if layout == 'NCHW':
            tensor_format = 0 # CUDNN_TENSOR_NCHW
            N, group_count, H, W = get_const_tuple(data.shape)
        elif layout == 'NHWC':
            tensor_format = 1 # CUDNN_TENSOR_NHWC
            N, H, W, group_count = get_const_tuple(data.shape)
        else:
            raise ValueError("Unsupported layout %s in cudnn" % layout)
        CO, CI, KH, KW = get_const_tuple(kernel.shape)

        # handle dilation
        stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

        # OH = (H + 2 * pad_h - KH) // stride_h + 1
        # OW = (W + 2 * pad_w - KW) // stride_w + 1
        # cfg.add_flop(2 * N * OH * OW * CO * CI * ((KH - 1) * dilation_h + 1) *\
        #             ((KW - 1) * dilation_w + 1))

        return cudnn.grouped_conv2d_forward(data,
                                            kernel,
                                            group_count,
                                            stride_h,
                                            stride_w,
                                            pad_h,
                                            pad_w,
                                            dilation_h,
                                            dilation_w,
                                            conv_mode=1,
                                            tensor_format=tensor_format,
                                            algo=algo)  # let CUDNN choose the best algo


@autotvm.register_topi_schedule(generic.schedule_depthwise_conv2d_nchw, ['cuda', 'gpu'], 'direct')
def schedule_depthwise_conv2d_nchw_cuda(cfg, outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    target = tvm.target.current_target()
    if 'cudnn' in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'depthwise_conv2d_nchw':
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            cfg.define_split("tile_f", f, num_outputs=4)
            cfg.define_split("tile_y", y, num_outputs=4)
            cfg.define_split("tile_x", x, num_outputs=4)
            cfg.define_knob("auto_unroll_max_step", [0, 256, 1500])

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            # fallback support
            if cfg.is_fallback:
                ref_log = autotvm.tophub.load_reference_log(
                    target.target_name, target.model, 'depthwise_conv2d_nchw', 'direct')
                cfg.fallback_with_reference_log(ref_log)
                # TODO(lmzheng): A bug here, set unroll_explicit to False as workaround
                cfg['unroll_explicit'].val = 0
            ##### space definition end #####

            s[pad_data].compute_inline()
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, 'local')
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope('local')
                OL = conv

            # create cache stage
            AA = s.cache_read(pad_data, 'shared', [OL])
            WW = s.cache_read(kernel, 'shared', [OL])
            AL = s.cache_read(AA, 'local', [OL])
            WL = s.cache_read(WW, 'local', [OL])

            # tile and bind spatial axes
            n, f, y, x = s[output].op.axis
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            kernel_scope, n = s[output].split(n, nparts=1)
            bf = s[output].fuse(n, bf)
            s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
            s[output].bind(by, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vf, tvm.thread_axis("vthread"))
            s[output].bind(vy, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))
            s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
            s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
            s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
            s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
            s[OL].compute_at(s[output], tx)

            # cooperative fetching
            s[AA].compute_at(s[output], bx)
            s[WW].compute_at(s[output], bx)
            s[AL].compute_at(s[output], tx)
            s[WL].compute_at(s[output], tx)

            for load in [AA, WW]:
                fused = s[load].fuse(*list(s[load].op.axis))
                fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
                fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
                fused, tz = s[load].split(fused, cfg["tile_f"].size[2])
                s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
                s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(s, outs[0].op, _callback)
    return s

@generic.schedule_depthwise_conv2d_nhwc.register(["cuda", "gpu"])
def schedule_depthwise_conv2d_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(temp, Filter, DepthwiseConv2d):
        s[temp].compute_inline()
        FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])
        if DepthwiseConv2d.op in s.outputs:
            Output = DepthwiseConv2d
            CL = s.cache_write(DepthwiseConv2d, "local")
        else:
            Output = outs[0].op.output(0)
            s[DepthwiseConv2d].set_scope("local")

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")

        b, h, w, c = s[Output].op.axis

        # num_thread here could be 728, it is larger than cuda.max_num_threads
        num_thread = tvm.ir_pass.Simplify(temp.shape[3]).value
        target = tvm.target.current_target()
        if target and (target.target_name not in ["cuda", "nvptx"]):
            num_thread = target.max_num_threads
        xoc, xic = s[Output].split(c, factor=num_thread)
        s[Output].reorder(xoc, b, h, w, xic)
        xo, yo, _, _ = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(fused, b)
        fused = s[Output].fuse(fused, xoc)

        s[Output].bind(fused, block_x)
        s[Output].bind(xic, thread_x)

        if DepthwiseConv2d.op in s.outputs:
            s[CL].compute_at(s[Output], xic)
        else:
            s[DepthwiseConv2d].compute_at(s[Output], xic)

        _, _, ci, fi = s[FS].op.axis
        s[FS].compute_at(s[Output], fused)
        fused = s[FS].fuse(fi, ci)
        s[FS].bind(fused, thread_x)

    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                # if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        if OP.tag == 'depthwise_conv2d_nhwc':
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            if isinstance(Filter.op, tvm.tensor.ComputeOp) and 'dilate' in Filter.op.tag:
                s[Filter].compute_inline()
            DepthwiseConv2d = OP.output(0)
            _schedule(PaddedInput, Filter, DepthwiseConv2d)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


def schedule_depthwise_conv2d_backward_input_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc backward wrt input.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        backward wrt input in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d backward
        wrt input with layout nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Padded_out_grad, In_grad):
        s[Padded_out_grad].compute_inline()

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        _, h, w, c = In_grad.op.axis

        fused_hwc = s[In_grad].fuse(h, w, c)
        xoc, xic = s[In_grad].split(fused_hwc, factor=128)

        s[In_grad].bind(xoc, block_x)
        s[In_grad].bind(xic, thread_x)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if OP.tag == 'depthwise_conv2d_backward_input_nhwc':
            Padded_out_grad = OP.input_tensors[0]
            Dilated_out_grad = Padded_out_grad.op.input_tensors[0]
            s[Dilated_out_grad].compute_inline()
            In_grad = OP.output(0)
            _schedule(Padded_out_grad, In_grad)
        else:
            raise ValueError("Depthwise conv backward wrt input for non-NHWC is not supported.")

    traverse(outs[0].op)
    return s

def schedule_depthwise_conv2d_backward_weight_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc backward wrt weight.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        backward wrt weight in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d backward
        wrt weight with layout nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Weight_grad):
        block_x = tvm.thread_axis("blockIdx.x")
        thread_y = tvm.thread_axis("threadIdx.y")
        thread_x = tvm.thread_axis("threadIdx.x")

        db, dh, dw = Weight_grad.op.reduce_axis

        fused_dbdhdw = s[Weight_grad].fuse(db, dh, dw)
        _, ki = s[Weight_grad].split(fused_dbdhdw, factor=8)
        BF = s.rfactor(Weight_grad, ki)

        fused_fwcm = s[Weight_grad].fuse(*s[Weight_grad].op.axis)

        xo, xi = s[Weight_grad].split(fused_fwcm, factor=32)

        s[Weight_grad].bind(xi, thread_x)
        s[Weight_grad].bind(xo, block_x)

        s[Weight_grad].bind(s[Weight_grad].op.reduce_axis[0], thread_y)
        s[BF].compute_at(s[Weight_grad], s[Weight_grad].op.reduce_axis[0])

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if OP.tag == 'depthwise_conv2d_backward_weight_nhwc':
            Padded_in = OP.input_tensors[1]
            s[Padded_in].compute_inline()
            Weight_grad = OP.output(0)
            _schedule(Weight_grad)
        else:
            raise ValueError("Depthwise conv backward wrt weight for non-NHWC is not supported.")

    traverse(outs[0].op)
    return s
