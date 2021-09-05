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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""Fused Conv2D operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from .pad import pad
from ..utils import simplify


def padding(layer_idx, Input, Filter, paddings, pack=False):
    if pack:
        _, _, FH, FW, _, _ = Filter.shape
    else:
        FH, FW, _, _ = Filter.shape

    # Only pad when it's not 1x1
    if FH > 1 and FW > 1:
        pad_top, pad_left, pad_down, pad_right = paddings

        if pack:
            # 5D PackedInput (NCHWc)
            pad_before = [0, 0, pad_top, pad_left, 0]
            pad_after = [0, 0, pad_down, pad_right, 0]
        else:
            # 4D Input (NHWC)
            pad_before = [0, pad_top, pad_left, 0]
            pad_after = [0, pad_down, pad_right, 0]

        PaddedInput = pad(Input, pad_before, pad_after, name='FusedConv2D_PaddedInput_{}'.format(layer_idx))
        return PaddedInput
    return Input


def make_depthwise_output(layer_idx, Input, Filter, stride, paddings, dilation, pack=False, out_dtype="float32"):
    # Pad if necessary
    Padded = padding(layer_idx, Input, Filter, paddings, pack=pack)

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if pack:
        OC_chunk, _, FH, FW, _, OC_vec = Filter.shape
        dilated_kernel_h = (FH - 1) * dilation_h + 1
        dilated_kernel_w = (FW - 1) * dilation_w + 1
        batch_size, _, padded_h, padded_w, _ = Padded.shape
        out_height = simplify((padded_h - dilated_kernel_h) // stride_h + 1)
        out_width = simplify((padded_w - dilated_kernel_w) // stride_w + 1)

        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')

        Output = te.compute((batch_size, OC_chunk, out_height, out_width, OC_vec),
            lambda n, c_chunk, h, w, c_vec: te.sum(
                                                (Filter[c_chunk, 0, ry, rx, 0, c_vec] *
                                                Padded[n, c_chunk,
                                                                h * stride_h + ry * dilation_h,
                                                                w * stride_w + rx * dilation_w,
                                                                c_vec])
                                                .astype(out_dtype),
                                                axis=[ry, rx]),
                                            name='FusedConv2D_DWConv2dOutput_{}'.format(layer_idx),
                                            tag='dwconv_nchwc')
    else:
        FH, FW, out_channel, _ = Filter.shape
        dilated_kernel_h = (FH - 1) * dilation_h + 1
        dilated_kernel_w = (FW - 1) * dilation_w + 1
        batch_size, padded_h, padded_w, _ = Padded.shape
        out_height = simplify((padded_h - dilated_kernel_h) // stride_h + 1)
        out_width = simplify((padded_w - dilated_kernel_w) // stride_w + 1)

        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')

        Output = te.compute((batch_size, out_height, out_width, out_channel),
                    lambda n, h, w, c: te.sum(
                                            (Filter[ry, rx, c, 0] *
                                            Padded[n,
                                                    h * stride_h + ry * dilation_h,
                                                    w * stride_w + rx * dilation_w,
                                                    c])
                                            .astype(out_dtype),
                                            axis=[ry, rx]),
                                        name='FusedConv2D_DWConv2dOutput_{}'.format(layer_idx),
                                        tag='dwconv_nhwc')
    return Output


def make_conv_output(layer_idx, Input, Filter, stride, paddings, dilation, pack=False, out_dtype="float32"):
    # Pad if necessary
    Padded = padding(layer_idx, Input, Filter, paddings, pack=pack)

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if pack:
        _, IC_chunk, _, _, IC_vec = Padded.shape
        OC_chunk, _, FH, FW, _, OC_vec = Filter.shape
        dilated_kernel_h = (FH - 1) * dilation_h + 1
        dilated_kernel_w = (FW - 1) * dilation_w + 1
        batch_size, _, padded_h, padded_w, _ = Padded.shape
        out_height = simplify((padded_h - dilated_kernel_h) // stride_h + 1)
        out_width = simplify((padded_w - dilated_kernel_w) // stride_w + 1)

        rco = te.reduce_axis((0, IC_chunk), name='rco')
        rci = te.reduce_axis((0, IC_vec), name='rci')
        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')
        Output = te.compute((batch_size, OC_chunk, out_height, out_width, OC_vec),
            lambda n, c_chunk, h, w, c_vec: te.sum(
                                                    (Filter[c_chunk, rco, ry, rx, rci, c_vec] *
                                                    Padded[n, rco,
                                                                h * stride_h + ry * dilation_h,
                                                                w * stride_w + rx * dilation_w,
                                                                rci])
                                                    .astype(out_dtype),
                                                    axis=[rco, ry, rx, rci]),
                                                name='FusedConv2D_Conv2dOutput_{}'.format(layer_idx),
                                                tag='conv2d_nchwc')
    else:
        _, _, _, IC = Padded.shape
        FH, FW, _, OC = Filter.shape
        dilated_kernel_h = (FH - 1) * dilation_h + 1
        dilated_kernel_w = (FW - 1) * dilation_w + 1
        batch_size, padded_h, padded_w, _ = Padded.shape
        out_height = simplify((padded_h - dilated_kernel_h) // stride_h + 1)
        out_width = simplify((padded_w - dilated_kernel_w) // stride_w + 1)

        rc = te.reduce_axis((0, IC), name='rc')
        ry = te.reduce_axis((0, FH), name='ry')
        rx = te.reduce_axis((0, FW), name='rx')
        Output = te.compute((batch_size, out_height, out_width, OC),
                    lambda n, h, w, c: te.sum(
                                                (Filter[ry, rx, rc, c] *
                                                Padded[n,
                                                        h * stride_h + ry * dilation_h,
                                                        w * stride_w + rx * dilation_w,
                                                        rc])
                                                .astype(out_dtype),
                                                axis=[rc, ry, rx]),
                                            name='FusedConv2D_Conv2dOutput_{}'.format(layer_idx),
                                            tag='conv2d_nhwc')
    return Output

def process_post_ops(layer_idx, Input, Bias, post_op, pack=False, out_dtype="float32"):
    if pack:
        _, _, _, _, OC_vec = Input.shape
        BiasAdd = te.compute(Input.shape, lambda n, c_chunk, h, w, c_vec: Input[n, c_chunk, h, w, c_vec] + Bias[c_chunk * OC_vec + c_vec],
                            name='FusedConv2D_BiasAdd_{}'.format(layer_idx),
                            tag='biasadd')
    else:
        BiasAdd = te.compute(Input.shape, lambda n, h, w, c: Input[n, h, w, c] + Bias[c],
                            name='FusedConv2D_BiasAdd_{}'.format(layer_idx),
                            tag='biasadd')

    # TODO: Recover this
    # if block_input is not None:
    #     inputs = block_input if isinstance(block_input, list) else [block_input]
    #     First = inputs[0] # TODO: Support multiple branches addition later
    #     Last = self.stages[-1][-1] # Output if post_op is None, BiasAdd if it's not None
    #     assert sorted(get_const_tuple(First.shape)) == sorted(get_const_tuple(Last.shape)), '{} is not the same as {}'.format(First.shape, Last.shape)
    #     if self.pack:
    #         Output = te.compute(self.output_shape,
    #                             lambda n, c_chunk, h, w, c_vec: (First[n, c_chunk, h, w, c_vec] + (Last[n, c_chunk, h, w, c_vec])),
    #                             name='ElementwiseAddOutput_{}'.format(self.layer_idx),
    #                             tag='elem_{}'.format(tag_suffix))
    #     else:
    #         Output = te.compute(self.output_shape,
    #                             lambda n, h, w, c: (First[n, h, w, c] + (Last[n, h, w, c])),
    #                             name='ElementwiseAddOutput_{}'.format(self.layer_idx),
    #                             tag='elem_{}'.format(tag_suffix))
    #     self.stages[-1].append(Output)
    # Last = self.stages[-1][-1] # BiasAdd if it's not a block, Output if it's a block

    # Else: only bias_add
    Last = BiasAdd
    if post_op == 'relu':
        Last = te.compute(Last.shape,
                        lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                        name='FusedConv2D_ReLU_{}'.format(layer_idx), tag='relu')
    elif post_op == 'sigmoid':
        Last = te.compute(Last.shape, 
                        lambda *i: te.sigmoid(Last(*i)),
                        name='FusedConv2D_Sigmoid_{}'.format(layer_idx), tag='sigmoid')
    elif post_op == 'relu6':
        Last = te.compute(Last.shape,
                        lambda *i: te.min(te.max(Last(*i), tvm.runtime.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                        name='FusedConv2D_ReLU6_{}'.format(layer_idx), tag='relu6')
    return Last


# FusionComposer produces arguments of fused_conv2d
def fused_conv2d(Input, Filters, Biases, num_layers, strides, paddings, dilations, is_dws, post_ops, out_dtype="float32", skip_post_op=False):
    assert num_layers == 2 # For now!
    assert len(Filters) == len(Biases) == len(strides) == len(paddings) == len(dilations) == len(is_dws) == len(post_ops) == num_layers

    Feature = Input
    pack = len(Feature.shape) == 5 # Packing decided by input tensor sizes
    for idx in range(num_layers):
        Filter = Filters[idx]
        assert ((len(Feature.shape) == 5 and len(Filter.shape) == 6) or \
                (len(Feature.shape) == 4 and len(Filter.shape) == 4))

        if is_dws[idx]:
            Feature = make_depthwise_output(idx, Feature, Filter, strides[idx], paddings[idx], dilations[idx], pack=pack, out_dtype=out_dtype)
        else:
            Feature = make_conv_output(idx, Feature, Filter, strides[idx], paddings[idx], dilations[idx], pack=pack, out_dtype=out_dtype)

        if (post_ops[idx] is not None) and (not skip_post_op):
            Feature = process_post_ops(idx, Feature, Biases[idx], post_ops[idx], pack=pack, out_dtype=out_dtype)

    return Feature


@tvm.target.generic_func
def fused_conv2d_alter_layout(attrs, inputs, tinfos, out_type):
    """Change Fused Conv2D layout.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    # not to change by default
    return None


@tvm.target.generic_func
def fused_conv2d_infer_layout(workload, cfg):
    """Infer input/output shapes and layouts from a workload and cfg.

    Parameters
    ----------
    workload : tuple
        fused_conv2d workload

    cfg : tuple
        tvm.autotvm config

    Returns
    -------
    Output : [tuple of tuple and str, tuple of tuple and str]
        Input shapes and layouts, and output shapes and layouts
    """
    raise ValueError("missing register for topi.nn.fused_conv2d_infer_layout")
