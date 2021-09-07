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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Fused Conv2D schedule on x86"""

import logging

import tvm
from tvm import autotvm
from .. import nn
from ..fusion_composer import FusionComposer
from ..nn.fused_conv2d import fused_conv2d_infer_layout
from ..utils import tensors_to_fusion_param

logger = logging.getLogger("topi")


@fused_conv2d_infer_layout.register("cpu")
def _fused_conv2d_infer_layout(workload, cfg):
    if cfg.is_fallback:
        raise Exception("Don't accept FallBack config")
    num_layers = workload[4]
    p = tensors_to_fusion_param(num_layers=num_layers, 
                                Input=workload[1][1], 
                                Filters=[w[1] for w in workload[2]], 
                                strides=workload[5], 
                                is_dws=workload[8], 
                                post_ops=workload[9], 
                                layouts=workload[10])

    fc = FusionComposer(p, pack=True, use_autotvm=True, target=tvm.target.Target("llvm -mcpu=tracing"))
    fc.update_all_shapes_from_best_cfg(cfg)
    layers = fc.layers

    in_shape = layers[0][0].get_shape()
    vlen_i = cfg['vlen_input'].val
    in_layout = "NCHW%dc" % vlen_i
    conv_count = 0
    for layer in layers[0:num_layers]: # Last element is the output tensor
        if not layer[1].depthwise:
            conv_count += 1
    out_shape = layers[num_layers][0].get_shape()
    vlen_o = cfg['vlen_conv_{}'.format(conv_count-1)].val
    out_layout = "NCHW%dc" % vlen_o

    return ((in_shape, in_layout),), ((out_shape, out_layout),)


def _pack_data(Input, Filters):
    input_shape = tvm.topi.FUSION_COMPOSER.get_input_cfg(0).get_shape()
    Input = tvm.te.compute(
        input_shape,
        lambda bs, c, h, w, vc: Input[bs, h, w, c * input_shape[-1] + vc],
        name="Input",
    )

    New_Filters = []
    for idx in range(len(Filters)):
        FILTER_CFG = tvm.topi.FUSION_COMPOSER.get_filter_cfg(idx)
        filter_shape = FILTER_CFG.get_shape()
        if FILTER_CFG.depthwise:
            filter = tvm.te.compute(
                filter_shape,
                lambda occ, icc, k_h, k_w, icb, ocb: Filters[idx][k_h, k_w, occ * filter_shape[-1] + ocb, 0],
                name="Filter_{}".format(idx),
            )
        else:
            filter = tvm.te.compute(
                filter_shape,
                lambda occ, icc, k_h, k_w, icb, ocb: Filters[idx][k_h, k_w, icc * filter_shape[-2] + icb, occ * filter_shape[-1] + ocb],
                name="Filter_{}".format(idx),
            )
        New_Filters.append(filter)

    return Input, New_Filters


@autotvm.register_topi_compute("fused_conv2d.x86")
def fused_conv2d(cfg, Input, Filters, Biases, num_layers, strides, paddings, dilations, is_dws, post_ops, layouts, out_dtype="float32"):
    target = tvm.target.Target.current()
    _4D = len(Input.shape) == 4

    p = tensors_to_fusion_param(num_layers, Input, Filters, strides, is_dws, post_ops, layouts)
    if tvm.topi.FUSION_COMPOSER is None or p != tvm.topi.FUSION_COMPOSER.parameters:
        tvm.topi.FUSION_COMPOSER = FusionComposer(p, pack=True, use_autotvm=True, target=target)
    tvm.topi.FUSION_COMPOSER.define_search_space(cfg)

    if _4D:
        # Same treatment as topi.x86.conv2d_nchwc
        if autotvm.GLOBAL_SCOPE.in_tuning:
            input_shape = tvm.topi.FUSION_COMPOSER.get_input_cfg(0).get_shape()
            Input = tvm.te.placeholder(input_shape, out_dtype, name="Input")
            Filters = []
            for idx in range(num_layers):
                FILTER_CFG = tvm.topi.FUSION_COMPOSER.get_filter_cfg(idx)
                Filters.append(tvm.te.placeholder(FILTER_CFG.get_shape(), out_dtype, name="Filter_{}".format(idx)))
        else:
            Input, Filters = _pack_data(Input, Filters)
    else:
        # Inputs are already 5D. Update config accordingly.
        tvm.topi.FUSION_COMPOSER.update_all_shapes_from_tensors(Input.shape, [f.shape for f in Filters])
    skip_post_op = not tvm.topi.FUSION_COMPOSER.tuned # Skip when the task is just created and not tuned

    return nn.fused_conv2d(Input, Filters, Biases, num_layers, strides, paddings, dilations, is_dws, post_ops, out_dtype=out_dtype, skip_post_op=skip_post_op)


@autotvm.register_topi_schedule("fused_conv2d.x86")
def schedule_fused_conv2d(cfg, outs):
    assert tvm.topi.FUSION_COMPOSER is not None
    from .fused_conv2d_schedules.schedule_utils import cpu_schedules as sch
    f = sch(tvm.topi.FUSION_COMPOSER.get_pattern(), (cfg is not None), \
            tuning=(not tvm.topi.FUSION_COMPOSER.tuned or autotvm.GLOBAL_SCOPE.in_tuning)) # Either it has never been tuned (task creation) or it's tuning.
    s = f(cfg, outs)
    return s
