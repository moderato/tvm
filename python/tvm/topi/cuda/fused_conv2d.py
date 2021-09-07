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
# pylint: disable=invalid-name, unused-argument
"""Compute definition for fused_conv2d with cuda backend"""

import logging

import tvm
from tvm import autotvm
from .. import nn
from ..fusion_composer import FusionComposer
from ..utils import tensors_to_fusion_param

logger = logging.getLogger("topi")


@autotvm.register_topi_compute("fused_conv2d.cuda")
def fused_conv2d(cfg, Input, Filters, Biases, num_layers, strides, paddings, dilations, is_dws, post_ops, layouts, out_dtype="float32"):
    target = tvm.target.Target.current()

    p = tensors_to_fusion_param(num_layers, Input, Filters, strides, is_dws, post_ops, layouts)
    if tvm.topi.FUSION_COMPOSER is None or p != tvm.topi.FUSION_COMPOSER.parameters:
        tvm.topi.FUSION_COMPOSER = FusionComposer(p, pack=False, use_autotvm=True, target=target)
    tvm.topi.FUSION_COMPOSER.define_search_space(cfg)

    return nn.fused_conv2d(Input, Filters, Biases, num_layers, strides, paddings, dilations, is_dws, post_ops, out_dtype=out_dtype, skip_post_op=False)


@autotvm.register_topi_schedule("fused_conv2d.cuda")
def schedule_fused_conv2d(cfg, outs):
    assert tvm.topi.FUSION_COMPOSER is not None
    from .fused_conv2d_schedules.schedule_utils import gpu_schedules as sch
    f = sch(tvm.topi.FUSION_COMPOSER.get_pattern(), (cfg is not None))
    s = f(cfg, outs)
    return s
