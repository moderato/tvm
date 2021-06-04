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
from ..fusion_composer import FusionComposer
from ..nn.fused_conv2d import fused_conv2d_infer_layout
from ..utils import get_4D_shapes_from_params

logger = logging.getLogger("topi")


@fused_conv2d_infer_layout.register("cpu")
def _fused_conv2d_infer_layout(workload, cfg):
    if cfg.is_fallback:
        raise Exception("Don't accept FallBack config")

    layers = get_4D_shapes_from_params(workload[1])
    num_layers = len(layers) - 1

    # Input
    first_feature, first_filter = layers[0]
    if first_filter.depthwise:
        vlen_i = cfg['vlen_conv_0'].val
    else:
        vlen_i = cfg['vlen_input'].val
    first_feature.update_shape(vlen_i)
    in_layout = "NCHW%dc" % vlen_i
    in_shape = first_feature.shape

    # Output
    output, = layers[-1]
    vlen_o = cfg['vlen_conv_{}'.format(num_layers-1)].val
    output.update_shape(vlen_o)
    out_layout = "NCHW%dc" % vlen_o
    out_shape = output.shape

    return ((in_shape, in_layout),), ((out_shape, out_layout),)


@autotvm.template('fused_conv2d.x86')
def get_schedule_tuning_x86(parameters):
    target = tvm.target.Target('llvm')

    # A workaround for CPU autotuning
    tmp = []
    for idx in range(len(parameters)):
        if parameters[idx] == 'relu' or parameters[idx] == 'relu6' or parameters[idx] == 'bias':
            tmp.append(None)
        else:
            tmp.append(parameters[idx])
    parameters = tmp
    fc = FusionComposer(parameters, target=target)

    # Get schedule
    schedule = fc.get_schedule(tuning=True)

    # Get compute
    compute = fc.get_compute()
    input_tensors = fc.make_placeholders()
    output_tensor = compute(input_tensors)
    all_tensors = input_tensors + [output_tensor]

    s = schedule(output_tensor)
    return s, all_tensors
