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
import tvm
from tvm import autotvm

from ..fusion_composer import FusionComposer

@autotvm.template('fused_conv2d.cuda')
def get_schedule_tuning_cuda(parameters):
    target = tvm.target.Target('cuda')
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
