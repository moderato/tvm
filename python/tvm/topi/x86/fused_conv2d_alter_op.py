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
"""Fused Conv2D alter op and legalize functions for x86"""

import logging, re
import tvm
from tvm import relay
from tvm import autotvm
from ..nn import fused_conv2d_alter_layout

logger = logging.getLogger("topi")

_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWio_matcher = re.compile("^OIHW[0-9]+i[0-9]+o$")


@fused_conv2d_alter_layout.register("cpu")
def _alter_fused_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
        cfg = dispatch_ctx.query(target, None)
        workload = cfg.workload
    else:
        _, outs = relay.backend.compile_engine.select_implementation(
            relay.op.get("nn.fused_conv2d"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
        if workload is None:
            # The best implementation is not an AutoTVM template,
            # we then assume it's not necessary to alter this op.
            return None
        cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]
    assert (topi_tmpl == "fused_conv2d.x86")
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    num_layers = attrs['num_layers']
    data_layout_array = list(attrs['data_layout_array'])
    kernel_layout_array = list(attrs['kernel_layout_array'])
    out_layout_array = list(attrs['out_layout_array'])
    groups_array = list(attrs['groups_array'])
    vlen_i = -1
    conv_count = 0

    for l in range(num_layers):
        data_layout = data_layout_array[l]
        kernel_layout = kernel_layout_array[l]
        groups = groups_array[l]
        depthwise = (groups > 1)

        if (data_layout == "NCHW" and kernel_layout == "OIHW") or \
            (data_layout == "NHWC" and kernel_layout == "HWOI" and depthwise) or \
                (data_layout == "NHWC" and kernel_layout == "HWIO" and not depthwise):

            if cfg.is_fallback:
                raise Exception("Don't accept FallBack config")

            if l == 0:
                vlen_i = cfg['vlen_input'].val
            if depthwise:
                vlen_o = vlen_i
            else:
                vlen_o = cfg['vlen_conv_{}'.format(conv_count)].val # Count convs
                conv_count += 1

            # update new attrs
            data_layout_array[l] = "NCHW%dc" % vlen_i
            if depthwise:
                kernel_layout_array[l] = "OIHW1i%do" % vlen_o
            else:
                kernel_layout_array[l] = "OIHW%di%do" % (vlen_i, vlen_o)
            out_layout_array[l] = "NCHW%dc" % vlen_o # Duplicated with the input of the next layer

            # vlen_o of this layer is vlen_i of next layer
            vlen_i = vlen_o

        else:
            assert _NCHWc_matcher.match(data_layout)
            assert _OIHWio_matcher.match(kernel_layout)

    new_attrs['data_layout_array'] = data_layout_array
    new_attrs['kernel_layout_array'] = kernel_layout_array
    new_attrs['out_layout_array'] = out_layout_array

    # TODO: Skip num_layers for now
    del new_attrs['num_layers']

    return relay.op.nn.fused_conv2d(*inputs, **new_attrs)
