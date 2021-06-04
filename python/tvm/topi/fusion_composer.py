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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Fused Conv2D Composer"""
import tvm
from tvm import te, autotvm
from tvm.autotvm.task import TaskExtractEnv
from .nn.pad import pad
from .utils import get_vlen, get_4D_shapes_from_params, get_CPU_vlen_from_config
import numpy as np

class FusionComposer:
    def get_input_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][0]

    def get_filter_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1]

    def get_output_cfg(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx+1][0]

    def get_post_op(self, idx):
        assert(idx >= 0 and idx < self.layer_num)
        return self.layers[idx][1].post_op

    def make_placeholders(self, skip_post_op=False):
        placeholders = []
        placeholders.append(te.placeholder(self.get_input_cfg(0).get_shape(), name='Input'))
        for idx in range(self.layer_num):
            filter_cfg = self.get_filter_cfg(idx)
            placeholders.append(te.placeholder(filter_cfg.get_shape(), name='Filter_{}'.format(idx)))

            if self.get_post_op(idx) and not skip_post_op:
                output_cfg = self.get_output_cfg(idx)
                placeholders.append(te.placeholder((output_cfg.C,), name='Bias_{}'.format(idx)))
        return placeholders

    def define_search_space(self):
        conv_count = 0
        for idx in range(self.layer_num):
            is_first_stage = (idx == 0)
            is_final_stage = (idx == self.layer_num - 1)

            DATA = self.get_input_cfg(idx)
            FILTER = self.get_filter_cfg(idx)
            OUTPUT = self.get_output_cfg(idx)

            # Split axes, etc
            if self.cfg is not None:
                # Vector length
                if self.pack:
                    if idx == 0: # Define input vlen for the first layer, no matter what it is
                        self.cfg.define_knob('vlen_input', get_vlen(DATA.C, self.target.kind.name))
                    if not FILTER.depthwise: # ONLY DEFINE vlen FOR CONV, because dw-conv uses the vlen of the previous layer
                        self.cfg.define_knob('vlen_conv_{}'.format(conv_count), get_vlen(OUTPUT.C, self.target.kind.name))
                        conv_count += 1

                    # Assuming no two dw-convs come together
                    if idx == 0 or conv_count < 2: #
                        vlen_i = self.cfg['vlen_input'].val
                    else: #
                        vlen_i = self.cfg['vlen_conv_{}'.format(conv_count-2)].val
                    if FILTER.depthwise: # dw-convs have same vlen_o and vlen_i
                        vlen_o = vlen_i
                    else: # Convs have their own vlen_o
                        vlen_o = self.cfg['vlen_conv_{}'.format(conv_count-1)].val

                    DATA.update_shape(vlen_i)
                    FILTER.update_shape(vlen_i, vlen_o)
                    OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer

                    if is_final_stage:
                        if FILTER.depthwise:
                            self.cfg.define_knob('bind_axis', [0, 1, 2, 3]) # 'oc', 'h', 'w', 'root'
                        else:
                            self.cfg.define_knob('bind_axis', [0, 1, 2, 3, 4]) # 'oc', 'ic', 'h', 'w', 'root'

                if self.target.kind.name == 'cuda' or self.target.device_name == 'tracing':
                    _, OH, OW, OC = OUTPUT.get_shape()
                    c_filter = lambda x: x.size[-1] in get_vlen(OC, self.target.kind.name)

                    if FILTER.depthwise:
                        self.cfg.define_split('split_{}_c'.format(idx), self.cfg.axis(int(OC)), num_outputs=3, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 4
                            W_num_outputs = 3 # 3 for depthwise + 1x1, 4 for 3x3 + 1x1

                            self.cfg.define_split('split_h', self.cfg.axis(int(OH)),
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                            self.cfg.define_split('split_w', self.cfg.axis(int(OW)),
                                                num_outputs=W_num_outputs,
                                                policy='factors')

                        self.cfg.define_split('split_{}_c'.format(idx), self.cfg.axis(int(OC)),
                                        num_outputs=3,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            self.cfg.define_split('split_0_rc', self.cfg.axis(int(OC)),
                                            num_outputs=3,
                                            policy='factors')
                else:
                    _, OC_chunk, OH, OW, _ = OUTPUT.get_shape()
                    c_filter = lambda x: x.size[-1] >= -1 # dummy

                    if FILTER.depthwise:
                        self.cfg.define_split('split_{}_c'.format(idx), self.cfg.axis(int(OC_chunk)), num_outputs=2, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 3
                            W_num_outputs = 3

                            self.cfg.define_split('split_h', self.cfg.axis(int(OH)),
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                            self.cfg.define_split('split_w', self.cfg.axis(int(OW)),
                                                num_outputs=W_num_outputs,
                                                policy='factors')

                        self.cfg.define_split('split_{}_c'.format(idx), self.cfg.axis(int(OC_chunk)),
                                        num_outputs=2,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            self.cfg.define_split('split_0_rc', self.cfg.axis(int(OC_chunk)),
                                            num_outputs=2,
                                            policy='factors')

        # Add flop
        if self.cfg:
            self.cfg.add_flop(self.get_FLOP())

    def update_all_shapes_from_best_cfg(self, best_config):
        if self.pack:
            conv_count = 0
            for idx in range(self.layer_num):
                DATA = self.get_input_cfg(idx)
                FILTER = self.get_filter_cfg(idx)
                OUTPUT = self.get_output_cfg(idx)

                if not FILTER.depthwise:
                    conv_count += 1
                cfg_key = 'vlen_input' if (idx == 0 or conv_count < 2) else\
                            'vlen_conv_{}'.format(conv_count-2)
                vlen_i = get_CPU_vlen_from_config(best_config, cfg_key)
                vlen_o = get_CPU_vlen_from_config(best_config, cfg_key if FILTER.depthwise else 'vlen_conv_{}'.format(conv_count-1))

                DATA.update_shape(vlen_i)
                FILTER.update_shape(vlen_i, vlen_o)
                OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer

    def get_FLOP(self):
        flop = 0
        for l in range(0, self.layer_num):
            fcfg = self.get_filter_cfg(l)
            ocfg = self.get_output_cfg(l)

            if fcfg.depthwise:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
            else:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)
            if fcfg.post_op:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
        return flop

    def get_theoretical_mem_bytes(self, dtype="float32"):
        mem = 0
        icfg = self.get_input_cfg(0)
        mem += 4 * (icfg.N * icfg.H * icfg.W * icfg.C)
        for l in range(0, self.layer_num):
            fcfg = self.get_filter_cfg(l)
            mem += 4 * (fcfg.H * fcfg.W * fcfg.I * fcfg.O)

            if fcfg.post_op:
                mem += 4 * (fcfg.I)
        ocfg = self.get_output_cfg(self.layer_num - 1)
        mem += 4 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
        return mem

    def get_FLOP_per_layer(self):
        flop_list = []
        for l in range(0, self.layer_num):
            flop = 0
            fcfg = self.get_filter_cfg(l)
            ocfg = self.get_output_cfg(l)

            if fcfg.depthwise:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
            else:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)
            if fcfg.post_op:
                flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
            flop_list.append(flop)
        return flop_list

    def get_theoretical_mem_bytes_per_layer(self, dtype="float32"):
        mem = []
        for l in range(0, self.layer_num):
            icfg = self.get_input_cfg(l)
            fcfg = self.get_filter_cfg(l)
            ocfg = self.get_output_cfg(l)

            mem.append(4 * (icfg.N * icfg.H * icfg.W * icfg.C + fcfg.H * fcfg.W * fcfg.I * fcfg.O + ocfg.N * ocfg.H * ocfg.W * ocfg.C))
        return mem

    def get_pattern(self):
        assert self.layers is not None

        is_depthwise_array = [l[1].depthwise for l in self.layers[:-1]]
        if is_depthwise_array[0] and not is_depthwise_array[1]:
            return 'depth_conv'
        if not is_depthwise_array[0] and is_depthwise_array[1]:
            return 'conv_depth'
        if not is_depthwise_array[0] and not is_depthwise_array[1]:
            return 'conv_conv'
        if not is_depthwise_array[0] and is_depthwise_array[1]:
            return 'conv_depth'

        return 'block'

    def __init__(self, p, pack=None, use_autotvm=True, use_auto_scheduler=False, target=None, dtype='float32', workload_name=None, workspace='/tmp'):
        self.cfg = None
        self.parameters = p
        self.use_autotvm = use_autotvm
        self.use_auto_scheduler = use_auto_scheduler
        self.target = target
        if isinstance(self.target, str):
            self.target = tvm.target.Target(self.target)
        if use_auto_scheduler:
            self.pack = False
        else:
            self.pack = (self.target.kind.name != 'cuda' and self.target.device_name != 'tracing') if pack is None else pack
        self.output_dtype=dtype
        self.task_name = 'fused_conv2d.{}'.format('cuda' if self.target.kind.name == 'cuda' else 'x86')
        self.is_block = False
        self.layers = []
        self.placeholders = []

        self.layers = get_4D_shapes_from_params(p)
        self.layer_num = len(self.layers) - 1 # Excluding input
        self.workload_name = workload_name # mv1_1, res_2x, etc
        self.workspace = workspace

        # Temporary variables for composing compute
        self.filter_cfg = None
        self.output_cfg = None
        self.layer_idx = -1

        # Temporary variables for returning the best_config
        self.cfg = None
        device = 'cpu' if 'llvm' in self.target.kind.name else 'gpu'
        self.dir_name = '{}/logs/{}/layer/{}'.format(workspace, 'auto_scheduler' if use_auto_scheduler else 'autotvm', device)
        self.log_name = '{}_fused_{}.log'.format(self.get_pattern(), self.workload_name)

    def padding(self, Input, Filter):
        if self.pack:
            _, _, FH, FW, _, _ = Filter.shape
        else:
            FH, FW, _, _ = Filter.shape

        # Only pad when it's not 1x1
        if FH > 1 and FW > 1:
            pad_top, pad_left, pad_down, pad_right = self.filter_cfg.get_padding_shape()

            if self.pack:
                # 5D PackedInput (NCHWc)
                pad_before = [0, 0, pad_top, pad_left, 0]
                pad_after = [0, 0, pad_down, pad_right, 0]
            else:
                # 4D Input (NHWC)
                pad_before = [0, pad_top, pad_left, 0]
                pad_after = [0, pad_down, pad_right, 0]

            PaddedInput = pad(Input, pad_before, pad_after, name='PaddedInput_{}'.format(self.layer_idx))
            return PaddedInput
        return Input

    def make_depthwise_output(self, Input, Filter):
        # Pad if necessary
        Padded = self.padding(Input, Filter)

        stride_h, stride_w = self.filter_cfg.get_stride()
        dilation_h, dilation_w = self.filter_cfg.get_dilation()

        if self.pack:
            _, _, FH, FW, _, _ = Filter.shape

            # Don't consider 1by1 depthwise
            assert not (self.filter_cfg.depthwise and FH == 1 and FW == 1)

            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')

            Output = te.compute(self.output_cfg.get_shape(),
                lambda n, c_chunk, h, w, c_vec: te.sum(
                                                    (Filter[c_chunk, 0, ry, rx, 0, c_vec] *
                                                    Padded[n, c_chunk,
                                                                    h * stride_h + ry * dilation_h,
                                                                    w * stride_w + rx * dilation_w,
                                                                    c_vec])
                                                    .astype(self.output_dtype),
                                                    axis=[ry, rx]),
                                                name='DepthwiseConv2dOutput_{}'.format(self.layer_idx),
                                                tag='depthwise_nchwc')
        else:
            FH, FW, _, _ = Filter.shape

            # Don't consider 1by1 depthwise
            assert not (self.filter_cfg.depthwise and FH == 1 and FW == 1)

            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')

            Output = te.compute(self.output_cfg.get_shape(),
                        lambda n, h, w, c: te.sum(
                                                (Filter[ry, rx, c, 0] *
                                                Padded[n,
                                                        h * stride_h + ry * dilation_h,
                                                        w * stride_w + rx * dilation_w,
                                                        c])
                                                .astype(self.output_dtype),
                                                axis=[ry, rx]),
                                            name='DepthwiseConv2dOutput_{}'.format(self.layer_idx),
                                            tag='depthwise_nhwc')
        return Output

    def make_conv_output(self, Input, Filter):
        # Pad if necessary
        Padded = self.padding(Input, Filter)

        stride_h, stride_w = self.filter_cfg.get_stride()
        dilation_h, dilation_w = self.filter_cfg.get_dilation()

        if self.pack:
            _, IC_chunk, _, _, IC_vec = Padded.shape
            _, _, FH, FW, _, _ = Filter.shape
            rco = te.reduce_axis((0, IC_chunk), name='rco')
            rci = te.reduce_axis((0, IC_vec), name='rci')
            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')
            Output = te.compute(self.output_cfg.get_shape(),
                lambda n, c_chunk, h, w, c_vec: te.sum(
                                                        (Filter[c_chunk, rco, ry, rx, rci, c_vec] *
                                                        Padded[n, rco,
                                                                    h * stride_h + ry * dilation_h,
                                                                    w * stride_w + rx * dilation_w,
                                                                    rci])
                                                        .astype(self.output_dtype),
                                                        axis=[rco, ry, rx, rci]),
                                                    name='Conv2dOutput_{}'.format(self.layer_idx),
                                                    tag='conv2d_nchwc')
        else:
            _, _, _, IC = Padded.shape
            FH, FW, _, _ = Filter.shape
            rc = te.reduce_axis((0, IC), name='rc')
            ry = te.reduce_axis((0, FH), name='ry')
            rx = te.reduce_axis((0, FW), name='rx')
            Output = te.compute(self.output_cfg.get_shape(),
                        lambda n, h, w, c: te.sum(
                                                    (Filter[ry, rx, rc, c] *
                                                    Padded[n,
                                                            h * stride_h + ry * dilation_h,
                                                            w * stride_w + rx * dilation_w,
                                                            rc])
                                                    .astype(self.output_dtype),
                                                    axis=[rc, ry, rx]),
                                                name='Conv2dOutput_{}'.format(self.layer_idx),
                                                tag='conv2d_nhwc')
        return Output

    def process_post_ops(self, Input, Bias):
        if self.pack:
            _, _, _, _, OC_vec = Input.shape
            BiasAdd = te.compute(Input.shape, lambda n, c_chunk, h, w, c_vec: Input[n, c_chunk, h, w, c_vec] + Bias[c_chunk * OC_vec + c_vec],
                                name='BiasAdd_{}'.format(self.layer_idx),
                                tag='biasadd')
        else:
            BiasAdd = te.compute(Input.shape, lambda n, h, w, c: Input[n, h, w, c] + Bias[c],
                                name='BiasAdd_{}'.format(self.layer_idx),
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
        if self.filter_cfg.post_op == 'relu':
            Last = te.compute(Last.shape,
                            lambda *i: te.max(Last(*i), tvm.runtime.const(0, Last.dtype)),
                            name='ReLU_{}'.format(self.layer_idx), tag='relu')
        elif self.filter_cfg.post_op == 'sigmoid':
            Last = te.compute(Last.shape, 
                            lambda *i: te.sigmoid(Last(*i)),
                            name='Sigmoid_{}'.format(self.layer_idx), tag='sigmoid')
        elif self.filter_cfg.post_op == 'relu6':
            Last = te.compute(Last.shape,
                            lambda *i: te.min(te.max(Last(*i), tvm.runtime.const(0, Last.dtype)), tvm.runtime.const(6, Last.dtype)),
                            name='ReLU6_{}'.format(self.layer_idx), tag='relu6')
        return Last

    # TODO: integrate with TOPI
    def get_compute(self, raw_compute=False, skip_post_op=False):
        def compute(input_tensors):
            Feature = input_tensors[0]
            tensor_idx = 1
            for idx in range(self.layer_num):
                Filter = input_tensors[tensor_idx]

                # Updates:
                self.filter_cfg = self.get_filter_cfg(idx)
                self.output_cfg = self.get_output_cfg(idx)
                self.layer_idx = idx

                if self.get_filter_cfg(idx).depthwise:
                    Feature = self.make_depthwise_output(Feature, Filter)
                else:
                    Feature = self.make_conv_output(Feature, Filter)

                if (self.get_post_op(idx) is not None) and (not skip_post_op):
                    Bias = input_tensors[tensor_idx + 1]
                    tensor_idx += 2
                    Feature = self.process_post_ops(Feature, Bias)
                else:
                    tensor_idx += 1
            return Feature

        def autotvm_wrapper(input_tensors):
            task_env = TaskExtractEnv.current
            args = autotvm.task.topi_integration.serialize_args([self.parameters])
            if task_env is not None and task_env.tracing:
                task_env.add_task(self.task_name, args)
            workload = ((self.task_name),) + args

            # attach workload to return op
            node = compute(input_tensors)
            op = node.op
            attrs = {}
            for k, v in node.op.attrs.items():
                attrs[k] = v
            attrs["workload"] = workload
            if isinstance(op, te.tensor.ComputeOp):
                op = te._ffi_api.ComputeOp(op.name, op.tag, attrs, op.axis, op.body)
            elif isinstance(op, te.tensor.ExternOp):
                op = te._ffi_api.ExternOp(
                    op.name,
                    op.tag,
                    attrs,
                    op.inputs,
                    op.input_placeholders,
                    op.output_placeholders,
                    op.body,
                )
            else:
                raise RuntimeError("Unsupported op type: " + str(type(op)))
            if isinstance(node, te.tensor.Tensor):
                return op.output(0)
            return [op.output(i) for i in range(len(node))]

        self.filter_cfg = None
        self.output_cfg = None
        self.layer_idx = -1

        return compute if raw_compute else autotvm_wrapper

    def get_schedule(self, target=None, tuning=False):
        assert not (not tuning and target is None)
        task_env = TaskExtractEnv.current

        if not self.use_autotvm:
            cfg = None
            self.update_all_shapes_from_best_cfg(cfg)
        else:
            if tuning:
                # Define search space
                self.cfg = autotvm.get_config()
                self.define_search_space()
                cfg = self.cfg
            else: # inference
                dispatch_ctx = autotvm.task.DispatchContext.current
                if not dispatch_ctx or isinstance(dispatch_ctx, autotvm.task.FallbackContext):
                    log_name = '{}/fused/{}'.format(self.dir_name, self.log_name)
                    dispatch_ctx = autotvm.apply_history_best(log_name)
                workload = ((self.task_name),) + autotvm.task.topi_integration.serialize_args([self.parameters])
                cfg = dispatch_ctx.query(target, workload)

                if task_env and not task_env.tracing and cfg.is_fallback:
                    print("---[[[ AutoTVM cfg not found! ]]]---")

                # Update the tensor shapes with the best config
                self.update_all_shapes_from_best_cfg(cfg)

        def wrapper(outs):
            def raw_schedule():
                if self.target.kind.name == 'cuda':
                    from .cuda.fused_conv2d_schedules.schedule_utils import gpu_schedules as sch
                else:
                    from .x86.fused_conv2d_schedules.schedule_utils import cpu_schedules as sch
                return sch(self.get_pattern(), (cfg is not None), tuning=tuning)
            f = raw_schedule()
            if self.pack:
                inputs_cfg = {}
                filters_cfg = {}
                outputs_cfg = {}
                for l in range(self.layer_num):
                    inputs_cfg['Layer_{}'.format(l)] = self.get_input_cfg(l)
                    filters_cfg['Layer_{}'.format(l)] = self.get_filter_cfg(l)
                    outputs_cfg['Layer_{}'.format(l)] = self.get_output_cfg(l)
                if cfg is not None:
                    s = f(cfg, outs, inputs_cfg=inputs_cfg, filters_cfg=filters_cfg, outputs_cfg=outputs_cfg)
                else:
                    s = f(outs, inputs_cfg=inputs_cfg, filters_cfg=filters_cfg, outputs_cfg=outputs_cfg)
            elif self.target.kind.name == 'cuda': # CUDA
                if cfg is not None:
                    s = f(cfg, outs)
                else:
                    s = f(outs)
            elif self.target.device_name == 'tracing':
                # Return empty schedule
                outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
                s = te.create_schedule([x.op for x in outs])
            else:
                raise Exception("Case unrecognizable!")
            return s

        return wrapper

    def get_schedule_inference(self, target):
        # Get schedule (comes first as tensor shapes need to be updated)
        schedule = self.get_schedule(target)

        # Get compute
        compute = self.get_compute()
        input_tensors = self.make_placeholders()
        output_tensor = compute(input_tensors)
        all_tensors = input_tensors + [output_tensor]

        s = schedule(output_tensor)
        return s, all_tensors

    def print_info(self):
        for i in range(self.layer_num):
            DATA, KERNEL = self.layers[i]
            print('Input_{} size: {}'.format(i, DATA.get_shape()))
            print('Filter_{} size: {}, depthwise: {}, post_op: {}'.format(i, KERNEL.get_shape(), KERNEL.depthwise, KERNEL.post_op))
            print('Is a block: {}'.format(self.is_block))
        # OUTPUT = self.layers[-1][0]
        print('Output size: {}'.format(DATA.get_shape()))

    def tensor_transformation(self, data, tensor_cfg, tensor_type):
        if self.pack:
            if tensor_type == 'data': # NHWC -> NCHWc
                n, c_chunk, h, w, vlen = tensor_cfg.get_shape()
                nchwc = data.reshape(n, h, w, c_chunk, vlen)
                return np.array(nchwc.transpose(0, 3, 1, 2, 4), order='C')
            else: # kernel: HWIO -> OIHWio
                o_chunk, i_chunk, h, w, vlen_i, vlen_o = tensor_cfg.get_shape()
                if tensor_cfg.depthwise:
                    oihwio = data.reshape(h, w, o_chunk, vlen_o, i_chunk, vlen_i)
                    np_array = np.array(oihwio.transpose(2, 4, 0, 1, 5, 3), order='C')
                else:
                    oihwio = data.reshape(h, w, i_chunk, vlen_i, o_chunk, vlen_o)
                    np_array = np.array(oihwio.transpose(4, 2, 0, 1, 3, 5), order='C')
                return np_array
        return data


# def test_compute():
#     parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
#     target = tvm.target.Target('cuda')
#     print(target)
#     fc = FusionComposer(parameters, target=target)
#     f = fc.get_compute()
#     input_tensors = fc.make_placeholders()
#     from pprint import pprint
#     pprint(input_tensors)
#     print(f(input_tensors))
#     print(fc.cfg)


# def test_schedule():
#     parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
#     with tvm.target.Target('cuda'):
#         s, flatten_params = get_schedule_tuning_cuda(parameters)
#     print(tvm.lower(s, flatten_params, simple_mode=True))


# if __name__ == '__main__':
#     test_compute()
#     test_schedule()
