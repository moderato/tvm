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
from tvm import te
from .utils import get_vlen, get_4D_shapes_from_params, get_CPU_vlen_from_config

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


    def define_search_space(self, cfg):
        # Add flop
        cfg.add_flop(self.get_FLOP())

        conv_count = 0
        for idx in range(self.layer_num):
            is_first_stage = (idx == 0)
            is_final_stage = (idx == self.layer_num - 1)

            DATA = self.get_input_cfg(idx)
            FILTER = self.get_filter_cfg(idx)
            OUTPUT = self.get_output_cfg(idx)

            # Split axes, etc
            if cfg is not None:
                # Vector length
                if self.pack:
                    if idx == 0: # Define input vlen for the first layer, no matter what it is
                        cfg.define_knob('vlen_input', get_vlen(DATA.C, self.target.kind.name))
                    if not FILTER.depthwise: # ONLY DEFINE vlen FOR CONV, because dw-conv uses the vlen of the previous layer
                        cfg.define_knob('vlen_conv_{}'.format(conv_count), get_vlen(OUTPUT.C, self.target.kind.name))
                        conv_count += 1

                    # Assuming no two dw-convs come together
                    if idx == 0 or conv_count < 2: #
                        vlen_i = cfg['vlen_input'].val
                    else: #
                        vlen_i = cfg['vlen_conv_{}'.format(conv_count-2)].val
                    if FILTER.depthwise: # dw-convs have same vlen_o and vlen_i
                        vlen_o = vlen_i
                    else: # Convs have their own vlen_o
                        vlen_o = cfg['vlen_conv_{}'.format(conv_count-1)].val

                    DATA.update_shape(vlen_i)
                    FILTER.update_shape(vlen_i, vlen_o)
                    OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer

                    if is_final_stage:
                        if FILTER.depthwise:
                            cfg.define_knob('bind_axis', [0, 1, 2, 3]) # 'oc', 'h', 'w', 'root'
                        else:
                            cfg.define_knob('bind_axis', [0, 1, 2, 3, 4]) # 'oc', 'ic', 'h', 'w', 'root'

                if not self.pack: # CUDA or tracing
                    OH, OW, OC = OUTPUT.H, OUTPUT.W, OUTPUT.C
                    c_filter = lambda x: x.size[-1] in get_vlen(OC, self.target.kind.name)

                    if FILTER.depthwise:
                        cfg.define_split('split_{}_c'.format(idx), cfg.axis(int(OC)), num_outputs=3, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 4
                            W_num_outputs = 3 if self.get_pattern() == "depth_conv" else 4

                            cfg.define_split('split_h', cfg.axis(int(OH)),
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                            cfg.define_split('split_w', cfg.axis(int(OW)),
                                                num_outputs=W_num_outputs,
                                                policy='factors')

                        cfg.define_split('split_{}_c'.format(idx), cfg.axis(int(OC)),
                                        num_outputs=3,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            cfg.define_split('split_0_rc', cfg.axis(int(OC)),
                                            num_outputs=3,
                                            policy='factors')
                else:
                    _, OC_chunk, OH, OW, _ = OUTPUT.get_shape()
                    c_filter = lambda x: x.size[-1] >= -1 # dummy

                    if FILTER.depthwise:
                        cfg.define_split('split_{}_c'.format(idx), cfg.axis(int(OC_chunk)), num_outputs=2, policy='factors', filter=c_filter)
                    else:
                        if is_final_stage:
                            H_num_outputs = 3
                            W_num_outputs = 3

                            cfg.define_split('split_h', cfg.axis(int(OH)),
                                            num_outputs=H_num_outputs,
                                            policy='factors')
                            cfg.define_split('split_w', cfg.axis(int(OW)),
                                                num_outputs=W_num_outputs,
                                                policy='factors')

                        cfg.define_split('split_{}_c'.format(idx), cfg.axis(int(OC_chunk)),
                                        num_outputs=2,
                                        policy='factors', filter=c_filter)

                        if is_first_stage:
                            cfg.define_split('split_0_rc', cfg.axis(int(OC_chunk)),
                                            num_outputs=2,
                                            policy='factors')


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
        self.tuned = True


    def update_all_shapes_from_tensors(self, input_shape, filter_shapes):
        if self.pack:
            feature_shape = None
            for idx in range(self.layer_num):
                DATA = self.get_input_cfg(idx)
                FILTER = self.get_filter_cfg(idx)
                OUTPUT = self.get_output_cfg(idx)

                feature_shape = input_shape if feature_shape is None else OUTPUT.get_shape()
                filter_shape = filter_shapes[idx]
                assert len(feature_shape) == 5 and len(filter_shape) == 6
                vlen_i = feature_shape[-1]
                vlen_o = filter_shape[-1]

                DATA.update_shape(vlen_i)
                FILTER.update_shape(vlen_i, vlen_o)
                OUTPUT.update_shape(vlen_o) # Actually overlapped with the input of next layer
        self.tuned = True


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
        self.out_dtype = dtype
        self.task_name = 'fused_conv2d.{}'.format('cuda' if self.target.kind.name == 'cuda' else 'x86')
        self.is_block = False
        self.layers = get_4D_shapes_from_params(p)
        self.layer_num = len(self.layers) - 1 # Excluding input
        self.workspace = workspace
        self.tuned = False

        # Temporary variables for returning the best_config
        device = 'cpu' if 'llvm' in self.target.kind.name else 'gpu'
        self.dir_name = '{}/logs/{}/layer/{}'.format(workspace, 'auto_scheduler' if use_auto_scheduler else 'autotvm', device)
        self.log_name = '{}_fused_{}.log'.format(self.get_pattern(), workload_name) # mv1_1, res_2x, etc


    def make_params(self, raw=True, layout='NHWC'):
        return {
            "Input": te.placeholder(self.get_input_cfg(0).get_shape(raw, layout), name='Input'),
            "Filters": [te.placeholder(self.get_filter_cfg(idx).get_shape(raw, layout), name='Filter_{}'.format(idx)) for idx in range(self.layer_num)],
            "Biases": [te.placeholder((self.get_output_cfg(idx).C,), name='Bias_{}'.format(idx)) for idx in range(self.layer_num)],
            "num_layers": self.layer_num,
            "strides": [[self.get_filter_cfg(idx).stride_h, self.get_filter_cfg(idx).stride_w] for idx in range(self.layer_num)],
            "paddings": [self.get_filter_cfg(idx).get_padding_shape() for idx in range(self.layer_num)], 
            "dilations": [[self.get_filter_cfg(idx).dilation_h, self.get_filter_cfg(idx).dilation_w] for idx in range(self.layer_num)], 
            "is_dws": [self.get_filter_cfg(idx).depthwise for idx in range(self.layer_num)], 
            "post_ops": [self.get_filter_cfg(idx).post_op for idx in range(self.layer_num)],
            "layouts": ["NCHW{}c".format(self.get_filter_cfg(idx).get_shape()[-1]) if len(self.get_filter_cfg(idx).get_shape()) != 4 else layout for idx in range(self.layer_num)],
            "out_dtype": self.out_dtype, 
        }


    def print_info(self):
        print("{} layers".format(self.layer_num))
        for i in range(self.layer_num):
            DATA, FILTER = self.layers[i]
            print('Input_{} size: {}'.format(i, DATA.get_shape()))
            print('Filter_{} size: {}, depthwise: {}, post_op: {}'.format(i, FILTER.get_shape(), FILTER.depthwise, FILTER.post_op))
        print('Is a block: {}'.format(self.is_block))
        print('Output size: {}'.format(DATA.get_shape()))


if __name__ == '__main__':
    parameters = (1, 56, 56, 128, 3, 1, 1, True, 'relu', 1, 64, 1, False, 'relu', False)
    target = tvm.target.Target('cuda')
    fc = FusionComposer(parameters, target=target)
    fc.print_info()


FUSION_COMPOSER = None
