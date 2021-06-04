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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals, too-many-branches
"""Fused Conv2D Reference Data"""
import numpy as np
from .conv2d_nhwc_python import conv2d_nhwc_python
from .depthwise_conv2d_python import depthwise_conv2d_python_nhwc
from scipy.special import expit
import os

def get_fused_conv2d_ref_data(fc,
                                workload_name,
                                best_config=None,
                                save_data=False):
        if best_config:
            fc.update_all_shapes_from_best_cfg(best_config)
        ref_data = []
        ref_data_no_transform = []
        workspace = fc.workspace

        # Pretending the input_data is some output_data from stage -1
        input_cfg = fc.get_input_cfg(0)
        output_data = np.random.uniform(0.0, 0.1, size=(input_cfg.N, input_cfg.H, input_cfg.W, input_cfg.C)).astype(fc.output_dtype)
        ref_data_no_transform.append(output_data)
        ref_data.append(fc.tensor_transformation(output_data, input_cfg, 'data'))
        # params names for saving data
        params_name = ['input']

        for idx in range(fc.layer_num):
            f = fc.get_filter_cfg(idx)
            f_size = (f.H, f.W, f.O, f.I) if f.depthwise else (f.H, f.W, f.I, f.O)
            filter_data = np.random.uniform(0.0, 0.1, size=f_size).astype(fc.output_dtype)
            ref_data_no_transform.append(filter_data)
            ref_data.append(fc.tensor_transformation(filter_data, f, 'kernel'))
            input_data = np.copy(output_data)

            if f.depthwise:
                output_data = depthwise_conv2d_python_nhwc(input_data, filter_data, stride=[f.stride_h, f.stride_w], padding='SAME').astype(fc.output_dtype)
                params_name.append('filter_{}_d'.format(idx+1)) # Mark depthwise filter
            else: # Normal convolution
                output_data = conv2d_nhwc_python(input_data, filter_data, f.stride_h, padding=f.padding).astype(fc.output_dtype)
                params_name.append('filter_{}'.format(idx+1))

            if f.post_op is not None:
                n, h, w, oc = output_data.shape
                bias_np = np.random.uniform(0.0, 0.1, size=(oc,)).astype(fc.output_dtype)
                ref_data_no_transform.append(bias_np)
                ref_data.append(bias_np)

                post_op_scipy = np.zeros(shape=(n, h, w, oc))
                for c in range(oc):
                    post_op_scipy[:,:,:,c] = output_data[:,:,:,c] + bias_np[c]

                    # For ResNet / DenseNet blocks, etc
                    if fc.is_block:
                        post_op_scipy[:,:,:,c] = post_op_scipy[:,:,:,c] + input_data[:,:,:,c]

                    if f.post_op == 'relu':
                        post_op_scipy[:,:,:,c] = np.maximum(post_op_scipy[:,:,:,c], 0)
                    elif f.post_op == 'relu6':
                        post_op_scipy[:,:,:,c] = np.maximum(post_op_scipy[:,:,:,c], 0)
                        post_op_scipy[:,:,:,c] = np.minimum(post_op_scipy[:,:,:,c], 6)
                    elif f.post_op == 'sigmoid':
                        post_op_scipy[:,:,:,c] = expit(post_op_scipy[:,:,:,c])
                output_data = post_op_scipy.astype(fc.output_dtype)
                params_name.append('bias_{}'.format(idx+1))

            if idx == fc.layer_num - 1: # At the last stage, append output_data as the final output for reference
                output_cfg = fc.get_output_cfg(idx)
                ref_data_no_transform.append(output_data)
                ref_data.append(fc.tensor_transformation(output_data, output_cfg, 'data'))
        params_name.append('output')

        if save_data:
            # Save ref data
            folder_name = '{}/npy/fused/{}/'.format(workspace, workload_name)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            for i in range(0, len(ref_data)):
                filename = folder_name + params_name[i]
                # Transpose filter for cudnn: should be non-fortran order
                if fc.target.kind.name == 'cuda':
                    np.save(filename, ref_data[i])
                    if 'filter' in filename:
                        np.save(filename+'_transposed', np.array(ref_data[i].transpose(3, 2, 0, 1), order='C'))
                    else:
                        if len(ref_data[i].shape) == 4: # Don't need to save NCHW format for bias data
                            np.save(filename+'_NCHW', np.array(ref_data[i].transpose(0, 3, 1, 2), order='C'))
                        else:
                            np.save(filename, ref_data[i])
                else:
                    if 'filter' in filename:
                        np.save(filename+'_NCHWc', ref_data[i]) # NCHWc data
                        np.save(filename+'_transposed', np.array(ref_data_no_transform[i].transpose(3, 2, 0, 1), order='C'))
                    else:
                        if len(ref_data[i].shape) == 5: # Don't need to save NCHW format for bias data
                            np.save(filename+'_NCHWc', ref_data[i]) # NCHWc data
                            np.save(filename+'_NCHW', np.array(ref_data_no_transform[i].transpose(0, 3, 1, 2), order='C')) # NHWC to NCHW
                        else:
                            np.save(filename, ref_data[i])

        return ref_data
