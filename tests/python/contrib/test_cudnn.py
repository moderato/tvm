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
import tvm
from tvm.contrib import cudnn
import numpy as np
import topi.testing


def verify_conv2d(data_dtype, conv_dtype, tensor_format=0):
    in_channel = 4
    out_channel = 16
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    batch = 3
    height = 32
    weight = 32

    if not tvm.module.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cudnn.conv.output_shape", True):
        print("skip because cudnn is not enabled...")
        return
    if tensor_format == 0:
        xshape = [batch, in_channel, height, weight]
        wshape = [out_channel, in_channel, filter_h, filter_w]
    else:
        xshape = [batch, height, weight, in_channel]
        wshape = [out_channel, filter_h, filter_w, in_channel]

    X = tvm.placeholder(xshape, name='X', dtype=data_dtype)
    W = tvm.placeholder(wshape, name='W', dtype=data_dtype)
    Y = cudnn.conv_forward(X,
                           W,
                           [pad_h, pad_w],
                           [stride_h, stride_w],
                           [dilation_h, dilation_w],
                           conv_mode=1,
                           tensor_format=tensor_format,
                           conv_dtype=conv_dtype,
                           algo=-1)
    yshape = [x.value for x in Y.shape]
    s = tvm.create_schedule(Y.op)

    def verify():
        ctx = tvm.gpu(0)
        f = tvm.build(s, [X, W, Y], "cuda", target_host="llvm", name="conv2d")
        x_np = np.random.uniform(-1, 1, xshape).astype(data_dtype)
        w_np = np.random.uniform(-1, 1, wshape).astype(data_dtype)
        y_np = np.zeros(yshape).astype(data_dtype)
        x = tvm.nd.array(x_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        y = tvm.nd.array(y_np, ctx)
        if tensor_format == 0:
            c_np = topi.testing.conv2d_nchw_python(x_np, w_np, 1, 1)
        elif tensor_format == 1:
            wt = w_np.transpose((1, 2, 3, 0))  #OHWI => HWIO
            c_np = topi.testing.conv2d_nhwc_python(x_np, wt, 1, 1)

        f(x, w, y)
        tvm.testing.assert_allclose(y.asnumpy(), c_np, atol=1e-5, rtol=1e-3)

    verify()

def test_conv2d():
    verify_conv2d("float32", "float32", tensor_format=0)
    verify_conv2d("float16", "float32", tensor_format=1)
    #Not pass accuracy test, need check
    #verify_conv2d("float16", "float16", tensor_format=0)
    verify_conv2d("int8", "int32", tensor_format=1)


def verify_conv3d(data_dtype, conv_dtype, tensor_format=0):
    in_channel = 4
    out_channel = 16
    filter_d = 3
    filter_h = 3
    filter_w = 3
    pad_d = 1
    pad_h = 1
    pad_w = 1
    stride_d = 1
    stride_h = 1
    stride_w = 1
    dilation_d = 1
    dilation_h = 1
    dilation_w = 1
    batch = 3
    depth = 32
    height = 32
    weight = 32

    if not tvm.module.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cudnn.conv.output_shape", True):
        print("skip because cudnn is not enabled...")
        return

    xshape = [batch, in_channel, depth, height, weight]
    wshape = [out_channel, in_channel, filter_d, filter_h, filter_w]

    X = tvm.placeholder(xshape, name='X', dtype=data_dtype)
    W = tvm.placeholder(wshape, name='W', dtype=data_dtype)
    Y = cudnn.conv_forward(X,
                           W,
                           [pad_d, pad_h, pad_w],
                           [stride_d, stride_h, stride_w],
                           [dilation_d, dilation_h, dilation_w],
                           conv_mode=1,
                           tensor_format=tensor_format,
                           algo=-1,
                           conv_dtype=conv_dtype)
    yshape = [x.value for x in Y.shape]
    s = tvm.create_schedule(Y.op)

    def verify():
        ctx = tvm.gpu(0)
        f = tvm.build(s, [X, W, Y], "cuda", target_host="llvm", name="conv3d")
        x_np = np.random.uniform(-1, 1, xshape).astype(data_dtype)
        w_np = np.random.uniform(-1, 1, wshape).astype(data_dtype)
        y_np = np.zeros(yshape).astype(data_dtype)
        x = tvm.nd.array(x_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        y = tvm.nd.array(y_np, ctx)
        if tensor_format == 0:
            c_np = topi.testing.conv3d_ncdhw_python(x_np, w_np, 1, 1)
        else:
            raise AssertionError("For now, conv3d tensor format only support: 0(NCHW)")

        f(x, w, y)
        tvm.testing.assert_allclose(y.asnumpy(), c_np, atol=1e-5, rtol=1e-4)

    verify()

def test_conv3d():
    verify_conv3d("float32", "float32", tensor_format=0)

def test_grouped_conv2d():
    in_channel = 32
    out_channel = 32

    group_count = in_channel

    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [1, 32, 112, 112] # NCHW

    if not tvm.module.enabled("cuda"):
        print("skip because cuda is not enabled...")
        return
    if not tvm.get_global_func("tvm.contrib.cudnn.grouped_conv2d.output_shape", True):
        print("skip because cudnn is not enabled...")
        return
    wshape = cudnn.grouped_conv2d_w_shape(group_count,
                                          in_channel,
                                          out_channel,
                                          filter_h,
                                          filter_w)

    X = tvm.placeholder(xshape, name='X')
    W = tvm.placeholder(wshape, name='W')
    Y = cudnn.grouped_conv2d_forward(X,
                                     W,
                                     group_count,
                                     stride_h,
                                     stride_w,
                                     pad_h,
                                     pad_w,
                                     dilation_h,
                                     dilation_w,
                                     conv_mode=1,
                                     tensor_format=0,
                                     algo=1)
    yshape = [x.value for x in Y.shape]
    s =  tvm.create_schedule(Y.op)
    dtype = X.dtype

    def get_ref_data():
        x_np = np.random.uniform(size=xshape).astype(dtype)
        w_np = np.random.uniform(size=wshape).astype(dtype)
        y_np = topi.testing.depthwise_conv2d_python_nchw(x_np, w_np, stride=[stride_h, stride_w], padding="SAME")

        return x_np, w_np, y_np
    x_np, w_np, y_np = get_ref_data()
    
    def verify():
        ctx = tvm.gpu(0)
        x = tvm.nd.array(x_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        y = tvm.nd.array(np.zeros(yshape, dtype=dtype), ctx)
        f = tvm.build(s, [X, W, Y], "cuda", target_host="llvm", name="grouped_conv2d")
        f(x, w, y)
        np.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5)
    
    verify()

    
if __name__ == "__main__":
    test_conv2d()
    test_conv3d()
    test_grouped_conv2d()
