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
"""
Port of NNVM version of MobileNet to Relay.
"""
# pylint: disable=invalid-name

from tvm import relay
from . import layers
from .init import create_workload


def conv_block(
    data,
    name,
    channels,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
):
    """Helper function to construct conv_bn-relu"""
    bn_axis = layout.index('C')
    # convolution + bn + relu
    conv = layers.conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name+'_conv')
    bn = layers.batch_norm_infer(data=conv, epsilon=epsilon, axis=bn_axis, name=name + '_bn')
    act = relay.nn.relu(data=bn)
    return act


def separable_conv_block(
    data,
    name,
    depthwise_channels,
    pointwise_channels,
    kernel_size=(3, 3),
    downsample=False,
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
    dtype="float32",
):
    """Helper function to get a separable conv block"""
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    # depthwise convolution + bn + relu
    if layout == "NCHW":
        wshape = (depthwise_channels, 1) + kernel_size
    elif layout == "NHWC":
        wshape = kernel_size + (depthwise_channels, 1)
    else:
        raise ValueError("Invalid layout: " + layout)
    bn_axis = layout.index("C")
    weight = relay.var(name + "_weight", shape=wshape, dtype=dtype)
    conv1 = layers.conv2d(
        data=data,
        weight=weight,
        channels=depthwise_channels,
        groups=depthwise_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout, is_depthwise=True),
        name=name + "_depthwise_conv1",
    )
    bn1 = layers.batch_norm_infer(data=conv1, epsilon=epsilon, axis=bn_axis, name=name + "_bn1")
    act1 = relay.nn.relu(data=bn1)
    # pointwise convolution + bn + relu
    conv2 = layers.conv2d(
        data=act1,
        channels=pointwise_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + "_conv2",
    )
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, axis=bn_axis, name=name + "_bn2")
    act2 = relay.nn.relu(data=bn2)
    return act2


def bottleneck_block(data, name, input_channels, output_channels, t, s,
                        epsilon=1e-5, layout='NCHW', dtype="float32"):
    bn_axis = layout.index('C')
    residual = (input_channels == output_channels) and (s == 1)

    conv1 = layers.conv2d(
        data=data,
        channels=t*input_channels, # Expansion on channels
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name+'_conv1')
    bn1 = layers.batch_norm_infer(data=conv1, epsilon=epsilon, axis=bn_axis, name=name+'_bn1')
    act1 = relay.nn.relu(data=bn1)

    if layout == "NCHW":
        wshape = (t*input_channels, 1) + (3, 3)
    elif layout == "NHWC":
        wshape = (3, 3) + (t*input_channels, 1)
    else:
        raise ValueError("Invalid layout: " + layout)
    weight = relay.var(name + "_depthwise_conv_weight", shape=wshape, dtype=dtype)
    conv2 = layers.conv2d(
        data=act1,
        weight=weight,
        channels=t*input_channels,
        groups=t*input_channels,
        kernel_size=(3, 3),
        strides=(s, s),
        padding=(1, 1),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout, is_depthwise=True),
        name=name + '_depthwise_conv')
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, axis=bn_axis, name=name+'_bn2')
    act2 = relay.nn.relu(data=bn2)

    conv3 = layers.conv2d(
        data=act2,
        channels=output_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + '_conv2')
    bn3 = layers.batch_norm_infer(data=conv3, epsilon=epsilon, axis=bn_axis, name=name+'_bn3')

    if residual:
        output = relay.add(data, bn3)
    else:
        output = bn3
    return output


def mobile_net(
    num_classes=1000,
    data_shape=(1, 3, 224, 224),
    dtype="float32",
    alpha=1.0,
    is_shallow=False,
    layout="NCHW",
):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    body = conv_block(data, "conv_block_1", int(32 * alpha), strides=(2, 2), layout=layout)
    body = separable_conv_block(
        body, "separable_conv_block_1", int(32 * alpha), int(64 * alpha), layout=layout, dtype=dtype
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_2",
        int(64 * alpha),
        int(128 * alpha),
        downsample=True,
        layout=layout,
        dtype=dtype,
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_3",
        int(128 * alpha),
        int(128 * alpha),
        layout=layout,
        dtype=dtype,
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_4",
        int(128 * alpha),
        int(256 * alpha),
        downsample=True,
        layout=layout,
        dtype=dtype,
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_5",
        int(256 * alpha),
        int(256 * alpha),
        layout=layout,
        dtype=dtype,
    )
    body = separable_conv_block(
        body,
        "separable_conv_block_6",
        int(256 * alpha),
        int(512 * alpha),
        downsample=True,
        layout=layout,
        dtype=dtype,
    )
    if is_shallow:
        body = separable_conv_block(
            body,
            "separable_conv_block_7",
            int(512 * alpha),
            int(1024 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
        body = separable_conv_block(
            body,
            "separable_conv_block_8",
            int(1024 * alpha),
            int(1024 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
    else:
        for i in range(7, 12):
            body = separable_conv_block(
                body,
                "separable_conv_block_%d" % i,
                int(512 * alpha),
                int(512 * alpha),
                layout=layout,
                dtype=dtype,
            )
        body = separable_conv_block(
            body,
            "separable_conv_block_12",
            int(512 * alpha),
            int(1024 * alpha),
            downsample=True,
            layout=layout,
            dtype=dtype,
        )
        body = separable_conv_block(
            body,
            "separable_conv_block_13",
            int(1024 * alpha),
            int(1024 * alpha),
            layout=layout,
            dtype=dtype,
        )
    pool = relay.nn.global_avg_pool2d(data=body, layout=layout)
    flatten = relay.nn.batch_flatten(data=pool)
    weight = relay.var("fc_weight")
    bias = relay.var("fc_bias")
    fc = relay.nn.dense(data=flatten, weight=weight, units=num_classes)
    fc = relay.nn.bias_add(fc, bias)
    softmax = relay.nn.softmax(data=fc)
    return relay.Function(relay.analysis.free_vars(softmax), softmax)


def mobile_net_v2(num_classes=1000, data_shape=(1, 3, 224, 224),
                    dtype='float32', alpha=1.0, layout='NCHW'):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    body = conv_block(data, 'conv_block_1', 32, strides=(2, 2),
                      layout=layout)

    cfgs = [(1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)]

    ic = 32
    for idx, (t, oc, n, s) in enumerate(cfgs):
        for i in range(n):
            body = bottleneck_block(body, 'bottleneck_block_{}_{}'.format(idx+1, i+1),
                                    ic, oc, t, s if i == 0 else 1,
                                    layout=layout, dtype=dtype)
            ic = oc

    body = conv_block(body, 'conv_block_2', 
                        channels=1280, 
                        kernel_size=(1, 1), 
                        strides=(1, 1), 
                        padding=(0, 0), 
                        layout=layout)
    pool = relay.nn.global_avg_pool2d(data=body, layout=layout)
    flatten = relay.nn.batch_flatten(data=pool)
    weight = relay.var("fc_weight")
    bias = relay.var("fc_bias")
    fc = relay.nn.dense(data=flatten, weight=weight, units=num_classes)
    fc = relay.nn.bias_add(fc, bias)
    return relay.Function(relay.analysis.free_vars(fc), fc)


def get_workload(batch_size=1, num_classes=1000, image_shape=(3, 224, 224), version='v1',
                 dtype='float32', layout='NCHW'):
    """Get benchmark workload for mobilenet

    Parameters
    ----------
    batch_size : int, optional
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape, cooperate with layout

    version: str, optional
        The version of mnasnet, by default v1.

    dtype : str, optional
        The data type

    layout : str, optional
        The data layout of image_shape and the operators
        cooperate with image_shape

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a MobileNet network.

    params : dict of str to NDArray
        The parameters.
    """
    data_shape = tuple([batch_size] + list(image_shape))
    if version == 'v1':
        net = mobile_net(num_classes=num_classes, data_shape=data_shape,
                        dtype=dtype, alpha=1.0, is_shallow=False,
                        layout=layout)
    elif version == 'v2':
        net = mobile_net_v2(num_classes=num_classes, data_shape=data_shape,
                    dtype=dtype, alpha=1.0, layout=layout)

    return create_workload(net)
