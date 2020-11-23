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


def bottleneck_block(data, name, input_channels, t, output_channels, s,
                        kernel_size=(3, 3), insert_se=False, epsilon=1e-5, layout='NCHW', dtype="float32"):
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
        wshape = (t*input_channels, 1) + kernel_size
    elif layout == "NHWC":
        wshape = kernel_size + (t*input_channels, 1)
    else:
        raise ValueError("Invalid layout: " + layout)
    weight = relay.var(name + "_weight", shape=wshape, dtype=dtype)

    if kernel_size == (3, 3):
        p = 1
    elif kernel_size == (5, 5):
        p = 2
    else:
        p = 0
    conv2 = layers.conv2d(
        data=act1,
        weight=weight,
        channels=t*input_channels,
        groups=t*input_channels,
        kernel_size=kernel_size,
        strides=(s, s),
        padding=(p, p),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout, is_depthwise=True),
        name=name + '_depthwise_conv1')
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, axis=bn_axis, name=name+'_bn2')
    act2 = relay.nn.relu(data=bn2)

    if insert_se:
        act2 = se_block(act2, name=name + '_squeeze_excitation1', input_channels=t*input_channels, layout=layout, dtype=dtype)

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
    act3 = relay.nn.relu(data=bn3)

    if residual:
        output = relay.add(data, act3)
    else:
        output = act3
    return output


def se_block(data, name, input_channels, ratio=0.25, layout='NCHW', dtype='float32'):
    pooled = relay.nn.global_avg_pool2d(data, layout=layout)
    dense1 = layers.dense_add_bias(pooled, units=int(input_channels*ratio), name=name + '_dense1')
    relu = relay.nn.relu(data=dense1)
    dense2 = layers.dense_add_bias(relu, units=input_channels, name=name + '_dense2')
    sigmoid = relay.sigmoid(dense2)
    mul = relay.multiply(data, sigmoid)
    return mul


def mnasnet(
    cfgs,
    num_classes=1000,
    data_shape=(1, 3, 224, 224),
    dtype="float32",
    layout="NCHW",
):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    body = conv_block(data, "conv_block_1", 32, strides=(2, 2), layout=layout)
    body = separable_conv_block(
        body, "separable_conv_block_1", 32, 16, layout=layout, dtype=dtype
    )

    ic = 16
    for idx, (t, oc, n, s, k, se) in enumerate(cfgs):
        for i in range(n):
            body = bottleneck_block(body, 'bottleneck_block_{}_{}'.format(idx+1, i+1),
                                    ic, t, oc, s=(s if i == 0 else 1),
                                    kernel_size=(k, k), insert_se=se, layout=layout, dtype=dtype)
            ic = oc

    body = conv_block(body, 'conv_block_2', 
                        channels=1280, 
                        kernel_size=(1, 1), 
                        strides=(1, 1), 
                        padding=(0, 0), 
                        layout=layout)
    pool = relay.nn.global_avg_pool2d(data=body, layout=layout)
    output = conv_block(pool, 'conv_block_3',
                        channels=num_classes,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding=(0, 0),
                        layout=layout)
    return relay.Function(relay.analysis.free_vars(output), output)


def get_workload(batch_size=1, num_classes=1000, image_shape=(3, 224, 224), version='a1',
                 dtype='float32', layout='NCHW'):
    """Get benchmark workload for mnasnet

    Parameters
    ----------
    batch_size : int, optional
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape, cooperate with layout

    version: str, optional
        The version of mnasnet, by default a1.

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
    if version == 'a1':
        cfgs = [(6, 24, 2, 2, 3, False),
                (3, 40, 3, 2, 5, True),
                (6, 80, 4, 2, 3, False),
                (6, 112, 2, 1, 3, True),
                (6, 160, 3, 2, 5, True),
                (6, 320, 1, 1, 3, False)]
    elif version == 'b1':
        cfgs = [(3, 24, 3, 2, 3, False),
                (3, 40, 3, 2, 5, False),
                (6, 80, 3, 2, 5, False),
                (6, 96, 2, 1, 3, False),
                (6, 192, 4, 2, 5, False),
                (6, 320, 1, 1, 3, False)]
    net = mnasnet(cfgs, num_classes=num_classes, data_shape=data_shape, dtype=dtype, layout=layout)

    return create_workload(net)
