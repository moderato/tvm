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
"""Common topi utilities"""
from __future__ import absolute_import as _abs
from numbers import Integral
import numpy as np
import os


import tvm
from tvm import te
from tvm.tir import layout, bijective_layout
from . import tag, cpp


class InvalidShapeError(ValueError):
    """Invalid shape for a topi function. i.e. call winograd template for non-3x3 kernel)"""


def nchw_pack_layout(layout_info):
    """Check whether the layout type is NCHWinic"""
    return layout_info[:4] == "NCHW" and "c" in layout_info and "n" in layout_info


def nchw_xc_layout(layout_info):
    """Check whether the layout type is NCHWxc"""
    return layout_info[:4] == "NCHW" and "c" in layout_info and layout_info[4:-1].isnumeric()


def traverse_inline(s, final_op, callback):
    """Traverse computation graph and do auto inline

    Parameters
    ----------
    s: schedule
        The schedule
    final_op: Operation
        The final output operator.
    callback: callable
        The callback function on each op
    """
    visited = set()

    def _traverse(op):
        if op in visited:
            return
        visited.add(op)
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    _traverse(tensor.op)
        callback(op)

    _traverse(final_op)


def prod(x):
    """Get the product of every items in the tuple.

    Parameters
    ----------
    x: tuple
        Input tuple

    Returns
    -------
    value : Expr
        The result value
    """
    if not x:
        return tvm.tir.const(1, "int32")
    res = x[0]
    for i in range(1, len(x)):
        res = res * x[i]
    return res


def get_const_int(expr):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(expr, Integral):
        return expr
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        raise ValueError("Expect value to be constant int")
    return int(expr.value)


def get_const_float(expr):
    """Verifies expr is a floating point and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or float
        The input expression.

    Returns
    -------
    out_value : float
        The output.
    """
    if isinstance(expr, float):
        return float(expr)
    if not isinstance(expr, tvm.tir.FloatImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.FloatImm):
        raise ValueError("Expect value to be constant float")
    return float(expr.value)


def equal_const_int(expr, value):
    """Returns if expr equals value.

    Parameters
    ----------
    expr : tvm.Expr
        The input expression.

    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        return False
    return expr.value == value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    ana = None
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            ana = tvm.arith.Analyzer() if ana is None else ana
            elem = ana.simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)


def const_vector(vector, name="const_vector"):
    """convert a const numpy 1-dimensional vector to tvm tensor

    Parameters
    ----------
    vector: numpy.ndarray
        Const input array
    name: str, optional
        The name of output op

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    row = vector.shape[0]
    dtype = str(vector.dtype)
    idxm = tvm.tir.indexmod

    def select_array(i):
        now = tvm.tir.const(0.0, dtype)
        for ii in range(row):
            now = tvm.tir.Select(
                tvm.tir.all(idxm(i, row) == ii),
                tvm.tir.const(vector[ii], dtype),
                now,
            )
        return now

    return te.compute(vector.shape, select_array, name=name)


def get_float_tuple(in_tuple):
    """Verifies input tuple is FloatImm, returns tuple of float.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of float
        The output.
    """
    return tuple(get_const_float(elem) for elem in in_tuple)


def simplify(expr):
    """Simplify the expression if it is Expr, directly return if it is int.

    Parameters
    ----------
    expr : Expr or int
        The input.

    Returns
    -------
    out : Expr or int
        The simplified output
    """
    return tvm.arith.Analyzer().simplify(expr) if isinstance(expr, tvm.tir.PrimExpr) else expr


def ravel_index(indices, shape):
    """Flatten the index tuple to 1D

    Parameters
    ----------
    indices : tuple of int or tvm.tir.IntImm
        The input coordinates

    shape : tuple of int
        Shape of the tensor.

    Returns
    -------
    idx : int or Expr
        The index after flattening
    """
    idx = None
    for i, (shape_val, ind) in enumerate(zip(shape, indices)):
        if i != 0:
            idx = idx * shape_val + ind
        else:
            idx = ind
    return idx


def unravel_index(idx, shape):
    """Convert the flattened ind to the coordinate array

    Parameters
    ----------
    idx : int or tvm.tir.IntImm
        The 1D index

    shape : tuple of int
        Shape of the tensor

    Returns
    -------
    indices : tuple of int or tvm.tir.IntImm
        Corresponding coordinate of the 1D index
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod
    indices = []
    for i in range(len(shape) - 1, -1, -1):
        indices.append(idxm(idx, shape[i]))
        idx = idxd(idx, shape[i])
    indices = indices[::-1]
    return indices


def const_matrix(matrix, name="const_matrix"):
    """convert a const numpy 2-dimensional matrix to tvm tensor

    Parameters
    ----------
    matrix: numpy.ndarray
        Const input array
    name: str, optional
        The name of output op

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    row, col = matrix.shape
    dtype = str(matrix.dtype)
    idxm = tvm.tir.indexmod

    def select_array(i, j):
        now = tvm.tir.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.tir.Select(
                    tvm.tir.all(idxm(i, row) == ii, idxm(j, col) == jj),
                    tvm.tir.const(matrix[ii][jj], dtype),
                    now,
                )
        return now

    return te.compute(matrix.shape, select_array, name=name, attrs={"const_matrix": True})


def get_max_power2_factor(n, max_value=None):
    """Get max factor of n in power of 2. If max_value is specificed, max factor
    value will be no more max_value,

    Parameter
    ---------
    n : int
        The input value

    max_value : int, optional
        The max value for the factor

    Returns
    -------
    factor : int
        The max factor in power of 2.
    """
    x = 1
    while n % 2 == 0:
        if max_value is not None and max_value < x * 2:
            break
        x *= 2
        n /= 2
    return x


def get_shape(src_shape, src_layout, dst_layout):
    """Given a source shape, a source layout and a destination layout, infer
    the destination shape.

    Parameter
    ---------
    src_shape : tuple of int or IntImm
        Source shape

    src_layout : str or Layout
        Source layout

    dst_layout : str or Layout
        Destination layout

    Returns
    -------
    dst_shape : tuple of int
        Destination shape
    """
    if src_layout == dst_layout:
        return get_const_tuple(src_shape)

    if isinstance(src_layout, str):
        src_layout = layout(src_layout)
    if isinstance(dst_layout, str):
        dst_layout = layout(dst_layout)

    assert len(src_layout) == len(dst_layout), "Incompatible layout %s vs %s" % (
        src_layout,
        dst_layout,
    )

    layout_mapping = bijective_layout(src_layout, dst_layout)
    dst_indices = layout_mapping.forward_index(tvm.runtime.convert(list(range(len(src_layout)))))

    return get_const_tuple(tuple([src_shape[i.value] for i in dst_indices]))


def within_index(b, e, s, i):
    """Return a boolean value that indicates if i is within the given index.

    Parameters
    ----------
    b : Expr
      beginning of the index

    e : Expr
      end of the index

    s : Expr
      strides of index

    i : Expr
      array position

    Returns
    -------
    selected: Expr
        bool expression that is True is the array position would be selected
        by the index and False otherwise
    """
    bc = tvm.tir.Select(s < 0, i <= e, i < b)
    ec = tvm.tir.Select(s < 0, i > b, i >= e)
    ss = te.if_then_else(s < 0, ((i - e) + (e % te.abs(s)) + 1) % te.abs(s), (i - b) % s)
    return tvm.tir.Select(tvm.tir.Or(bc, ec), tvm.tir.const(False), ss.equal(0))


def make_idx(b, e, s, z, i):
    """Return the array position in the selection that corresponds to an
    array position in the full array.

    The returned value is only meaningful if within_index() returns True
    for the same set of parameters.

    Parameters
    ----------
    b : Expr
      beginning of the index

    e : Expr
      end of the index

    s : Expr
      strides of index

    z : Expr
      size of the indexed dimension

    i : Expr
      array position

    Returns
    -------
    postion: Expr
        int expression that corresponds to an array position in the selection.
    """
    bc = tvm.tir.Select(s < 0, i <= e, i < b)
    ec = tvm.tir.Select(s < 0, i > b, i >= e)

    # Clamp to array size
    b = tvm.tir.Select(z < b, z - 1, b)

    ss = tvm.tir.if_then_else(s < 0, (b - i) // te.abs(s), (i - b) // s)
    return tvm.tir.if_then_else(tvm.tir.Or(bc, ec), 88, ss)


def is_empty_shape(shape):
    """Check whether an input shape has dimesion with size 0.

    Parameter
    ---------
    shape : list of Expr
      Input shape

    Returns
    -------
    is_empty: bool
      Whether input shape is empty or has dimesion with size 0.
    """
    return cpp.utils.is_empty_shape(shape)


class FeatureConfig:
    def __init__(self, N, H, W, C):
        self.N = int(N)
        self.H = int(H)
        self.W = int(W)
        self.C = int(C)
        self.vlen = -1
        self.shape = (N, H, W, C)
    def update_shape(self, vlen):
        self.vlen = vlen
        C_chunk = tvm.tir.indexdiv(self.C, vlen).value
        self.shape = (self.N, C_chunk, self.H, self.W, vlen)
    def get_shape(self, raw=False, layout='NHWC'):
        if raw:
            return (self.N, self.H, self.W, self.C) if layout == 'NHWC' else (self.N, self.C, self.H, self.W) # NCHW
        return self.shape


class FilterConfig:
    def __init__(self, H, W, I, O, stride_h, stride_w, depthwise, post_op, dilation=1, padding='SAME', layout='NHWC'):
        assert post_op in [None, 'bias', 'relu', 'relu6', 'sigmoid']
        self.H = int(H)
        self.W = int(W)
        self.I = int(I)
        self.O = int(O)
        self.stride_h = int(stride_h)
        self.stride_w = int(stride_w)
        self.depthwise = bool(depthwise)
        self.post_op = post_op
        self.dilation_h = int(dilation)
        self.dilation_w = int(dilation)
        self.shape = (int(H), int(W), int(O), int(I)) if depthwise else (int(H), int(W), int(I), int(O))
        self.vlen_i = -1
        self.vlen_o = -1
        if isinstance(padding, str):
            self.padding = padding
            self.padding_shape = None
        else:
            self.padding = None
            self.padding_shape = padding
    def update_shape(self, vlen_i, vlen_o):
        self.vlen_i = vlen_i
        self.vlen_o = vlen_o
        IC_chunk = tvm.tir.indexdiv(self.I, vlen_i).value
        OC_chunk = tvm.tir.indexdiv(self.O, vlen_o).value
        self.shape = (OC_chunk, IC_chunk, self.H, self.W, vlen_i, vlen_o) if not self.depthwise else (OC_chunk, 1, self.H, self.W, 1, vlen_o)
    def get_shape(self, raw=False, layout='NHWC'):
        if raw:
            if layout == 'NHWC':
                return (int(self.H), int(self.W), int(self.O), int(self.I)) if self.depthwise else (int(self.H), int(self.W), int(self.I), int(self.O))
            else: # NCHW
                return (int(self.O), int(self.I), int(self.H), int(self.W))
        return self.shape
    def get_padding_shape(self):
        assert(self.padding_shape is not None)
        return self.padding_shape[0], self.padding_shape[1], self.padding_shape[2], self.padding_shape[3]
    def get_stride(self):
        return self.stride_h, self.stride_w
    def get_dilation(self):
        return self.dilation_h, self.dilation_w


def get_vlen(axis_length, device=None):
    if device == 'cuda':
        candidates = [16, 24, 32, 64, 128]
    elif 'llvm' in device:
        candidates = [8, 16, 24, 32, 64] # Non-c axes don't matter
    vlens = []
    for i in candidates:
        if axis_length % i == 0:
            vlens.append(i)
    assert vlens != []
    return vlens


def get_4D_shapes_from_params(p):
    from .nn import get_pad_tuple
    idx = 0
    OUTPUT = None
    layers = []
    while 1:
        if idx + 5 > len(p): # Skip is_block for now
            break

        if not OUTPUT:
            FEATURE = FeatureConfig(*p[idx:(idx+4)])
            idx += 4
        else:
            FEATURE = OUTPUT

        is_depthwise = p[idx+3]
        # Depthwise: I: 1 (channel_multiplier), O: same as FEATURE's C
        # Normal: I: same as FEATURE's C, O: same as output's C
        FILTER = FilterConfig(p[idx], p[idx], 1 if is_depthwise else FEATURE.C, FEATURE.C if is_depthwise else p[idx+1],\
                                    p[idx+2], *p[(idx+2):(idx+5)])
        idx += 5
        layers.append((FEATURE, FILTER))

        # Compute the output shape with the original input size, i.e. WITHOUT INPUT PACKING
        dilated_kernel_h = (FILTER.H - 1) * FILTER.dilation_h + 1
        dilated_kernel_w = (FILTER.W - 1) * FILTER.dilation_w + 1
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            FILTER.padding, (dilated_kernel_h, dilated_kernel_w))
        if FILTER.padding_shape is None:
            FILTER.padding_shape = (pad_top, pad_left, pad_down, pad_right)

        # Make output
        ON = FEATURE.N
        OH = simplify((FEATURE.H - dilated_kernel_h + pad_top + pad_down) // FILTER.stride_h + 1)
        OW = simplify((FEATURE.W - dilated_kernel_w + pad_left + pad_right) // FILTER.stride_w + 1)
        OC = FILTER.I * FILTER.O if FILTER.depthwise else FILTER.O
        OUTPUT = FeatureConfig(ON, OH, OW, OC)

    layers.append((OUTPUT,))

    return layers


def export_kernel_launch_config(workload_name, output_shape, best_config, target, unfused=False):
    assert best_config is not None
    config_dict = best_config.to_json_dict()

    if target == 'cuda':
        if not os.path.exists('generated_kernels/gpu/fused/kernel_launch_config'):
            os.mkdir('generated_kernels/gpu/fused/kernel_launch_config')
        n = output_shape[0]
        ho = output_shape[1]
        wo = output_shape[2]
        recompute = output_shape[3]

        # print('n: {}, ho: {}, wo: {}, recompute: {}'.format(n, ho, wo, recompute))
        for e in config_dict['entity']:
            if e[0] == 'split_1_h': # TODO: Fix it layer with a layer num
                thz = e[2][1]
                thy = e[2][2]
                for ee in e[2][1:]:
                    ho = (ho + ee - 1) // ee
                    # print('ho: {}', ho)
            elif e[0] == 'split_1_w':
                for ee in e[2][1:]:
                    wo = (wo + ee - 1) // ee
                    # print('wo: {}', wo)
            elif e[0] == 'split_1_c':
                thx = e[2][2]
                for ee in e[2][1:]:
                    recompute = (recompute + ee - 1) // ee
                    # print('recompute: {}', recompute)
        blx = n * ho * wo * recompute
        print('n: {}, ho: {}, wo: {}, recompute: {}'.format(n, ho, wo, recompute))
        print('thx: {}, thy: {}, thz: {}, blx: {}'.format(thx, thy, thz, blx))

        with open('generated_kernels/gpu/fused/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
            f.write('{},{},{},{}'.format(thx, thy, thz, blx))
    else:
        if not os.path.exists('generated_kernels/cpu/{}/kernel_launch_config'.format('unfused' if unfused else 'fused')):
            os.mkdir('generated_kernels/cpu/{}/kernel_launch_config'.format('unfused' if unfused else 'fused'))
        if unfused:
            vlen_ic, vlen_oc = -1, -1
            for e in config_dict['entity']:
                if e[0] == 'tile_ic':
                    vlen_ic = e[2][-1]
                if e[0] == 'tile_oc':
                    vlen_oc = e[2][-1]
            assert vlen_ic != -1 and vlen_oc != -1
            with open('generated_kernels/cpu/unfused/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
                f.write('{},{}'.format(vlen_ic, vlen_oc))
        else:
            vlens = get_CPU_vlen_from_config(best_config, 'all')
            vlens = [str(v) for v in vlens]
            with open('generated_kernels/cpu/fused/kernel_launch_config/{}_config.csv'.format(workload_name), 'w') as f:
                f.write(','.join(vlens))


def get_CPU_vlen_from_config(best_config=None, cfg_key=''):
    from tvm.autotvm.task.space import FallbackConfigEntity
    if best_config is None or isinstance(best_config, FallbackConfigEntity):
        return 16
    config_dict = best_config.to_json_dict()
    if cfg_key != 'all':
        for e in config_dict['entity']:
            if e[0] == cfg_key:
                return int(e[2])
    else: # Get all vlens, sort by keys and return values
        vlens_dict = {}
        for e in config_dict['entity']:
            if 'vlen' in e[0]:
                vlens_dict[e[0]] = int(e[2])
        vlens = []
        for k in sorted (vlens_dict.keys()):
            vlens.append(vlens_dict[k])
        return vlens


def attrs_to_fusion_param(attrs, inputs):
    import re
    _NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
    param = []

    num_layers = attrs.num_layers
    for l in range(num_layers):
        layout = attrs.data_layout_array[l]
        if l == 0:
            if layout == 'NHWC':
                param.append(inputs[0].shape[0])
                param.append(inputs[0].shape[1])
                param.append(inputs[0].shape[2])
                param.append(inputs[0].shape[3])
            elif layout == 'NCHW':
                param.append(inputs[0].shape[0])
                param.append(inputs[0].shape[2])
                param.append(inputs[0].shape[3])
                param.append(inputs[0].shape[1])
            elif _NCHWc_matcher.match(layout):
                param.append(inputs[0].shape[0])
                param.append(inputs[0].shape[2])
                param.append(inputs[0].shape[3])
                param.append(inputs[0].shape[1] * inputs[0].shape[4])
            else:
                raise Exception("Layout {} is not supported!".format(layout))
        param.append(attrs.kernel_size_array[l][0])
        param.append(attrs.channels_array[l] // attrs.groups_array[l])
        param.append(attrs.strides_array[l][0])
        param.append(bool(attrs.groups_array[l] > 1))
        param.append(attrs.post_op_array[l])
    param.append(False)

    return param


def fused_conv2d_workload_to_fusion_param(workload):
    return tensors_to_fusion_param(num_layers=workload[4], 
                                    Input=workload[1][1], 
                                    Filters=[w[1] for w in workload[2]], 
                                    strides=workload[5], 
                                    is_dws=workload[8], 
                                    post_ops=workload[9], 
                                    layouts=workload[10])


def tensors_to_fusion_param(num_layers, Input, Filters, strides, is_dws, post_ops, layouts):
    """
        Accept Input and Filters as either te.Tensor or tuple
    """
    import re
    _NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
    param = []

    for l in range(num_layers):
        layout = layouts[l]
        Filter = Filters[l]
        input_shape = Input.shape if isinstance(Input, te.Tensor) else Input
        filter_shape = Filter.shape if isinstance(Filter, te.Tensor) else Filter
        if layout == 'NHWC':
            if l == 0:
                param.append(input_shape[0])
                param.append(input_shape[1])
                param.append(input_shape[2])
                param.append(input_shape[3])
            h, _, _, cm_or_oc = filter_shape # Channel multiplier or OC
        elif layout == 'NCHW':
            if l == 0:
                param.append(input_shape[0])
                param.append(input_shape[2])
                param.append(input_shape[3])
                param.append(input_shape[1])
            cm_or_oc, _, h, _ = filter_shape
        elif _NCHWc_matcher.match(layout):
            if l == 0:
                param.append(input_shape[0])
                param.append(input_shape[2])
                param.append(input_shape[3])
                param.append(input_shape[1] * input_shape[4])
            oc_chunk, _, h, _, _, vec = filter_shape
            if is_dws[l]:
                cm_or_oc = 1
            else:
                cm_or_oc = oc_chunk * vec
        else:
            raise Exception("Layout {} is not supported!".format(layout))

        param.append(h)
        param.append(cm_or_oc)
        param.append(strides[l][0])
        param.append(is_dws[l])
        param.append(post_ops[l])
    param.append(False)

    return param


def get_FLOP(p):
    layers = get_4D_shapes_from_params(p)
    flop = 0
    for l in range(len(layers)):
        fcfg = layers[l][1]
        ocfg = layers[l+1][0]

        if fcfg.depthwise:
            flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W)
        else:
            flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C) * (fcfg.H * fcfg.W * fcfg.I)

        if fcfg.post_op:
            flop += 2 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
    return flop


def get_theoretical_mem_bytes(p):
    layers = get_4D_shapes_from_params(p)
    mem = 0
    for l in range(len(layers)):
        icfg = layers[l][0]
        fcfg = layers[l][1]
        ocfg = layers[l+1][0]

        mem += 4 * (fcfg.H * fcfg.W * fcfg.I * fcfg.O)
        if l == 0:
            mem += 4 * (icfg.N * icfg.H * icfg.W * icfg.C)
        elif l == len(layers) - 1:
            mem += 4 * (ocfg.N * ocfg.H * ocfg.W * ocfg.C)
        if fcfg.post_op:
            mem += 4 * ocfg.C
    return mem


def get_stages_and_cfgs(s, outs):
    stage_dict = {}
    layer_output_dict = {}
    param_dict = {}
    def get_tensors(outs):
        def traverse(prev_tensor, tensors):
            for t in tensors:
                op = t.op
                name = op.name
                prev_op_name = prev_tensor.op.name if prev_tensor is not None else None
                if op not in s.outputs and (prev_op_name == 'T_add' or prev_op_name == 'T_relu') and isinstance(op, te.ComputeOp):
                    s[op].compute_inline()
                if 'PaddedInput' in name:
                    stage_dict[name] = t
                elif 'BiasAdd' in name or 'ReLU' in name or 'ReLU6' in name or 'Sigmoid' in name:
                    _, n, i = name.split('_')
                    stage_dict['Output_{}_{}'.format(i, n)] = t
                elif 'Bias' in name or 'Filter' in name:
                    param_dict[name] = t
                elif 'Conv2dOutput' in name:
                    i = name.split('_')[-1]
                    stage_dict['Output_{}'.format(i)] = t
                elif 'Input' in name:
                    if 'PaddedInput_0' not in stage_dict.keys():
                        stage_dict[name] = t
                elif 'placeholder' in name:
                    i = prev_op_name.split('_')[-1]
                    if 'Conv2d' in prev_op_name: # Filter
                        param_dict['Filter_{}'.format(i)] = t
                    elif 'BiasAdd' in prev_op_name: # Bias
                        param_dict['{}_{}'.format('Bias', i)] = t
                    else:
                        continue
                elif 'T_add' in name or 'T_relu' in name: # Handle it later in the outside function
                    pass
                else:
                    raise Exception("Unknown tensor type: {}!".format(name))
                traverse(t, op.input_tensors)

        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        traverse(None, outs)

    get_tensors(outs)
    layer_num = 0
    post_ops = []
    padded = []
    while 1:
        if 'Output_{}'.format(layer_num) not in stage_dict.keys():
            break
        layer_num += 1
    for idx in range(layer_num):
        if 'Output_{}_ReLU6'.format(idx) in stage_dict.keys():
            post_ops.append('relu6')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_ReLU6'.format(idx)]
        elif 'Output_{}_ReLU'.format(idx) in stage_dict.keys():
            post_ops.append('relu')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_ReLU'.format(idx)]
        elif 'Output_{}_Sigmoid'.format(idx) in stage_dict.keys():
            post_ops.append('sigmoid')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_Sigmoid'.format(idx)]
        elif 'Output_{}_BiasAdd'.format(idx) in stage_dict.keys():
            post_ops.append('bias')
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}_BiasAdd'.format(idx)]
        else:
            post_ops.append(None)
            layer_output_dict['Layer_{}'.format(idx)] = stage_dict['Output_{}'.format(idx)]

        if 'FusedConv2D_PaddedInput_{}'.format(idx) in stage_dict.keys():
            padded.append(True)
        else:
            padded.append(False)

    # The final output is some extra add or relu
    if 'T_add' in outs[0].op.name or 'T_relu' in outs[0].op.name:
        layer_output_dict['Layer_{}'.format(layer_num-1)] = outs[0]

    return stage_dict, layer_output_dict, param_dict, layer_num, post_ops, padded
