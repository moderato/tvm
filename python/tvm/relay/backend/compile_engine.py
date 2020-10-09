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
# pylint: disable=len-as-condition,no-else-return,invalid-name
"""Backend code generation engine."""
from __future__ import absolute_import

import logging
import numpy as np
import tvm
from tvm import te
from tvm.runtime import Object
from ... import target as _target
from ... import autotvm
from .. import function as _function
from .. import ty as _ty
from . import _backend

logger = logging.getLogger('compile_engine')
autotvm_logger = logging.getLogger('autotvm')
FUSION_PATTERNS = [
    ["nn.conv2d", "multiply", "add", "nn.relu", "nn.conv2d", "multiply", "add", "nn.relu"],
    ["nn.conv2d", "nn.conv2d"],
]

@tvm._ffi.register_object("relay.LoweredOutput")
class LoweredOutput(Object):
    """Lowered output"""
    def __init__(self, outputs, implement):
        self.__init_handle_by_constructor__(
            _backend._make_LoweredOutput, outputs, implement)


@tvm._ffi.register_object("relay.CCacheKey")
class CCacheKey(Object):
    """Key in the CompileEngine.

    Parameters
    ----------
    source_func : tvm.relay.Function
        The source function.

    target : tvm.Target
        The target we want to run the function on.
    """
    def __init__(self, source_func, target):
        self.__init_handle_by_constructor__(
            _backend._make_CCacheKey, source_func, target)


@tvm._ffi.register_object("relay.CCacheValue")
class CCacheValue(Object):
    """Value in the CompileEngine, including usage statistics.
    """


def _get_cache_key(source_func, target):
    if isinstance(source_func, _function.Function):
        if isinstance(target, str):
            target = _target.create(target)
            if not target:
                raise ValueError("Need target when source_func is a Function")
        return CCacheKey(source_func, target)
    if not isinstance(source_func, CCacheKey):
        raise TypeError("Expect source_func to be CCacheKey")
    return source_func


def get_shape(shape):
    """Convert the shape to correct dtype and vars."""
    ret = []
    for dim in shape:
        if isinstance(dim, tvm.tir.IntImm):
            val = int(dim)
            assert val <= np.iinfo(np.int32).max
            ret.append(tvm.tir.IntImm("int32", val))
        elif isinstance(dim, tvm.tir.Any):
            ret.append(te.var("any_dim", "int32"))
        else:
            ret.append(dim)
    return ret


def get_valid_implementations(op, attrs, inputs, out_type, target):
    """Get all valid implementations from the op strategy.

    Note that this function doesn't support op with symbolic input shapes.

    Parameters
    ----------
    op : tvm.ir.Op
        Relay operator.

    attrs : object
        The op attribute.

    inputs : List[tvm.te.Tensor]
        Input tensors to the op.

    out_type : relay.Type
        The output type.

    target : tvm.target.Target
        The target to compile the op.

    Returns
    -------
    ret : List[relay.op.OpImplementation]
        The list of all valid op implementations.
    """
    fstrategy = op.get_attr("FTVMStrategy")
    assert fstrategy is not None, "%s doesn't have FTVMStrategy registered" % op.name
    with target:
        strategy = fstrategy(attrs, inputs, out_type, target)
    analyzer = tvm.arith.Analyzer()
    ret = []
    for spec in strategy.specializations:
        if spec.condition:
            # check if all the clauses in the specialized condition are true
            flag = True
            for clause in spec.condition.clauses:
                clause = analyzer.canonical_simplify(clause)
                if isinstance(clause, tvm.tir.IntImm) and clause.value:
                    continue
                flag = False
                break
            if flag:
                for impl in spec.implementations:
                    ret.append(impl)
        else:
            for impl in spec.implementations:
                ret.append(impl)
    return ret


def select_implementation(op, attrs, inputs, out_type, target, use_autotvm=True):
    """Select the best implementation from the op strategy.

    If use_autotvm is True, it'll first try to find the best implementation
    based on AutoTVM profile results. If no AutoTVM profile result is found,
    it'll choose the implementation with highest plevel.

    If use_autotvm is False, it'll directly choose the implementation with
    highest plevel.

    Note that this function doesn't support op with symbolic input shapes.

    Parameters
    ----------
    op : tvm.ir.Op
        Relay operator.

    attrs : object
        The op attribute.

    inputs : List[tvm.te.Tensor]
        Input tensors to the op.

    out_type : relay.Type
        The output type.

    target : tvm.target.Target
        The target to compile the op.

    use_autotvm : bool
        Whether query AutoTVM to pick the best.

    Returns
    -------
    ret : tuple(relay.op.OpImplementation, List[tvm.te.Tensor])
        The best op implementation and the corresponding output tensors.
    """
    all_impls = get_valid_implementations(op, attrs, inputs, out_type, target)

    best_plevel_impl = max(all_impls, key=lambda x: x.plevel)
    if not use_autotvm:
        logger.info(
            "Using %s for %s based on highest priority (%d)",
            best_plevel_impl.name,
            op.name,
            best_plevel_impl.plevel,
        )
        outs = best_plevel_impl.compute(attrs, inputs, out_type)
        return best_plevel_impl, outs

    outputs = {}
    workloads = {}
    best_autotvm_impl = None
    best_cfg = None
    dispatch_ctx = autotvm.task.DispatchContext.current
    autotvm.GLOBAL_SCOPE.silent = True
    for impl in all_impls:
        outs = impl.compute(attrs, inputs, out_type)
        outputs[impl] = outs
        workload = autotvm.task.get_workload(outs)
        workloads[impl] = workload
        if workload is None:
            # Not an AutoTVM tunable implementation
            continue
        cfg = dispatch_ctx.query(target, workload)
        if cfg.is_fallback:
            # Skip fallback config
            continue
        logger.info(
            "Implementation %s for %s has cost %.2e", impl.name, op.name, cfg.cost
        )
        if best_cfg is None or best_cfg.cost > cfg.cost:
            best_autotvm_impl = impl
            best_cfg = cfg
    autotvm.GLOBAL_SCOPE.silent = False
    if best_autotvm_impl:
        # The best autotvm implementation definitely doesn't use fallback config
        logger.info(
            "Using %s for %s based on lowest cost (%.2e)",
            best_autotvm_impl.name,
            op.name,
            best_cfg.cost,
        )
        return best_autotvm_impl, outputs[best_autotvm_impl]
    # Use the implementation with highest plevel
    if workloads[best_plevel_impl] is not None:
        msg = "Cannot find config for target=%s, workload=%s. A fallback configuration "\
              "is used, which may bring great performance regression." \
              % (target, workloads[best_plevel_impl])
        if msg not in autotvm.task.DispatchContext.warning_messages:
            autotvm.task.DispatchContext.warning_messages.add(msg)
            autotvm_logger.warning(msg)
    logger.info(
        "Using %s for %s based on highest priority (%s)",
        best_plevel_impl.name,
        op.name,
        best_plevel_impl.plevel,
    )
    return best_plevel_impl, outputs[best_plevel_impl]


def detect_fusion_pattern(call):
    """Detect if the call matches any of the supported fusion patterns"""
    tmp = call
    op_names = []
    while True:
        op_names.append(tmp.op.name)
        tmp = tmp.args[0]
        if not isinstance(tmp, tvm.relay.expr.Call):
            break
    op_names.reverse()
    if op_names in FUSION_PATTERNS:
        return "depth_conv"
    return None


def extract_attrs(call):
    tmp = call
    attrs_list = []
    input_shape = None
    while True:
        print(tmp.op.name)
        if tmp.op.name == "nn.relu":
            attrs_list.append("relu")
        if tmp.op.name == "nn.conv2d":
            attrs_list.append(tmp.attrs)
            if 'type_annotation' in dir(tmp.args[0]):
                input_shape = tmp.args[0].type_annotation.shape # All I want is the input tensor shape!!
        tmp = tmp.args[0]
        if not isinstance(tmp, tvm.relay.expr.Call):
            break
    assert input_shape is not None
    attrs_list.reverse()
    return input_shape, attrs_list


def attrs_list_to_parameters(input_shape, attrs_list):
    from tvm.topi.util import get_const_tuple
    param = []
    for x in input_shape:
        param.append(x)

    idx = 0
    while 1:
        attrs = attrs_list[idx]
        print(attrs)
        if attrs == "multiply" or attrs == "add":
            idx += 1
            continue
        else: # Assuming conv2dattrs here. TODO: Fix that.
            H, _ = get_const_tuple(attrs.kernel_size)
            stride_h, _ = get_const_tuple(attrs.strides)
            is_depthwise = not (attrs.groups == 1)
            bn_relu = "relu" if idx < len(attrs_list) - 1 and attrs_list[idx+1] == "relu" else None
            print(bn_relu)

            param.append(H) # Filter hw
            param.append(1 if is_depthwise else attrs.channels) # Filter oc
            param.append(stride_h) # Filter stride
            param.append(is_depthwise)
            param.append(bn_relu)

            if bn_relu:
                idx += 2
            else:
                idx += 1
        if idx >= len(attrs_list):
            break
    # For block by default. TODO: Fix this.
    param.append(False)

    return param


# TODO: Move this to the right place and try to register it
def conv2d_fusion_strategy_cuda(p, all_inputs, pattern, target):
    """conv2d cuda strategy (NHWC)"""
    from .. import op as _op
    from fusion_composer import FusionComposer
    from schedules.schedule_utils import gpu_schedules as sch

    def wrap_compute_conv2d_fusion(topi_compute):
        """Wrap conv2d fusion topi compute"""
        # The API for compute in a strategy op is always FIXED (attrs, inputs, ret_type), while the computes for different ops are usually DIFFERENT.
        # Needs to pull in all inputs (including every tensors involved in fusion) of the call.
        def _compute_conv2d_fusion(attrs, inputs, ret_type):
            return [topi_compute(inputs)]
        return _compute_conv2d_fusion

    def wrap_schedule_conv2d_fusion(topi_schedule):
        """Wrap fusion schedule"""
        # The API for schedule in a strategy op is always FIXED (attrs, outs, target), while the schedules for different ops are usually DIFFERENT.
        def wrapper(attrs, outs, target):
            with target:
                return topi_schedule(outs)
        return wrapper

    fc = FusionComposer(p, True, target.kind.name)
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d_fusion(fc.get_compute()),
        wrap_schedule_conv2d_fusion(fc.get_schedule(pattern)),
        name="conv2d_fusion.cuda")
    return strategy


def select_fusion_implementation(call, inputs, pattern, ret_type, target):
    # TODO: Fix this
    # from fusion_composer import FusionComposer
    #
    print("select fusion implementation")
    input_shape, attrs_list = extract_attrs(call)
    parameters = attrs_list_to_parameters(input_shape, attrs_list)
    fstrategy = conv2d_fusion_strategy_cuda # TODO: Modify this if the strategy function is moved to somewhere else
    with target:
        strategy = fstrategy(parameters, inputs, pattern, target)
    impl = strategy.specializations[0].implementations[0] # Assuming only one single implementation
    autotvm.GLOBAL_SCOPE.silent = True

    # print(impl.compute)
    # print(impl.schedule)
    # print(impl.same_as)
    # print(impl.name)

    print("------=====")
    outs = impl.compute(None, inputs, ret_type)
    workload = ("fused",) + autotvm.task.topi_integration.serialize_args([parameters, True, target.kind.name, pattern])
    dispatch_ctx = autotvm.task.DispatchContext.current
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        raise Exception("AutoTVM cfg not found!")
    logger.info(
        "Using %s for fusing %s based on lowest cost (%.2e)",
        impl.name, pattern, cfg.cost,
    )

    print(type(impl))
    print(type(outs))
    return impl, outs


@tvm._ffi.register_func("relay.backend.lower_call")
def lower_call(call, inputs, target):
    """Lower the call expression to op implementation and tensor outputs."""
    assert isinstance(call.op, tvm.ir.Op)
    op = call.op

    # Prepare the call_node->checked_type(). For the call node inputs, we ensure that
    # the shape is Int32. Following code ensures the same for the output as well.
    # TODO(@icemelon9): Support recursive tuple
    ret_type = call.checked_type
    if isinstance(ret_type, _ty.TensorType):
        ret_type = _ty.TensorType(get_shape(ret_type.shape), ret_type.dtype)
    elif isinstance(ret_type, _ty.TupleType):
        new_fields = []
        for field in ret_type.fields:
            if isinstance(field, _ty.TensorType):
                new_fields.append(_ty.TensorType(get_shape(field.shape), field.dtype))
            else:
                new_fields.append(field)
        ret_type = _ty.TupleType(new_fields)

    is_dyn = _ty.is_dynamic(call.checked_type)
    for arg in call.args:
        is_dyn = is_dyn or _ty.is_dynamic(arg.checked_type)

    # check if in the AutoTVM tracing mode, and disable if op is not in wanted list
    env = autotvm.task.TaskExtractEnv.current
    reenable_tracing = False
    if env is not None and env.tracing:
        if env.wanted_relay_ops is not None and op not in env.wanted_relay_ops:
            env.tracing = False
            reenable_tracing = True

    # If fusion pattern detected
    pattern = detect_fusion_pattern(call)
    if pattern:
        print("    Fusion detected")
        best_impl, outputs = select_fusion_implementation(call, inputs, pattern, ret_type, target)
        # pprint(call)
        # print(inputs)
        # print(ret_type)
        # print(target)
        print("    Fusion implementation selected")
    else:
        if not is_dyn:
            print("    select implementation")
            best_impl, outputs = select_implementation(
                op, call.attrs, inputs, ret_type, target)
            print("    select implementation finished")
        else:
            # TODO(@icemelon9): Allow tvm to generate multiple kernels for dynamic shapes.
            #   Currently, we just use the implementation with highest plevel
            best_impl, outputs = select_implementation(
                op, call.attrs, inputs, ret_type, target, use_autotvm=False)

    # re-enable AutoTVM tracing
    if reenable_tracing:
        env.tracing = True
    return LoweredOutput(outputs, best_impl)


@tvm._ffi.register_object("relay.CompileEngine")
class CompileEngine(Object):
    """CompileEngine to get lowered code.
    """
    def __init__(self):
        raise RuntimeError("Cannot construct a CompileEngine")

    def lower(self, source_func, target=None):
        """Lower a source_func to a CachedFunc.

        Parameters
        ----------
        source_func : Union[tvm.relay.Function, CCacheKey]
            The source relay function.

        target : tvm.Target
            The target platform.

        Returns
        -------
        cached_func: CachedFunc
            The result of lowering.
        """
        # pylint: disable=broad-except, import-outside-toplevel
        try:
            key = _get_cache_key(source_func, target)
            return _backend._CompileEngineLower(self, key)
        except Exception:
            import traceback
            msg = traceback.format_exc()
            msg += "Error during compile func\n"
            msg += "--------------------------\n"
            msg += source_func.astext(show_meta_data=False)
            msg += "--------------------------\n"
            raise RuntimeError(msg)

    def lower_shape_func(self, source_func, target=None):
        key = _get_cache_key(source_func, target)
        return _backend._CompileEngineLowerShapeFunc(self, key)

    def jit(self, source_func, target=None):
        """JIT a source_func to a tvm.runtime.PackedFunc.

        Parameters
        ----------
        source_func : Union[tvm.relay.Function, CCacheKey]
            The source relay function.

        target : tvm.Target
            The target platform.

        Returns
        -------
        jited_func: tvm.runtime.PackedFunc
            The result of jited function.
        """
        key = _get_cache_key(source_func, target)
        return _backend._CompileEngineJIT(self, key)

    def clear(self):
        """clear the existing cached functions"""
        _backend._CompileEngineClear(self)

    def items(self):
        """List items in the cache.

        Returns
        -------
        item_list : List[Tuple[CCacheKey, CCacheValue]]
            The list of items.
        """
        res = _backend._CompileEngineListItems(self)
        assert len(res) % 2 == 0
        return [(res[2*i], res[2*i+1]) for i in range(len(res) // 2)]

    def dump(self):
        """Return a string representation of engine dump.

        Returns
        -------
        dump : str
            The dumped string representation
        """
        items = self.items()
        res = "====================================\n"
        res += "CompilerEngine dump, %d items cached\n" % len(items)
        for k, v in items:
            res += "------------------------------------\n"
            res += "target={}\n".format(k.target)
            res += "use_count={}\n".format(v.use_count)
            res += "func_name={}\n".format(v.cached_func.func_name)
            res += k.source_func.astext() + "\n"
        res += "===================================\n"
        return res


def get():
    """Get the global compile engine.

    Returns
    -------
    engine : tvm.relay.backend.CompileEngine
        The compile engine.
    """
    return _backend._CompileEngineGlobal()
