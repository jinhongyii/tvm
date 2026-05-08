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

"""CUDA unary operator dispatch: local-memory variant.

Registered ops: zero, fill, reciprocal, exp, exp2, sqrt, silu.

Sub-path A -- view_full (warp/warpgroup/cta scope, full region):

Before:
    with Tx.warp():
        Tx.exp(res_view[0:16, 0:128], acc_view[0:16, 0:128])
        # res_view, acc_view: local bufs with WGMMA layout

After (flattened local view, local_total=64, vec_len=8):
    with Tx.thread():
        base_dst = Tx.decl_buffer((64,), "float16", res.data, scope="local")
        base_src = Tx.decl_buffer((64,), "float16", acc.data, scope="local")
        for s in Tx.serial(64 // 8):
            for vec in Tx.vectorized(8):
                base_dst[s * 8 + vec] = Tx.exp(base_src[s * 8 + vec])

Sub-path B -- view_sliced (warp/warpgroup/cta scope, partial region):
    Like view_full but uses buf.local(*shape) + index decomposition per element.

Sub-path C -- thread_wise (thread scope):

Before:
    with Tx.thread():
        Tx.zero(a_local[0:32], a_local[0:32])     # a_local: local (32,) float16

After (vec_len=8):
    with Tx.thread():
        for s in Tx.serial(32 // 8):
            for vec in Tx.vectorized(8):
                a_local[s * 8 + vec] = 0.0
"""

import functools
import operator

from tvm.arith.analyzer import Analyzer
from tvm.ir.expr import PrimExpr
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.expr import FloatImm
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, fail
from tvm.tirx.operator.tile_primitive_dispatch.dispatcher import predicate, register_dispatch

from ...common import MapOpType
from ..common import get_indices, get_st_extent, get_thread_cnt
from ..layout_utils import get_local_region, sig_equal
from .utils import (
    _LOCAL_CASE_THREAD_WISE,
    _LOCAL_CASE_VIEW_FULL,
    _LOCAL_CASE_VIEW_SLICED,
    _basic_shape_layout_dtype_checks,
    _classify_unary_local_case,
    _infer_unary_vec_len,
    _match_storage_scope,
    _slice_andlayout_signature,
    _unary_args,
    unary_op_table,
)


def validate_unary_local(op: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    _dst, _src, _bias, _scale = _unary_args(op)
    local_case = _classify_unary_local_case(_dst, _src, _bias, sctx)
    if local_case is None:
        return False, f"unsupported exec_scope {sctx.scope_kind} for local unary op"

    # For thread-wise path, allow layout=None (flat local buffers).
    # Only require scope="local" and basic dtype checks.
    if local_case == _LOCAL_CASE_THREAD_WISE:
        if not (
            _dst.buffer.scope() == "local"
            and (_src.buffer.scope() == "local" if isinstance(_src, BufferRegion) else True)
            and (_bias.buffer.scope() == "local" if isinstance(_bias, BufferRegion) else True)
        ):
            return False, "invalid storage scope for thread-wise local unary op"
        compute_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
        if _scale is not None and _scale.dtype != compute_dtype:
            return (
                False,
                f"dtype mismatch for scale in local unary op;"
                f" expected {compute_dtype} but got {_scale.dtype}",
            )
        if isinstance(_bias, BufferRegion) and _bias.buffer.dtype != compute_dtype:
            return (
                False,
                f"dtype mismatch for bias in local unary op;"
                f" expected {compute_dtype} but got {_bias.buffer.dtype}",
            )
        return True, None

    # For view_full / view_sliced paths: require layout on all buffers.
    if not (
        _dst.buffer.scope() == "local"
        and _dst.buffer.layout is not None
        and (_src.buffer.scope() == "local" if isinstance(_src, BufferRegion) else True)
        and (_src.buffer.layout is not None if isinstance(_src, BufferRegion) else True)
        and (_bias.buffer.scope() == "local" if isinstance(_bias, BufferRegion) else True)
        and (_bias.buffer.layout is not None if isinstance(_bias, BufferRegion) else True)
    ):
        return (
            False,
            "invalid storage scope or missing layout for local unary op; expected local scope and "
            "valid layout for src/dst/bias if applicable",
        )

    compute_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
    if _scale is not None and _scale.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for scale in local unary op;"
            f" expected {compute_dtype} but got {_scale.dtype}",
        )
    if isinstance(_bias, BufferRegion) and _bias.buffer.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for bias in local unary op;"
            f" expected {compute_dtype} but got {_bias.buffer.dtype}",
        )

    analyzer = Analyzer()
    if isinstance(_src, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_src, _dst, analyzer, disallow_swizzle=True):
            return False, "shape or layout mismatch between src and dst for local unary op"
    if isinstance(_bias, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_bias, _dst, analyzer, disallow_swizzle=True):
            return False, "shape or layout mismatch between bias and dst for local unary op"

    dst_st, dst_extent, dst_sliced, dst_sig = _slice_andlayout_signature(_dst)
    src_st, src_extent = get_st_extent(_src) if isinstance(_src, BufferRegion) else (None, None)
    bias_st, bias_extent = get_st_extent(_bias) if isinstance(_bias, BufferRegion) else (None, None)

    check_regions = [(_dst.buffer, dst_st, dst_extent)]
    if isinstance(_src, BufferRegion):
        check_regions.append((_src.buffer, src_st, src_extent))
    if isinstance(_bias, BufferRegion):
        check_regions.append((_bias.buffer, bias_st, bias_extent))
    for buf, st, ext in check_regions:
        layout = buf.layout
        for it in layout.shard:
            if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                return (
                    False,
                    "thread-shared dimension with zero stride is not supported for local unary op",
                )
        replica = getattr(layout, "replica", None) or []
        if any(it.axis.is_thread() for it in replica):
            return False, "thread-shared dimension with replica is not supported for local unary op"
        if get_local_region(layout, buf.shape, st, ext) is None:
            return False, "invalid region for local-view unary op"

    src_sig = _slice_andlayout_signature(_src)[3] if isinstance(_src, BufferRegion) else None
    bias_sig = _slice_andlayout_signature(_bias)[3] if isinstance(_bias, BufferRegion) else None

    # Here check the canonicalized layouts are semantically equal.
    if src_sig and not sig_equal(analyzer, src_sig, dst_sig):
        return False, "cannot validate src and dst layout signatures for local unary op"
    if bias_sig and not sig_equal(analyzer, bias_sig, dst_sig):
        return False, "cannot validate bias and dst layout signatures for local unary op"

    # Validate launch-thread consistency against dst layout thread partition.
    thr_extents = []
    for it in dst_sliced.shard:
        if it.axis.is_thread():
            thr_extents.append(it.extent)

    expected = functools.reduce(operator.mul, thr_extents, 1)
    actual = get_thread_cnt(sctx)
    if thr_extents and not analyzer.can_prove_equal(expected, actual):
        return (
            False,
            f"thread count mismatch for local unary op; expected {expected} but got {actual}",
        )
    return True, None


def _emit_unary_local_view_full(
    op: TilePrimitiveCall,
    op_type: MapOpType,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    _scale: FloatImm | None,
    dst_local_info: tuple[list, list, list],
) -> PrimFunc:
    dst_local_shape, _, dst_local_ext = dst_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    vec_len = op.config.get("vec_len", None)
    if vec_len is None:
        analyzer = Analyzer()
        ele_size = DataType(_dst.buffer.dtype).bits  # in bits
        src_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
        ele_size = max(ele_size, DataType(src_dtype).bits)
        for v in [128 // ele_size, 64 // ele_size, 32 // ele_size, 1]:
            if v > 0 and analyzer.can_prove_equal(local_total % v, 0):
                vec_len = v
                break
    assert vec_len is not None

    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                base_dst = Tx.decl_buffer((local_total,), dst.dtype, dst.data, scope=dst.scope())
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        local_idx = Tx.meta_var(s * vec_len + vec)
                        base_dst[local_idx] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, FloatImm):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    base_src = Tx.decl_buffer(
                        (local_total,), src.dtype, src.data, scope=src.scope()
                    )
                    base_dst = Tx.decl_buffer(
                        (local_total,), dst.dtype, dst.data, scope=dst.scope()
                    )
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            local_idx = Tx.meta_var(s * vec_len + vec)
                            base_dst[local_idx] = Tx.cast(
                                op_func(base_src[local_idx], _scale, _bias), dst.dtype
                            )
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    base_src = Tx.decl_buffer(
                        (local_total,), src.dtype, src.data, scope=src.scope()
                    )
                    base_dst = Tx.decl_buffer(
                        (local_total,), dst.dtype, dst.data, scope=dst.scope()
                    )
                    base_bias = Tx.decl_buffer(
                        (local_total,), bias.dtype, bias.data, scope=bias.scope()
                    )
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            local_idx = Tx.meta_var(s * vec_len + vec)
                            base_dst[local_idx] = Tx.cast(
                                op_func(base_src[local_idx], _scale, base_bias[local_idx]),
                                dst.dtype,
                            )
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")
    return impl


def _emit_unary_local_view_sliced(
    op: TilePrimitiveCall,
    op_type: MapOpType,
    sctx: DispatchContext,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    _scale: FloatImm | None,
    dst_local_info: tuple[list, list, list],
) -> PrimFunc:
    thread_cnt = get_thread_cnt(sctx)
    assert thread_cnt is not None
    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    vec_len = _infer_unary_vec_len(
        op, _dst, _src, _bias, thread_cnt=thread_cnt, fallback_to_scalar=False
    )
    assert vec_len is not None

    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                        dst_local[tuple(dst_indices)] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        src_st, src_extent = get_st_extent(_src)
        src_local_info = get_local_region(src.layout, src.shape, src_st, src_extent)
        if not src_local_info:
            fail("src layout is not supported for local-view cast")
        src_local_shape, src_local_st, src_local_ext = src_local_info
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, FloatImm):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    src_local = src.local(*src_local_shape)
                    dst_local = dst.local(*dst_local_shape)
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src_indices = Tx.meta_var(
                                get_indices(fused, src_local_st, src_local_ext)
                            )
                            dst_local[tuple(dst_indices)] = Tx.cast(
                                op_func(src_local[tuple(src_indices)], _scale, _bias), dst.dtype
                            )
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer
            bias_st, bias_extent = get_st_extent(_bias)
            bias_local_info = get_local_region(bias.layout, bias.shape, bias_st, bias_extent)
            if not bias_local_info:
                fail("bias layout is not supported for local-view cast")
            bias_local_shape, bias_local_st, bias_local_ext = bias_local_info

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    src_local = src.local(*src_local_shape)
                    dst_local = dst.local(*dst_local_shape)
                    bias_local = bias.local(*bias_local_shape)
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src_indices = Tx.meta_var(
                                get_indices(fused, src_local_st, src_local_ext)
                            )
                            bias_indices = Tx.meta_var(
                                get_indices(fused, bias_local_st, bias_local_ext)
                            )
                            dst_local[tuple(dst_indices)] = Tx.cast(
                                op_func(
                                    src_local[tuple(src_indices)],
                                    _scale,
                                    bias_local[tuple(bias_indices)],
                                ),
                                dst.dtype,
                            )
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")
    return impl


def _emit_unary_local_thread_wise(
    op: TilePrimitiveCall,
    op_type: MapOpType,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    _scale: FloatImm | None,
) -> PrimFunc:
    dst_st, dst_extent = get_st_extent(_dst)
    local_total = functools.reduce(operator.mul, dst_extent, 1)
    vec_len = _infer_unary_vec_len(op, _dst, _src, _bias, thread_cnt=1, fallback_to_scalar=False)
    assert vec_len is not None

    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                        dst[tuple(dst_indices)] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        src_st, src_extent = get_st_extent(_src)
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, FloatImm):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                            dst[tuple(dst_indices)] = Tx.cast(
                                op_func(src[tuple(src_indices)], _scale, None), dst.dtype
                            )
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer
            bias_st, bias_extent = get_st_extent(_bias)

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                with Tx.thread():
                    for s in Tx.serial(0, local_total // vec_len):
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var(s * vec_len + vec)
                            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                            bias_indices = Tx.meta_var(get_indices(fused, bias_st, bias_extent))
                            dst[tuple(dst_indices)] = Tx.cast(
                                op_func(src[tuple(src_indices)], _scale, bias[tuple(bias_indices)]),
                                dst.dtype,
                            )
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")
    return impl


def unary_local_impl(
    op: TilePrimitiveCall, op_type: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    _dst, _src, _bias, _scale = _unary_args(op)
    local_case = _classify_unary_local_case(_dst, _src, _bias, sctx)
    if local_case is None:
        fail(f"unsupported exec_scope {sctx.scope_kind} for unary map impl")

    if local_case in (_LOCAL_CASE_VIEW_FULL, _LOCAL_CASE_VIEW_SLICED):
        dst_st, dst_extent = get_st_extent(_dst)
        dst_local_info = get_local_region(
            _dst.buffer.layout, list(_dst.buffer.shape), dst_st, dst_extent
        )
        if not dst_local_info:
            fail("dst layout is not supported for local unary op")
        if local_case == _LOCAL_CASE_VIEW_FULL:
            return _emit_unary_local_view_full(
                op, op_type, _dst, _src, _bias, _scale, dst_local_info
            )
        return _emit_unary_local_view_sliced(
            op, op_type, sctx, _dst, _src, _bias, _scale, dst_local_info
        )

    if local_case == _LOCAL_CASE_THREAD_WISE:
        return _emit_unary_local_thread_wise(op, op_type, _dst, _src, _bias, _scale)
    fail(f"unsupported local case {local_case} for unary map impl")


# ---------------------------------------------------------------------------
# Registration: bind each unary op name to its CUDA local schedule.
# ---------------------------------------------------------------------------

for _op_name, _op_type in {
    "zero": MapOpType.ZERO,
    "fill": MapOpType.FILL,
    "reciprocal": MapOpType.RECIPROCAL,
    "exp": MapOpType.EXP,
    "exp2": MapOpType.EXP2,
    "sqrt": MapOpType.SQRT,
    "silu": MapOpType.SILU,
}.items():

    @register_dispatch(
        _op_name,
        "cuda",
        variant="local",
        priority=10,
        when=[
            predicate("storage_scope", _match_storage_scope, expected_scope=["local"]),
            predicate("local_valid", validate_unary_local),
        ],
    )
    def _local_dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _ty=_op_type) -> PrimFunc:
        return unary_local_impl(op, _ty, sctx)
