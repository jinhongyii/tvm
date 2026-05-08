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

"""Shared helpers for unary operator dispatches on CUDA targets."""

from typing import Literal

from tvm.arith.analyzer import Analyzer
from tvm.ir.expr import PrimExpr
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, TilePrimitiveCall
from tvm.tirx.expr import FloatImm
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext

from ...common import MapOpType
from ..common import get_st_extent, get_vec_len, match_scope
from ..layout_utils import get_sublayout_from_region, layout_signature

unary_op_table = {
    MapOpType.ZERO: lambda x, s, b: 0.0,
    MapOpType.FILL: lambda x, s, b: x,
    MapOpType.SQRT: lambda x, s, b: Tx.sqrt(x * s + b) if b is not None else Tx.sqrt(x * s),
    MapOpType.RECIPROCAL: lambda x, s, b: Tx.FloatImm(x.dtype, 1.0) / x,
    MapOpType.EXP: lambda x, s, b: Tx.exp(x * s + b) if b is not None else Tx.exp(x * s),
    MapOpType.EXP2: lambda x, s, b: Tx.exp2(x * s + b) if b is not None else Tx.exp2(x * s),
    MapOpType.SILU: lambda x, s, b: (
        x / (Tx.FloatImm(x.dtype, 1.0) + Tx.exp(Tx.FloatImm(x.dtype, 0.0) - x))
    ),
}


def _match_storage_scope(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    expected_scope: list[Literal["global", "shared*", "local"]],
) -> tuple[bool, str | None]:
    dst_scope = op_call.args[0].buffer.scope()
    if isinstance(op_call.args[1], BufferRegion):
        src_scope = op_call.args[1].buffer.scope()
    else:
        src_scope = None
    if len(op_call.args) > 2 and isinstance(op_call.args[2], BufferRegion):
        bias_scope = op_call.args[2].buffer.scope()
    else:
        bias_scope = None

    ok = any(
        match_scope(dst_scope, scope)
        and match_scope(src_scope, scope)
        and match_scope(bias_scope, scope)
        for scope in expected_scope
    )
    return (
        ok,
        None
        if ok
        else (
            f"storage scope mismatch: dst {dst_scope}, src {src_scope},"
            f" bias {bias_scope}; expected {expected_scope}"
        ),
    )


def _unary_args(
    op: TilePrimitiveCall,
) -> tuple[BufferRegion, BufferRegion | PrimExpr, BufferRegion | FloatImm | None, FloatImm | None]:
    """Parse unary op-call args as (dst, src, bias, scale)."""
    _dst: BufferRegion = op.args[0]
    _src: BufferRegion | PrimExpr = op.args[1]
    _bias: BufferRegion | FloatImm | None = op.args[2] if len(op.args) > 2 else None
    _scale: FloatImm | None = op.args[3] if len(op.args) > 2 else None
    return _dst, _src, _bias, _scale


def _slice_andlayout_signature(buf_region: BufferRegion):
    """Slice a layout by region and return (start, extent, sliced_layout, canonical_signature)."""
    st, ext = get_st_extent(buf_region)
    sliced = get_sublayout_from_region(buf_region.buffer.layout, buf_region.buffer.shape, st, ext)
    canonical = sliced.canonicalize() if hasattr(sliced, "canonicalize") else sliced
    return st, ext, sliced, layout_signature(canonical)


def _basic_shape_layout_dtype_checks(
    cur_buf_region: BufferRegion,
    ref_buf_region: BufferRegion,
    analyzer: Analyzer,
    *,
    disallow_swizzle: bool,
) -> bool:
    cur_buf, ref_buf = cur_buf_region.buffer, ref_buf_region.buffer
    cur_region = [r.extent for r in cur_buf_region.region]
    ref_region = [r.extent for r in ref_buf_region.region]
    return (
        len(cur_region) == len(ref_region)
        and all(analyzer.can_prove_equal(r, rr) for r, rr in zip(cur_region, ref_region))
        and (cur_buf.layout is not None and ref_buf.layout is not None)
        and isinstance(cur_buf.layout, TileLayout)
        and isinstance(ref_buf.layout, TileLayout)
        and getattr(cur_buf.layout, "shard", None)
        and getattr(ref_buf.layout, "shard", None)
        and not (disallow_swizzle and (cur_buf.layout.is_swizzle() or ref_buf.layout.is_swizzle()))
    )


def _infer_unary_vec_len(
    op: TilePrimitiveCall,
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    thread_cnt: int,
    *,
    fallback_to_scalar: bool,
) -> int | None:
    vec_len = op.config.get("vec_len", None)
    if vec_len is not None:
        return vec_len

    ele_size = DataType(_dst.buffer.dtype).bits  # in bits
    if isinstance(_src, BufferRegion):
        ele_size = max(ele_size, DataType(_src.buffer.dtype).bits)
    possible_vec_lens = [128 // ele_size, 64 // ele_size, 32 // ele_size, 1]
    if isinstance(_src, BufferRegion):
        vec_len = get_vec_len(_src, _dst, possible_vec_lens, thread_cnt)
        if vec_len is None:
            return 1 if fallback_to_scalar else None
        possible_vec_lens = [vl for vl in possible_vec_lens if vl <= vec_len]
        if isinstance(_bias, BufferRegion):
            vec_len = get_vec_len(_bias, _dst, possible_vec_lens, thread_cnt)
    else:
        vec_len = get_vec_len(_dst, _dst, possible_vec_lens, thread_cnt)

    if vec_len is None and fallback_to_scalar:
        return 1
    return vec_len


_LOCAL_CASE_VIEW_FULL = "view_full"
_LOCAL_CASE_VIEW_SLICED = "view_sliced"
_LOCAL_CASE_THREAD_WISE = "thread_wise"


def _classify_unary_local_case(
    _dst: BufferRegion,
    _src: BufferRegion | PrimExpr,
    _bias: BufferRegion | FloatImm | None,
    sctx: DispatchContext,
) -> str | None:
    """Classify local unary implementation path without changing public registration."""

    def _full_region(buf_region: BufferRegion | PrimExpr | None) -> bool:
        if not isinstance(buf_region, BufferRegion):
            return True
        st, ext = get_st_extent(buf_region)
        analyzer = Analyzer()
        zero_st = [0] * len(st)
        return all(
            analyzer.can_prove_equal(e, s) for e, s in zip(ext, buf_region.buffer.shape)
        ) and all(analyzer.can_prove_equal(s, z) for s, z in zip(st, zero_st))

    if sctx.is_thread:
        return _LOCAL_CASE_THREAD_WISE
    if sctx.scope_kind in ["warp", "warpgroup", "cta"]:
        if _full_region(_dst) and _full_region(_src) and _full_region(_bias):
            return _LOCAL_CASE_VIEW_FULL
        return _LOCAL_CASE_VIEW_SLICED
    return None
