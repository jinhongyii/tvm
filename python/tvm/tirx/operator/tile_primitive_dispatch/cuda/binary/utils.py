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

"""Shared helpers for binary operator dispatches on CUDA targets."""

import functools
import operator
from typing import Literal, NamedTuple

from tvm.arith.analyzer import Analyzer
from tvm.error import InternalError
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimExpr, TilePrimitiveCall
from tvm.tirx.expr import FloatImm
from tvm.tirx.layout import TileLayout, laneid
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext

from ...common import MapOpType
from ..common import get_st_extent, get_thread_cnt, get_vec_len, match_scope, sm_version_ok
from ..layout_utils import (
    get_local_region,
    get_sublayout_from_region,
    layout_signature,
    resolve_thread_var,
    sig_equal,
)

binary_op_table = {
    MapOpType.ADD: lambda a, b: a + b,
    MapOpType.SUB: lambda a, b: a - b,
    MapOpType.MUL: lambda a, b: a * b,
    MapOpType.FDIV: lambda a, b: a / b,
}

binary_op_f32x2_table = {
    MapOpType.ADD: Tx.ptx.add_packed_f32x2,
    MapOpType.SUB: Tx.ptx.sub_packed_f32x2,
    MapOpType.MUL: Tx.ptx.mul_packed_f32x2,
}


def get_indices_zero_out(indices, src1_start, src1_extent, src2_start, src2_extent):
    """Compute src2 indices for broadcasting based on src1 indices."""
    len_diff = len(src1_extent) - len(src2_extent)
    return [
        (
            (indices[i + len_diff] - src1_start[i + len_diff]) + src2_start[i]
            if src2_extent[i] != 1
            else src2_start[i]
        )
        for i in range(len(src2_extent))
    ]


def _match_storage_scope(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    expected_scope: list[Literal["global", "shared*", "local"]],
) -> tuple[bool, str | None]:
    dst_scope = op_call.args[0].buffer.scope()
    if isinstance(op_call.args[1], BufferRegion):
        src1_scope = op_call.args[1].buffer.scope()
    else:
        src1_scope = None
    if isinstance(op_call.args[2], BufferRegion):
        src2_scope = op_call.args[2].buffer.scope()
    else:
        src2_scope = None

    ok = any(
        match_scope(dst_scope, scope)
        and match_scope(src1_scope, scope)
        and match_scope(src2_scope, scope)
        for scope in expected_scope
    )
    return (
        ok,
        None
        if ok
        else f"storage scope mismatch: dst {dst_scope}, src1 {src1_scope}, "
        f"src2 {src2_scope}; expected {expected_scope}",
    )


def _dtype_ok(op: TilePrimitiveCall, sctx: DispatchContext, expected_dtype: str):
    """Check if src buffer dtype matches."""
    dst, src1, src2 = op.args[:3]
    if dst.buffer.dtype != expected_dtype:
        return (False, f"dst dtype {dst.buffer.dtype} != {expected_dtype}")
    for i, src in enumerate([src1, src2], 1):
        if isinstance(src, BufferRegion) and src.buffer.dtype != expected_dtype:
            return (False, f"src{i} dtype {src.buffer.dtype} != {expected_dtype}")
        elif isinstance(src, PrimExpr) and src.dtype != expected_dtype:
            return (False, f"src{i} dtype {src.dtype} != {expected_dtype}")
    return (True, None)


def _validate_binary(
    op_call: TilePrimitiveCall, sctx: DispatchContext, op_type: MapOpType
) -> tuple[bool, str | None]:
    """Basic validation for binary ops."""
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]

    # Check that op_type is supported.
    if binary_op_table.get(op_type) is None:
        return (False, f"unsupported binary op: {op_type}")
    _, _, _, msg = _normalize_binary_args(_src1, _src2, op_type)
    return (msg is None, msg)


class _BinaryMapInfo(NamedTuple):
    dst_br: BufferRegion
    src1_br: BufferRegion
    src2_br: BufferRegion | None
    const: FloatImm | None
    dst_start: list
    dst_extent: list
    src1_start: list
    src1_extent: list
    src2_start: list | None
    src2_extent: list | None


def _is_scalar(x) -> bool:
    """Check if x is a scalar (non-BufferRegion PrimExpr, including FloatImm)."""
    return not isinstance(x, BufferRegion)


def _normalize_binary_args(
    _src1: BufferRegion | PrimExpr, _src2: BufferRegion | PrimExpr, op_type: MapOpType
) -> tuple[BufferRegion | PrimExpr, BufferRegion | PrimExpr, PrimExpr | None, str | None]:
    """Normalize binary args: move const to rhs and ensure rhs is broadcastable to lhs if needed."""
    # Ensure at least one source is not a scalar constant.
    if _is_scalar(_src1) and _is_scalar(_src2):
        return _src1, _src2, None, "both inputs are constants; unsupported for binary map"

    # If src1 is scalar, swap (only allowed for commutative ops).
    if _is_scalar(_src1):
        if op_type not in (MapOpType.ADD, MapOpType.MUL):
            return _src1, _src2, None, "commutativity required to swap constant as lhs"
        _src1, _src2 = _src2, _src1

    const = _src2 if _is_scalar(_src2) else None
    if const is not None:
        return _src1, _src2, const, None

    # For non-constant rhs, switch broadcasting direction if needed.
    src1_num = functools.reduce(operator.mul, [r.extent for r in _src1.region], 1)
    src2_num = functools.reduce(operator.mul, [r.extent for r in _src2.region], 1)
    if src1_num < src2_num:
        if op_type not in (MapOpType.ADD, MapOpType.MUL):
            return _src1, _src2, None, "non-commutative op cannot broadcast second source"
        _src1, _src2 = _src2, _src1
    return _src1, _src2, None, None


def _try_prepare_binary_map(
    op_call: TilePrimitiveCall,
    op_type: MapOpType,
    require_trivial_layout: bool,
    allow_no_layout: bool = False,
) -> tuple[_BinaryMapInfo | None, str | None]:
    _dst: BufferRegion = op_call.args[0]
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]

    _src1, _src2, const, msg = _normalize_binary_args(_src1, _src2, op_type)
    if msg is not None:
        return None, msg

    dst = _dst.buffer
    src1 = _src1.buffer
    src2_br = _src2 if const is None else None
    src2 = src2_br.buffer if src2_br is not None else None
    dtype = dst.dtype
    has_layout = dst.layout and src1.layout and (src2.layout if src2 else True)
    if not has_layout and not allow_no_layout:
        return None, "unsupported layout/dtype for binary map"
    if not (src1.dtype == dtype and ((src2.dtype == dtype) if src2 else (const.dtype == dtype))):
        return None, "unsupported dtype for binary map"
    if (
        has_layout
        and require_trivial_layout
        and not (
            dst.layout.is_trivial()
            and src1.layout.is_trivial()
            and (src2.layout.is_trivial() if src2 else True)
        )
    ):
        return None, "unsupported non-trivial layout for binary map"

    analyzer = Analyzer()
    dst_extent = [r.extent for r in _dst.region]
    src1_extent = [r.extent for r in _src1.region]
    dst_non1 = [e for e in dst_extent if e != 1]
    src1_non1 = [e for e in src1_extent if e != 1]
    if not (
        len(dst_non1) == len(src1_non1)
        and all(analyzer.can_prove_equal(s, d) for s, d in zip(src1_non1, dst_non1))
    ):
        return None, "shape mismatch between dst and src1 for binary map"

    src2_extent = [r.extent for r in src2_br.region] if src2_br is not None else None
    if src2_br is not None:
        for i in range(1, len(src2_extent) + 1):
            if src2_extent[-i] not in (1, src1_extent[-i]):
                return None, "src2 not broadcastable to src1 for binary map"

    info = _BinaryMapInfo(
        dst_br=_dst,
        src1_br=_src1,
        src2_br=src2_br,
        const=const,
        dst_start=[r.min for r in _dst.region],
        dst_extent=dst_extent,
        src1_start=[r.min for r in _src1.region],
        src1_extent=src1_extent,
        src2_start=[r.min for r in src2_br.region] if src2_br is not None else None,
        src2_extent=src2_extent,
    )
    return info, None


def _infer_binary_vec_len(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    _dst: BufferRegion,
    _src1: BufferRegion,
    _src2: BufferRegion | None,
) -> tuple[int | None, str | None]:
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is not None:
        return vec_len, None
    tx = get_thread_cnt(sctx)
    if tx is None:
        return None, f"unsupported exec_scope {sctx.scope_kind} for vec_len"

    elem_size = DataType(_dst.buffer.dtype).bits  # in bits
    possible_vec_len = [128 // elem_size, 64 // elem_size, 32 // elem_size, 1]
    vec_len = get_vec_len(_dst, _src1, possible_vec_len, thread_cnt=tx)
    if vec_len is None:
        return None, "no valid vector length; check alignment/extents/thread-count"
    possible_vec_len = [vl for vl in possible_vec_len if vl <= vec_len]
    if _src2 is not None:
        vec_len = get_vec_len(_dst, _src2, possible_vec_len, thread_cnt=tx)
    if vec_len is None:
        return None, "no valid vector length; check alignment/extents/thread-count"
    return vec_len, None


def _is_binary_local_packed_f32x2_case(
    op_call: TilePrimitiveCall, op_type: MapOpType, sctx: DispatchContext
) -> bool:
    """Check whether local-thread trivial layout can use packed f32x2 path."""
    if not sctx.is_thread:
        return False
    if op_type not in binary_op_f32x2_table:
        return False
    if not sm_version_ok(op_call, sctx, min_version=100)[0]:
        return False
    if not _dtype_ok(op_call, sctx, expected_dtype="float32")[0]:
        return False

    _dst: BufferRegion = op_call.args[0]
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]
    _src1, _src2, const, msg = _normalize_binary_args(_src1, _src2, op_type)
    if msg is not None:
        return False
    if get_vec_len(_dst, _src1, [2], thread_cnt=1) != 2:
        return False
    if const is None:
        if not isinstance(_src2, BufferRegion) or get_vec_len(_dst, _src2, [2], thread_cnt=1) != 2:
            return False
    return True


def _is_binary_local_wgmma_row_red_view_case(
    op_call: TilePrimitiveCall, op_type: MapOpType, sctx: DispatchContext
) -> bool:
    """Check whether op fits the local WGMMA/ROW_RED view implementation."""
    if sctx.scope_kind not in ["warp", "warpgroup", "cta"]:
        return False
    _dst: BufferRegion = op_call.args[0]
    _src1: BufferRegion | FloatImm = op_call.args[1]
    _src2: BufferRegion | FloatImm = op_call.args[2]

    _src1, _src2, const, msg = _normalize_binary_args(_src1, _src2, op_type)
    if msg is not None:
        return False

    dst, src1 = _dst.buffer, _src1.buffer
    src2 = None if const is not None else _src2.buffer
    dst_region, src1_region = _dst.region, _src1.region
    src2_region = None if const is not None else _src2.region

    # no slicing allowed, since op is on local tensor
    if not (
        len(src1_region) == 2
        and len(dst_region) == 2
        and (len(src2_region) == 2 if src2 else True)
        and len(src1.shape) == 2
        and (len(src2.shape) == 2 if src2 else True)
        and len(dst.shape) == 2
        and src1_region[0].min == 0
        and src1_region[1].min == 0
        and ((src2_region[0].min == 0 and src2_region[1].min == 0) if src2 else True)
        and dst_region[0].min == 0
        and dst_region[1].min == 0
        and src1_region[0].extent == src1.shape[0]
        and src1_region[1].extent == src1.shape[1]
        and (
            (src2_region[0].extent == src2.shape[0] and src2_region[1].extent == src2.shape[1])
            if src2
            else True
        )
        and dst_region[0].extent == dst.shape[0]
        and dst_region[1].extent == dst.shape[1]
    ):
        return False

    # basic shape/layout check
    if (
        len(src1.shape) != 2
        or (len(src2.shape) != 2 if src2 else False)
        or len(dst.shape) != 2
        or src1.shape[0] != 16
        or not (src1.shape[1] % 8 == 0 or src1.shape[1] == 4)
        or (src2.shape[0] != 16 if src2 else False)
        or (not (src2.shape[1] % 8 == 0 or src2.shape[1] == 4) if src2 else False)
        or dst.shape[0] != 16
        or not (dst.shape[1] % 8 == 0 or dst.shape[1] == 4)
        or src1.layout is None
        or (src2.layout is None if src2 else False)
        or dst.layout is None
        or src1.layout.is_swizzle()
        or (src2.layout.is_swizzle() if src2 else False)
        or dst.layout.is_swizzle()
    ):
        return False

    src1_extent = [r.extent for r in src1_region]
    dst_extent = [r.extent for r in dst_region]
    src2_extent = [r.extent for r in src2_region] if src2 else None
    broadcast = False
    if src2 is not None:
        for i in range(1, len(src2_extent) + 1):
            if src1_extent[-i] != dst_extent[-i]:
                return False
            if src2_extent[-i] not in (4, src1_extent[-i]):
                return False
            if src2_extent[-i] == 4 and src1_extent[-i] != 4:
                broadcast = True

    atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
    warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
    warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
    red_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
    red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))

    def _kind(buf) -> str | None:
        try:
            if warp_atom.is_tile_inner(buf.layout, buf.shape, [8, 8]):
                return "wgmma"
        except InternalError:
            pass
        try:
            if red_warp_atom.is_tile_inner(buf.layout.canonicalize(), (64,), (32,)):
                return "row_red"
        except InternalError:
            pass
        return None

    kind_dst = _kind(dst)
    kind_src1 = _kind(src1)
    if kind_dst is None or kind_src1 is None:
        return False
    if const is not None:
        return kind_dst == kind_src1

    kind_src2 = _kind(src2)
    if kind_src2 is None:
        return False
    if broadcast:
        return kind_dst == "wgmma" and kind_src1 == "wgmma" and kind_src2 == "row_red"
    return kind_dst == kind_src1 == kind_src2


def _validate_binary_local_view_case(
    op_call: TilePrimitiveCall, sctx: DispatchContext, op_type: MapOpType
) -> tuple[bool, str | None]:
    if sctx.scope_kind not in ["cta", "warpgroup", "warp"]:
        return False, f"unsupported exec_scope {sctx.scope_kind} for local-view binary op"

    info, msg = _try_prepare_binary_map(op_call, op_type, require_trivial_layout=False)
    if msg is not None:
        return False, msg
    assert info is not None

    _dst, _src1, _src2 = info.dst_br, info.src1_br, info.src2_br
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None

    analyzer = Analyzer()
    dst_st, dst_extent = get_st_extent(_dst)
    src1_st, src1_extent = get_st_extent(_src1)
    src2_st, src2_extent = get_st_extent(_src2) if _src2 is not None else (None, None)

    check_regions = [(dst, dst_st, dst_extent), (src1, src1_st, src1_extent)]
    if src2 is not None:
        check_regions.append((src2, src2_st, src2_extent))

    for buf, st, ext in check_regions:
        layout = buf.layout
        if layout is None:
            return False, "missing layout for local-view binary op"
        if not isinstance(layout, TileLayout) or not getattr(layout, "shard", None):
            return False, "non-TileLayout is not supported for local-view binary op"
        if layout.is_swizzle():
            return False, "swizzle layout is not supported for local-view binary op"
        for it in layout.shard:
            if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                return False, "thread-shared dimension with zero stride is not supported"
        replica = getattr(layout, "replica", None) or []
        if any(it.axis.is_thread() for it in replica):
            return False, "thread-shared dimension with replica is not supported"
        if get_local_region(layout, buf.shape, st, ext) is None:
            return False, "invalid region for local-view binary op"

    # src1 and dst should represent the same sliced thread/local partition.
    dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)
    src1_sliced = get_sublayout_from_region(src1.layout, src1.shape, src1_st, src1_extent)
    dst_sig = layout_signature(
        dst_sliced.canonicalize() if hasattr(dst_sliced, "canonicalize") else dst_sliced
    )
    src1_sig = layout_signature(
        src1_sliced.canonicalize() if hasattr(src1_sliced, "canonicalize") else src1_sliced
    )
    if not sig_equal(analyzer, src1_sig, dst_sig):
        return False, "cannot validate src1 and dst layout signatures for local-view binary op"

    thread_vars_list = []
    thr_extents = []
    for it in dst_sliced.shard:
        if it.axis.is_thread():
            var = resolve_thread_var(it.axis, sctx)
            if var is None:
                return False, "cannot resolve thread variable"
            thread_vars_list.append(var)
            thr_extents.append(it.extent)

    if thread_vars_list and "threadIdx.x" in sctx.launch_params:
        expected = functools.reduce(operator.mul, thr_extents, 1)
        actual = get_thread_cnt(sctx)
        if len(set(id(v) for v in thread_vars_list)) == 1:
            if thread_vars_list[0] is sctx.launch_params["threadIdx.x"].var:
                if not analyzer.can_prove_equal(actual, expected):
                    return False, f"thread count mismatch; expected {expected} but got {actual}"

    return True, None


def validate_binary_shared(
    op_call: TilePrimitiveCall, sctx: DispatchContext, op_type: MapOpType
) -> tuple[bool, str | None]:
    if sctx.scope_kind not in ["cta", "warpgroup", "warp", "thread"]:
        return False, f"unsupported exec_scope {sctx.scope_kind} for shared binary op"
    _, msg = _try_prepare_binary_map(op_call, op_type, require_trivial_layout=False)
    return (msg is None, msg)


_BINARY_LOCAL_CASE_SUBCTA = "subcta_view"
_BINARY_LOCAL_CASE_THREAD = "thread"


def _classify_binary_local_case(
    op_call: TilePrimitiveCall, op_type: MapOpType, sctx: DispatchContext
) -> tuple[str | None, str | None]:
    """Classify local binary path by layout capability.

    For non-thread scopes (cta/warpgroup/warp): WGMMA/ROW_RED-view and
    generic local-view (including thread-axis distributed layouts) are supported.
    For thread scope: trivial layout path is supported, with optional packed_f32x2 optimization.
    """
    scope = sctx.scope_kind
    if scope not in ["cta", "warpgroup", "warp", "thread"]:
        return None, f"unsupported exec_scope {sctx.scope_kind}"

    if scope in ["cta", "warpgroup", "warp"]:
        if _is_binary_local_wgmma_row_red_view_case(op_call, op_type, sctx):
            return _BINARY_LOCAL_CASE_SUBCTA, None
        ok, msg = _validate_binary_local_view_case(op_call, sctx, op_type)
        if ok:
            return _BINARY_LOCAL_CASE_SUBCTA, None
        return None, msg
    elif scope == "thread":
        _, msg = _try_prepare_binary_map(
            op_call, op_type, require_trivial_layout=True, allow_no_layout=True
        )
        if msg is not None:
            return None, msg
        return _BINARY_LOCAL_CASE_THREAD, None
    else:
        return None, f"unsupported exec_scope {sctx.scope_kind}"


def validate_binary_local(
    op_call: TilePrimitiveCall, sctx: DispatchContext, op_type: MapOpType
) -> tuple[bool, str | None]:
    local_case, err = _classify_binary_local_case(op_call, op_type, sctx)
    return (local_case is not None, err)
