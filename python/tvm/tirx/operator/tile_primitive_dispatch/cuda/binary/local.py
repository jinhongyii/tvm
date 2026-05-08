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

"""CUDA binary operator dispatch: local-memory variant.

Registered ops: add, sub, mul, fdiv.

Four sub-paths within binary_local_impl:

(A) wgmma_row_red_view (warp/warpgroup/cta scope, layout has laneid shard
    matching WGMMA accumulator pattern):
    Decomposes the logical layout into per-warp iteration with physical
    indices computed from layout decomposition.

(B) generic view (warp/warpgroup/cta scope, non-WGMMA layout):
    Flat local view; uses packed f32x2 PTX when eligible (SM100+, float32,
    even per-thread element count, op in {ADD, SUB, MUL}), scalar fallback
    otherwise.

(C) packed_f32x2 (thread scope, SM100+, float32, vec_len=2):
    Uses PTX add.f32x2 instructions.

(D) trivial_layout (thread scope, generic fallback):
    Simple per-element loop: serial(n/vec) x vectorized(vec).
"""

import functools
import operator
import re

from tvm.arith.analyzer import Analyzer
from tvm.error import InternalError
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.layout import laneid
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, fail
from tvm.tirx.operator.tile_primitive_dispatch.dispatcher import predicate, register_dispatch

from ...common import MapOpType
from ..common import get_indices, get_st_extent, get_vec_len
from ..layout_utils import get_local_region
from .utils import (
    _BINARY_LOCAL_CASE_SUBCTA,
    _BINARY_LOCAL_CASE_THREAD,
    _classify_binary_local_case,
    _infer_binary_vec_len,
    _is_binary_local_packed_f32x2_case,
    _is_binary_local_wgmma_row_red_view_case,
    _match_storage_scope,
    _normalize_binary_args,
    _try_prepare_binary_map,
    _validate_binary,
    binary_op_f32x2_table,
    binary_op_table,
    get_indices_zero_out,
    validate_binary_local,
)


def _emit_binary_local_trivial_layout(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    """Emit local trivial-layout binary map."""
    if not sctx.is_thread:
        fail(f"unsupported exec_scope {sctx.scope_kind} for local thread-trivial binary op")

    info, msg = _try_prepare_binary_map(
        op, binary_op, require_trivial_layout=True, allow_no_layout=True
    )
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, CONST = info.dst_br, info.src1_br, info.src2_br, info.const
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_start, dst_extent = info.dst_start, info.dst_extent
    src1_start, src1_extent = info.src1_start, info.src1_extent
    src2_start, src2_extent = info.src2_start, info.src2_extent
    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    vec_len, msg = _infer_binary_vec_len(op, sctx, _dst, _src1, _src2)
    if msg is not None:
        vec_len = 1
    assert vec_len is not None

    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for s in Tx.serial(0, Tx.ceildiv(n_elements, vec_len)):
            for vec in Tx.vectorized(vec_len):
                fused = Tx.meta_var(s * vec_len + vec)
                if fused < n_elements:
                    dst_indices = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                    src1_indices = Tx.meta_var(get_indices(fused, src1_start, src1_extent))
                    if CONST is not None:
                        dst[tuple(dst_indices)] = op_func(src1[tuple(src1_indices)], CONST)
                    else:
                        src2_indices = Tx.meta_var(
                            get_indices_zero_out(
                                src1_indices, src1_start, src1_extent, src2_start, src2_extent
                            )
                        )
                        dst[tuple(dst_indices)] = op_func(
                            src1[tuple(src1_indices)], src2[tuple(src2_indices)]
                        )

    return impl


def _local_view_eligible_for_packed_f32x2(
    binary_op: MapOpType,
    sctx: DispatchContext,
    dst,
    src1,
    src2,
    const,
    local_total,
) -> bool:
    """Check if a per-thread local-view binary op can use packed f32x2 PTX."""
    if binary_op not in binary_op_f32x2_table:
        return False
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    if sm_version < 100:
        return False
    if dst.dtype != "float32" or src1.dtype != "float32":
        return False
    if const is None:
        if src2 is None or src2.dtype != "float32":
            return False
    else:
        if const.dtype != "float32":
            return False
    analyzer = Analyzer()
    return analyzer.can_prove_equal(local_total % 2, 0)


def _try_emit_binary_local_view_packed_f32x2(
    op: TilePrimitiveCall,
    binary_op: MapOpType,
    sctx: DispatchContext,
    dst,
    src1,
    src2,
    const,
    dst_local_shape,
    src1_local_shape,
    src2_local_shape,
    dst_local_st,
    dst_local_ext,
    src1_local_st,
    src1_local_ext,
    src2_local_st,
    src2_local_ext,
    local_total,
) -> PrimFunc | None:
    """Emit packed f32x2 PTX for per-thread local view if eligible, else None."""
    if not _local_view_eligible_for_packed_f32x2(
        binary_op, sctx, dst, src1, src2, const, local_total
    ):
        return None
    op_func_f32x2 = binary_op_f32x2_table[binary_op]
    rounding_mode = op.config.get("rounding_mode", "rz")
    pair_count = local_total // 2

    if const is None:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                src1_local = src1.local(*src1_local_shape)
                src2_local = src2.local(*src2_local_shape)
                for s in Tx.unroll(pair_count):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_local_st, dst_local_ext))
                    src1_idx_1 = Tx.meta_var(get_indices(2 * s, src1_local_st, src1_local_ext))
                    src1_idx_2 = Tx.meta_var(get_indices(2 * s + 1, src1_local_st, src1_local_ext))
                    src2_idx_1 = Tx.meta_var(
                        get_indices_zero_out(
                            src1_idx_1,
                            src1_local_st,
                            src1_local_ext,
                            src2_local_st,
                            src2_local_ext,
                        )
                    )
                    src2_idx_2 = Tx.meta_var(
                        get_indices_zero_out(
                            src1_idx_2,
                            src1_local_st,
                            src1_local_ext,
                            src2_local_st,
                            src2_local_ext,
                        )
                    )
                    op_func_f32x2(
                        src1_local[tuple(src1_idx_1)],
                        src1_local[tuple(src1_idx_2)],
                        src2_local[tuple(src2_idx_1)],
                        src2_local[tuple(src2_idx_2)],
                        Tx.address_of(dst_local[tuple(dst_idx)]),
                        rounding_mode=rounding_mode,
                    )

        return impl

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl_const():
        with Tx.thread():
            dst_local = dst.local(*dst_local_shape)
            src1_local = src1.local(*src1_local_shape)
            for s in Tx.unroll(pair_count):
                dst_idx = Tx.meta_var(get_indices(2 * s, dst_local_st, dst_local_ext))
                src1_idx_1 = Tx.meta_var(get_indices(2 * s, src1_local_st, src1_local_ext))
                src1_idx_2 = Tx.meta_var(get_indices(2 * s + 1, src1_local_st, src1_local_ext))
                op_func_f32x2(
                    src1_local[tuple(src1_idx_1)],
                    src1_local[tuple(src1_idx_2)],
                    const,
                    const,
                    Tx.address_of(dst_local[tuple(dst_idx)]),
                    rounding_mode=rounding_mode,
                )

    return impl_const


def _emit_binary_local_view(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    info, msg = _try_prepare_binary_map(op, binary_op, require_trivial_layout=False)
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, const = info.dst_br, info.src1_br, info.src2_br, info.const

    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_st, dst_extent = get_st_extent(_dst)
    src1_st, src1_extent = get_st_extent(_src1)
    src2_st, src2_extent = get_st_extent(_src2) if _src2 is not None else (None, None)

    dst_local_info = get_local_region(dst.layout, dst.shape, dst_st, dst_extent)
    src1_local_info = get_local_region(src1.layout, src1.shape, src1_st, src1_extent)
    if not dst_local_info or not src1_local_info:
        fail("dst/src1 layout is not supported for local-view binary op")
    src2_local_info = (
        get_local_region(src2.layout, src2.shape, src2_st, src2_extent)
        if src2 is not None
        else None
    )
    if src2 is not None and not src2_local_info:
        fail("src2 layout is not supported for local-view binary op")

    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    src1_local_shape, src1_local_st, src1_local_ext = src1_local_info
    if src2_local_info is not None:
        src2_local_shape, src2_local_st, src2_local_ext = src2_local_info
    else:
        src2_local_shape, src2_local_st, src2_local_ext = None, None, None

    local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    vec_len, msg = _infer_binary_vec_len(op, sctx, _dst, _src1, _src2)
    if msg is not None:
        vec_len = 1
    assert vec_len is not None

    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    # Try packed f32x2 PTX path on per-thread local view.
    packed_impl = _try_emit_binary_local_view_packed_f32x2(
        op,
        binary_op,
        sctx,
        dst,
        src1,
        src2,
        const,
        dst_local_shape,
        src1_local_shape,
        src2_local_shape,
        dst_local_st,
        dst_local_ext,
        src1_local_st,
        src1_local_ext,
        src2_local_st,
        src2_local_ext,
        local_total,
    )
    if packed_impl is not None:
        return packed_impl

    if const is None:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                src1_local = src1.local(*src1_local_shape)
                src2_local = src2.local(*src2_local_shape)

                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src1_indices = Tx.meta_var(
                                get_indices(fused, src1_local_st, src1_local_ext)
                            )
                            src2_indices = Tx.meta_var(
                                get_indices_zero_out(
                                    src1_indices,
                                    src1_local_st,
                                    src1_local_ext,
                                    src2_local_st,
                                    src2_local_ext,
                                )
                            )
                            dst_local[tuple(dst_indices)] = op_func(
                                src1_local[tuple(src1_indices)], src2_local[tuple(src2_indices)]
                            )
    else:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                src1_local = src1.local(*src1_local_shape)

                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            dst_indices = Tx.meta_var(
                                get_indices(fused, dst_local_st, dst_local_ext)
                            )
                            src1_indices = Tx.meta_var(
                                get_indices(fused, src1_local_st, src1_local_ext)
                            )
                            dst_local[tuple(dst_indices)] = op_func(
                                src1_local[tuple(src1_indices)], const
                            )

    return impl


def _emit_binary_local_wgmma_row_red_view(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    """
    Schedule binary map operation on CUDA on warp-level logical tensor.

    Note: for now, is_tile_inner only support warp-level buffer view verification.
    Warpgroup-level buffer view and cta-level buffer view need sharding on the
    outermost level, thus not supported to be checked and verified at the moment.
    User should pass in warp-level buffer view for src and dst whenever possible.

    Since broadcast checking for arbitrary layout is complicated, for now,
    we only support two kinds of layouts: WGMMA and ROW_RED.
    ROW_RED layout is the row-reduced form of WGMMA, which is:

                            T0 T1 T2 T3
                            T4 T5 T6 T7
                                ...
                            T28 T29 T30 T31
                            T0 T1 T2 T3
                            T4 T5 T6 T7
                                ...
                            T28 T29 T30 T31

    More layouts can be supported in the future.
    """
    op = TilePrimitiveCall.downcast(op)
    _dst: BufferRegion = op.output
    _src1: BufferRegion = op.lhs
    _src2: BufferRegion = op.rhs
    _src1, _src2, CONST, msg = _normalize_binary_args(_src1, _src2, binary_op)
    if msg is not None:
        fail(msg)
    if not isinstance(_src1, BufferRegion):
        fail("normalized src1 is not a BufferRegion for local WGMMA/ROW_RED-view binary map")

    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = None if CONST is not None else _src2.buffer
    dst_region, src1_region = _dst.region, _src1.region
    src2_region = None if CONST is not None else _src2.region
    dtype = dst.dtype

    _src1_start = [r.min for r in src1_region]
    src1_extent = [r.extent for r in src1_region]
    _dst_start = [r.min for r in dst_region]
    dst_extent = [r.extent for r in dst_region]
    if src2_region is not None:
        _src2_start = [r.min for r in src2_region]
        src2_extent = [r.extent for r in src2_region]
    else:
        _src2_start = src2_extent = None

    # basic validation checks
    if not all(
        [
            src1.layout is not None,
            src2.layout is not None if src2 else True,
            dst.layout is not None,
            src1.dtype == dtype,
            src2.dtype == dtype if src2 else CONST.dtype == dtype,
        ]
    ):
        fail("unsupported layout/dtype or exec_scope for local WGMMA/ROW_RED-view binary map")

    # get binary op
    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    # no slicing allowed, since op is on local tensor
    Analyzer()
    if not (
        len(src1_region) == 2
        and len(dst_region) == 2
        and (len(src2_region) == 2 if src2 else True)
        and len(src1.shape) == 2
        and len(dst.shape) == 2
        and (len(src2.shape) == 2 if src2 else True)
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
        fail("unsupported layout/dtype or exec_scope for local WGMMA/ROW_RED-view binary map")

    # For buffer src2, ensure it is broadcastable to src1,
    # and non-broadcasting dimensions match.
    BROADCAST = False
    if CONST is None:
        for i in range(1, len(src2_extent) + 1):
            if src1_extent[-i] != dst_extent[-i]:
                fail("src1 does not match dst extent in binary map")
            if src2_extent[-i] not in (4, src1_extent[-i]):
                fail("src2 not broadcastable to src1 for binary map")
            if src2_extent[-i] == 4 and src1_extent[-i] != 4:
                BROADCAST = True

    # basic shape check
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
        or src1.layout.is_swizzle()
        or (src2.layout.is_swizzle() if src2 else False)
        or dst.layout.is_swizzle()
    ):
        fail("basic shape/layout check failed for WGMMA/ROW_RED-view binary map")

    # layout check:
    # (dst, src1, src2) layout must adhere to one of the five cases below:
    # 1. (WGMMA, WGMMA, WGMMA)
    # 2. (WGMMA, WGMMA, ROW_RED)
    # 3. (ROW_RED, ROW_RED, ROW_RED)
    # 4. (WGMMA, WGMMA, const)
    # 5. (ROW_RED, ROW_RED, const)

    # WGMMA layout check
    atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
    warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
    warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))

    def check_wgmma(buf):
        try:
            return warp_atom.is_tile_inner(buf.layout, buf.shape, [8, 8])
        except InternalError:
            return None

    # ROW_RED layout check
    red_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
    red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))

    def check_row_red(buf):
        try:
            return red_warp_atom.is_tile_inner(buf.layout.canonicalize(), (64,), (32,))
        except InternalError:
            return None

    if CONST is not None:
        # check for case 4 and 5
        num_rows = 2
        if check_wgmma(dst) and check_wgmma(src1):
            num_cols = check_wgmma(src1).size()
        elif check_row_red(dst) and check_row_red(src1):
            num_cols = 1
        else:
            fail("layout check failed for const binary map case")

        src1_local_shape = dst_local_shape = (num_rows, num_cols)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl_const():
            with Tx.thread():
                src1_local = src1.local(*src1_local_shape)
                dst_local = dst.local(*dst_local_shape)
                for i in Tx.serial(num_rows):
                    for j in Tx.serial(num_cols):
                        dst_local[i, j] = op_func(src1_local[i, j], CONST)
        # fmt: on

        return impl_const

    if BROADCAST:
        # check for case 2
        if not (check_wgmma(dst) and check_wgmma(src1) and check_row_red(src2)):
            fail("layout check failed for broadcast binary map case")

        num_rows = 2
        src1_local_shape = dst_local_shape = (num_rows, check_wgmma(src1).size())
        src2_local_shape = (num_rows, 1)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl_broadcast():
            with Tx.thread():
                src1_local = src1.local(*src1_local_shape)
                src2_local = src2.local(*src2_local_shape)
                dst_local = dst.local(*dst_local_shape)
                for i in Tx.serial(num_rows):
                    for j in Tx.serial(dst_local_shape[1]):
                        dst_local[i, j] = op_func(src1_local[i, j], src2_local[i, 0])
        # fmt: on

        return impl_broadcast

    # check for case 1 and 3
    num_rows = 2
    if check_wgmma(dst) and check_wgmma(src1) and check_wgmma(src2):
        num_cols = check_wgmma(src1).size()
    elif check_row_red(dst) and check_row_red(src1) and check_row_red(src2):
        num_cols = 1
    else:
        fail("layout check failed for binary map (WGMMA/ROW_RED)")

    src1_local_shape = src2_local_shape = dst_local_shape = (num_rows, num_cols)

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            src1_local = src1.local(*src1_local_shape)
            src2_local = src2.local(*src2_local_shape)
            dst_local = dst.local(*dst_local_shape)
            for i in Tx.serial(num_rows):
                for j in Tx.serial(num_cols):
                    dst_local[i, j] = op_func(src1_local[i, j], src2_local[i, j])
    # fmt: on

    return impl


def _emit_binary_local_packed_f32x2(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    _dst: BufferRegion = op.args[0]

    op_func_f32x2 = binary_op_f32x2_table.get(binary_op)
    if op_func_f32x2 is None:
        fail(f"binary op {binary_op} does not support f32x2 vectorization")

    info, msg = _try_prepare_binary_map(op, binary_op, require_trivial_layout=True)
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, CONST = info.dst_br, info.src1_br, info.src2_br, info.const
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_start, dst_extent = info.dst_start, info.dst_extent
    src1_start, src1_extent = info.src1_start, info.src1_extent
    src2_start, src2_extent = info.src2_start, info.src2_extent

    # f32x2 check
    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    vec_len = op.config.get("vec_len", None)
    if vec_len is None:
        if get_vec_len(_dst, _src1, [2], thread_cnt=1) is None:
            fail("src1/dst cannot be accessed as float32x2")
        if _src2 is not None and get_vec_len(_dst, _src2, [2], thread_cnt=1) is None:
            fail("src2/dst cannot be accessed as float32x2")
    else:
        if vec_len != 2:
            fail("vec_len must be 2 for f32x2 vectorization")

    rounding_mode = op.config.get("rounding_mode", "rz")

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        for s in Tx.serial(0, n_elements // 2):
            dst_indices = Tx.meta_var(get_indices(2 * s, dst_start, dst_extent))
            src1_indices_1 = Tx.meta_var(get_indices(2 * s, src1_start, src1_extent))
            src1_indices_2 = Tx.meta_var(get_indices(2 * s + 1, src1_start, src1_extent))
            if CONST is not None:
                op_func_f32x2(
                    src1[tuple(src1_indices_1)],
                    src1[tuple(src1_indices_2)],
                    CONST,
                    CONST,
                    Tx.address_of(dst[tuple(dst_indices)]),
                    rounding_mode=rounding_mode,
                )
            else:
                src2_indices_1 = Tx.meta_var(
                    get_indices_zero_out(
                        src1_indices_1, src1_start, src1_extent, src2_start, src2_extent
                    )
                )
                src2_indices_2 = Tx.meta_var(
                    get_indices_zero_out(
                        src1_indices_2, src1_start, src1_extent, src2_start, src2_extent
                    )
                )
                op_func_f32x2(
                    src1[tuple(src1_indices_1)],
                    src1[tuple(src1_indices_2)],
                    src2[tuple(src2_indices_1)],
                    src2[tuple(src2_indices_2)],
                    Tx.address_of(dst[tuple(dst_indices)]),
                    rounding_mode=rounding_mode,
                )

    return impl


def binary_local_impl(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    local_case, err = _classify_binary_local_case(op, binary_op, sctx)
    if local_case is None:
        fail(err if err is not None else "unknown error in classifying local binary case")
    if local_case == _BINARY_LOCAL_CASE_SUBCTA:
        if _is_binary_local_wgmma_row_red_view_case(op, binary_op, sctx):
            return _emit_binary_local_wgmma_row_red_view(op, binary_op, sctx)
        return _emit_binary_local_view(op, binary_op, sctx)
    if local_case == _BINARY_LOCAL_CASE_THREAD:
        if _is_binary_local_packed_f32x2_case(op, binary_op, sctx):
            return _emit_binary_local_packed_f32x2(op, binary_op, sctx)
        return _emit_binary_local_trivial_layout(op, binary_op, sctx)
    fail(f"unsupported local case {local_case} for local binary map impl")


# ---------------------------------------------------------------------------
# Registration: bind each binary op name to its CUDA local schedule.
# ---------------------------------------------------------------------------

for _op_name, _op_type in {
    "add": MapOpType.ADD,
    "sub": MapOpType.SUB,
    "mul": MapOpType.MUL,
    "fdiv": MapOpType.FDIV,
}.items():

    @register_dispatch(
        _op_name,
        "cuda",
        variant="local",
        priority=10,
        when=[
            predicate("validate_binary", _validate_binary, op_type=_op_type),
            predicate("storage_scope", _match_storage_scope, expected_scope=["local"]),
            predicate("local_valid", validate_binary_local, op_type=_op_type),
        ],
    )
    def _local_dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _ty=_op_type) -> PrimFunc:
        return binary_local_impl(op, _ty, sctx)
