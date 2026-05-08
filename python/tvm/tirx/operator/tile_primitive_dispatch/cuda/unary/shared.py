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

"""CUDA unary operator dispatch: shared-memory variant.

Registered ops: zero, fill, reciprocal, exp, exp2, sqrt, silu.

When: dst and src are both shared-memory TileLayout buffers with the same
canonical layout signature (i.e. same axis structure after canonicalize()).

Before (TilePrimitiveCall):
    with Tx.cta():
        A_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
        B_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
        Tx.sqrt(A_smem[0:32, 0:32], B_smem[0:32, 0:32])

After (scheduled PrimFunc, thread_cnt=64, vec_len=8):
    for s in Tx.serial(ceildiv(1024, 8 * 64)):          # outer serial loop
        for vec in Tx.vectorized(8):                     # inner vectorized
            fused = s * 512 + threadIdx.x * 8 + vec
            if fused < 1024:
                idx = [fused // 32, fused % 32]
                A_smem[idx] = Tx.sqrt(B_smem[idx])
    Tx.cuda.cta_sync()                                  # scope-level barrier

With bias+scale (Tx.exp(dst, src, bias=B, scale=1.5)):
    dst[idx] = Tx.cast(Tx.exp(src[idx] * 1.5 + B[idx]), dst.dtype)
"""

import functools
import operator

from tvm.arith.analyzer import Analyzer
from tvm.ir.expr import PrimExpr
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, fail
from tvm.tirx.operator.tile_primitive_dispatch.dispatcher import predicate, register_dispatch

from ...common import MapOpType
from ..common import get_indices, get_st_extent, get_thread_cnt
from ..layout_utils import sig_equal
from .utils import (
    _basic_shape_layout_dtype_checks,
    _infer_unary_vec_len,
    _match_storage_scope,
    _slice_andlayout_signature,
    _unary_args,
    unary_op_table,
)


def validate_unary_shared(op: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    _dst, _src, _bias, _scale = _unary_args(op)

    # support local-view and local-thread-wise unary ops
    if sctx.scope_kind not in ["thread", "warp", "warpgroup", "cta"]:
        return False, f"unsupported exec_scope {sctx.scope_kind} for shared unary op"

    if not (
        _dst.buffer.scope().startswith("shared")
        and _dst.buffer.layout is not None
        and (_src.buffer.scope().startswith("shared") if isinstance(_src, BufferRegion) else True)
        and (_src.buffer.layout is not None if isinstance(_src, BufferRegion) else True)
        and (_bias.buffer.scope().startswith("shared") if isinstance(_bias, BufferRegion) else True)
        and (_bias.buffer.layout is not None if isinstance(_bias, BufferRegion) else True)
    ):
        return (
            False,
            "invalid storage scope or missing layout for shared unary op;"
            " expected shared scope and valid layout for src/dst/bias if applicable",
        )

    compute_dtype = _src.buffer.dtype if isinstance(_src, BufferRegion) else _src.dtype
    if _scale is not None and _scale.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for scale in shared unary op;"
            f" expected {compute_dtype} but got {_scale.dtype}",
        )
    if isinstance(_bias, BufferRegion) and _bias.buffer.dtype != compute_dtype:
        return (
            False,
            f"dtype mismatch for bias in shared unary op;"
            f" expected {compute_dtype} but got {_bias.buffer.dtype}",
        )

    analyzer = Analyzer()
    if isinstance(_src, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_src, _dst, analyzer, disallow_swizzle=False):
            return False, "shape or layout mismatch between src and dst for shared unary op"
    if isinstance(_bias, BufferRegion):
        if not _basic_shape_layout_dtype_checks(_bias, _dst, analyzer, disallow_swizzle=False):
            return False, "shape or layout mismatch between bias and dst for shared unary op"

    dst_sig = _slice_andlayout_signature(_dst)[3]
    src_sig = _slice_andlayout_signature(_src)[3] if isinstance(_src, BufferRegion) else None
    bias_sig = _slice_andlayout_signature(_bias)[3] if isinstance(_bias, BufferRegion) else None

    # Here check the canonicalized layouts are semantically equal.
    if src_sig and not sig_equal(analyzer, src_sig, dst_sig):
        return False, "cannot validate src and dst layout signatures for shared unary op"
    if bias_sig and not sig_equal(analyzer, bias_sig, dst_sig):
        return False, "cannot validate bias and dst layout signatures for shared unary op"

    return True, None


def unary_shared_impl(
    op: TilePrimitiveCall, op_type: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    _dst, _src, _bias, _scale = _unary_args(op)

    dst_start, dst_extent = get_st_extent(_dst)
    num_elements = functools.reduce(operator.mul, dst_extent, 1)
    thread_cnt = get_thread_cnt(sctx)
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
    vec_len = _infer_unary_vec_len(
        op, _dst, _src, _bias, thread_cnt=thread_cnt, fallback_to_scalar=True
    )
    assert vec_len is not None

    dst = _dst.buffer
    op_func = unary_op_table.get(op_type)
    assert op_func is not None
    exec_scope_name = sctx.scope_kind

    def get_tid_in_scope():
        tx_var = sctx.launch_params["threadIdx.x"].var
        if exec_scope_name == "cta":
            return tx_var
        elif exec_scope_name in ("warp", "warpgroup"):
            return tx_var % thread_cnt
        elif exec_scope_name == "thread":
            return 0

    @Tx.inline
    def sync():
        if exec_scope_name == "cta":
            Tx.cuda.cta_sync()
        elif exec_scope_name == "warpgroup":
            Tx.cuda.warpgroup_sync(8)  # TODO: fix this hardcoded value
        elif exec_scope_name == "warp":
            Tx.cuda.warp_sync()
        elif exec_scope_name == "thread":
            pass

    if isinstance(_src, PrimExpr):

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            tid = get_tid_in_scope()
            for s in Tx.serial(0, Tx.ceildiv(num_elements, vec_len * thread_cnt)):
                # for tid in Tx.thread_binding(thread_st, thread_st + thread_cnt, "threadIdx.x"):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                    if fused < num_elements:
                        idx_dst = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                        dst[tuple(idx_dst)] = Tx.cast(op_func(_src, 1.0, None), dst.dtype)
            sync()
    elif isinstance(_src, BufferRegion):
        src = _src.buffer
        src_start, src_extent = get_st_extent(_src)
        if _scale is None:
            _scale = Tx.FloatImm(src.dtype, 1.0)
        if _bias is None or isinstance(_bias, PrimExpr):

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                tid = get_tid_in_scope()
                for s in Tx.serial(0, Tx.ceildiv(num_elements, vec_len * thread_cnt)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                        if fused < num_elements:
                            idx_dst = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                            idx_src = Tx.meta_var(get_indices(fused, src_start, src_extent))
                            dst[tuple(idx_dst)] = Tx.cast(
                                op_func(src[tuple(idx_src)], _scale, _bias), dst.dtype
                            )
                sync()
        elif isinstance(_bias, BufferRegion):
            bias = _bias.buffer
            bias_start, bias_extent = get_st_extent(_bias)

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                tid = get_tid_in_scope()
                for s in Tx.serial(0, Tx.ceildiv(num_elements, vec_len * thread_cnt)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                        if fused < num_elements:
                            idx_dst = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                            idx_src = Tx.meta_var(get_indices(fused, src_start, src_extent))
                            idx_bias = Tx.meta_var(get_indices(fused, bias_start, bias_extent))
                            dst[tuple(idx_dst)] = Tx.cast(
                                op_func(src[tuple(idx_src)], _scale, bias[tuple(idx_bias)]),
                                dst.dtype,
                            )
                sync()
        else:
            fail(f"unsupported bias type {_bias} for unary map with bias/scale impl")
    else:
        fail(f"unsupported src type {_src} for unary map impl")

    return impl


# ---------------------------------------------------------------------------
# Registration: bind each unary op name to its CUDA shared schedule.
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
        variant="shared",
        priority=10,
        when=[
            predicate("storage_scope", _match_storage_scope, expected_scope=["shared*"]),
            predicate("shared_valid", validate_unary_shared),
        ],
    )
    def _shared_dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _ty=_op_type) -> PrimFunc:
        return unary_shared_impl(op, _ty, sctx)
