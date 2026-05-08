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

"""CUDA binary operator dispatch: shared-memory variant.

Registered ops: add, sub, mul, fdiv.

When: dst, src1, src2 are all shared-memory TileLayout buffers (or one src
is a FloatImm constant -- but only for commutative ops like add/mul).
Non-commutative ops (sub, fdiv) reject constant LHS.

Before (TilePrimitiveCall):
    with Tx.cta():
        A_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
        B_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
        C_smem = Tx.alloc_buffer([32, 32], "float16", scope="shared", layout=...)
        Tx.add(C_smem[0:32, 0:32], A_smem[0:32, 0:32], B_smem[0:32, 0:32])

After (scheduled PrimFunc, thread_cnt=64, vec_len=8):
    for s in Tx.serial(ceildiv(1024, 8 * 64)):
        for vec in Tx.vectorized(8):
            fused = s * 512 + threadIdx.x * 8 + vec
            if fused < 1024:
                idx = [fused // 32, fused % 32]
                C_smem[idx] = A_smem[idx] + B_smem[idx]
    Tx.cuda.cta_sync()

With constant RHS: Tx.mul(C_smem, A_smem, Tx.float16(2.0))
    -> C_smem[idx] = A_smem[idx] * float16(2.0)
"""

import functools
import operator

from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, fail
from tvm.tirx.operator.tile_primitive_dispatch.dispatcher import predicate, register_dispatch

from ...common import MapOpType
from ..common import get_indices, get_thread_cnt
from .utils import (
    _infer_binary_vec_len,
    _match_storage_scope,
    _try_prepare_binary_map,
    _validate_binary,
    binary_op_table,
    get_indices_zero_out,
    validate_binary_shared,
)


def _emit_binary_shared(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc:
    """Emit shared-memory binary map for cta/warpgroup/warp/thread scope."""
    info, msg = _try_prepare_binary_map(op, binary_op, require_trivial_layout=False)
    if msg is not None:
        fail(msg)
    assert info is not None
    _dst, _src1, _src2, const = info.dst_br, info.src1_br, info.src2_br, info.const
    dst = _dst.buffer
    src1 = _src1.buffer
    src2 = _src2.buffer if _src2 is not None else None
    dst_start, dst_extent = info.dst_start, info.dst_extent
    src1_start, src1_extent = info.src1_start, info.src1_extent
    src2_start, src2_extent = info.src2_start, info.src2_extent

    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    vec_len, msg = _infer_binary_vec_len(op, sctx, _dst, _src1, _src2)
    if msg is not None:
        vec_len = 1  # fallback to scalar
    assert vec_len is not None and vec_len >= 1
    thread_cnt = get_thread_cnt(sctx)
    if thread_cnt is None:
        fail(f"unsupported exec_scope {sctx.scope_kind} for shared binary map impl")
    exec_scope_name = sctx.scope_kind

    op_func = binary_op_table.get(binary_op)
    assert op_func is not None

    def get_tid_in_scope():
        tx_var = sctx.launch_params["threadIdx.x"].var
        if exec_scope_name == "cta":
            return tx_var
        if exec_scope_name in ("warp", "warpgroup"):
            return tx_var % thread_cnt
        if exec_scope_name == "thread":
            return 0
        fail(f"unsupported exec_scope {exec_scope_name} for shared binary map impl")

    @Tx.inline
    def sync():
        if exec_scope_name == "cta":
            Tx.cuda.cta_sync()
        elif exec_scope_name == "warpgroup":
            Tx.cuda.warpgroup_sync(8)  # TODO: derive from launch config
        elif exec_scope_name == "warp":
            Tx.cuda.warp_sync()
        elif exec_scope_name == "thread":
            pass

    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        tid = get_tid_in_scope()
        for s in Tx.serial(0, Tx.ceildiv(n_elements, vec_len * thread_cnt)):
            for vec in Tx.vectorized(vec_len):
                fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                if fused < n_elements:
                    dst_indices = Tx.meta_var(get_indices(fused, dst_start, dst_extent))
                    src1_indices = Tx.meta_var(get_indices(fused, src1_start, src1_extent))
                    if const is not None:
                        dst[tuple(dst_indices)] = op_func(src1[tuple(src1_indices)], const)
                    else:
                        src2_indices = Tx.meta_var(
                            get_indices_zero_out(
                                src1_indices, src1_start, src1_extent, src2_start, src2_extent
                            )
                        )
                        dst[tuple(dst_indices)] = op_func(
                            src1[tuple(src1_indices)], src2[tuple(src2_indices)]
                        )
        sync()

    return impl


def binary_shared_impl(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    return _emit_binary_shared(op, binary_op, sctx)


# ---------------------------------------------------------------------------
# Registration: bind each binary op name to its CUDA shared schedule.
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
        variant="shared",
        priority=10,
        when=[
            predicate("validate_binary", _validate_binary, op_type=_op_type),
            predicate("storage_scope", _match_storage_scope, expected_scope=["shared*"]),
            predicate("shared_valid", validate_binary_shared, op_type=_op_type),
        ],
    )
    def _shared_dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _ty=_op_type) -> PrimFunc:
        return binary_shared_impl(op, _ty, sctx)
