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
import functools

import numpy as np
import pytest
import torch
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from torch.nn import functional as F
from triton import language as tl

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.bench import ProtonContext, bench
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

test_configs = [
    {
        "batch_size": 128,  # M
        "hidden_size": 2048,  # K
        "num_experts": 128,  # E
        "top_k": 8,  # Top-K
        "intermediate_size": 768,  # N
    }
]

DEBUG = False
AUOTUNE = False


def _ensure_sglang_server_args():
    """Newer sglang builds require scheduler args before querying MoE configs."""

    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def compute_routing(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def prepare_group_gemm(BLK_M, num_experts, selected_experts):
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        selected_experts, BLK_M, num_experts
    )
    sorted_token_ids = sorted_token_ids[:]
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def gen_input(batch_size, hidden_size, num_experts, top_k, intermediate_size):
    torch.manual_seed(42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    otype = torch.float16
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    x1 = torch.randn(m, k, dtype=otype).cuda()
    w13 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype)
    x2 = torch.randn(m, top_k, n, dtype=otype).reshape(batch_size * top_k, intermediate_size).cuda()
    w2 = torch.randn((e, k, n), device="cuda", dtype=otype)

    selected_experts = selected_experts.to(torch.int)

    return (x1, w13, routing_weights, selected_experts, x2, w2)


def ceildiv(a, b):
    return (a + b - 1) // b


def get_group_gemm_kernel(M, K, E, top_k, N, config, mul_routed_weight):
    print(
        f"M: {M}, K: {K}, E: {E}, top_k: {top_k}, N: {N}, config: {config}, mul_routed_weight: {mul_routed_weight}"  # noqa: E501
    )

    FP16_BYTES = 2
    VEC_LEN = 128 // (FP16_BYTES * 8)
    SM_COUNT = 148
    WARP_GROUP_COUNT = 2
    WARP_COUNT = 4

    BLK_M, BLK_N, BLK_K, NUM_STAGES = (
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["num_stages"],
    )
    if BLK_M == 16:
        NUM_WARPS_M = 1
        NUM_WARPS_N = 4
    else:
        NUM_WARPS_M = 2
        NUM_WARPS_N = 2
    assert BLK_M % 16 == 0
    assert BLK_N % 16 == 0
    assert BLK_K % 16 == 0
    assert N % BLK_N == 0
    assert K % BLK_K == 0
    assert WARP_COUNT == NUM_WARPS_M * NUM_WARPS_N

    MMA_M = BLK_M // 16
    MMA_N = BLK_N // 16
    MMA_K = BLK_K // 16
    assert MMA_M % NUM_WARPS_M == 0
    assert MMA_N % NUM_WARPS_N == 0
    NUM_MMA_M = MMA_M // NUM_WARPS_M
    NUM_MMA_N = MMA_N // NUM_WARPS_N

    N_TILE_CNT = ceildiv(N, BLK_N)
    K_TILE_CNT = ceildiv(K, BLK_K)

    AB_THR_LAYOUT_COL = min(32, BLK_K // VEC_LEN)
    AB_THR_LAYOUT_ROW = 32 // AB_THR_LAYOUT_COL
    assert 32 % AB_THR_LAYOUT_COL == 0
    assert AB_THR_LAYOUT_COL >= 8

    C_THR_LAYOUT_COL = min(32, BLK_N // VEC_LEN)
    C_THR_LAYOUT_ROW = 32 // C_THR_LAYOUT_COL
    assert 32 % C_THR_LAYOUT_COL == 0
    assert C_THR_LAYOUT_COL >= 8

    UPCAST_STRIDE_K = BLK_K // VEC_LEN
    UPCAST_STRIDE_N = BLK_N // VEC_LEN

    # A load
    assert BLK_K % (AB_THR_LAYOUT_COL * VEC_LEN) == 0
    assert BLK_M % (AB_THR_LAYOUT_ROW * WARP_COUNT) == 0

    # B load
    assert BLK_K % (AB_THR_LAYOUT_COL * VEC_LEN) == 0
    assert BLK_N % (AB_THR_LAYOUT_ROW * WARP_COUNT) == 0

    # C store
    assert BLK_N % (C_THR_LAYOUT_COL * VEC_LEN) == 0
    assert BLK_M % (C_THR_LAYOUT_ROW * WARP_COUNT) == 0

    SMEM_SIZE = max(
        NUM_STAGES * BLK_M * BLK_K * FP16_BYTES + NUM_STAGES * BLK_N * BLK_K * FP16_BYTES,
        BLK_K * BLK_K * FP16_BYTES,
    )
    print(f"SMEM_SIZE: {SMEM_SIZE}")
    assert SMEM_SIZE * WARP_GROUP_COUNT <= 232448
    assert SMEM_SIZE % VEC_LEN == 0

    def int_cell(value):
        buf = Tx.local_scalar("int32")
        if value is not None:
            Tx.buffer_store(buf.buffer, value, 0)
        return buf

    def get_permuted_offset(stride, i, j):
        return i * stride + (j ^ (i % 8))

    def advance_offset_by_column(step_size: int, offset, step_idx: int):
        if not (step_size == 2 or step_size == 4 or step_size % 8 == 0):
            raise ValueError(f"Unsupported step_size {step_size} for K128B mode")

        if step_size == 2:
            return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + ((step_idx % 4 == 3) * 8)
        elif step_size == 4:
            return (offset ^ 0x4) + ((step_idx % 2 == 1) * 8)
        else:  # This condition implies step_size % 8 == 0
            return offset + step_size

    def advance_offset_by_row(step_size: int, row_stride: int, offset):
        if not (step_size == 4 or step_size % 8 == 0):
            raise ValueError(f"Unsupported step_size: {step_size}. Must be 4 or a multiple of 8.")
        if step_size % 8 == 0:
            return offset + step_size * row_stride
        return (offset ^ 0x4) + step_size * row_stride

    def half_to_float(x):
        func_name = "tvm_builtin_half_to_float"
        source_code = f"""
__device__ __forceinline__ float {func_name}(half x) {{
  return __half2float(x);
}}
    """
        return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")

    # fmt: off
    @Tx.inline
    def mma_sync_m16n16k16_row_col_f16f16f32(C_in, c_offset, A_in, a_offset, B_in, b_offset, init: bool):  # noqa: E501
        with Tx.thread():
            C_mma = Tx.decl_buffer([8], dtype="float32", data=C_in.data, byte_offset=c_offset)
            A_mma = Tx.decl_buffer([4], dtype="uint32", data=A_in.data, byte_offset=a_offset)
            B_mma = Tx.decl_buffer([4], dtype="uint32", data=B_in.data, byte_offset=b_offset)
            if init:
                Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                        C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]))
                Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                        C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]))
            else:
                Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                        C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]), C_mma.ptr_to([0]))
                Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                        C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]), C_mma.ptr_to([4]))
    # fmt: on

    def store_128b(dst_ptr, src_ptr):
        func_name = "store_128b"
        source_code = f"""
__device__ __forceinline__ void {func_name}(void* dst_ptr, void* src_ptr) {{
  using b128_t = uint4;
  b128_t* dst_ptr_b128 = reinterpret_cast<b128_t*>(dst_ptr);
  b128_t* src_ptr_b128 = reinterpret_cast<b128_t*>(src_ptr);
  *dst_ptr_b128 = *src_ptr_b128;
}}
"""
        return Tx.cuda.func_call(func_name, dst_ptr, src_ptr, source_code=source_code)

    # fmt: off

    @Tx.prim_func(tirx=True)
    def group_gemm(
        A: Tx.Buffer((M, K), "float16"),
        B: Tx.Buffer((E, N, K), "float16"),
        C: Tx.Buffer((M, top_k, N), "float16"),
        topk_weights: Tx.Buffer((M, top_k), "float32"),
        topk_ids: Tx.Buffer((M, top_k), "int32"),
        sorted_token_ids_ptr: Tx.handle,
        expert_ids_ptr: Tx.handle,
        num_tokens_post_padded: Tx.Buffer((1), "int32"),
    ):
        MAX_SORTED_TOKEN_IDS = Tx.int32()
        MAX_EXPERT_IDS = Tx.int32()
        sorted_token_ids = Tx.match_buffer(sorted_token_ids_ptr, (MAX_SORTED_TOKEN_IDS), "int32")
        expert_ids = Tx.match_buffer(expert_ids_ptr, (MAX_EXPERT_IDS), "int32")

        with Tx.kernel():
            cta_cnt = Tx.meta_var(SM_COUNT) # persistent kernel
            # cta_cnt = Tx.meta_var(ceildiv(MAX_SORTED_TOKEN_IDS, BLK_M) * ceildiv(N, BLK_N)) # non-persistent kernel  # noqa: E501
            bx = Tx.cta_id([cta_cnt])
            wg_id = Tx.warpgroup_id([WARP_GROUP_COUNT])
            warp_id = Tx.warp_id_in_wg([WARP_COUNT])
            lane_id = Tx.lane_id([32])

            buf = Tx.alloc_shared([SMEM_SIZE * WARP_GROUP_COUNT], "uint8", scope="shared.dyn")
            pool = Tx.SMEMPool(buf.data)
            A_smem = pool.alloc([NUM_STAGES *  BLK_M * BLK_K], "float16", align=16)
            B_smem = pool.alloc([NUM_STAGES *  BLK_N * BLK_K], "float16", align=16)
            pool.move_base_to(0)
            C_smem = pool.alloc([BLK_M * BLK_N], "float16", align=16)

            A_gmem_1d = A.view(-1)
            B_gmem_1d = B.view(-1)

            @Tx.inline
            def scope_sync():
                Tx.cuda.warpgroup_sync(wg_id)

            with Tx.thread():
                @Tx.inline
                def compute_tile(m_idx, n_idx):
                    # init states
                    s_frag = Tx.alloc_local([NUM_MMA_M, NUM_MMA_N, 8], "float32")

                    for mma_m, mma_n, i in Tx.grid(NUM_MMA_M, NUM_MMA_N, 8):
                        s_frag[mma_m, mma_n, i] = Tx.float32(0)

                    # per-row routing and mask
                    rmap = Tx.alloc_local([BLK_M], "int32")
                    Tx.copy(rmap[:], sorted_token_ids[m_idx * BLK_M : (m_idx + 1) * BLK_M])
                    rbound = Tx.meta_var(M * top_k)

                    # expert
                    eid = int_cell(expert_ids[m_idx])

                    M_rows_per_thread = Tx.meta_var(ceildiv(BLK_M, AB_THR_LAYOUT_ROW * WARP_COUNT))
                    N_rows_per_thread = Tx.meta_var(ceildiv(BLK_N, AB_THR_LAYOUT_ROW * WARP_COUNT))

                    # prefetch global
                    thr_local_A_offset = Tx.alloc_local([M_rows_per_thread], "int32")
                    thr_local_B_offset = Tx.alloc_local([N_rows_per_thread], "int32")

                    @Tx.inline
                    def prefetch_A_offset():
                        for i in range(M_rows_per_thread):
                            row = Tx.meta_var(rmap[i * AB_THR_LAYOUT_ROW * WARP_COUNT + warp_id * AB_THR_LAYOUT_ROW + lane_id // AB_THR_LAYOUT_COL] // top_k)  # noqa: E501
                            col = Tx.meta_var((lane_id % AB_THR_LAYOUT_COL) * VEC_LEN)
                            thr_local_A_offset[i] = A.elem_offset_of([row, col])

                    @Tx.inline
                    def prefetch_B_offset():
                        for i in range(N_rows_per_thread):
                            row = Tx.meta_var(n_idx * BLK_N + i * AB_THR_LAYOUT_ROW * WARP_COUNT + warp_id * AB_THR_LAYOUT_ROW + lane_id // AB_THR_LAYOUT_COL)  # noqa: E501
                            col = Tx.meta_var((lane_id % AB_THR_LAYOUT_COL) * VEC_LEN)
                            thr_local_B_offset[i] = B.elem_offset_of([eid, row, col])

                    prefetch_A_offset()
                    prefetch_B_offset()

                    warp_id_m = Tx.meta_var(warp_id // NUM_WARPS_N)
                    warp_id_n = Tx.meta_var(warp_id % NUM_WARPS_N)
                    smem_offset_A_w = int_cell(get_permuted_offset(UPCAST_STRIDE_K, warp_id * AB_THR_LAYOUT_ROW + lane_id // AB_THR_LAYOUT_COL, lane_id % AB_THR_LAYOUT_COL))  # noqa: E501
                    smem_offset_B_w = int_cell(get_permuted_offset(UPCAST_STRIDE_K, warp_id * AB_THR_LAYOUT_ROW + lane_id // AB_THR_LAYOUT_COL, lane_id % AB_THR_LAYOUT_COL))  # noqa: E501
                    smem_offset_A_r = int_cell(get_permuted_offset(UPCAST_STRIDE_K, warp_id_m * NUM_MMA_M * 16 + lane_id % 16, lane_id // 16))  # noqa: E501
                    smem_offset_B_r = int_cell(get_permuted_offset(UPCAST_STRIDE_K, warp_id_n * NUM_MMA_N * 16 + 8 * (lane_id // 16) + lane_id % 8, lane_id % 16 // 8))  # noqa: E501

                    smem_offset_A_w += wg_id * (SMEM_SIZE // FP16_BYTES // VEC_LEN)
                    smem_offset_B_w += wg_id * (SMEM_SIZE // FP16_BYTES // VEC_LEN)
                    smem_offset_A_r += wg_id * (SMEM_SIZE // FP16_BYTES // VEC_LEN)
                    smem_offset_B_r += wg_id * (SMEM_SIZE // FP16_BYTES // VEC_LEN)

                    @Tx.inline
                    def async_load_A_to_smem(stage):
                        row_in_blk = int_cell(warp_id * AB_THR_LAYOUT_ROW + lane_id // AB_THR_LAYOUT_COL)  # noqa: E501
                        for i in range(M_rows_per_thread):
                            Tx.ptx.cp_async(A_smem.ptr_to([stage * BLK_M * BLK_K + smem_offset_A_w * VEC_LEN]), A_gmem_1d.ptr_to([thr_local_A_offset[i]]), cp_size=16, prefetch_size=128,  # noqa: E501
                                            predicate=rmap[row_in_blk] < rbound)
                            row_in_blk += AB_THR_LAYOUT_ROW * WARP_COUNT
                            thr_local_A_offset[i] += BLK_K
                            smem_offset_A_w = advance_offset_by_row(AB_THR_LAYOUT_ROW * WARP_COUNT, UPCAST_STRIDE_K, smem_offset_A_w)  # noqa: E501
                        smem_offset_A_w -= M_rows_per_thread * AB_THR_LAYOUT_ROW * WARP_COUNT * UPCAST_STRIDE_K  # noqa: E501

                    @Tx.inline
                    def async_load_B_to_smem(stage):
                        for i in range(N_rows_per_thread):
                            Tx.ptx.cp_async(B_smem.ptr_to([stage * BLK_N * BLK_K + smem_offset_B_w * VEC_LEN]), B_gmem_1d.ptr_to([thr_local_B_offset[i]]), cp_size=16, prefetch_size=128)  # noqa: E501
                            thr_local_B_offset[i] += BLK_K
                            smem_offset_B_w = advance_offset_by_row(AB_THR_LAYOUT_ROW * WARP_COUNT, UPCAST_STRIDE_K, smem_offset_B_w)  # noqa: E501
                        smem_offset_B_w -= N_rows_per_thread * AB_THR_LAYOUT_ROW * WARP_COUNT * UPCAST_STRIDE_K  # noqa: E501

                    @Tx.inline
                    def compute_gemm(stage):
                        a_frag = Tx.alloc_local([NUM_MMA_M, 8], "float16")
                        b_frag = Tx.alloc_local([8], "float16")

                        for mma_k in range(MMA_K):
                            for mma_m in range(NUM_MMA_M):
                                Tx.ptx.ldmatrix(False, 4, ".b16", a_frag.ptr_to([mma_m, 0]), A_smem.ptr_to([stage * BLK_M * BLK_K + smem_offset_A_r * VEC_LEN]))  # noqa: E501
                                smem_offset_A_r = advance_offset_by_row(16, UPCAST_STRIDE_K, smem_offset_A_r)  # noqa: E501
                            smem_offset_A_r = advance_offset_by_column(2, smem_offset_A_r, mma_k) - NUM_MMA_M * 16 * UPCAST_STRIDE_K  # noqa: E501
                            for mma_n in range(NUM_MMA_N):
                                Tx.ptx.ldmatrix(False, 4, ".b16", b_frag.ptr_to([0]), B_smem.ptr_to([stage * BLK_N * BLK_K + smem_offset_B_r * VEC_LEN]))  # noqa: E501
                                smem_offset_B_r = advance_offset_by_row(16, UPCAST_STRIDE_K, smem_offset_B_r)  # noqa: E501
                                for mma_m in range(NUM_MMA_M):
                                    mma_sync_m16n16k16_row_col_f16f16f32(s_frag, s_frag.byte_offset_of([mma_m, mma_n, 0]),  # noqa: E501
                                                                            a_frag, a_frag.byte_offset_of([mma_m, 0]),  # noqa: E501
                                                                            b_frag, b_frag.byte_offset_of([0]), False)  # noqa: E501
                            smem_offset_B_r = advance_offset_by_column(2, smem_offset_B_r, mma_k) - NUM_MMA_N * 16 * UPCAST_STRIDE_K  # noqa: E501
                        smem_offset_A_r -= MMA_K * 2
                        smem_offset_B_r -= MMA_K * 2

                    # prelogue
                    for stage in range(min(NUM_STAGES, K_TILE_CNT)):
                        async_load_A_to_smem(stage)
                        async_load_B_to_smem(stage)
                        Tx.ptx.cp_async.commit_group()

                    # main loop
                    for k_tile in range(K_TILE_CNT - NUM_STAGES):
                        stage = int_cell(k_tile % NUM_STAGES)
                        # wait for the stage to complete
                        Tx.ptx.cp_async.wait_group(NUM_STAGES - 1)
                        scope_sync()
                        # compute gemm for this tile
                        compute_gemm(stage)
                        scope_sync()
                        # prefetch next tile for this stage
                        async_load_A_to_smem(stage)
                        async_load_B_to_smem(stage)
                        Tx.ptx.cp_async.commit_group()

                    # epilogue
                    Tx.ptx.cp_async.wait_group(0)
                    scope_sync()
                    for k_tile in range(min(NUM_STAGES, K_TILE_CNT)):
                        stage = int_cell((k_tile + max(0, K_TILE_CNT - NUM_STAGES)) % NUM_STAGES)
                        compute_gemm(stage)
                    scope_sync()

                    # write back
                    @Tx.inline
                    def store_C_to_smem():
                        for mma_m in range(NUM_MMA_M):
                            for mma_n in range(NUM_MMA_N):
                                s_frag_f16 = Tx.alloc_local([8], "float16")
                                Tx.cast(s_frag_f16[:], s_frag[mma_m, mma_n, :])
                                c_smem_offset_w = int_cell(get_permuted_offset(UPCAST_STRIDE_N, warp_id_m * NUM_MMA_M * 16 + mma_m * 16 + lane_id % 16, warp_id_n * NUM_MMA_N * 2 + mma_n * 2 + lane_id // 16))  # noqa: E501
                                c_smem_offset_w += wg_id * (SMEM_SIZE // FP16_BYTES // VEC_LEN)
                                Tx.ptx.stmatrix(C_smem.ptr_to([c_smem_offset_w * VEC_LEN]), s_frag_f16.ptr_to([0]), num=4)  # noqa: E501

                    @Tx.inline
                    def write_C_to_gmem():
                        C_gmem_1d = C.view(-1)
                        C_gmem_2d = C.view(-1, N)
                        row_in_blk = int_cell(warp_id * C_THR_LAYOUT_ROW + lane_id // C_THR_LAYOUT_COL)  # noqa: E501
                        c_gmem_offset_base = int_cell(C_gmem_2d.elem_offset_of([0, n_idx * BLK_N + lane_id % C_THR_LAYOUT_COL * VEC_LEN]))  # noqa: E501
                        c_smem_offset_r = int_cell(get_permuted_offset(UPCAST_STRIDE_N, row_in_blk, lane_id % C_THR_LAYOUT_COL))  # noqa: E501
                        c_smem_offset_r += wg_id * (SMEM_SIZE // FP16_BYTES // VEC_LEN)
                        for i in range(ceildiv(BLK_M, C_THR_LAYOUT_ROW * WARP_COUNT)):
                            if rmap[row_in_blk] < rbound:
                                c_gmem_offset = int_cell(c_gmem_offset_base + C_gmem_2d.elem_offset_of([rmap[row_in_blk], 0]))  # noqa: E501
                                store_128b(C_gmem_1d.ptr_to([c_gmem_offset]), C_smem.ptr_to([c_smem_offset_r * VEC_LEN]))  # noqa: E501
                            row_in_blk += C_THR_LAYOUT_ROW * WARP_COUNT
                            c_smem_offset_r = advance_offset_by_row(C_THR_LAYOUT_ROW * WARP_COUNT, UPCAST_STRIDE_N, c_smem_offset_r)  # noqa: E501

                    store_C_to_smem()
                    scope_sync()
                    write_C_to_gmem()

                M_TILE_CNT = Tx.meta_var(ceildiv(num_tokens_post_padded[0], BLK_M))
                scheduler = ClusterPersistentScheduler2D("sched", num_m_tiles=M_TILE_CNT, num_n_tiles=N_TILE_CNT, num_clusters=cta_cnt * 2, l2_group_size=1)  # noqa: E501
                scheduler.init(bx * 2 + wg_id)
                m_idx = Tx.meta_var(scheduler.m_idx)
                n_idx = Tx.meta_var(scheduler.n_idx)

                while scheduler.valid():
                    compute_tile(m_idx, n_idx)
                    scheduler.next_tile()
    # fmt: on
    return group_gemm


@pytest.mark.parametrize("task", test_configs)
def test_group_gemm(task):
    batch_size = task["batch_size"]
    hidden_size = task["hidden_size"]
    num_experts = task["num_experts"]
    top_k = task["top_k"]
    intermediate_size = task["intermediate_size"]

    (x1, w13, routing_weights, selected_experts, x2, w2) = gen_input(
        batch_size, hidden_size, num_experts, top_k, intermediate_size
    )

    def tir():
        def torch_to_tvm(tensor):
            return tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

        target = tvm.target.Target("cuda")
        dev = tvm.cuda(0)
        x1_tvm = torch_to_tvm(x1)
        w13_tvm = torch_to_tvm(w13)
        routing_weights_tvm_1 = torch_to_tvm(routing_weights)
        selected_experts_tvm_1 = torch_to_tvm(selected_experts)
        routing_weights_tvm_2 = torch_to_tvm(routing_weights.reshape(batch_size * top_k, 1))
        selected_experts_tvm_2 = torch_to_tvm(selected_experts.reshape(batch_size * top_k, 1))
        x2_tvm = torch_to_tvm(x2)
        w2_tvm = torch_to_tvm(w2)

        out1_tvm = tvm.runtime.empty(
            (batch_size, top_k, 2 * intermediate_size), dtype="float16", device=dev
        )
        out2_tvm = tvm.runtime.empty(
            (batch_size * top_k, 1, hidden_size), dtype="float16", device=dev
        )

        def get_config(batch_size):
            config = {
                "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "num_stages": 5},
                "8": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "num_stages": 3},
                "32": [
                    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "num_stages": 4},
                    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "num_stages": 2},
                ],
                "64": {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "num_stages": 3,
                },
                "128": {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "num_stages": 3,
                },
            }
            return config[str(batch_size)]

        def get_config_space():
            space = []
            for block_size_m in [16, 32]:
                for block_size_n in [64, 128, 256]:
                    for block_size_k in [64, 128, 256]:
                        for num_stages in [2, 3, 4, 5]:
                            space.append(
                                {
                                    "BLOCK_SIZE_M": block_size_m,
                                    "BLOCK_SIZE_N": block_size_n,
                                    "BLOCK_SIZE_K": block_size_k,
                                    "num_stages": num_stages,
                                }
                            )
            return space

        config_space = get_config_space() if AUOTUNE else [get_config(batch_size)]

        for tir_config in config_space:
            tir_config_1 = tir_config if not isinstance(tir_config, list) else tir_config[0]
            tir_config_2 = tir_config if not isinstance(tir_config, list) else tir_config[1]
            assert tir_config_1["BLOCK_SIZE_M"] == tir_config_2["BLOCK_SIZE_M"]
            sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_group_gemm(
                tir_config_1["BLOCK_SIZE_M"], num_experts, selected_experts
            )
            sorted_token_ids_tvm = torch_to_tvm(sorted_token_ids)
            expert_ids_tvm = torch_to_tvm(expert_ids)
            num_tokens_post_padded_tvm = torch_to_tvm(num_tokens_post_padded)
            try:
                grp_gemm1 = get_group_gemm_kernel(
                    batch_size,
                    hidden_size,
                    num_experts,
                    top_k,
                    intermediate_size * 2,
                    tir_config_1,
                    False,
                )
                grp_gemm2 = get_group_gemm_kernel(
                    batch_size * top_k,
                    intermediate_size,
                    num_experts,
                    1,
                    hidden_size,
                    tir_config_2,
                    True,
                )
            except Exception as e:
                print(f"Error compiling tir kernel: {e}")
                continue
            with target:
                mod1 = tvm.IRModule({"main": grp_gemm1})
                mod1 = tvm.compile(mod1, target=target, tir_pipeline="tirx")

                mod2 = tvm.IRModule({"main": grp_gemm2})
                mod2 = tvm.compile(mod2, target=target, tir_pipeline="tirx")

                def func1():
                    return mod1(
                        x1_tvm,
                        w13_tvm,
                        out1_tvm,
                        routing_weights_tvm_1,
                        selected_experts_tvm_1,
                        sorted_token_ids_tvm,
                        expert_ids_tvm,
                        num_tokens_post_padded_tvm,
                    )

                ms1 = bench(
                    func1,
                    warmup=3,
                    repeat=30,
                    proton_name=f"tvm gemm1 + {tir_config_1}",
                    debug=DEBUG,
                )
                print("tvm gemm1", ms1)

                def func2():
                    return mod2(
                        x2_tvm,
                        w2_tvm,
                        out2_tvm,
                        routing_weights_tvm_2,
                        selected_experts_tvm_2,
                        sorted_token_ids_tvm,
                        expert_ids_tvm,
                        num_tokens_post_padded_tvm,
                    )

                ms2 = bench(
                    func2,
                    warmup=3,
                    repeat=30,
                    proton_name=f"tvm gemm2 + {tir_config_2}",
                    debug=DEBUG,
                )
                print("tvm gemm2", ms2)

        return out1_tvm.numpy(), out2_tvm.numpy().reshape(batch_size, top_k, hidden_size)

    def sglang():
        _ensure_sglang_server_args()

        def get_config(batch_size):
            get_config_func = functools.partial(
                try_get_optimal_moe_config,
                w13.shape,
                (w2.shape[0], w2.shape[1], w2.shape[2]),
                selected_experts.shape[1],
                "float16",
                block_shape=None,
            )
            return get_config_func(batch_size)

        sgL_config = get_config(batch_size)
        print(f"sgL_config: {sgL_config}")
        sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_group_gemm(
            sgL_config["BLOCK_SIZE_M"], num_experts, selected_experts
        )

        out1 = torch.empty(
            (batch_size, top_k, 2 * intermediate_size), dtype=torch.float16, device="cuda"
        )
        out2 = torch.empty((batch_size, top_k, hidden_size), dtype=torch.float16, device="cuda")

        def func1():
            return invoke_fused_moe_kernel(
                x1,
                w13,
                None,  # bias
                out1,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=routing_weights,
                topk_ids=selected_experts,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=top_k,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

        ms1 = bench(func1, warmup=3, repeat=30, proton_name="sglang gemm1", debug=DEBUG)
        print("sglang gemm1", ms1)

        def func2():
            return invoke_fused_moe_kernel(
                x2,
                w2,
                None,  # bias
                out2,
                None,  # A_scale
                None,  # B_scale
                None,  # B_zp
                topk_weights=routing_weights,
                topk_ids=selected_experts,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=1,
                config=sgL_config,
                compute_type=tl.float16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
            )

        ms2 = bench(func2, warmup=3, repeat=30, proton_name="sglang gemm2", debug=DEBUG)
        print("sglang gemm2", ms2)
        return out1.cpu().numpy(), out2.cpu().numpy()

    with ProtonContext("group_gemm", debug=DEBUG):
        out1_tir, out2_tir = tir()
        out1_sglang, out2_sglang = sglang()
        np.testing.assert_allclose(out1_tir, out1_sglang, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(out2_tir, out2_sglang, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    for config in test_configs:
        test_group_gemm(config)
