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

import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.bench import ProtonContext, bench
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D


@tvm.testing.requires_cuda_compute_version(9, exact=True)
def test_fp8_gemm_hopper_no_ws():
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    f8_bytes = 1
    f16_bytes = 2
    SM_COUNT = DEV.multi_processor_count
    M, N, K = 8192, 8192, 8192
    BLK_M, BLK_N, BLK_K = 128, 256, 128
    WGMMA_M, WGMMA_N, WGMMA_K = 64, 256, 32

    swizzleA = swizzleB = 3  # 128B swizzle
    swizzleC = 3  # 128B swizzle
    GROUP_SIZE = 16
    STAGES_TMA = 4
    STAGES_WGMMA = 2
    STAGES_EPI = 2
    TMA_BYTES = BLK_M * BLK_K * f8_bytes + BLK_K * BLK_N * f8_bytes

    m_blocks = (M + BLK_M - 1) // BLK_M
    n_blocks = (N + BLK_N - 1) // BLK_N

    # fmt: off
    @Tx.inline
    def tma_load(tid, m_idx, n_idx, k_tile, A_smem: tvm.tirx.Buffer, B_smem: tvm.tirx.Buffer, A_map, B_map, bars: tvm.tirx.Buffer):  # noqa: E501
        if Tx.filter(tid, tid == 0):
            with Tx.thread():
                stage = Tx.meta_var(k_tile % STAGES_TMA)
                Tx.ptx.mbarrier.arrive.expect_tx(bars.ptr_to([stage]), TMA_BYTES)
                Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.ptr_to([stage, 0]), bars.ptr_to([stage]), Tx.address_of(A_map), 0, 1, "", k_tile * BLK_K, m_idx * BLK_M)  # noqa: E501
                Tx.ptx.cp_async.bulk.tensor.g2c(2, B_smem.ptr_to([stage, 0]), bars.ptr_to([stage]), Tx.address_of(B_map), 0, 1, "", k_tile * BLK_K, n_idx * BLK_N)  # noqa: E501

    def get_accum_list(C, C_elems):
        return [C[i] for i in range(C_elems)]

    @Tx.inline
    def mma_compute(wg_id, k_tile, A_smem: tvm.tirx.Buffer, B_smem: tvm.tirx.Buffer, accum, bars: tvm.tirx.Buffer, descA, descB):  # noqa: E501
        stage = Tx.meta_var(k_tile % STAGES_TMA)
        parity = Tx.meta_var((k_tile // STAGES_TMA) % 2)
        Tx.ptx.mbarrier.try_wait(bars.ptr_to([stage]), parity)
        for inner_k in Tx.serial(BLK_K // WGMMA_K):
            A_offset = Tx.meta_var(wg_id * BLK_M * BLK_K // 2 + inner_k * WGMMA_K)
            B_offset = Tx.meta_var(inner_k * WGMMA_K)
            Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descA), A_smem.ptr_to([stage, A_offset]), 1, 64, swizzle=3)  # noqa: E501
            Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descB), B_smem.ptr_to([stage, B_offset]), 1, 64, swizzle=3)  # noqa: E501
            Tx.ptx.wgmma.mma_async.ss(descA, descB, *get_accum_list(accum, 128), M=WGMMA_M, N=WGMMA_N, K=WGMMA_K, in_dtype="float8_e4m3fn", out_dtype="float32", transA=False, transB=False, scaleA=1.0, scaleB=1.0, scaleD=True)  # noqa: E501
        Tx.ptx.wgmma.commit_group()

    @Tx.inline
    def r2S(warp_id, lane_id, C_smem: tvm.tirx.Buffer, accum, accum_half, n_tile):
        Tx.cuda.cta_sync()
        for st_tile in Tx.serial(4):
            for i in Tx.serial(8):
                accum_half[i] = accum[n_tile * 32 + st_tile * 8 + i]
            col_noswizzle = Tx.meta_var(st_tile * 2 + lane_id // 16)
            col = Tx.meta_var((lane_id % 8) ^ col_noswizzle)
            row = Tx.meta_var(warp_id * 16 + lane_id % 16)
            Tx.ptx.stmatrix(C_smem.ptr_to([n_tile % STAGES_EPI, row, col * 8]), accum_half.ptr_to([0]), num=4)  # noqa: E501

    @Tx.inline
    def s2G(warp_id, lane_id, C_smem: tvm.tirx.Buffer, C_map, m_idx, n_idx, n_tile):
        Tx.ptx.fence.proxy_async("shared::cta")
        Tx.cuda.cta_sync()
        if Tx.filter(tid, warp_id == 0 and lane_id == 0):  # noqa: F821
            with Tx.thread():
                Tx.ptx.cp_async.bulk.tensor.s2g(2, C_smem.ptr_to([n_tile % STAGES_EPI, 0, 0]), Tx.address_of(C_map), "", n_idx * BLK_N + n_tile * 64, m_idx * BLK_M)  # noqa: E501
                Tx.ptx.cp_async.bulk.commit_group()
                Tx.ptx.cp_async.bulk.wait_group(1, read=True)

    @Tx.inline
    def write_epilogue(warp_id, lane_id, m_idx, n_idx, C_smem: tvm.tirx.Buffer, C_map, accum, accum_half):  # noqa: E501
        Tx.cuda.cta_sync()
        for n_tile in Tx.serial(BLK_N // 64):
            if n_tile != 0:
                # s2G for the previous stage
                s2G(warp_id, lane_id, C_smem, C_map, m_idx, n_idx, n_tile - 1)
            # r2S for the current stage
            r2S(warp_id, lane_id, C_smem, accum, accum_half, n_tile)
        # s2G for the last stage
        s2G(warp_id, lane_id, C_smem, C_map, m_idx, n_idx, BLK_N // 64 - 1)
        Tx.ptx.cp_async.bulk.wait_group(0, read=False)

    @Tx.prim_func(tirx=True)
    def manual(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (M, K), "float8_e4m3fn", scope="global", layout=Tx.TileLayout(Tx.S[M, K]))  # noqa: E501
        B = Tx.match_buffer(B_ptr, (N, K), "float8_e4m3fn", scope="global", layout=Tx.TileLayout(Tx.S[N, K]))  # noqa: E501
        C = Tx.match_buffer(C_ptr, (M, N), "float16", scope="global", layout=Tx.TileLayout(Tx.S[M, N]))  # noqa: E501

        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        C_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)

        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float8_e4m3fn", 2, A.data, K, M, f8_bytes * K, BLK_K, BLK_M, 1, 1, 0, swizzleA, 0, 0)  # noqa: E501
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float8_e4m3fn", 2, B.data, K, N, f8_bytes * K, BLK_K, BLK_N, 1, 1, 0, swizzleB, 0, 0)  # noqa: E501
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", C_map, "float16", 2, C.data, N, M, f16_bytes * N, 64, BLK_M, 1, 1, 0, swizzleC, 0, 0)  # noqa: E501

        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT])
            tid = Tx.thread_id([128 * 2])
            warp_id = Tx.warp_id([8])
            lane_id = Tx.lane_id([32])
            wg_id = Tx.warpgroup_id([2])

            with Tx.cta():
                # tensor stroage
                A_smem = Tx.alloc_buffer([STAGES_TMA, BLK_M * BLK_K], "float8_e4m3fn", scope="shared.dyn", align=128)  # noqa: E501
                B_smem = Tx.alloc_buffer([STAGES_TMA, BLK_K * BLK_N], "float8_e4m3fn", scope="shared.dyn", align=128)  # noqa: E501
                C_smem = Tx.alloc_buffer([STAGES_EPI, BLK_M, 64], "float16", scope="shared.dyn", align=1024)  # noqa: E501
                # barriers
                bars = Tx.alloc_buffer([STAGES_TMA], "uint64", scope="shared.dyn", align=8)
                desc_A: Tx.uint64
                desc_B: Tx.uint64

                with Tx.thread():
                    # index
                    tma_index: Tx.int32
                    mma_index: Tx.int32
                    # acuumulators
                    accum = Tx.alloc_buffer([128], "float32", scope="local")
                    accum_half = Tx.alloc_buffer([8], "float16", scope="local")
                    # tile scheduler
                    tile_scheduler = ClusterPersistentScheduler2D(
                        "tile_scheduler",
                        num_m_tiles=m_blocks,
                        num_n_tiles=n_blocks,
                        num_clusters=SM_COUNT,
                        l2_group_size=GROUP_SIZE,
                    )

                    # initialize the tile scheduler
                    tile_scheduler.init(bx)

                    while (tile_scheduler.valid()):
                        # initialize the barriers
                        if Tx.filter(tid, tid == 0):
                            with Tx.thread():
                                for i in range(STAGES_TMA):
                                    Tx.ptx.mbarrier.init(bars.ptr_to([i]), 1)
                        Tx.cuda.cta_sync()
                        # initialize the index
                        tma_index = 0
                        mma_index = 0
                        # initialize the accumulators
                        for i in range(128):
                            accum[i] = 0
                        Tx.cuda.cta_sync()

                        m_idx = Tx.meta_var(tile_scheduler.m_idx)
                        n_idx = Tx.meta_var(tile_scheduler.n_idx)
                        # prelogue
                        for _ in range(STAGES_TMA):
                            tma_load(tid, m_idx, n_idx, tma_index, A_smem, B_smem, A_map, B_map, bars)  # noqa: E501
                            tma_index = tma_index + 1
                        for _ in range(STAGES_WGMMA - 1):
                            mma_compute(wg_id, mma_index, A_smem, B_smem, accum, bars, desc_A, desc_B)  # noqa: E501, F821
                            mma_index = mma_index + 1

                        # mainloop
                        k_tile_count = Tx.meta_var((K + BLK_K - 1) // BLK_K)
                        for _ in range(k_tile_count):
                            if mma_index < k_tile_count:
                                mma_compute(wg_id, mma_index, A_smem, B_smem, accum, bars, desc_A, desc_B)  # noqa: E501, F821
                                mma_index = mma_index + 1
                            # wait for oldest one stage to finish
                            if _ == k_tile_count - 1:
                                Tx.ptx.wgmma.wait_group(0)
                            else:
                                Tx.ptx.wgmma.wait_group(STAGES_WGMMA - 1)
                            Tx.cuda.cta_sync()
                            # load the next tile
                            if tma_index < k_tile_count:
                                tma_load(tid, m_idx, n_idx, tma_index, A_smem, B_smem, A_map, B_map, bars)  # noqa: E501
                                tma_index = tma_index + 1

                        # epilogue
                        write_epilogue(warp_id, lane_id, m_idx, n_idx, C_smem, C_map, accum, accum_half)  # noqa: E501

                        # move to the next tile
                        tile_scheduler.next_tile()
    # fmt: on

    A_np = np.random.randn(M, K).astype(ml_dtypes.float8_e4m3fn)
    B_np = np.random.randn(N, K).astype(ml_dtypes.float8_e4m3fn)
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)

    def flops(ms):
        return M * N * K * 2 / ms / 1e9

    def cublas_gemm():
        C_np = np.empty((M, N)).astype(np.float16)

        gemm_func = tvm.get_global_func("cublaslt.fp8_gemm")

        C_tvm = tvm.runtime.tensor(C_np, device=DEV)
        workspace = tvm.runtime.empty((32 * 1024 * 1024,), dtype="uint8", device=DEV)
        scale = tvm.runtime.tensor(np.array([1.0], dtype="float32"), device=DEV)

        def func():
            return gemm_func(A_tvm, B_tvm, workspace, scale, C_tvm)

        ms = bench(func, warmup=0, repeat=10, proton_name="cublas")
        print(f"CUBLAS flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")
        return C_tvm

    def tir_gemm():
        C_np = -np.ones((M, N), dtype=np.float16)
        C_tvm = tvm.runtime.tensor(C_np, device=DEV)

        with target:
            mod = tvm.IRModule({"main": manual})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            def func():
                return mod(A_tvm, B_tvm, C_tvm)

            ms = bench(func, warmup=0, repeat=10, proton_name="tir")
            print(f"TIR flops: {flops(ms)} GFLOPS, time: {ms:.3f} ms")

        return C_tvm

    with ProtonContext("matmul"):
        C_cublas = cublas_gemm()
        C_tir = tir_gemm()

    tvm.testing.assert_allclose(C_tir.numpy(), C_cublas.numpy(), rtol=2e-2, atol=1e-4)


if __name__ == "__main__":
    test_fp8_gemm_hopper_no_ws()
