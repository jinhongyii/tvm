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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.bench import ProtonContext, bench


@pytest.mark.skip
@tvm.testing.requires_cuda_compute_version(8)
@tvm.testing.requires_cublas
def test_hgemm_ampere():
    M, N, K = 8192, 8192, 8192
    MI, NI, KI = 128, 128, 32
    MII, NII, _KII = 64, 64, 16
    wmmaM, wmmaN = 16, 16

    _BLK_M, _BLK_N, _BLK_K = 128, 128, 32

    @Tx.inline
    def load_global_to_shared_A(
        tz: int,
        ty: int,
        tx: int,
        by: int,
        SA: Tx.Buffer,
        slice_idx: int,
        A: Tx.Buffer,
        ko: int,
        KI: int,
        K: int,
    ):
        tid = Tx.meta_var(tz * 64 + ty * 32 + tx)
        for i in Tx.serial(4):
            logic_row = Tx.meta_var(i * 32 + tid // 4)
            logic_col = Tx.meta_var((tid % 4) * 8)
            row = Tx.meta_var(i * 16 + tid // 8)
            col = Tx.meta_var(((tid % 8) * 8) ^ ((row & 3) << 3))

            smem_offset = Tx.meta_var(slice_idx * (MI * KI) + row * 64 + col)
            global_offset = Tx.meta_var((by * 128 + logic_row) * K + (ko * KI + logic_col))
            Tx.ptx.cp_async(
                dst_ptr=SA.access_ptr("rw", offset=smem_offset),
                src_ptr=A.access_ptr("rw", offset=global_offset),
                cp_size=16,
            )

    @Tx.inline
    def load_global_to_shared_B(
        tz: int,
        ty: int,
        tx: int,
        bx: int,
        SB: Tx.Buffer,
        slice_idx: int,
        B: Tx.Buffer,
        ko: int,
        KI: int,
        K: int,
    ):
        tid = Tx.meta_var(tz * 64 + ty * 32 + tx)
        for i in Tx.serial(4):
            logic_row = Tx.meta_var(i * 32 + tid // 4)
            logic_col = Tx.meta_var((tid % 4) * 8)
            row = Tx.meta_var(i * 16 + tid // 8)
            col = Tx.meta_var((((tid // 4) % 2) * 32 + (tid % 4) * 8) ^ ((row & 3) << 3))

            smem_offset = Tx.meta_var(slice_idx * (NI * KI) + row * 64 + col)
            global_offset = Tx.meta_var((bx * 128 + logic_row) * K + (ko * KI + logic_col))
            Tx.ptx.cp_async(
                dst_ptr=SB.access_ptr("rw", offset=smem_offset),
                src_ptr=B.access_ptr("rw", offset=global_offset),
                cp_size=16,
            )

    @Tx.inline
    def loadFragA(
        frag: Tx.Buffer, slice_frag: int, SA: Tx.Buffer, slice_idx: int, ki: int, tx: int, tz: int
    ):
        """Loads a 64x16 fragment from shared memory into `frag` from SA[slice_idx]."""
        for i in Tx.serial(4):
            row = Tx.meta_var((tz * 64 + i * 16 + tx // 16 * 8 + tx % 8) // 2)
            col = Tx.meta_var(
                (
                    ((tz * 64 + i * 16 + tx // 16 * 8 + tx % 8) % 2) * 32
                    + (ki * 16 + tx // 8 % 2 * 8)
                )
                ^ ((row & 3) << 3)
            )
            smem_offset = Tx.meta_var(slice_idx * (MI * KI) + row * 64 + col)

            Tx.ptx.ldmatrix(
                dtype="float16",
                trans=False,
                num=4,
                type=".b16",
                local_ptr=frag.data,
                local_offset=i * 4 + slice_frag * 16,
                smem_ptr=SA.access_ptr("r", offset=0),
                smem_offset=smem_offset,
            )

            tmp = frag[slice_frag, i * 4 + 1]
            frag[slice_frag, i * 4 + 1] = frag[slice_frag, i * 4 + 2]
            frag[slice_frag, i * 4 + 2] = tmp

    @Tx.inline
    def loadFragB(
        frag: Tx.Buffer, slice_frag: int, SB: Tx.Buffer, slice_idx: int, ki: int, tx: int, ty: int
    ):
        for i in Tx.serial(4):
            row = Tx.meta_var((ty * 64 + i * 16 + tx // 16 * 8 + tx % 8) // 2)
            col = Tx.meta_var(
                (
                    ((ty * 64 + i * 16 + tx // 16 * 8 + tx % 8) % 2) * 32
                    + (ki * 16 + tx // 8 % 2 * 8)
                )
                ^ ((row & 3) << 3)
            )
            smem_offset = Tx.meta_var(slice_idx * (NI * KI) + row * 64 + col)

            Tx.ptx.ldmatrix(
                dtype="float16",
                trans=False,
                num=4,
                type=".b16",
                local_ptr=frag.data,
                local_offset=i * 4 + slice_frag * 16,
                smem_ptr=SB.access_ptr("r", offset=0),
                smem_offset=smem_offset,
            )

    @Tx.inline
    def store_accum_to_shared(tz, ty, tx, C_smem: tvm.tirx.Buffer, Accum):
        """Helper for storing accumulator results to shared memory with optimized indexing"""
        for i in Tx.serial(4):
            for j in Tx.serial(4):
                for r in Tx.serial(2):
                    for c in Tx.serial(2):
                        # Calculate row and column indices
                        row = Tx.meta_var(tz * 64 + i * 16 + r * 8 + tx // 4)
                        col = Tx.meta_var(ty * 64 + j * 16 + c * 8 + tx % 4 * 2)

                        # Apply crosswise transformation to column
                        scol = Tx.meta_var(col ^ ((row & 3) << 3))

                        # Calculate accumulator offset
                        acc_offset = Tx.meta_var(i * 32 + j * 8 + r * 4 + c * 2)
                        if acc_offset % 8 == 4 or acc_offset % 8 == 5:
                            C_smem[row, scol] = Accum[acc_offset - 2]
                            C_smem[row, scol + 1] = Accum[acc_offset - 2 + 1]
                        elif acc_offset % 8 == 2 or acc_offset % 8 == 3:
                            C_smem[row, scol] = Accum[acc_offset + 2]
                            C_smem[row, scol + 1] = Accum[acc_offset + 2 + 1]
                        else:
                            C_smem[row, scol] = Accum[acc_offset]
                            C_smem[row, scol + 1] = Accum[acc_offset + 1]

    @Tx.inline
    def store_shared_to_global(tz, ty, tx, bx, by, C: tvm.tirx.Buffer, C_smem: tvm.tirx.Buffer):
        """Helper for storing shared memory results to global memory"""
        tid = Tx.meta_var(tz * 64 + ty * 32 + tx)

        for i in Tx.serial(128):
            row = Tx.meta_var(i)
            col = Tx.meta_var(tid)

            scol = Tx.meta_var(col ^ ((row & 3) << 3))

            global_row = Tx.meta_var(by * 128 + row)
            global_col = Tx.meta_var(bx * 128 + col)

            C[global_row, global_col] = Tx.Cast("float16", C_smem[row, scol])

    @Tx.inline
    def mmaSync(
        fragA: tvm.tirx.Buffer,
        fragA_offset: int,
        fragB: tvm.tirx.Buffer,
        fragB_offset: int,
        Accum: tvm.tirx.Buffer,
        accum_offset: int,
    ):
        """Matrix multiply-accumulate operation using tensor cores"""

        # First MMA operation - for accum[0:4]
        Tx.ptx.mma(
            dtype="float32",
            shape="m16n8k16",
            A_layout="row",
            B_layout="col",
            A_dtype="float16",
            B_dtype="float16",
            C_dtype="float32",
            multiplicand_a=fragA.data,
            a_index=fragA_offset,
            multiplicand_b=fragB.data,
            b_index=fragB_offset,
            accumulator=Accum.data,
            c_index=accum_offset,
            saturate=False,
        )

        # Second MMA operation - for accum[4:8]
        Tx.ptx.mma(
            dtype="float32",
            shape="m16n8k16",
            A_layout="row",
            B_layout="col",
            A_dtype="float16",
            B_dtype="float16",
            C_dtype="float32",
            multiplicand_a=fragA.data,
            a_index=fragA_offset,
            multiplicand_b=fragB.data,
            b_index=fragB_offset + 2,
            accumulator=Accum.data,
            c_index=accum_offset + 4,
            saturate=False,
        )

    @Tx.prim_func(tirx=True)
    def efficient(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (M, K), "float16", scope="global")
        B = Tx.match_buffer(B_ptr, (N, K), "float16", scope="global")
        C = Tx.match_buffer(C_ptr, (M, N), "float32", scope="global")

        with Tx.kernel():
            bx, by = Tx.cta_id([M // MI, N // NI])
            tx, ty, tz = Tx.thread_id([32, 2, 2])

            with Tx.cta():
                # 4-pipeline + ld_matrix + ptx_mma
                SA = Tx.alloc_buffer([4, MI, KI], dtype="float16", scope="shared.dyn")
                SB = Tx.alloc_buffer([4, NI, KI], dtype="float16", scope="shared.dyn")
                C_smem = Tx.alloc_buffer([MI, NI], dtype="float32", scope="shared.dyn")

                fragA = Tx.alloc_buffer([2, 16], dtype="uint32", scope="local")
                fragB = Tx.alloc_buffer([2, 16], dtype="uint32", scope="local")
                Accum = Tx.alloc_buffer([128], dtype="float32", scope="local")

                with Tx.thread():
                    for idx in Tx.serial(128):
                        Accum[idx] = 0.0

                    load_global_to_shared_A(tz, ty, tx, by, SA, 0, A, 0, KI, K)
                    load_global_to_shared_B(tz, ty, tx, bx, SB, 0, B, 0, KI, K)
                    Tx.ptx.cp_async.commit_group()

                    load_global_to_shared_A(tz, ty, tx, by, SA, 1, A, 1, KI, K)
                    load_global_to_shared_B(tz, ty, tx, bx, SB, 1, B, 1, KI, K)
                    Tx.ptx.cp_async.commit_group()

                    load_global_to_shared_A(tz, ty, tx, by, SA, 2, A, 2, KI, K)
                    load_global_to_shared_B(tz, ty, tx, bx, SB, 2, B, 2, KI, K)
                    Tx.ptx.cp_async.commit_group()

                    Tx.ptx.cp_async.wait_group(2)
                    Tx.cuda.cta_sync()

                    loadFragA(fragA, 0, SA, 0, 0, tx, tz)
                    loadFragB(fragB, 0, SB, 0, 0, tx, ty)

                    for ko_idx in Tx.serial(K // KI):
                        ko = Tx.meta_var(ko_idx)

                        slice_in: Tx.let = ko % 4
                        loadFragA(fragA, 1, SA, slice_in, 1, tx, tz)
                        loadFragB(fragB, 1, SB, slice_in, 1, tx, ty)

                        for mii in Tx.serial(MII // wmmaM):
                            for nii in Tx.serial(NII // wmmaN):
                                n = (NII // wmmaN - 1 - nii) if Tx.meta_var(mii & 1) else nii
                                mmaSync(fragA, mii * 4, fragB, n * 4, Accum, mii * 32 + n * 8)

                        if ko + 3 < K // KI:
                            slice_out = (ko + 3) % 4
                            load_global_to_shared_A(tz, ty, tx, by, SA, slice_out, A, ko + 3, KI, K)
                            load_global_to_shared_B(tz, ty, tx, bx, SB, slice_out, B, ko + 3, KI, K)

                        Tx.ptx.cp_async.commit_group()
                        Tx.ptx.cp_async.wait_group(2)
                        Tx.cuda.cta_sync()

                        next_slice = (ko + 1) % 4
                        loadFragA(fragA, 0, SA, next_slice, 0, tx, tz)
                        loadFragB(fragB, 0, SB, next_slice, 0, tx, ty)

                        for mii in Tx.serial(MII // wmmaM):
                            for nii in Tx.serial(NII // wmmaN):
                                n = (NII // wmmaN - 1 - nii) if Tx.meta_var(mii & 1) else nii
                                mmaSync(
                                    fragA,
                                    16 * 1 + mii * 4,
                                    fragB,
                                    16 * 1 + n * 4,
                                    Accum,
                                    mii * 32 + n * 8,
                                )

                    store_accum_to_shared(tz, ty, tx, C_smem, Accum)
                    Tx.cuda.cta_sync()
                    store_shared_to_global(tz, ty, tx, bx, by, C, C_smem)

    np.random.seed(0)
    A_np = np.random.randn(M, K).astype(np.float16)
    B_np = np.random.randn(N, K).astype(np.float16)

    DEV = tvm.cuda()
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)

    target = tvm.target.Target("cuda")
    print(target)

    def tvm_gemm(func):
        with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
            mod = tvm.IRModule({"main": func})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
            C_np = np.zeros((M, N), dtype=np.float32)
            C_tvm = tvm.runtime.tensor(C_np, device=DEV)

            def func():
                return mod(A_tvm, B_tvm, C_tvm)

            ms = bench(func, warmup=0, repeat=10, proton_name="tir")
            print(f"TIR time: {ms} ms")
        return C_tvm.numpy()

    # cublas
    def cublas_gemm():
        import torch

        torch_dev = torch.device("cuda")
        A_torch = torch.tensor(A_np, device=torch_dev)
        B_torch = torch.tensor(B_np, device=torch_dev)
        C_torch = torch.zeros((M, N), device=torch_dev)

        def func():
            return torch.matmul(A_torch, B_torch.T)

        ms = bench(func, warmup=0, repeat=10, proton_name="cublas")
        print(f"cublas time: {ms} ms")
        C_torch = func()
        return C_torch.cpu().numpy()

    with target:
        with ProtonContext("hgemm"):
            print(f"M, N, K: {M}, {N}, {K}")
            C_tvm_efficient = tvm_gemm(efficient)
            C_cublas = cublas_gemm()

    tvm.testing.assert_allclose(C_tvm_efficient, C_cublas, rtol=1e-3, atol=1e-3)
    print("test passed")


if __name__ == "__main__":
    test_hgemm_ampere()
