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

import tvm
import tvm.testing
from tvm.script import tirx as Tx

M = 1024
N = 1024
BLOCK_M = 64
BLOCK_N = 64
NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N


@Tx.meta_class
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = Tx.alloc_buffer([1], "int32", scope="local", align=4)

    @Tx.inline
    def semaphore_wait(self, *coord):
        with Tx.thread():
            while 1:
                Tx.ptx.ld_global_acquire(
                    self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                )
                if Tx.cuda.syncthreads_and(self.state[0] == self.cnt):
                    break
                Tx.cuda.nano_sleep(40)

    @Tx.inline
    def semaphore_notify(self, *coord):
        tid = Tx.thread_id([1024])
        with Tx.thread():
            Tx.cuda.cta_sync()
            if Tx.filter(tid, 0, 1):
                with Tx.thread():
                    Tx.cuda.atomic_add(
                        self.sem.access_ptr("rw", offset=self.sem.elem_offset_of(coord)), 1
                    )
            Tx.cuda.thread_fence()


@Tx.prim_func(tirx=True)
def partial_reduction_ref_stage1(A: Tx.handle, B: Tx.handle):
    A_ptr = Tx.match_buffer(A, (M, N), "float32")
    B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    with Tx.kernel():
        bx, by = Tx.cta_id([NUM_BLOCK_M, NUM_BLOCK_N])
        Tx.thread_id([1024])
        with Tx.cta():
            A_smem = Tx.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
            B_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            Tx.copy(
                A_smem, A_ptr[bx * BLOCK_M : (bx + 1) * BLOCK_M, by * BLOCK_N : (by + 1) * BLOCK_N]
            )
            Tx.sum(B_smem, A_smem)
            Tx.copy(B_ptr[bx * BLOCK_M : (bx + 1) * BLOCK_M, by], B_smem)


@Tx.prim_func(tirx=True)
def partial_reduction_ref_stage2(B: Tx.handle, C: Tx.handle):
    B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    C_ptr = Tx.match_buffer(C, (M, 1), "float32")
    with Tx.kernel():
        bx = Tx.cta_id([NUM_BLOCK_M])
        Tx.thread_id([1024])
        with Tx.cta():
            B_smem = Tx.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
            C_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
            Tx.copy(B_smem, B_ptr[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])
            Tx.sum(C_smem, B_smem)
            Tx.copy(C_ptr[bx * BLOCK_M : (bx + 1) * BLOCK_M, 0], C_smem)


STAGE_1_SM_CNT = 116
STAGE_2_SM_CNT = 16
TOTAL_SM_CNT = STAGE_1_SM_CNT + STAGE_2_SM_CNT


@Tx.meta_class
class SpatialTileScheduler:
    def __init__(self, prefix: str, tile_num: tuple[int, int], sm_cnt: int):
        self.tile_num = tile_num
        self.sm_cnt = sm_cnt
        self.m_idx = Tx.alloc_local([1], "int32")
        self.n_idx = Tx.alloc_local([1], "int32")
        self.linear_idx = Tx.alloc_local([1], "int32")

    def get_current_m_n_idx(self, linear_idx):
        row = linear_idx // self.tile_num[1]
        col = linear_idx % self.tile_num[1]
        return (row, col)

    @Tx.inline
    def init(self, linear_init):
        self.linear_idx[0] = linear_init
        self.m_idx[0] = self.get_current_m_n_idx(linear_init)[0]
        self.n_idx[0] = self.get_current_m_n_idx(linear_init)[1]

    @Tx.inline
    def next_tile(self):
        self.linear_idx[0] = self.linear_idx[0] + self.sm_cnt
        self.m_idx[0] = self.get_current_m_n_idx(self.linear_idx[0])[0]
        self.n_idx[0] = self.get_current_m_n_idx(self.linear_idx[0])[1]

    def valid(self):
        return self.linear_idx[0] < functools.reduce(lambda x, y: x * y, self.tile_num)


@Tx.prim_func(tirx=True)
def partial_reduction_fused(A: Tx.handle, B: Tx.handle, C: Tx.handle, semaphore: Tx.handle):
    A_ptr = Tx.match_buffer(A, (M, N), "float32")
    B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
    C_ptr = Tx.match_buffer(C, (M, 1), "float32")
    sem_ptr = Tx.match_buffer(semaphore, (NUM_BLOCK_M,), "int32")
    with Tx.kernel():
        bx = Tx.cta_id([TOTAL_SM_CNT])
        Tx.thread_id([1024])
        sem = Semaphore(NUM_BLOCK_N, sem_ptr)
        if Tx.filter(bx, 0, STAGE_1_SM_CNT):
            with Tx.cta():
                A_smem = Tx.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
                B_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                stage1_scheduler = SpatialTileScheduler(
                    "stage1", (NUM_BLOCK_M, NUM_BLOCK_N), STAGE_1_SM_CNT
                )
                stage1_scheduler.init(bx)
                while stage1_scheduler.valid():
                    m_idx = Tx.meta_var(stage1_scheduler.m_idx[0])
                    n_idx = Tx.meta_var(stage1_scheduler.n_idx[0])
                    Tx.copy(
                        A_smem,
                        A_ptr[
                            m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M,
                            n_idx * BLOCK_N : (n_idx + 1) * BLOCK_N,
                        ],
                    )
                    Tx.sum(B_smem, A_smem)
                    Tx.copy(B_ptr[m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M, n_idx], B_smem)
                    sem.semaphore_notify(m_idx)
                    stage1_scheduler.next_tile()
        if Tx.filter(bx, STAGE_1_SM_CNT, TOTAL_SM_CNT):
            with Tx.cta():
                B_smem = Tx.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
                C_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                stage2_scheduler = SpatialTileScheduler("stage2", (NUM_BLOCK_M, 1), STAGE_2_SM_CNT)
                stage2_scheduler.init(bx - STAGE_1_SM_CNT)
                while stage2_scheduler.valid():
                    m_idx = Tx.meta_var(stage2_scheduler.m_idx[0])
                    sem.semaphore_wait(m_idx)
                    Tx.copy(B_smem, B_ptr[m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M, :])
                    Tx.sum(C_smem, B_smem)
                    Tx.copy(C_ptr[m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M, 0], C_smem)
                    stage2_scheduler.next_tile()


@tvm.testing.requires_cuda_compute_version(8)
def test_partial_reduction():
    A_np = np.random.randn(M, N).astype(np.float32)
    B_np = np.zeros((M, NUM_BLOCK_N), dtype=np.float32)
    C_np = np.zeros((M, 1), dtype=np.float32)
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    C_tvm = tvm.runtime.tensor(C_np, device=DEV)
    C_tvm_fused = tvm.runtime.tensor(C_np, device=DEV)
    sem_tvm = tvm.runtime.tensor(np.zeros((NUM_BLOCK_M,), dtype=np.int32), device=DEV)
    with target:
        ref_mod_stage_1 = tvm.IRModule({"main": partial_reduction_ref_stage1})
        ref_mod_stage_1 = tvm.compile(ref_mod_stage_1, target=target, tir_pipeline="tirx")
        ref_mod_stage_1(A_tvm, B_tvm)
        ref_mod_stage_2 = tvm.IRModule({"main": partial_reduction_ref_stage2})
        ref_mod_stage_2 = tvm.compile(ref_mod_stage_2, target=target, tir_pipeline="tirx")
        ref_mod_stage_2(B_tvm, C_tvm)
        ret_ref_stage_2 = C_tvm.numpy()
        fused_mod = tvm.IRModule({"main": partial_reduction_fused})
        fused_mod = tvm.compile(fused_mod, target=target, tir_pipeline="tirx")
        fused_mod(A_tvm, B_tvm, C_tvm_fused, sem_tvm)
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_ref_stage_2, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    test_partial_reduction()
