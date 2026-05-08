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
import enum

import numpy as np

import tvm
import tvm.testing
from tvm.script import tirx as Tx


def extract_values(filename):
    # List to store all extracted values
    extracted_values = []

    # Open and read the file
    with open(filename) as file:
        for line in file:
            # Use string manipulation to extract values
            parts = line.strip().split(", ")

            # Extract value a from "block_id: a"
            a_part = parts[0].split(": ")[1]
            a = int(a_part)

            # Extract value b from "fetched_task_type[0]: b"
            b_part = parts[1].split(": ")[1]
            b = int(b_part)

            # Extract value c from "fetched_task_idx[0]: c"
            c_part = parts[2].split(": ")[1]
            c = int(c_part)

            # Extract value d from "fetched_task_idx[1]: d"
            d_part = parts[3].split(": ")[1]
            d = int(d_part)

            # Add the extracted values as a tuple to the result list
            extracted_values.append((a, b, c, d))

    return extracted_values


@tvm.testing.requires_cuda_compute_version(8)
def test_partial_reduction():
    M = 1024
    N = 1024

    BLOCK_M = 64
    BLOCK_N = 64

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    # fmt: off

    def int_var(name: str, scope="local"):
        return Tx.alloc_buffer([1], "int32", scope=scope, align=4)

    class TaskType(enum.Enum):
        PARTIAL = 0
        REDUCE = 1


    @Tx.meta_class
    class StaticTileScheduler:
        def __init__(self, tasks_indptr: Tx.Buffer, task_types: Tx.Buffer, task_indices: Tx.Buffer):
           self.linear_idx = int_var("linear_idx")
           self.linear_lim = int_var("linear_lim")
           self.task_types = task_types
           self.task_indices = task_indices
           self.tasks_indptr = tasks_indptr
           self.task_type = int_var("task_type")
           self.task_idx = Tx.alloc_buffer([2], "int32", scope="local")

        @Tx.inline
        def get_block_coord(self):
            self.task_type[0] = self.task_types[self.linear_idx[0]]
            self.task_idx[0] = self.task_indices[self.linear_idx[0], 0]
            self.task_idx[1] = self.task_indices[self.linear_idx[0], 1]

        @Tx.inline
        def init(self, linear_init):
            self.linear_idx[0] = self.tasks_indptr[linear_init]
            self.linear_lim[0] = self.tasks_indptr[linear_init + 1]
            self.get_block_coord()

        @Tx.inline
        def next_tile(self):
            self.linear_idx[0] = self.linear_idx[0] + 1
            self.get_block_coord()

        def valid(self):
            return self.linear_idx[0] < self.linear_lim[0]

    @Tx.meta_class
    class Semaphore:
        def __init__(self, cnt, buffer, tid):
            self.cnt = cnt
            self.sem = buffer
            self.tid = tid
            self.state = Tx.alloc_buffer([1], "int32", scope="local", align=4)

        @Tx.inline
        def semaphore_wait(self, *coord):
            with Tx.thread():
                while 1:
                    Tx.ptx.ld_global_acquire(self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)))  # noqa: E501
                    if Tx.cuda.syncthreads_and(self.state[0] == self.cnt):
                        break
                    Tx.cuda.nano_sleep(40)

        @Tx.inline
        def semaphore_notify(self, *coord):
            with Tx.thread():
                Tx.cuda.cta_sync()
                if Tx.filter(self.tid, 0, 1):
                    with Tx.thread():
                        Tx.cuda.atomic_add(self.sem.access_ptr("rw", offset=self.sem.elem_offset_of(coord)), 1)  # noqa: E501
                Tx.cuda.thread_fence()


    # reduction on N
    @Tx.prim_func(tirx=True)
    def partial_reduction_ref_stage1(A: Tx.handle, B: Tx.handle):
        A_ptr = Tx.match_buffer(A, (M, N), "float32")
        B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")

        with Tx.kernel():
            bx, by = Tx.cta_id([NUM_BLOCK_M, NUM_BLOCK_N])
            tid = Tx.thread_id([1024])

            with Tx.cta():
                A_smem = Tx.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
                B_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                Tx.copy(A_smem, A_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, by * BLOCK_N: (by + 1) * BLOCK_N])  # noqa: E501
                Tx.sum(B_smem, A_smem)
                Tx.copy(B_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, by], B_smem)


    @Tx.prim_func(tirx=True)
    def partial_reduction_ref_stage2(B: Tx.handle, C: Tx.handle):
        B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
        C_ptr = Tx.match_buffer(C, (M, 1), "float32")

        with Tx.kernel():
            bx = Tx.cta_id([NUM_BLOCK_M])
            tid = Tx.thread_id([1024])
            with Tx.cta():
                B_smem = Tx.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
                C_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                Tx.copy(B_smem, B_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, :])
                Tx.sum(C_smem, B_smem)
                Tx.copy(C_ptr[bx * BLOCK_M: (bx + 1) * BLOCK_M, 0], C_smem)

    TOTAL_SM_CNT = 132

    @Tx.prim_func(tirx=True)
    def partial_reduction_fused(A: Tx.handle, B: Tx.handle, C: Tx.handle, semaphore: Tx.handle, task_types: Tx.handle, task_indices: Tx.handle, tasks_indptr: Tx.handle):  # noqa: E501
        A_ptr = Tx.match_buffer(A, (M, N), "float32")
        B_ptr = Tx.match_buffer(B, (M, NUM_BLOCK_N), "float32")
        C_ptr = Tx.match_buffer(C, (M, 1), "float32")
        sem_ptr = Tx.match_buffer(semaphore, (NUM_BLOCK_M,), "int32")
        task_types_ptr = Tx.match_buffer(task_types, (NUM_BLOCK_M * NUM_BLOCK_N + NUM_BLOCK_M,), "int32")  # noqa: E501
        task_indices_ptr = Tx.match_buffer(task_indices, (NUM_BLOCK_M * NUM_BLOCK_N + NUM_BLOCK_M, 2), "int32")  # noqa: E501
        tasks_indptr_ptr = Tx.match_buffer(tasks_indptr, (TOTAL_SM_CNT + 1,), "int32")
        with Tx.kernel():
            bx = Tx.cta_id([TOTAL_SM_CNT])
            tid = Tx.thread_id([1024])
            sem = Semaphore(NUM_BLOCK_N, sem_ptr, tid)
            with Tx.cta():
                A_smem = Tx.alloc_buffer([BLOCK_M, BLOCK_N], "float32", scope="shared")
                B_smem_1 = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                B_smem_2 = Tx.alloc_buffer([BLOCK_M, NUM_BLOCK_N], "float32", scope="shared")
                C_smem = Tx.alloc_buffer([BLOCK_M, 1], "float32", scope="shared")
                tile_scheduler = StaticTileScheduler(tasks_indptr_ptr, task_types_ptr, task_indices_ptr)  # noqa: E501
                tile_scheduler.init(bx)
                while tile_scheduler.valid():
                    if tile_scheduler.task_type[0] == TaskType.PARTIAL.value:
                        m_idx = Tx.meta_var(tile_scheduler.task_idx[0])
                        n_idx = Tx.meta_var(tile_scheduler.task_idx[1])
                        Tx.copy(A_smem, A_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx * BLOCK_N: (n_idx + 1) * BLOCK_N])  # noqa: E501
                        Tx.sum(B_smem_1, A_smem)
                        Tx.copy(B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, n_idx], B_smem_1)
                        sem.semaphore_notify(m_idx)
                    elif tile_scheduler.task_type[0] == TaskType.REDUCE.value:
                        m_idx = Tx.meta_var(tile_scheduler.task_idx[0])
                        sem.semaphore_wait(m_idx)
                        Tx.copy(B_smem_2, B_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, :])
                        Tx.sum(C_smem, B_smem_2)
                        Tx.copy(C_ptr[m_idx * BLOCK_M: (m_idx + 1) * BLOCK_M, 0], C_smem)
                    tile_scheduler.next_tile()

    # fmt: on

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
    from pathlib import Path

    script_directory = Path(__file__).parent.resolve()
    file_path = script_directory / "partial_reduction_sm_log.txt"
    sm_trace = extract_values(file_path)
    sm_trace = sorted(sm_trace, key=lambda x: x[0])
    indices = np.zeros((TOTAL_SM_CNT + 1,), dtype=np.int32)
    for num in sm_trace:
        indices[num[0] + 1] += 1
    indptr = np.cumsum(indices)
    task_types = np.array(list(map(lambda x: x[1], sm_trace)), dtype=np.int32)
    task_indices = np.array(list(map(lambda x: x[2:], sm_trace)), dtype=np.int32)
    task_types_tvm = tvm.runtime.tensor(task_types, device=DEV)
    task_indices_tvm = tvm.runtime.tensor(task_indices, device=DEV)
    task_indptr_tvm = tvm.runtime.tensor(np.array(indptr, dtype=np.int32), device=DEV)

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
        fused_mod(
            A_tvm, B_tvm, C_tvm_fused, sem_tvm, task_types_tvm, task_indices_tvm, task_indptr_tvm
        )
        ret_fused = C_tvm_fused.numpy()
        tvm.testing.assert_allclose(ret_fused, ret_ref_stage_2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_partial_reduction()
