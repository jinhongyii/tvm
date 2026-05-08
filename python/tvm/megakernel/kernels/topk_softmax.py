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

import tvm
from tvm.megakernel.utils.base import SmemManager, Tile
from tvm.megakernel.utils.config import KernelConfig
from tvm.megakernel.utils.utils import find_power_of_two, is_power_of_two
from tvm.script import tirx as Tx


class TopkSoftmaxTile(Tile):
    WARPS_PER_TB = 8
    MAX_BYTES_PER_LDG = 16
    TPB = 256
    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx

    def __init__(
        self, num_experts: Tx.int32, num_tokens: Tx.int32, topk: Tx.int32, dtype="float32"
    ):
        super().__init__()
        Tx.Assert(
            is_power_of_two(num_experts),
            f"number of experts is {num_experts}, which is not a power of 2",
        )
        Tx.Assert(
            tvm.tirx.all(1 <= num_experts, num_experts <= 256),
            f"number of experts is {num_experts}, which is not within [1, 256]",
        )
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.topk = topk
        self.dtype = dtype
        if dtype in ["float16", "bfloat16"]:
            self.n_bytes = 2
        elif dtype in ["float32"]:
            self.n_bytes = 4
        else:
            raise ValueError(f"{dtype} is not supported in topk_softmax")
        Tx.Assert(
            self.n_bytes * self.num_experts >= self.MAX_BYTES_PER_LDG,
            f"n_bytes * num_experts is {self.n_bytes * self.num_experts}, which is less than 16",
        )
        # self.BYTES_PER_LDG = Tx.min(self.MAX_BYTES_PER_LDG, self.n_bytes * self.num_experts)
        self.BYTES_PER_LDG = self.MAX_BYTES_PER_LDG
        Tx.Assert(
            tvm.tirx.all(is_power_of_two(self.BYTES_PER_LDG), self.BYTES_PER_LDG <= 16),
            f"BYTES_PER_LDG is {self.BYTES_PER_LDG}, which is not a power of 2 or is larger than 16",  # noqa: E501
        )
        self.ELTS_PER_LDG = self.BYTES_PER_LDG // self.n_bytes
        Tx.Assert(
            tvm.tirx.any(
                num_experts // (self.ELTS_PER_LDG * 32) == 0,
                num_experts % (self.ELTS_PER_LDG * 32) == 0,
            ),
            f"num_experts is {num_experts}, which is not multiple or less than {(self.ELTS_PER_LDG * 32)}",  # noqa: E501
        )
        self.ELTS_PER_ROW = num_experts
        Tx.Assert(
            num_experts // (self.ELTS_PER_LDG * 32) <= 1,
            f"num_experts is {num_experts}, which is larger than {self.ELTS_PER_LDG * 32}",
        )
        # self.VECs_PER_THREAD = Tx.max(1, num_experts // (self.ELTS_PER_LDG * 32))
        self.VECs_PER_THREAD = 1
        self.VPT = self.VECs_PER_THREAD * self.ELTS_PER_LDG
        Tx.Assert(is_power_of_two(self.VPT), f"VPT is {self.VPT}, which is not a power of 2")
        self.LDG_PER_THREAD = self.VPT // self.ELTS_PER_LDG
        Tx.Assert(
            self.VPT % self.ELTS_PER_LDG == 0,
            f"VPT is {self.VPT}, which is not a multiple of {self.ELTS_PER_LDG}",
        )
        self.THREADS_PER_ROW = num_experts // self.VPT
        Tx.Assert(
            32 % self.THREADS_PER_ROW == 0,
            f"THREADS_PER_ROW is {self.THREADS_PER_ROW}, which is not divisible by 32",
        )
        Tx.Assert(
            is_power_of_two(self.THREADS_PER_ROW),
            f"THREADS_PER_ROW is {self.THREADS_PER_ROW}, which is not a power of 2",
        )
        Tx.Assert(
            self.THREADS_PER_ROW <= 32,
            f"THREADS_PER_ROW is {self.THREADS_PER_ROW}, which is not less or equal to 32",
        )
        self.COLS_PER_GROUP_LDG = self.ELTS_PER_LDG * self.THREADS_PER_ROW
        self.ELTS_PER_WARP = self.VPT * 32
        self.ROWS_PER_WARP = self.ELTS_PER_WARP // self.ELTS_PER_ROW
        self.ROWS_PER_CTA = self.WARPS_PER_TB * self.ROWS_PER_WARP
        Tx.Assert(
            self.ELTS_PER_WARP % self.ELTS_PER_ROW == 0,
            f"ELTS_PER_WARP is {self.ELTS_PER_WARP}, which is not a multiple of {self.ELTS_PER_ROW}",  # noqa: E501
        )
        self.num_warps = (num_tokens + self.ROWS_PER_WARP - 1) // self.ROWS_PER_WARP
        self.num_blocks = (self.num_warps + self.WARPS_PER_TB - 1) // self.WARPS_PER_TB
        self.MAX_OCCUPANCY = 8
        self.PERSISTENT_SM_NUMBER = KernelConfig.SM_NUMBER * 1

    def _alloc_local(self):
        # alloc local memory
        self.row_chunk_temp = Tx.alloc_local([self.VPT], self.dtype)
        self.row_chunk = Tx.alloc_local([self.VPT], "float32")
        self.thread_max = Tx.alloc_local([1], self.dtype)
        self.row_sum = Tx.alloc_local([1], "float32")
        self.reciprocal_row_sum = Tx.alloc_local([1], "float32")
        self.row_sum_for_renormalize = Tx.alloc_local([1], "float32")
        self.col = Tx.alloc_local([1], "int32")
        self.max_val = Tx.alloc_local([1], "float32")
        self.expert = Tx.alloc_local([1], "int32")
        self.token_idx = Tx.alloc_local([1], "int32")

    @Tx.inline
    def init(self, smem_manager: SmemManager = None):
        self._alloc_local()

    # fmt: off
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, gating_output, topk_weights, topk_indices, renormalize=False):  # noqa: E501
        with Tx.cta():
            warp_id_in_cta = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([self.bdx])

            with Tx.thread():
                self.token_idx[0] = m_idx * self.ROWS_PER_CTA

                while self.token_idx[0] < self.num_tokens:
                    cta_base_row = Tx.meta_var(self.token_idx[0])
                    warp_base_row = Tx.meta_var(cta_base_row + warp_id_in_cta * self.ROWS_PER_WARP)
                    thread_row_in_warp = Tx.meta_var(lane_id // self.THREADS_PER_ROW)
                    thread_row = Tx.meta_var(warp_base_row + thread_row_in_warp)

                    thread_group_idx = Tx.meta_var(lane_id % self.THREADS_PER_ROW)
                    first_elt_read_by_thread = Tx.meta_var(thread_group_idx * self.ELTS_PER_LDG)
                    thread_read_ptr_offset = Tx.meta_var(thread_row * self.ELTS_PER_ROW + first_elt_read_by_thread)  # noqa: E501

                    if thread_row < self.num_tokens:
                        # copy global to reg
                        for ii in Tx.unroll(self.LDG_PER_THREAD):
                            thread_read_ptr_offset1 = Tx.meta_var(thread_read_ptr_offset + ii * self.THREADS_PER_ROW * self.ELTS_PER_LDG)  # noqa: E501
                            for vec in Tx.vectorized(self.ELTS_PER_LDG):
                                thread_read_ptr_offset2 = Tx.meta_var(thread_read_ptr_offset1 + vec)
                                idx0 = Tx.meta_var(thread_read_ptr_offset2 // self.num_experts)
                                idx1 = Tx.meta_var(thread_read_ptr_offset2 % self.num_experts)
                                self.row_chunk_temp[ii * self.ELTS_PER_LDG + vec] = gating_output[idx0, idx1]  # noqa: E501
                    else:
                        for ii in Tx.unroll(self.VPT):
                            self.row_chunk_temp[ii] = 0.0

                    # max reduce within thread
                    self.thread_max[0] = self.row_chunk_temp[0]
                    for ii in Tx.unroll(1, self.VPT):
                        self.thread_max[0] = Tx.max(self.thread_max[0], self.row_chunk_temp[ii])

                    # max reduce within thread group
                    # now, thread_max in all threads have the max within the row
                    for kr in Tx.unroll(find_power_of_two(self.THREADS_PER_ROW // 2) + 1):
                        self.thread_max[0] = Tx.max(self.thread_max[0], Tx.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.thread_max[0], (self.THREADS_PER_ROW // 2) >> kr, self.THREADS_PER_ROW, 32  # noqa: E501
                        ))

                    # take exp and compute local sum
                    self.row_sum[0] = 0
                    for ii in Tx.unroll(self.VPT):
                        self.row_chunk[ii] = Tx.exp(Tx.Cast("float32", self.row_chunk_temp[ii]) - Tx.Cast("float32", self.thread_max[0]))  # noqa: E501
                        self.row_sum[0] += self.row_chunk[ii]

                    # sum reduce within thread group
                    # now, all threads have the sum within the row
                    for kr in Tx.unroll(find_power_of_two(self.THREADS_PER_ROW // 2) + 1):
                        self.row_sum[0] = self.row_sum[0] + Tx.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.row_sum[0], (self.THREADS_PER_ROW // 2) >> kr, self.THREADS_PER_ROW, 32  # noqa: E501
                        )

                    # scale the rows
                    self.reciprocal_row_sum[0] = 1.0 / self.row_sum[0]
                    for ii in Tx.unroll(self.VPT):
                        self.row_chunk[ii] = self.row_chunk[ii] * self.reciprocal_row_sum[0]

                    # now, start to find topk elements in each row
                    start_col = Tx.meta_var(first_elt_read_by_thread)
                    self.row_sum_for_renormalize[0] = 0

                    for k_idx in Tx.serial(self.topk):
                        # first step, each thread does local argmax
                        self.max_val[0] = self.row_chunk[0]
                        self.expert[0] = start_col
                        self.col[0] = start_col
                        for ldg in Tx.unroll(self.LDG_PER_THREAD):
                            for ii in Tx.unroll(self.ELTS_PER_LDG):
                                val = Tx.meta_var(self.row_chunk[ldg * self.ELTS_PER_LDG + ii])
                                if val > self.max_val[0]:
                                    self.max_val[0] = val
                                    self.expert[0] = self.col[0] + ii
                            self.col[0] = self.col[0] + self.COLS_PER_GROUP_LDG

                        # second step, perform argmax reduce among threads
                        for ki in Tx.unroll(find_power_of_two(self.THREADS_PER_ROW // 2) + 1):
                            other_max = Tx.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.max_val[0], (self.THREADS_PER_ROW // 2) >> ki, self.THREADS_PER_ROW, 32  # noqa: E501
                            )
                            other_expert = Tx.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.expert[0], (self.THREADS_PER_ROW // 2) >> ki, self.THREADS_PER_ROW, 32  # noqa: E501
                            )
                            if (other_max > self.max_val[0]) or ((other_max == self.max_val[0]) and (other_expert < self.expert[0])):  # noqa: E501
                                self.max_val[0] = other_max
                                self.expert[0] = other_expert

                        # write max to global memory
                        if thread_group_idx == 0 and thread_row < self.num_tokens:
                            start_expert = Tx.meta_var(0)
                            end_expert = Tx.meta_var(self.num_experts)
                            should_process_row = Tx.meta_var(Tx.bool((self.expert[0] >= start_expert) and (self.expert[0] < end_expert)))  # noqa: E501
                            idx = Tx.meta_var(self.topk * thread_row + k_idx)
                            idx0 = Tx.meta_var(idx // self.topk)
                            idx1 = Tx.meta_var(idx % self.topk)
                            topk_weights[idx0, idx1] = self.max_val[0]
                            topk_indices[idx0, idx1] = Tx.if_then_else(should_process_row, self.expert[0] - start_expert, self.num_experts)  # noqa: E501
                            self.row_sum_for_renormalize[0] = self.row_sum_for_renormalize[0] + self.max_val[0]  # noqa: E501

                        # clear value in thread
                        if k_idx + 1 < self.topk:
                            ldg_group_for_expert = Tx.meta_var(self.expert[0] // self.COLS_PER_GROUP_LDG)  # noqa: E501
                            thread_to_clear_in_group = Tx.meta_var((self.expert[0] // self.ELTS_PER_LDG) % self.THREADS_PER_ROW)  # noqa: E501
                            if thread_group_idx == thread_to_clear_in_group:
                                offset_for_expert = Tx.meta_var(self.expert[0] % self.ELTS_PER_LDG)
                                idx = Tx.meta_var(ldg_group_for_expert * self.ELTS_PER_LDG + offset_for_expert)  # noqa: E501
                                self.row_chunk[idx] = -10000.0

                    # handle renormalize of top k weights
                    if renormalize and thread_group_idx == 0 and thread_row < self.num_tokens:
                        row_sum_for_renormalize_inv = Tx.meta_var(1.0 / self.row_sum_for_renormalize[0])  # noqa: E501
                        for k_idx in Tx.unroll(self.topk):
                            idx = Tx.meta_var(self.topk * thread_row + k_idx)
                            idx0 = Tx.meta_var(idx // self.topk)
                            idx1 = Tx.meta_var(idx % self.topk)
                            topk_weights[idx0, idx1] = topk_weights[idx0, idx1] * row_sum_for_renormalize_inv  # noqa: E501

                    self.token_idx[0] += self.PERSISTENT_SM_NUMBER * self.ROWS_PER_CTA
    # fmt: on
