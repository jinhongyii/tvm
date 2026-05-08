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

from tvm.megakernel.utils.base import SmemManager, Tile
from tvm.megakernel.utils.config import F16_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv, next_power_of_two
from tvm.script import tirx as Tx


class MOEAlignTile(Tile):
    def __init__(self, num_experts, numel, block_size, pad_sorted_token_ids=False):
        super().__init__()
        self.num_experts = num_experts
        self.scan_size = next_power_of_two(num_experts)
        self.numel = numel  # numel = num_tokens * num_experts
        self.block_size = block_size
        self.pad_sorted_token_ids = pad_sorted_token_ids

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.shared_counts = smem_manager.alloc([self.num_experts], "int32")
        self.prefix = smem_manager.alloc([self.num_experts + 1], "int32")
        self.scan_buf = smem_manager.alloc([self.scan_size], "int32")
        self.warp_sums = smem_manager.alloc(
            [KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], "int32"
        )
        self.s_total_tokens_post_pad = smem_manager.alloc([1], "int32")

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @Tx.inline
    def warp_exclusive_scan(self, v, output, mask=0xFFFFFFFF):
        # offset = Tx.alloc_scalar("int32")
        # original = Tx.alloc_scalar("int32")
        # with Tx.warp():
        #     lane_id = Tx.lane_id(32)
        #     offset = 1
        #     original = v
        #     while offset < 32:
        #         n = Tx.tvm_warp_shuffle_up(mask, v, offset)
        #         if lane_id >= offset:
        #             v += n
        #         offset = offset << 1
        #     output = v - original
        #     v = original
        output[0] = Tx.cuda.func_call(
            "warp_exclusive_scan",
            v,
            mask,
            source_code="""
            __device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
            int original = v;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int n = __shfl_up_sync(mask, v, offset);
                if ((threadIdx.x & (32 - 1)) >= offset) v += n;
            }
            return v - original;
            }
                                  """,
            return_type="int32",
        )

    @Tx.inline
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        cumsum_buffer,
        num_valid_tokens=None,
    ):
        idx = Tx.alloc_local([1], "int32")
        pre = Tx.alloc_local([1], "int32")
        sum_val = Tx.alloc_local([1], "int32")
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            lane_id = Tx.lane_id([32])
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            self.smem_manager.wait_all("cta")
            if tid < self.num_experts:
                self.shared_counts[tid] = 0
            Tx.tvm_storage_sync("shared")
            idx[0] = tid
            while idx[0] < self.numel:
                expert_id: Tx.let = topk_ids[
                    idx[0]
                ]  # TODO: the reference expert use topk_ids[idx[0]] + 1. Is it correct?
                Tx.cuda.atomic_add(Tx.address_of(self.shared_counts[expert_id]), 1)
                idx[0] += KernelConfig.NUM_THREADS
            Tx.tvm_storage_sync("shared")
            if tid < self.num_experts:
                padded_count: Tx.let = (
                    ceildiv(self.shared_counts[tid], self.block_size) * self.block_size
                )
                self.scan_buf[tid] = padded_count
            elif tid < self.scan_size:
                self.scan_buf[tid] = 0
            Tx.tvm_storage_sync("shared")
            v: Tx.let = Tx.if_then_else(tid < self.num_experts, self.scan_buf[tid], 0)
            self.warp_exclusive_scan(v, pre)
            if lane_id == 31:
                self.warp_sums[warp_id] = pre[0] + v
            Tx.tvm_storage_sync("shared")
            num_warps_for_scan = Tx.meta_var(ceildiv(self.scan_size, 32))
            if warp_id == 0:
                val: Tx.let = Tx.if_then_else(
                    lane_id < num_warps_for_scan, self.warp_sums[lane_id], 0
                )
                self.warp_exclusive_scan(val, sum_val)
                if lane_id == num_warps_for_scan - 1:
                    self.prefix[self.num_experts] = sum_val[0] + val
                    self.s_total_tokens_post_pad[0] = sum_val[0] + val
                    total_tokens_post_pad[0] = self.s_total_tokens_post_pad[0]
                self.warp_sums[lane_id] = sum_val[0]
            Tx.tvm_storage_sync("shared")
            if tid < self.scan_size:
                self.scan_buf[tid] = pre[0] + self.warp_sums[warp_id]
            if tid < self.num_experts:
                self.prefix[tid] = self.scan_buf[tid]
            if tid <= self.num_experts:
                cumsum_buffer[tid] = self.prefix[tid]
            Tx.tvm_storage_sync("shared")
            num_blocks: Tx.let = self.s_total_tokens_post_pad[0] // self.block_size
            idx[0] = tid
            while idx[0] < num_blocks:
                block_start: Tx.let = idx[0] * self.block_size
                left = Tx.alloc_local([1], "int32")
                right = Tx.alloc_local([1], "int32")
                with Tx.thread():
                    left[0] = 0
                    right[0] = self.num_experts
                    while left[0] < right[0]:
                        mid: Tx.let = (left[0] + right[0]) // 2
                        if self.prefix[mid] <= block_start:
                            left[0] = mid + 1
                        else:
                            right[0] = mid
                expert_ids[idx[0]] = left[0] - 1  # TODO: the reference expert use left - 2
                if num_valid_tokens is not None:
                    if idx[0] < cumsum_buffer[left[0]] // self.block_size - 1:
                        num_valid_tokens[idx[0]] = self.block_size
                    else:
                        num_valid_tokens[idx[0]] = (
                            self.shared_counts[left[0] - 1] - 1
                        ) % self.block_size + 1

                idx[0] += KernelConfig.NUM_THREADS
            if self.pad_sorted_token_ids:
                VEC_SIZE: Tx.let = 4
                idx[0] = tid * VEC_SIZE
                out_ptr = sorted_token_ids.view(-1)
                fill_vec = Tx.alloc_buffer([4], "int32", scope="local")
                for vec in Tx.vectorized(VEC_SIZE):
                    fill_vec[vec] = self.numel
                while idx[0] < self.s_total_tokens_post_pad[0]:
                    for vec in Tx.vectorized(VEC_SIZE):
                        out_ptr[idx[0] + vec] = fill_vec[vec]
                    idx[0] += VEC_SIZE * KernelConfig.NUM_THREADS
            self.smem_manager.arrive_all("cta")
            self.smem_manager.advance()


class CountAndSortExpertTokens(Tile):
    VEC_SIZE = 16 // F16_BYTES
    PIPE_DEPTH = 8

    def __init__(self, numel, hidden_size, topk):
        super().__init__()
        self.numel = numel  # numel = num_tokens * num_experts
        self.hidden_size = hidden_size
        assert self.hidden_size <= KernelConfig.NUM_THREADS * self.VEC_SIZE
        self.topk = topk

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.s_rank_post_pad = smem_manager.alloc([KernelConfig.NUM_THREADS], "int64")
        self.fetched_data = smem_manager.alloc(
            [self.PIPE_DEPTH, KernelConfig.NUM_THREADS, self.VEC_SIZE], "float16"
        )

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    # fmt: off
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, topk_ids, sorted_token_ids, cumsum_buffer, data, reordered_data):  # noqa: E501
        idx = Tx.alloc_local([1], "int32")
        cnt = Tx.alloc_local([1], "int32")
        col_idx = Tx.alloc_local([1], "int32")
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            process_token_idx: Tx.let = m_idx + tid * KernelConfig.SM_NUMBER
            self.smem_manager.wait_all("cta")
            if process_token_idx < self.numel:
                expert_id: Tx.let = topk_ids[process_token_idx]
                rank_post_pad: Tx.let = Tx.cuda.atomic_add(Tx.address_of(cumsum_buffer[expert_id]), 1)  # noqa: E501
                sorted_token_ids[rank_post_pad] = process_token_idx
                self.s_rank_post_pad[tid] = rank_post_pad
            idx[0] = m_idx
            Tx.tvm_storage_sync("shared")
            col_idx[0] = tid * self.VEC_SIZE
            for i in Tx.unroll(self.PIPE_DEPTH - 1):
                with Tx.thread():
                    if idx[0] < self.numel and col_idx[0] < self.hidden_size:
                        Tx.copy_async(self.fetched_data[i, tid, :], data[idx[0] // self.topk, col_idx[0]:col_idx[0] + self.VEC_SIZE], dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)  # noqa: E501
                    Tx.ptx.cp_async.commit_group()
                idx[0] += KernelConfig.SM_NUMBER
            cnt[0] = 0
            while idx[0] < self.numel + (self.PIPE_DEPTH - 1) * KernelConfig.SM_NUMBER:
                with Tx.thread():
                    if idx[0] < self.numel and col_idx[0] < self.hidden_size:
                        cp_pipe_idx = Tx.meta_var((idx[0] // KernelConfig.SM_NUMBER) % self.PIPE_DEPTH)  # noqa: E501
                        Tx.copy_async(self.fetched_data[cp_pipe_idx, tid, :], data[idx[0] // self.topk, col_idx[0]:col_idx[0] + self.VEC_SIZE], dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)  # noqa: E501
                    Tx.ptx.cp_async.commit_group()
                    Tx.ptx.cp_async.wait_group(self.PIPE_DEPTH - 1)
                    rank_post_pad: Tx.let = self.s_rank_post_pad[cnt[0]]
                    pipe_idx = Tx.meta_var(cnt[0] % self.PIPE_DEPTH)
                    if col_idx[0] < self.hidden_size:
                        for vec in Tx.vectorized(self.VEC_SIZE):
                            reordered_data[rank_post_pad, col_idx[0] + vec] = self.fetched_data[pipe_idx, tid, vec]  # noqa: E501
                    idx[0] += KernelConfig.SM_NUMBER
                    cnt[0] += 1
            self.smem_manager.arrive_all("cta")
            self.smem_manager.advance()
    # fmt: on


class MOETokenDispatchTile(Tile):
    def __init__(self, num_experts, numel, block_size, pad_sorted_token_ids=True):
        super().__init__()
        self.num_experts = num_experts
        self.scan_size = next_power_of_two(num_experts)
        self.numel = numel
        self.block_size = block_size
        self.pad_sorted_token_ids = pad_sorted_token_ids

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.shared_counts = smem_manager.alloc([self.num_experts], "int32")
        self.prefix = smem_manager.alloc([self.num_experts + 1], "int32")
        self.scan_buf = smem_manager.alloc([self.scan_size], "int32")
        self.warp_sums = smem_manager.alloc(
            [KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], "int32"
        )
        self.s_total_tokens_post_pad = smem_manager.alloc([1], "int32")

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @Tx.inline
    def warp_exclusive_scan(self, v, output, mask=0xFFFFFFFF):
        output[0] = Tx.cuda.func_call(
            "warp_exclusive_scan",
            v,
            mask,
            source_code="""
            __device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
            int original = v;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int n = __shfl_up_sync(mask, v, offset);
                if ((threadIdx.x & (32 - 1)) >= offset) v += n;
            }
            return v - original;
            }
                                  """,
            return_type="int32",
        )

    @Tx.inline
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        cumsum_buffer,
        num_valid_tokens=None,
    ):
        idx = Tx.alloc_local([1], "int32")
        pre = Tx.alloc_local([1], "int32")
        sum_val = Tx.alloc_local([1], "int32")
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            lane_id = Tx.lane_id([32])
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            self.smem_manager.wait_all("cta")
            if tid < self.num_experts:
                self.shared_counts[tid] = 0
            Tx.tvm_storage_sync("shared")
            idx[0] = tid
            while idx[0] < self.numel:
                expert_id = topk_ids[idx[0]]
                Tx.cuda.atomic_add(Tx.address_of(self.shared_counts[expert_id]), 1)
                idx[0] += KernelConfig.NUM_THREADS
            Tx.tvm_storage_sync("shared")
            if tid < self.num_experts:
                padded_count = ceildiv(self.shared_counts[tid], self.block_size) * self.block_size
                self.scan_buf[tid] = padded_count
            elif tid < self.scan_size:
                self.scan_buf[tid] = 0
            Tx.tvm_storage_sync("shared")
            v = Tx.if_then_else(tid < self.num_experts, self.scan_buf[tid], 0)
            self.warp_exclusive_scan(v, pre)
            if lane_id == 31:
                self.warp_sums[warp_id] = pre[0] + v
            Tx.tvm_storage_sync("shared")
            num_warps_for_scan = Tx.meta_var(ceildiv(self.scan_size, 32))
            if warp_id == 0:
                val = Tx.if_then_else(lane_id < num_warps_for_scan, self.warp_sums[lane_id], 0)
                self.warp_exclusive_scan(val, sum_val)
                if lane_id == num_warps_for_scan - 1:
                    self.prefix[self.num_experts] = sum_val[0] + val
                    self.s_total_tokens_post_pad[0] = sum_val[0] + val
                    total_tokens_post_pad[0] = self.s_total_tokens_post_pad[0]
                self.warp_sums[lane_id] = sum_val[0]
            Tx.tvm_storage_sync("shared")
            if tid < self.scan_size:
                self.scan_buf[tid] = pre[0] + self.warp_sums[warp_id]
            if tid < self.num_experts:
                self.prefix[tid] = self.scan_buf[tid]
            if tid <= self.num_experts:
                cumsum_buffer[tid] = self.prefix[tid]
            Tx.tvm_storage_sync("shared")
            num_blocks = self.s_total_tokens_post_pad[0] // self.block_size
            idx[0] = tid
            while idx[0] < num_blocks:
                block_start = idx[0] * self.block_size
                left = Tx.alloc_local([1], "int32")
                right = Tx.alloc_local([1], "int32")
                with Tx.thread():
                    left[0] = 0
                    right[0] = self.num_experts
                    while left[0] < right[0]:
                        mid = (left[0] + right[0]) // 2
                        if self.prefix[mid] <= block_start:
                            left[0] = mid + 1
                        else:
                            right[0] = mid
                expert_ids[idx[0]] = left[0] - 1
                if num_valid_tokens is not None:
                    if idx[0] < cumsum_buffer[left[0]] // self.block_size - 1:
                        num_valid_tokens[idx[0]] = self.block_size
                    else:
                        num_valid_tokens[idx[0]] = (
                            self.shared_counts[left[0] - 1] - 1
                        ) % self.block_size + 1
                idx[0] += KernelConfig.NUM_THREADS
            if self.pad_sorted_token_ids:
                VEC_SIZE = 4
                idx[0] = tid * VEC_SIZE
                out_ptr = sorted_token_ids.view(-1)
                fill_vec = Tx.alloc_buffer([4], "int32", scope="local")
                for vec in Tx.vectorized(VEC_SIZE):
                    fill_vec[vec] = self.numel
                while idx[0] < self.s_total_tokens_post_pad[0]:
                    for vec in Tx.vectorized(VEC_SIZE):
                        out_ptr[idx[0] + vec] = fill_vec[vec]
                    idx[0] += VEC_SIZE * KernelConfig.NUM_THREADS
            Tx.tvm_storage_sync("shared")
            idx[0] = tid
            while idx[0] < self.numel:
                expert_id = topk_ids[idx[0]]
                rank_post_pad = Tx.cuda.atomic_add(Tx.address_of(cumsum_buffer[expert_id]), 1)
                sorted_token_ids[rank_post_pad] = idx[0]
                idx[0] += KernelConfig.NUM_THREADS
            self.smem_manager.arrive_all("cta")
            self.smem_manager.advance()
