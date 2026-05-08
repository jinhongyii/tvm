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
from tvm.megakernel.utils.config import KernelConfig
from tvm.script import tirx as Tx


class EPDispatchPrecomputeTile(Tile):
    """would need 16 CTAs"""

    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx
    num_blocks = KernelConfig.SM_NUMBER
    num_warps_per_cta = KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
    num_threads_per_cta = num_warps_per_cta * bdx

    def __init__(
        self,
        num_tokens: Tx.int32,
        total_num_experts: Tx.int32,
        topk: Tx.int32,
        hidden_dim: Tx.int32,
        in_dtype,
        out_dtype,
        world_size: Tx.int32,
        n_dp_groups: Tx.int32,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.total_num_experts = total_num_experts
        self.local_num_experts = total_num_experts // world_size
        self.topk = topk
        self.hidden_dim = hidden_dim
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.world_size = world_size
        self.n_dp_groups = n_dp_groups

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.smem_buf = smem_manager.alloc([self.num_tokens * self.topk], "uint32")

    def _alloc_local(self):
        # alloc local memory
        self.count = Tx.alloc_local([1], "uint32")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)
        self._alloc_local()

    # fmt: off
    @Tx.inline
    def run(
        self,

        # tile index
        dst_expert_st, # the CTA is responsible for [dst_expert_st : dst_expert_st + 8] experts

        # input
        route_experts, # (num_tokens, topk)
        target_wait, # (local_num_experts, world_size)
        rank,
    ):
        with Tx.cta():
            cta_id = Tx.cta_id([KernelConfig.SM_NUMBER])
            warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER * 32])

            # TODO: tune number of CTAs for precompute based on profiling results
            # for now, each warp in a CTA is responsible for one expert; total CTA: 16
            with Tx.thread():
                self.smem_manager.wait_all("cta")

                dst_expert = Tx.meta_var(Tx.uint32(dst_expert_st + warp_id))
                if dst_expert < self.total_num_experts:
                    # load from global to shared
                    vec_size = 4
                    row_idx = Tx.meta_var(Tx.int32(tid // 2))
                    col_idx = Tx.meta_var(Tx.int32((tid % 2) * 4))
                    if row_idx < self.num_tokens:
                        for vec in Tx.vectorized(vec_size):
                            self.smem_buf[tid * vec_size + vec] = route_experts[row_idx, col_idx + vec]  # noqa: E501
                    Tx.tvm_storage_sync("shared")

                    # thread count
                    self.count[0] = 0
                    for k in Tx.serial(Tx.ceildiv(self.num_tokens * self.topk, 32)):
                        idx = Tx.meta_var(k * 32 + lane_id)
                        if idx < self.num_tokens * self.topk:
                            if self.smem_buf[idx] == dst_expert:
                                self.count[0] += 1

                    # warp scan
                    for kr in Tx.unroll(5):
                        self.count[0] += Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, self.count[0], 16 >> kr, 32, 32)  # noqa: E501

                    # a single thread in warp signal the count
                    dst_local_expert = Tx.meta_var(Tx.int32(dst_expert % self.local_num_experts))
                    dst_rank = Tx.meta_var(Tx.int32(dst_expert // self.local_num_experts))
                    if lane_id == 0:
                        Tx.nvshmem.signal_op(
                            sig_addr=target_wait.access_ptr("w", offset=target_wait.elem_offset_of([dst_local_expert, rank])),  # noqa: E501
                            signal=self.count[0] + 1,
                            sig_op="set",
                            pe=dst_rank,
                        )

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on


class EPDispatchSendTile(Tile):
    """would need 1-128 (num_tokens) CTAs"""

    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx
    num_blocks = KernelConfig.SM_NUMBER
    num_warps_per_cta = KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
    num_threads_per_cta = num_warps_per_cta * bdx

    def __init__(
        self,
        num_tokens: Tx.int32,
        total_num_experts: Tx.int32,
        topk: Tx.int32,
        hidden_dim: Tx.int32,
        in_dtype,
        out_dtype,
        world_size: Tx.int32,
        n_dp_groups: Tx.int32,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.total_num_experts = total_num_experts
        self.local_num_experts = total_num_experts // world_size
        self.topk = topk
        self.hidden_dim = hidden_dim
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.world_size = world_size
        self.n_dp_groups = n_dp_groups
        dtype_dict = {"bfloat16": 2, "float16": 2, "float8_e4m3fn": 1}
        self.nbytes = dtype_dict[self.in_dtype]
        self.vec_len = 16 // self.nbytes

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager

    def _alloc_local(self):
        # alloc local memory
        self.dst_index = Tx.alloc_local([1], "int32")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_local()

    # fmt: off
    @Tx.inline
    def run(
        self,

        # tile index
        src_token_idx, # int, the token_idx that the CTA is responsible for in send_tokens

        # input
        send_tokens, # (num_tokens, hidden_dim), in_dtype
        route_experts, # (num_tokens, topk)
        recv_tokens, # (local_num_experts, world_size, num_tokens, hidden_dim)
        actual_wait, # (local_num_experts, world_size)
        dst_token_indices, # (num_tokens, topk)
        dst_token_idx, # (total_num_experts,), the token_idx that the CTA is sending to in recv_tokens  # noqa: E501
        rank,
    ):
        with Tx.cta():
            cta_id = Tx.cta_id([KernelConfig.SM_NUMBER])
            warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER * 32])
            # each warp sends to one dest expert
            if warp_id < self.topk:
                dst_expert = route_experts[src_token_idx, warp_id]
                dst_local_expert = Tx.meta_var(Tx.int32(dst_expert % self.local_num_experts))
                dst_rank = Tx.meta_var(Tx.int32(dst_expert // self.local_num_experts))
                if lane_id == 0:
                    self.dst_index[0] = Tx.cuda.atomic_add(dst_token_idx.access_ptr("rw", offset=dst_expert), 1)  # noqa: E501
                    dst_token_indices[src_token_idx, warp_id] = self.dst_index[0]
                self.dst_index[0] = Tx.tvm_warp_shuffle(0xffffffff, self.dst_index[0], 0, 32, 32)
                Tx.nvshmem.putmem_signal_nbi.warp(
                    dst=recv_tokens.access_ptr("w", offset=recv_tokens.elem_offset_of([dst_local_expert, rank, self.dst_index[0], 0])),  # noqa: E501
                    src=send_tokens.access_ptr("r", offset=send_tokens.elem_offset_of([src_token_idx, 0])),  # noqa: E501
                    nelems=self.hidden_dim * self.nbytes,
                    sig_addr=actual_wait.access_ptr("w", offset=actual_wait.elem_offset_of([dst_local_expert, rank])),  # noqa: E501
                    signal=1,
                    sig_op="add",
                    pe=dst_rank,
                )
    # fmt: on


class EPDispatchRecvTile(Tile):
    """would need 128 CTAs"""

    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx
    num_blocks = KernelConfig.SM_NUMBER
    num_warps_per_cta = KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER
    num_threads_per_cta = num_warps_per_cta * bdx

    def __init__(
        self,
        num_tokens: Tx.int32,
        total_num_experts: Tx.int32,
        topk: Tx.int32,
        hidden_dim: Tx.int32,
        in_dtype,
        out_dtype,
        world_size: Tx.int32,
        n_dp_groups: Tx.int32,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.total_num_experts = total_num_experts
        self.local_num_experts = total_num_experts // world_size
        self.topk = topk
        self.hidden_dim = hidden_dim
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.world_size = world_size
        self.n_dp_groups = n_dp_groups
        dtype_dict = {"bfloat16": 2, "float16": 2, "float8_e4m3fn": 1}
        self.nbytes = dtype_dict[self.in_dtype]
        self.vec_len = 16 // self.nbytes

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager

    def _alloc_local(self):
        # alloc local memory
        pass

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_local()

    # fmt: off
    @Tx.inline
    def run(
        self,

        # tile index
        local_expert_idx, # the local expert index that the CTA is responsible for
        src_rank_idx, # the source rank index that the CTA is responsible for

        # input
        num_recv_tokens, # (local_num_experts, world_size)
        target_wait, # (local_num_experts, world_size)
        actual_wait, # (local_num_experts, world_size)
        rank,
    ):
        with Tx.cta():
            cta_id = Tx.cta_id([KernelConfig.SM_NUMBER])
            warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER * 32])

            # TODO: can adjust the granularity of tile when fusing with GEMM
            # for now, each CTA is responsible for one expert and one source GPU
            Tx.nvshmem.wait_until(
                ivar=target_wait.access_ptr("r", offset=target_wait.elem_offset_of([local_expert_idx, src_rank_idx])),  # noqa: E501
                cmp="ne",
                cmp_value=0,
            )
            recv_num = target_wait[local_expert_idx, src_rank_idx] - 1
            Tx.nvshmem.wait_until(
                ivar=actual_wait.access_ptr("r", offset=actual_wait.elem_offset_of([local_expert_idx, src_rank_idx])),  # noqa: E501
                cmp="eq",
                cmp_value=recv_num,
            )
            if tid == 0:
                num_recv_tokens[local_expert_idx, src_rank_idx] = recv_num
    # fmt: on
