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


class EPCombineSendTile(Tile):
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
        self.nbytes = dtype_dict[self.out_dtype]

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
        send_tokens, # (local_num_experts, world_size, num_tokens, hidden_dim), out_dtype
        buf_recv, # (total_num_experts, num_tokens, hidden_dim)
        buf_wait, # (total_num_experts,)
        num_recv_tokens, # (local_num_experts, world_size)
        rank,
    ):
        with Tx.cta():
            cta_id = Tx.cta_id([KernelConfig.SM_NUMBER])
            warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER * 32])
            # each CTA is responsible for one expert and one source GPU
            dst_expert = Tx.meta_var(Tx.int32(self.local_num_experts * rank + local_expert_idx))
            num_recv = Tx.meta_var(num_recv_tokens[local_expert_idx, src_rank_idx])
            Tx.nvshmem.putmem_signal_nbi.block(
                dst=buf_recv.access_ptr("w", offset=buf_recv.elem_offset_of([dst_expert, 0, 0])),
                src=send_tokens.access_ptr("r", offset=send_tokens.elem_offset_of([local_expert_idx, src_rank_idx, 0, 0])),  # noqa: E501
                nelems=num_recv * self.hidden_dim * self.nbytes,
                sig_addr=buf_wait.access_ptr("w", offset=buf_wait.elem_offset_of([dst_expert])),
                signal=1,
                sig_op="add",
                pe=src_rank_idx,
            )
    # fmt: on


class EPCombineRecvTile(Tile):
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
        self.nbytes = dtype_dict[self.out_dtype]

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.weights_buf = smem_manager.alloc([self.topk], self.out_dtype)
        self.experts_buf = smem_manager.alloc([self.topk], "int32")

    def _alloc_local(self):
        # alloc local memory
        self.sum = Tx.alloc_local([1], self.out_dtype)

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)
        self._alloc_local()

    # fmt: off
    @Tx.inline
    def run(
        self,

        # tile index
        token_idx, # int, the token_idx that the CTA is responsible for in recv_tokens

        # input
        recv_tokens, # (num_tokens, hidden_dim)
        route_experts, # (num_tokens, topk)
        route_weights, # (num_tokens, topk)
        buf_recv, # (total_num_experts, num_tokens, hidden_dim)
        buf_wait, # (total_num_experts,)
        dst_token_indices, # (num_tokens, topk)
        rank,
    ):
        with Tx.cta():
            cta_id = Tx.cta_id([KernelConfig.SM_NUMBER])
            warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER * 32])

            # each CTA is responsible for one token
            if token_idx < self.num_tokens:
                if tid == 0:
                    for k in Tx.serial(self.topk):
                        self.weights_buf[k] = route_weights[token_idx, k]
                        self.experts_buf[k] = Tx.int32(route_experts[token_idx, k])
                Tx.tvm_storage_sync("shared")

                for k in Tx.serial(self.topk):
                    dst_expert = self.experts_buf[k]
                    Tx.nvshmem.wait_until(
                        ivar=buf_wait.access_ptr("r", offset=buf_wait.elem_offset_of([dst_expert])),
                        cmp="eq",
                        cmp_value=1,
                    )

                for ii in Tx.serial(Tx.ceildiv(self.hidden_dim, self.num_threads_per_cta)):
                    idx = Tx.meta_var(ii * self.num_threads_per_cta + tid)
                    if idx < self.hidden_dim:
                        self.sum[0] = 0
                        for kk in Tx.serial(self.topk):
                            expert = self.experts_buf[kk]
                            weight = self.weights_buf[kk]
                            dst_token_idx = dst_token_indices[token_idx, kk]
                            self.sum[0] += weight * buf_recv[expert, dst_token_idx, idx]
                        recv_tokens[token_idx, idx] = self.sum[0]
    # fmt: on
