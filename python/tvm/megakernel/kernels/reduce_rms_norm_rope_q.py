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

from tvm.megakernel.utils.base import Tile
from tvm.megakernel.utils.config import F16_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv, find_power_of_two, rsqrt
from tvm.script import tirx as Tx


# TODO: pipeline
class SplitKReduceRMSnormRopeQTile(Tile):
    # weight_tvm: [num_heads]
    # qk_tvm: [batch_size, num_heads, head_dim]

    loop_inner = 1
    min_bdy = 1

    def __init__(
        self,
        batch_size,
        rms_norm_eps,
        qo_heads,
        kv_heads,
        head_dim,
        split_k_factor,
        h_tile=1,
        use_rms_norm=True,
    ):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.qo_heads = qo_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.h_tile = h_tile
        self.use_rms_norm = use_rms_norm
        self.split_k_factor = split_k_factor
        self.vec_size = max(16 // F16_BYTES, self.head_dim // 32)
        self.bdx = self.head_dim // self.vec_size
        self.bdy = KernelConfig.NUM_THREADS // self.bdx

        self.batch_size = batch_size
        self.m_split = ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads)
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)

    def _alloc_local(self):
        self.idx = Tx.alloc_local([1], "int32")
        self.pos = Tx.alloc_local([1], "int32")
        self.mask = Tx.alloc_local([1], "uint32")
        self.weight_vec = Tx.alloc_local([self.vec_size], "float16")
        self.weight_vec_f32 = Tx.alloc_local([self.vec_size], "float32")
        self.sum_sq = Tx.alloc_local([1], "float32")
        self.rms_norm = Tx.alloc_local([1], "float32")
        self.cos = Tx.alloc_local([self.vec_size], "float32")
        self.sin = Tx.alloc_local([self.vec_size], "float32")
        self.q_vec = Tx.alloc_local([self.vec_size], "float16")
        self.q_vec32 = Tx.alloc_local([self.vec_size], "float32")
        self.q_vec32_other = Tx.alloc_local([self.vec_size], "float32")

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, partial, qkv, q_weight, rope_pos, cos_sin_cache):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var(tid // self.bdx)
            self._alloc_local()
            with Tx.thread():
                batch_idx = Tx.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                head_idx = Tx.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                st = Tx.meta_var(tx * self.vec_size)
                half_dim = Tx.meta_var(self.head_dim // 2)
                group_in_warp = Tx.meta_var(32 // self.bdx)
                cache_stx = Tx.meta_var(st % half_dim)
                handle_num = Tx.meta_var(
                    Tx.min(self.m_tile, self.batch_size - m_idx * self.m_tile) * self.h_tile
                )
                self.idx[0] = ty
                while self.idx[0] < handle_num:
                    self.pos[0] = rope_pos[batch_idx]
                    self.sum_sq[0] = 0.0
                    # reduce
                    qkv_stx = Tx.meta_var(head_idx * self.head_dim + tx * self.vec_size)
                    Tx.fill(self.q_vec32[:], 0.0)
                    for kt in Tx.serial(self.split_k_factor):
                        Tx.copy(
                            self.q_vec32_other[:],
                            partial[kt, batch_idx, qkv_stx : qkv_stx + self.vec_size],
                            vec_len=self.vec_size,
                        )
                        for kv in Tx.unroll(self.vec_size):
                            self.q_vec32[kv] += self.q_vec32_other[kv]
                    remain = handle_num - self.idx[0] // group_in_warp * group_in_warp
                    if remain >= group_in_warp:
                        self.mask[0] = 0xFFFFFFFF
                    else:
                        self.mask[0] = (1 << (remain * self.bdx)) - 1
                    if self.use_rms_norm:
                        # sum square
                        for kv in Tx.unroll(self.vec_size):
                            self.sum_sq[0] += self.q_vec32[kv] * self.q_vec32[kv]
                        # warp reduce sum
                        for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[0] = self.sum_sq[0] + Tx.tvm_warp_shuffle_xor(
                                self.mask[0], self.sum_sq[0], (self.bdx // 2) >> kr, 32, 32
                            )
                        # rms norm
                        self.rms_norm[0] = rsqrt(self.sum_sq[0] / self.head_dim + self.rms_norm_eps)
                        # handle the weight
                        Tx.copy(
                            self.weight_vec[:],
                            q_weight[st : st + self.vec_size],
                            vec_len=self.vec_size,
                        )
                        Tx.cast(self.weight_vec_f32[:], self.weight_vec[:])
                        Tx.mul(self.weight_vec_f32[:], self.weight_vec_f32[:], self.rms_norm[0:1])
                        Tx.mul(self.q_vec32[:], self.q_vec32[:], self.weight_vec_f32[:])
                    # load cache
                    Tx.copy(
                        self.cos[:],
                        cos_sin_cache[self.pos[0], cache_stx : cache_stx + self.vec_size],
                        vec_len=self.vec_size,
                    )
                    Tx.copy(
                        self.sin[:],
                        cos_sin_cache[
                            self.pos[0], half_dim + cache_stx : half_dim + cache_stx + self.vec_size
                        ],
                        vec_len=self.vec_size,
                    )
                    # shuffle q value
                    for kv in Tx.serial(self.vec_size):
                        self.q_vec32_other[kv] = Tx.tvm_warp_shuffle_xor(
                            self.mask[0], self.q_vec32[kv], self.bdx // 2, 32, 32
                        )
                    # compute rope
                    if st < half_dim:
                        for kv in Tx.unroll(self.vec_size):
                            self.q_vec32[kv] = (
                                self.q_vec32[kv] * self.cos[kv]
                                - self.q_vec32_other[kv] * self.sin[kv]
                            )
                    else:
                        for kv in Tx.unroll(self.vec_size):
                            self.q_vec32[kv] = (
                                self.q_vec32[kv] * self.cos[kv]
                                + self.q_vec32_other[kv] * self.sin[kv]
                            )
                    # store q
                    Tx.cast(self.q_vec[:], self.q_vec32[:])
                    Tx.copy(
                        qkv[batch_idx, head_idx, st : st + self.vec_size],
                        self.q_vec[:],
                        vec_len=self.vec_size,
                    )
                    self.idx[0] += self.bdy
