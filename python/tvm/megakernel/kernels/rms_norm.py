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


class RMSnormTile(Tile):
    # weight_tvm: [num_heads]
    # qk_tvm: [batch_size, num_heads, head_dim]

    loop_inner = 1
    min_bdy = 1
    h_tile = 1

    def __init__(self, batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.qo_heads = qo_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.vec_size = max(16 // F16_BYTES, self.head_dim // 32)
        self.bdx = self.head_dim // self.vec_size
        self.bdy = KernelConfig.NUM_THREADS // self.bdx

        self.batch_size = batch_size
        self.m_split = ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads)
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)

    def _alloc_local(self):
        self.idx = Tx.alloc_local([1], "int32")
        self.input_vec = Tx.alloc_local([self.vec_size], "float16")
        self.weight_vec = Tx.alloc_local([self.vec_size], "float16")
        self.input_vec_f32 = Tx.alloc_local([self.vec_size], "float32")
        self.weight_vec_f32 = Tx.alloc_local([self.vec_size], "float32")
        self.sum_sq = Tx.alloc_local([1], "float32")
        self.rms_norm = Tx.alloc_local([1], "float32")
        self.mask = Tx.alloc_local([1], "uint32")

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, qkv, q_weight, k_weight):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var(tid // self.bdx)
            self._alloc_local()

            with Tx.thread():
                self.idx[0] = ty
                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = Tx.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = Tx.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    st = Tx.meta_var(tx * self.vec_size)

                    # add & sum square
                    self.sum_sq[0] = 0.0
                    if batch_idx < self.batch_size and head_idx < self.kv_heads + self.qo_heads:
                        for kv in Tx.unroll(self.vec_size):
                            self.input_vec[kv] = qkv[batch_idx, head_idx, st + kv]
                        Tx.cast(self.input_vec_f32[:], self.input_vec[:], vec_len=self.vec_size)
                        for kv in Tx.unroll(self.vec_size):
                            self.sum_sq[0] += self.input_vec_f32[kv] * self.input_vec_f32[kv]

                        # warp reduce sum
                        if ty % 2 == 0 and (
                            batch_idx + 1 == self.batch_size
                            or self.idx[0] // self.h_tile + 1 == self.m_tile
                        ):
                            self.mask[0] = 0xFFFF
                        else:
                            self.mask[0] = 0xFFFFFFFF
                        for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[0] = self.sum_sq[0] + Tx.tvm_warp_shuffle_xor(
                                self.mask[0], self.sum_sq[0], (self.bdx // 2) >> kr, 32, 32
                            )
                        # rms norm
                        self.rms_norm[0] = rsqrt(self.sum_sq[0] / self.head_dim + self.rms_norm_eps)

                        # handle the weight
                        if n_idx * self.h_tile < self.qo_heads:
                            for kv in Tx.unroll(self.vec_size):
                                self.weight_vec[kv] = q_weight[st + kv]
                        else:
                            for kv in Tx.unroll(self.vec_size):
                                self.weight_vec[kv] = k_weight[st + kv]
                        Tx.cast(self.weight_vec_f32[:], self.weight_vec[:], vec_len=self.vec_size)
                        for kv in Tx.unroll(self.vec_size):
                            self.input_vec_f32[kv] = (
                                self.input_vec_f32[kv] * self.rms_norm[0] * self.weight_vec_f32[kv]
                            )
                        Tx.cast(self.input_vec[:], self.input_vec_f32[:], vec_len=self.vec_size)
                        for kv in Tx.unroll(self.vec_size):
                            qkv[batch_idx, head_idx, st + kv] = self.input_vec[kv]

                    self.idx[0] += self.bdy
