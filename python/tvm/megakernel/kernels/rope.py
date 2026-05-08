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

import tvm.script.tirx as Tx
from tvm.megakernel.utils.base import Tile
from tvm.megakernel.utils.config import F16_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv


class RopeTile(Tile):
    # qk_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # cos_sin_cache_tvm: [1, head_dim]

    loop_inner = 1
    min_bdy = 1
    h_tile = 1

    def __init__(self, batch_size, qo_heads, kv_heads, head_dim):
        super().__init__()
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
        self.cos = Tx.alloc_local([self.vec_size], "float32")
        self.sin = Tx.alloc_local([self.vec_size], "float32")
        self.qk_vec = Tx.alloc_local([self.vec_size], "float16")
        self.qk_vec32 = Tx.alloc_local([self.vec_size], "float32")
        self.qk_vec32_other = Tx.alloc_local([self.vec_size], "float32")
        self.mask = Tx.alloc_local([1], "uint32")

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, qkv, cos_sin_cache, rope_pos):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var(tid // self.bdx)
            half_dim = self.head_dim // 2

            self._alloc_local()

            with Tx.thread():
                self.idx[0] = ty

                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = Tx.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = Tx.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    stx = Tx.meta_var(tx * self.vec_size)
                    cache_stx = Tx.meta_var(stx % half_dim)
                    pos = Tx.meta_var(rope_pos[batch_idx])

                    if batch_idx < self.batch_size and head_idx < self.qo_heads + self.kv_heads:
                        # load cache
                        for kv in Tx.unroll(self.vec_size):
                            self.cos[kv] = cos_sin_cache[pos, cache_stx + kv]
                        for kv in Tx.unroll(self.vec_size):
                            self.sin[kv] = cos_sin_cache[pos, half_dim + cache_stx + kv]

                        # load qk
                        for kv in Tx.unroll(self.vec_size):
                            self.qk_vec[kv] = qkv[batch_idx, head_idx, stx + kv]
                        Tx.cast(self.qk_vec32[:], self.qk_vec[:], vec_len=self.vec_size)

                        # shuffle qk value
                        if ty % 2 == 0 and (
                            batch_idx + 1 == self.batch_size
                            or self.idx[0] // self.h_tile + 1 == self.m_tile
                        ):
                            self.mask[0] = 0xFFFF
                        else:
                            self.mask[0] = 0xFFFFFFFF
                        for kv in Tx.serial(self.vec_size):
                            self.qk_vec32_other[kv] = Tx.tvm_warp_shuffle_xor(
                                self.mask[0], self.qk_vec32[kv], self.bdx // 2, 32, 32
                            )

                        # compute rope
                        if stx < half_dim:
                            for kv in Tx.unroll(self.vec_size):
                                self.qk_vec32[kv] = (
                                    self.qk_vec32[kv] * self.cos[kv]
                                    - self.qk_vec32_other[kv] * self.sin[kv]
                                )
                        else:
                            for kv in Tx.unroll(self.vec_size):
                                self.qk_vec32[kv] = (
                                    self.qk_vec32[kv] * self.cos[kv]
                                    + self.qk_vec32_other[kv] * self.sin[kv]
                                )

                        # store qk
                        Tx.cast(self.qk_vec[:], self.qk_vec32[:], vec_len=self.vec_size)
                        for kv in Tx.unroll(self.vec_size):
                            qkv[batch_idx, head_idx, stx + kv] = self.qk_vec[kv]

                    self.idx[0] += self.bdy
