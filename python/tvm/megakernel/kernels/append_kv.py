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

from tvm.megakernel.utils.base import KernelConfig, Tile
from tvm.megakernel.utils.config import F16_BYTES
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


class AppendKVTile(Tile):
    # kv_cache_tvm: [max_page_num, 2, kv_heads, page_size, head_dim]
    # qkv_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # kv_indptr_tvm: [batch_size + 1]
    # kv_indices_tvm: [total_page_num]
    # kv_last_page_len_tvm: [batch_size]
    # pos_map_tvm: [batch_size]

    loop_inner = 1
    min_bdy = 1
    h_tile = 1

    def __init__(self, batch_size, num_attention_heads, num_key_value_heads, head_dim, page_size):
        super().__init__()
        self.qo_heads = num_attention_heads
        self.kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.page_size = page_size
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
        self.vec = Tx.alloc_local([self.vec_size], "float16")

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, kv_cache, qkv, pos_map):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var(tid // self.bdx)
            stx = Tx.meta_var(tx * self.vec_size)
            self._alloc_local()

            with Tx.thread():
                self.idx[0] = ty

                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = Tx.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = Tx.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    if batch_idx < self.batch_size and head_idx < self.kv_heads:
                        self.pos[0] = Tx.cuda.ldg(Tx.address_of(pos_map[batch_idx]), "int32")
                        page_id = Tx.meta_var(self.pos[0] // self.page_size)
                        offset = Tx.meta_var(self.pos[0] % self.page_size)
                        for vec in Tx.vectorized(self.vec_size):
                            self.vec[vec] = qkv[
                                batch_idx,
                                self.qo_heads + k_idx * self.kv_heads + head_idx,
                                stx + vec,
                            ]
                        for vec in Tx.vectorized(self.vec_size):
                            kv_cache[page_id, k_idx, head_idx, offset, stx + vec] = self.vec[vec]

                    self.idx[0] += self.bdy
