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
from tvm.megakernel.utils.config import F16_BYTES, F32_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


class SplitKReduceAppendVTile(Tile):
    VEC_SIZE_32 = 16 // F32_BYTES
    VEC_SIZE_16 = 16 // F16_BYTES

    def __init__(
        self, batch_size, kv_heads, qo_heads, head_dim, split_k_factor, page_size, h_tile=1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.kv_heads = kv_heads
        self.qo_heads = qo_heads
        self.head_dim = head_dim
        self.split_k_factor = split_k_factor
        self.h_tile = h_tile
        self.batch_size = batch_size
        self.m_split = ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads)
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)
        self.bdx = self.head_dim // self.VEC_SIZE_16
        self.bdy = KernelConfig.NUM_THREADS // self.bdx
        self.page_size = page_size
        assert self.bdx % 2 == 0
        assert KernelConfig.NUM_THREADS % self.bdx == 0

    def _alloc_local(self):
        self.idx = Tx.local_scalar("int32")
        self.pos = Tx.local_scalar("int32")
        self.vec_32 = Tx.alloc_local([self.VEC_SIZE_16], "float32")
        self.tmp = Tx.alloc_local([self.VEC_SIZE_16], "float32")
        self.vec_16 = Tx.alloc_local([self.VEC_SIZE_16], "float16")

    # handle: [batch_size, h_tile * head_dim]
    # bdx, bdy = head_dim // VEC_SIZE_16, NUM_THREADS // bdx
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, partial, kv_cache, pos_map):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var(tid // self.bdx)
            stx = Tx.meta_var(tx * self.VEC_SIZE_16)
            handle_num = Tx.meta_var(
                Tx.min(self.m_tile, self.batch_size - m_idx * self.m_tile) * self.h_tile
            )
            self._alloc_local()
            with Tx.thread():
                batch_idx = Tx.meta_var(m_idx * self.m_tile + self.idx // self.h_tile)
                head_idx = Tx.meta_var(n_idx * self.h_tile + self.idx % self.h_tile)
                self.idx = ty
                while self.idx < handle_num:
                    # reduce
                    qkv_stx = Tx.meta_var(
                        (self.qo_heads + self.kv_heads + head_idx) * self.head_dim
                        + tx * self.VEC_SIZE_16
                    )

                    Tx.fill(self.vec_32[:], 0.0)
                    for kt in Tx.serial(self.split_k_factor):
                        Tx.copy(
                            self.tmp[:],
                            partial[kt, batch_idx, qkv_stx : qkv_stx + self.VEC_SIZE_16],
                            vec_len=self.VEC_SIZE_16,
                        )
                        for kv in Tx.unroll(self.VEC_SIZE_16):
                            self.vec_32[kv] += self.tmp[kv]
                    Tx.cast(self.vec_16[:], self.vec_32[:])
                    # append
                    self.pos = Tx.cuda.ldg(Tx.address_of(pos_map[batch_idx]), "int32")
                    page_id = Tx.meta_var(self.pos // self.page_size)
                    offset = Tx.meta_var(self.pos % self.page_size)
                    Tx.copy(
                        kv_cache[page_id, 1, head_idx, offset, stx : stx + self.VEC_SIZE_16],
                        self.vec_16[:],
                        vec_len=self.VEC_SIZE_16,
                    )
                    self.idx += self.bdy
