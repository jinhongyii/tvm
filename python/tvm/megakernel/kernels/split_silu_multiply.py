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
from tvm.megakernel.utils.utils import silu
from tvm.script import tirx as Tx


class SiluMultiplyTile(Tile):
    TILE_SIZE = 128
    VEC_SIZE = 16 // F16_BYTES
    PIPE_DEPTH = 10

    def __init__(self, batch_size, intermediate_size, dtype):
        super().__init__()
        self.batch_size = batch_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # allocate smem
        self.buf = smem_manager.alloc(
            [self.PIPE_DEPTH, 2, KernelConfig.NUM_THREADS, self.VEC_SIZE], dtype=self.dtype
        )

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)
        self.prefetch_round = self.batch_size // 64

    def _alloc_local(self):
        # allocate register
        self.vec1 = Tx.alloc_local([self.VEC_SIZE], self.dtype)
        self.vec2 = Tx.alloc_local([self.VEC_SIZE], self.dtype)
        self.idx = Tx.local_scalar("int32")

    # fmt: off
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, input, output, tile_scheduler):
        with Tx.cta():
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            lane_id = Tx.lane_id([32])
            self._alloc_local()
            with Tx.thread():
                self.idx = tid * self.VEC_SIZE

                self.smem_manager.wait_all("cta")

                for ki in Tx.unroll(self.PIPE_DEPTH - 1):
                    token_idx = Tx.meta_var(self.idx // self.TILE_SIZE)
                    offset_imme = Tx.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                    if self.idx < self.batch_size * self.TILE_SIZE:
                        Tx.copy_async(self.buf[ki, 0, tid, :], input[token_idx, offset_imme:offset_imme + self.VEC_SIZE],  # noqa: E501
                                      dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                        Tx.copy_async(self.buf[ki, 1, tid, :], input[token_idx, self.intermediate_size + offset_imme:self.intermediate_size + offset_imme + self.VEC_SIZE],  # noqa: E501
                                      dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                    Tx.ptx.cp_async.commit_group()
                    self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS

                while self.idx < self.batch_size * self.TILE_SIZE + (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS:  # noqa: E501
                    token_idx = Tx.meta_var(self.idx // self.TILE_SIZE)
                    offset_imme = Tx.meta_var(m_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                    if self.idx < self.batch_size * self.TILE_SIZE:
                        cp_pipe_idx = Tx.meta_var((self.idx // (self.VEC_SIZE * KernelConfig.NUM_THREADS)) % self.PIPE_DEPTH)  # noqa: E501
                        Tx.copy_async(self.buf[cp_pipe_idx, 0, tid, :], input[token_idx, offset_imme:offset_imme + self.VEC_SIZE],  # noqa: E501
                                      dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                        Tx.copy_async(self.buf[cp_pipe_idx, 1, tid, :], input[token_idx, self.intermediate_size + offset_imme:self.intermediate_size + offset_imme + self.VEC_SIZE],  # noqa: E501
                                      dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                    Tx.ptx.cp_async.commit_group()
                    Tx.ptx.cp_async.wait_group(self.PIPE_DEPTH - 1)
                    pipe_idx = Tx.meta_var((self.idx // (self.VEC_SIZE * KernelConfig.NUM_THREADS) - (self.PIPE_DEPTH - 1)) % self.PIPE_DEPTH)  # noqa: E501
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        self.vec1[kv] = self.buf[pipe_idx, 0, tid, kv]
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        self.vec2[kv] = self.buf[pipe_idx, 1, tid, kv]
                    for kv in Tx.unroll(self.VEC_SIZE):
                        self.vec1[kv] = silu(self.vec1[kv]) * self.vec2[kv]
                    token_idx_write = Tx.meta_var((self.idx - (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS) // self.TILE_SIZE)  # noqa: E501
                    offset_imme_write = Tx.meta_var(m_idx * self.TILE_SIZE + (self.idx - (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS) % self.TILE_SIZE)  # noqa: E501
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        output[token_idx_write, offset_imme_write + kv] = self.vec1[kv]
                    self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on


class SiluMultiplyMOETile(SiluMultiplyTile):
    TILE_SIZE = 768

    def __init__(self, batch_size, intermediate_size, numel, BLK_M, dtype):
        super().__init__(batch_size, intermediate_size, dtype)
        self.BLK_M = BLK_M
        self.numel = numel

    # fmt: off
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, input, output, sorted_token_ids):
        with Tx.cta():
            warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            lane_id = Tx.lane_id([32])
            self._alloc_local()
            with Tx.thread():
                self.idx = tid * self.VEC_SIZE

                self.smem_manager.wait_all("cta")

                for ki in Tx.unroll(self.PIPE_DEPTH - 1):
                    token_idx = m_idx * self.BLK_M + self.idx // self.TILE_SIZE
                    offset_imme = n_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE
                    if self.idx < self.BLK_M * self.TILE_SIZE:
                        if sorted_token_ids[token_idx] < self.numel:
                            Tx.copy_async(self.buf[ki, 0, tid, :], input[token_idx, offset_imme:offset_imme + self.VEC_SIZE],  # noqa: E501
                                        dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                            Tx.copy_async(self.buf[ki, 1, tid, :], input[token_idx, self.intermediate_size + offset_imme:self.intermediate_size + offset_imme + self.VEC_SIZE],  # noqa: E501
                                        dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                    Tx.ptx.cp_async.commit_group()
                    self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS

                while self.idx < self.BLK_M * self.TILE_SIZE + (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS:  # noqa: E501
                    token_idx = Tx.meta_var(m_idx * self.BLK_M + self.idx // self.TILE_SIZE)
                    offset_imme = Tx.meta_var(n_idx * self.TILE_SIZE + self.idx % self.TILE_SIZE)
                    if self.idx < self.BLK_M * self.TILE_SIZE:
                        if sorted_token_ids[token_idx] < self.numel:
                            cp_pipe_idx = Tx.meta_var((self.idx // (self.VEC_SIZE * KernelConfig.NUM_THREADS)) % self.PIPE_DEPTH)  # noqa: E501
                            Tx.copy_async(self.buf[cp_pipe_idx, 0, tid, :], input[token_idx, offset_imme:offset_imme + self.VEC_SIZE],  # noqa: E501
                                        dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                            Tx.copy_async(self.buf[cp_pipe_idx, 1, tid, :], input[token_idx, self.intermediate_size + offset_imme:self.intermediate_size + offset_imme + self.VEC_SIZE],  # noqa: E501
                                        dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                    Tx.ptx.cp_async.commit_group()
                    Tx.ptx.cp_async.wait_group(self.PIPE_DEPTH - 1)
                    pipe_idx = Tx.meta_var((self.idx // (self.VEC_SIZE * KernelConfig.NUM_THREADS) - (self.PIPE_DEPTH - 1)) % self.PIPE_DEPTH)  # noqa: E501
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        self.vec1[kv] = self.buf[pipe_idx, 0, tid, kv]
                    for kv in Tx.vectorized(self.VEC_SIZE):
                        self.vec2[kv] = self.buf[pipe_idx, 1, tid, kv]
                    for kv in Tx.unroll(self.VEC_SIZE):
                        self.vec1[kv] = silu(self.vec1[kv]) * self.vec2[kv]
                    token_idx_write = Tx.meta_var(m_idx * self.BLK_M + (self.idx - (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS) // self.TILE_SIZE)  # noqa: E501
                    offset_imme_write = Tx.meta_var(n_idx * self.TILE_SIZE + (self.idx - (self.PIPE_DEPTH - 1) * self.VEC_SIZE * KernelConfig.NUM_THREADS) % self.TILE_SIZE)  # noqa: E501
                    if sorted_token_ids[token_idx_write] < self.numel:
                        for kv in Tx.vectorized(self.VEC_SIZE):
                            output[token_idx_write, offset_imme_write + kv] = self.vec1[kv]
                    else:
                        break
                    self.idx += self.VEC_SIZE * KernelConfig.NUM_THREADS

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on
