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

from typing import Any

from tvm.megakernel.utils.base import SmemManager, Tile
from tvm.megakernel.utils.config import F32_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv, exp2
from tvm.script import tirx as Tx


class DecodeMergeTile(Tile):
    @classmethod
    def class_config_init(cls, problem_config: dict[str, Any], use_device_call=False):
        cls.use_device_call = use_device_call
        cls.qo_heads = problem_config["num_attention_heads"]
        cls.kv_heads = problem_config["num_key_value_heads"]
        cls.head_dim = problem_config["head_dim"]
        cls.vec_size = max(16 // F32_BYTES, cls.head_dim // 32)
        cls.bdx = cls.head_dim // cls.vec_size
        cls.bdz = 1  # handle bdz qo heads in one task
        assert cls.bdz <= cls.qo_heads // cls.kv_heads
        cls.bdy = KernelConfig.NUM_THREADS // (cls.bdx * cls.bdz)
        assert cls.bdy > 0
        cls.pipe_depth = 1

    def __init__(
        self,
        o_indptr_tvm: Tx.handle,
        o_tmp_tvm: Tx.handle,
        o_tvm: Tx.handle,
        lse_tmp_tvm: Tx.handle,
        lse_tvm: Tx.handle,
    ):
        super().__init__()
        self.o_indptr_global = o_indptr_tvm
        self.o_tmp_global = o_tmp_tvm
        self.o_global = o_tvm
        self.lse_tmp_global = lse_tmp_tvm
        self.lse_global = lse_tvm
        self.batch_size = o_tvm.shape[0]
        self.new_batch_size = o_tmp_tvm.shape[0]
        # assert o_indptr_tvm.shape[0] == self.batch_size + 1
        assert o_tmp_tvm.shape[1] == self.qo_heads
        assert o_tmp_tvm.shape[2] == self.head_dim
        assert o_tvm.shape[1] == self.qo_heads
        assert o_tvm.shape[2] == self.head_dim
        assert lse_tmp_tvm.shape[0] == self.new_batch_size
        assert lse_tmp_tvm.shape[1] == self.qo_heads
        assert lse_tvm.shape[0] == self.batch_size
        assert lse_tvm.shape[1] == self.qo_heads

    def alloc_buffer(self, smem_manager: SmemManager):
        # allocate the smem
        offset = smem_manager.pool_allocator.offset
        self.o_tmp_smem = smem_manager.alloc(
            [self.bdz, self.pipe_depth, self.bdy, self.head_dim], "float32", align=16
        )
        self.lse_tmp_smem_load = smem_manager.alloc([self.bdz, self.bdy, self.bdx], "float32")
        self.lse_tmp_smem_use = Tx.reshape(
            self.lse_tmp_smem_load, [self.bdz, self.bdx, self.bdy], name="lse_tmp_smem_use"
        )
        smem_manager.pool_allocator.move_base_to(offset)
        self.o_epi_smem = smem_manager.alloc([self.bdz, self.bdy, self.head_dim], "float32")
        self.lse_epi_smem = smem_manager.alloc([self.bdz, self.bdy], "float32")

        # allocate the reg
        self.new_beg_batch_idx = Tx.alloc_local([1], "int32")
        self.num = Tx.alloc_local([1], "int32")
        self.tmp = Tx.alloc_local([self.vec_size], "float16")
        self.o = Tx.alloc_local([self.vec_size], "float32")
        self.m = Tx.alloc_local([2], "float32")
        self.d = Tx.alloc_local([2], "float32")
        self.m_tmp = Tx.alloc_local([1], "float32")
        self.o_tmp = Tx.alloc_local([self.vec_size], "float32")

    @Tx.inline
    def init(self, pool_allocator: Tx.SMEMPool):
        self.alloc_buffer(pool_allocator)

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var((tid // self.bdx) % self.bdy)
            tz = Tx.meta_var(tid // (self.bdx * self.bdy))

            tx_start = Tx.meta_var(tx * self.vec_size)

            with Tx.thread():
                batch_idx = Tx.meta_var(m_idx)
                head_idx = Tx.meta_var(n_idx * self.bdz + tz)

                if batch_idx < self.batch_size and head_idx < self.qo_heads:
                    self.new_beg_batch_idx[0] = self.o_indptr_global[batch_idx]
                    self.num[0] = self.o_indptr_global[batch_idx + 1] - self.new_beg_batch_idx[0]

                    if self.num[0] == 1:
                        if ty == 0:
                            Tx.copy(
                                self.o[:],
                                self.o_tmp_global[
                                    self.new_beg_batch_idx[0],
                                    head_idx,
                                    tx_start : tx_start + self.vec_size,
                                ],
                            )
                            Tx.cast(self.tmp[:], self.o[:])
                            Tx.copy(
                                self.o_global[
                                    batch_idx, head_idx, tx_start : tx_start + self.vec_size
                                ],
                                self.tmp[:],
                            )
                            self.lse_global[batch_idx, head_idx] = self.lse_tmp_global[
                                self.new_beg_batch_idx[0], head_idx
                            ]

                    # pipeline
                    for kp in Tx.unroll(self.pipe_depth):
                        if kp * self.bdy + ty < self.num[0]:
                            Tx.copy_async(
                                self.o_tmp_smem[tz, kp, ty, tx_start : tx_start + self.vec_size],
                                self.o_tmp_global[
                                    self.new_beg_batch_idx[0] + kp * self.bdy + ty,
                                    head_idx,
                                    tx_start : tx_start + self.vec_size,
                                ],
                                dispatch="non-bulk-copy",
                                vec_len=self.vec_size,
                            )
                        Tx.ptx.cp_async.commit_group()

                    # initialize the value
                    self.m[0] = Tx.float32("-inf")
                    self.d[0] = 1.0
                    for kv in Tx.unroll(self.vec_size):
                        self.o[kv] = 0.0

                    for ki in Tx.serial(ceildiv(self.num[0], self.bdy)):
                        if ki % self.bdx == 0:
                            # load lse
                            if ki * self.bdy + ty * self.bdx + tx < self.num[0]:
                                self.lse_tmp_smem_load[tz, ty, tx] = self.lse_tmp_global[
                                    self.new_beg_batch_idx[0] + ki * self.bdy + ty * self.bdx + tx,
                                    head_idx,
                                ]
                            else:
                                self.lse_tmp_smem_load[tz, ty, tx] = 0.0
                            Tx.ptx.bar.sync(2, KernelConfig.NUM_THREADS)

                        Tx.ptx.cp_async.wait_group(self.pipe_depth - 1)
                        Tx.ptx.bar.sync(2, KernelConfig.NUM_THREADS)

                        for kv in Tx.serial(self.vec_size):
                            self.o_tmp[kv] = self.o_tmp_smem[
                                tz, ki % self.pipe_depth, ty, tx * self.vec_size + kv
                            ]
                        if ki * self.bdy + ty < self.num[0]:
                            self.m_tmp[0] = self.lse_tmp_smem_use[tz, ki % self.bdx, ty]
                            m1 = self.m[0]
                            d1 = self.d[0]
                            self.m[0] = Tx.max(m1, self.m_tmp[0])
                            self.d[0] = d1 * exp2(m1 - self.m[0]) + exp2(self.m_tmp[0] - self.m[0])
                            for kv in Tx.unroll(self.vec_size):
                                self.o[kv] = self.o[kv] * exp2(m1 - self.m[0]) + self.o_tmp[
                                    kv
                                ] * exp2(self.m_tmp[0] - self.m[0])
                        Tx.ptx.bar.sync(2, KernelConfig.NUM_THREADS)
                        if (self.pipe_depth + ki) * self.bdy + ty < self.num[0]:
                            Tx.copy_async(
                                self.o_tmp_smem[
                                    tz,
                                    ki % self.pipe_depth,
                                    ty,
                                    tx_start : tx_start + self.vec_size,
                                ],
                                self.o_tmp_global[
                                    self.new_beg_batch_idx[0]
                                    + (ki + self.pipe_depth) * self.bdy
                                    + ty,
                                    head_idx,
                                    tx_start : tx_start + self.vec_size,
                                ],
                                dispatch="non-bulk-copy",
                                vec_len=self.vec_size,
                            )
                        Tx.ptx.cp_async.commit_group()
                    Tx.ptx.cp_async.wait_group(0)
                    Tx.ptx.bar.sync(2, KernelConfig.NUM_THREADS)
                    # normalize
                    for kv in Tx.unroll(self.vec_size):
                        self.o[kv] = self.o[kv] / self.d[0]

                    # reduce
                    for kv in Tx.serial(self.vec_size):
                        self.o_epi_smem[tz, ty, tx * self.vec_size + kv] = self.o[kv]
                    self.lse_epi_smem[tz, ty] = self.m[0] + Tx.log2(self.d[0])
                    self.m[0] = Tx.float32("-inf")
                    self.d[0] = 1.0
                    for kv in Tx.serial(self.vec_size):
                        self.o[kv] = 0.0
                    Tx.ptx.bar.sync(2, KernelConfig.NUM_THREADS)
                    if ty == 0:
                        for ky in Tx.serial(self.bdy):
                            self.m_tmp[0] = self.lse_epi_smem[tz, ky]
                            for kv in Tx.serial(self.vec_size):
                                self.o_tmp[kv] = self.o_epi_smem[tz, ky, tx * self.vec_size + kv]
                            self.m[1] = self.m[0]
                            self.d[1] = self.d[0]
                            self.m[0] = Tx.max(self.m[1], self.m_tmp[0])
                            self.d[0] = self.d[1] * exp2(self.m[1] - self.m[0]) + exp2(
                                self.m_tmp[0] - self.m[0]
                            )
                            for kv in Tx.unroll(self.vec_size):
                                self.o[kv] = self.o[kv] * exp2(self.m[1] - self.m[0]) + self.o_tmp[
                                    kv
                                ] * exp2(self.m_tmp[0] - self.m[0])

                        for kv in Tx.unroll(self.vec_size):
                            self.o[kv] = self.o[kv] / self.d[0]

                        # store to global mem
                        Tx.cast(self.tmp[:], self.o[:])
                        Tx.copy(
                            self.o_global[batch_idx, head_idx, tx_start : tx_start + self.vec_size],
                            self.tmp[:],
                        )
                        if tx == 0:
                            self.lse_global[batch_idx, head_idx] = self.m[0] + Tx.log2(self.d[0])
