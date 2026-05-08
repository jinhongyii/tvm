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
from tvm.megakernel.utils.config import F16_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv, exp2, find_power_of_two
from tvm.script import tirx as Tx
from tvm.script.ir_builder import IRBuilder


class DecodeTile(Tile):
    # qkv_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # kv_cache_tvm: [max_page_num, 2, kv_heads, page_size, head_dim]
    # o_tvm: [new_batch_size, qo_heads, head_dim]
    # lse_tvm: [new_batch_size, qo_heads]
    # kv_indptr_tvm: [batch_size + 1]
    # kv_last_page_len_tvm: [batch_size]
    # kv_indices_tvm: [total_page_num]
    # request_indices_tvm: [new_batch_size]
    # kv_tile_indices_tvm: [new_batch_size]
    # max_chunk_size_tvm: [1]

    @classmethod
    def class_config_init(cls, problem_config: dict[str, Any], use_device_call=False):
        cls.use_device_call = use_device_call
        cls.loop_inner = 1
        cls.pipe_depth = 1
        cls.tile_per_bdx = 4
        cls.max_blk_per_sm = 8
        cls.page_size = problem_config["page_size"]
        cls.qo_heads = problem_config["num_attention_heads"]
        cls.kv_heads = problem_config["num_key_value_heads"]
        cls.head_dim = problem_config["head_dim"]
        assert cls.qo_heads % cls.kv_heads == 0
        cls.vec_size = max(16 // F16_BYTES, cls.head_dim // 32)
        cls.bdx = cls.head_dim // cls.vec_size
        cls.bdy = cls.qo_heads // cls.kv_heads
        cls.bdz = KernelConfig.NUM_THREADS // (cls.bdx * cls.bdy)
        assert cls.bdz > 0
        cls.sm_scale = (1 / 0.6931471805599453) * (1 / cls.head_dim**0.5)

    def __init__(
        self,
        qkv_tvm: Tx.handle,
        kv_cache_tvm: Tx.handle,
        o_tvm: Tx.handle,
        lse_tvm: Tx.handle,
        o_tmp_tvm: Tx.handle,
        lse_tmp_tvm: Tx.handle,
        kv_indptr_tvm: Tx.handle,
        kv_last_page_len_tvm: Tx.handle,
        kv_indices_tvm: Tx.handle,
        request_indices_tvm: Tx.handle,
        kv_tile_indices_tvm: Tx.handle,
        max_chunk_size_tvm: Tx.handle,
    ):
        self.batch_size = qkv_tvm.shape[0]
        self.new_batch_size = request_indices_tvm.shape[0]
        self.qkv_global = qkv_tvm
        self.kv_cache_global = Tx.reshape(kv_cache_tvm, (-1,))
        self.o_global = o_tvm
        self.lse_global = lse_tvm
        self.o_tmp_global = o_tmp_tvm
        self.lse_tmp_global = lse_tmp_tvm
        self.kv_indptr_global = kv_indptr_tvm
        self.kv_last_page_len_global = kv_last_page_len_tvm
        self.kv_indices_global = kv_indices_tvm
        self.request_indices_global = request_indices_tvm
        self.kv_tile_indices_global = kv_tile_indices_tvm
        self.max_chunk_size_global = max_chunk_size_tvm
        self.smem_manager = None
        assert qkv_tvm.shape[1] == self.qo_heads + 2 * self.kv_heads
        assert qkv_tvm.shape[2] == self.head_dim
        assert kv_cache_tvm.shape[1] == 2
        assert kv_cache_tvm.shape[2] == self.kv_heads
        assert kv_cache_tvm.shape[3] == self.page_size
        assert kv_cache_tvm.shape[4] == self.head_dim
        # assert kv_indptr_tvm.shape[0] == self.batch_size + 1
        assert kv_last_page_len_tvm.shape[0] == self.batch_size
        assert request_indices_tvm.shape[0] == self.new_batch_size
        assert kv_tile_indices_tvm.shape[0] == self.new_batch_size
        assert max_chunk_size_tvm.shape[0] == 1
        assert o_tvm.shape[0] == self.batch_size
        assert o_tvm.shape[1] == self.qo_heads
        assert o_tvm.shape[2] == self.head_dim
        assert lse_tvm.shape[0] == self.batch_size
        assert lse_tvm.shape[1] == self.qo_heads
        assert o_tmp_tvm.shape[0] == self.new_batch_size
        assert o_tmp_tvm.shape[1] == self.qo_heads
        assert o_tmp_tvm.shape[2] == self.head_dim
        assert lse_tmp_tvm.shape[0] == self.new_batch_size
        assert lse_tmp_tvm.shape[1] == self.qo_heads

    def alloc_buffer(self, smem_manager: SmemManager):
        # allocate the smem
        self.k_smem = smem_manager.alloc(
            [
                self.pipe_depth,
                self.loop_inner,
                self.bdz,
                self.bdy,
                self.tile_per_bdx,
                self.head_dim,
            ],
            "float16",
            align=16,
        )
        self.v_smem = smem_manager.alloc(
            [
                self.pipe_depth,
                self.loop_inner,
                self.bdz,
                self.bdy,
                self.tile_per_bdx,
                self.head_dim,
            ],
            "float16",
            align=16,
        )
        self.kv_offset = smem_manager.alloc(
            [self.bdz, self.bdx, self.bdy, self.tile_per_bdx], "int32"
        )
        self.epi_o = smem_manager.alloc(
            [self.bdz, self.bdy, self.bdx, self.loop_inner, self.vec_size], "float32"
        )
        self.epi_md = smem_manager.alloc([self.bdz, self.bdy, self.loop_inner, 2], "float32")

        # allocate the reg
        self.idx = Tx.alloc_local([1], "int32")
        self.tmp = Tx.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.q = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.k = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.v = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.s = Tx.alloc_local([self.loop_inner, self.tile_per_bdx * self.bdy], "float32")
        self.batch_idx = Tx.alloc_local([1], "int32")
        self.chunk_start_logical = Tx.alloc_local([1], "int32")
        self.chunk_end_logical = Tx.alloc_local([1], "int32")
        self.chunk_size = Tx.alloc_local([1], "int32")
        self.indices = Tx.alloc_local([1], "int32")
        self.kv_offset_cp = Tx.alloc_local([self.tile_per_bdx], "int32")
        self.o = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.m = Tx.alloc_local([self.loop_inner, 2], "float32")
        self.d = Tx.alloc_local([self.loop_inner, 2], "float32")
        self.m_tmp = Tx.alloc_local([self.loop_inner, 1], "float32")
        self.d_tmp = Tx.alloc_local([self.loop_inner, 1], "float32")
        self.o_tmp = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        IRBuilder.current().name("idx", self.idx)
        IRBuilder.current().name("tmp", self.tmp)
        IRBuilder.current().name("q", self.q)
        IRBuilder.current().name("k", self.k)
        IRBuilder.current().name("v", self.v)
        IRBuilder.current().name("s", self.s)
        IRBuilder.current().name("batch_idx", self.batch_idx)
        IRBuilder.current().name("chunk_start_logical", self.chunk_start_logical)
        IRBuilder.current().name("chunk_end_logical", self.chunk_end_logical)
        IRBuilder.current().name("chunk_size", self.chunk_size)
        IRBuilder.current().name("indices", self.indices)
        IRBuilder.current().name("kv_offset_cp", self.kv_offset_cp)
        IRBuilder.current().name("o", self.o)
        IRBuilder.current().name("m", self.m)
        IRBuilder.current().name("d", self.d)
        IRBuilder.current().name("m_tmp", self.m_tmp)
        IRBuilder.current().name("d_tmp", self.d_tmp)
        IRBuilder.current().name("o_tmp", self.o_tmp)

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self.alloc_buffer(smem_manager)

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, split_kv):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            tx = Tx.meta_var(tid % self.bdx)
            ty = Tx.meta_var((tid // self.bdx) % self.bdy)
            tz = Tx.meta_var(tid // (self.bdx * self.bdy))

            tx_start = Tx.meta_var(tx * self.vec_size)

            @Tx.inline
            def _fetch_kv_offset(kt, kv_head_id_beg, offset):
                token_id = Tx.meta_var(self.chunk_start_logical[0] + offset)
                if token_id < self.chunk_end_logical[0]:
                    p = Tx.meta_var(token_id // self.page_size)
                    r = Tx.meta_var(token_id % self.page_size)
                    self.indices[0] = Tx.cuda.ldg(self.kv_indices_global.ptr_to([p]), "int32")
                    self.kv_offset[tz, tx, ty, kt] = (
                        self.indices[0] * 2 * self.kv_heads * self.page_size * self.head_dim
                        + kv_head_id_beg * self.page_size * self.head_dim
                        + r * self.head_dim
                    )
                else:
                    self.kv_offset[tz, tx, ty, kt] = 0

            @Tx.inline
            def _sync_blk():
                if self.bdz <= 4:
                    Tx.ptx.bar.sync(1 + tz, self.bdx * self.bdy)
                else:
                    Tx.ptx.bar.sync(1, KernelConfig.NUM_THREADS)

            with Tx.thread():
                new_batch_id = Tx.meta_var(m_idx)
                kv_head_id_beg = Tx.meta_var(n_idx)

                # fetch q
                self.batch_idx[0] = self.request_indices_global[new_batch_id]
                for kb in Tx.unroll(self.loop_inner):
                    Tx.copy(
                        self.tmp[kb, :],
                        self.qkv_global[
                            self.batch_idx[0],
                            (kv_head_id_beg + kb) * self.bdy + ty,
                            tx_start : tx_start + self.vec_size,
                        ],
                    )
                    Tx.cast(self.q[kb, :], self.tmp[kb, :])

                # get chunk size info
                self.chunk_start_logical[0] = (
                    self.kv_indptr_global[self.batch_idx[0]] * self.page_size
                )
                self.chunk_end_logical[0] = self.chunk_start_logical[0]
                if split_kv:
                    self.chunk_start_logical[0] += (
                        self.max_chunk_size_global[0] * self.kv_tile_indices_global[new_batch_id]
                    )
                    self.chunk_end_logical[0] = Tx.min(
                        self.chunk_start_logical[0] + self.max_chunk_size_global[0],
                        self.chunk_end_logical[0]
                        + (
                            self.kv_indptr_global[self.batch_idx[0] + 1]
                            - self.kv_indptr_global[self.batch_idx[0]]
                            - 1
                        )
                        * self.page_size
                        + self.kv_last_page_len_global[self.batch_idx[0]],
                    )
                else:
                    self.chunk_end_logical[0] += (
                        self.kv_indptr_global[self.batch_idx[0] + 1]
                        - self.kv_indptr_global[self.batch_idx[0]]
                        - 1
                    ) * self.page_size + self.kv_last_page_len_global[self.batch_idx[0]]
                self.chunk_size[0] = self.chunk_end_logical[0] - self.chunk_start_logical[0]

                # fetch kv-offset
                for kt in Tx.unroll(self.tile_per_bdx):
                    _fetch_kv_offset(
                        kt,
                        kv_head_id_beg,
                        ((tx * self.bdz + tz) * self.bdy + ty) * self.tile_per_bdx + kt,
                    )
                Tx.ptx.fence.proxy_async("shared::cta")
                _sync_blk()

                for kp in Tx.unroll(self.pipe_depth):
                    # get kv-offset used in cp
                    for kt in Tx.unroll(self.tile_per_bdx):
                        self.kv_offset_cp[kt] = self.kv_offset[tz, kp, ty, kt] + tx * self.vec_size

                    # fetch K
                    for kt in Tx.unroll(self.tile_per_bdx):
                        if (
                            (kp * self.bdz + tz) * self.bdy + ty
                        ) * self.tile_per_bdx + kt < self.chunk_size[0]:
                            for kb in Tx.unroll(self.loop_inner):
                                g_st = Tx.meta_var(
                                    self.kv_offset_cp[kt] + kb * self.page_size * self.head_dim
                                )
                                Tx.copy_async(
                                    self.k_smem[
                                        kp, kb, tz, ty, kt, tx_start : tx_start + self.vec_size
                                    ],
                                    self.kv_cache_global[g_st : g_st + self.vec_size],
                                    dispatch="non-bulk-copy",
                                    vec_len=self.vec_size,
                                )
                    Tx.ptx.cp_async.commit_group()

                    # fetch V
                    for kt in Tx.unroll(self.tile_per_bdx):
                        if (
                            (kp * self.bdz + tz) * self.bdy + ty
                        ) * self.tile_per_bdx + kt < self.chunk_size[0]:
                            for kb in Tx.unroll(self.loop_inner):
                                g_st = Tx.meta_var(
                                    self.kv_heads * self.page_size * self.head_dim
                                    + self.kv_offset_cp[kt]
                                    + kb * self.page_size * self.head_dim
                                )
                                Tx.copy_async(
                                    self.v_smem[
                                        kp, kb, tz, ty, kt, tx_start : tx_start + self.vec_size
                                    ],
                                    self.kv_cache_global[g_st : g_st + self.vec_size],
                                    dispatch="non-bulk-copy",
                                    vec_len=self.vec_size,
                                )
                    Tx.ptx.cp_async.commit_group()

                    # initilize the value
                    self.idx[0] = 0
                    for kb in Tx.unroll(self.loop_inner):
                        for kv in Tx.unroll(self.vec_size):
                            self.o[kb, kv] = 0.0
                        self.m[kb, 0] = Tx.min_value("float32")
                        self.d[kb, 0] = 1.0
                    # pipeline
                    for ki in Tx.serial(
                        ceildiv(self.chunk_size[0], (self.tile_per_bdx * self.bdy * self.bdz))
                    ):
                        # fetch new kv-offset
                        if (ki + self.pipe_depth) % self.bdx == 0:
                            for kt in Tx.unroll(self.tile_per_bdx):
                                _fetch_kv_offset(
                                    kt,
                                    kv_head_id_beg,
                                    (
                                        (ki + self.pipe_depth)
                                        * self.tile_per_bdx
                                        * self.bdy
                                        * self.bdz
                                        + ((tx * self.bdz + tz) * self.bdy + ty) * self.tile_per_bdx
                                        + kt
                                    ),
                                )
                            Tx.ptx.fence.proxy_async("shared::cta")

                        # compute qk
                        Tx.ptx.cp_async.wait_group(2 * self.pipe_depth - 1)  # wait for K
                        _sync_blk()
                        for kb in Tx.unroll(self.loop_inner):
                            self.m[kb, 1] = self.m[kb, 0]
                        for kt in Tx.unroll(self.tile_per_bdx * self.bdy):
                            for kb in Tx.unroll(self.loop_inner):
                                # cast k to f32
                                Tx.cast(
                                    self.k[kb, :],
                                    self.k_smem[
                                        self.idx[0],
                                        kb,
                                        tz,
                                        kt // self.tile_per_bdx,
                                        kt % self.tile_per_bdx,
                                        tx_start : tx_start + self.vec_size,
                                    ],
                                )
                                self.s[kb, kt] = 0.0
                                # local gemm
                                for kv in Tx.unroll(self.vec_size):
                                    self.s[kb, kt] += self.q[kb, kv] * self.k[kb, kv]
                                # reduce from other tx's sum
                                for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                                    self.s[kb, kt] = self.s[kb, kt] + Tx.tvm_warp_shuffle_xor(
                                        0xFFFFFFFF, self.s[kb, kt], (self.bdx // 2) >> kr, 32, 32
                                    )
                                self.s[kb, kt] *= self.sm_scale
                                if (
                                    ki * self.bdz + tz
                                ) * self.bdy * self.tile_per_bdx + kt >= self.chunk_size[0]:
                                    self.s[kb, kt] = Tx.min_value("float32")
                                # update max value
                                self.m[kb, 0] = Tx.max(self.m[kb, 0], self.s[kb, kt])

                        # update the sum for softmax
                        if self.tile_per_bdx * self.bdy * tz < self.chunk_size[0]:
                            for kb in Tx.unroll(self.loop_inner):
                                o_scale = Tx.meta_var(exp2(self.m[kb, 1] - self.m[kb, 0]))
                                self.d[kb, 0] *= o_scale
                                for kt in Tx.unroll(self.tile_per_bdx * self.bdy):
                                    self.s[kb, kt] = exp2(self.s[kb, kt] - self.m[kb, 0])
                                    self.d[kb, 0] += self.s[kb, kt]
                                for kv in Tx.unroll(self.vec_size):
                                    self.o[kb, kv] = self.o[kb, kv] * o_scale
                        _sync_blk()

                        # get kv-offset used in cp
                        for kt in Tx.unroll(self.tile_per_bdx):
                            self.kv_offset_cp[kt] = (
                                self.kv_offset[tz, (ki + self.pipe_depth) % self.bdx, ty, kt]
                                + tx * self.vec_size
                            )

                        # fetch K
                        for kt in Tx.unroll(self.tile_per_bdx):
                            if (
                                ((ki + self.pipe_depth) * self.bdz + tz) * self.bdy + ty
                            ) * self.tile_per_bdx + kt < self.chunk_size[0]:
                                for kb in Tx.unroll(self.loop_inner):
                                    g_st = Tx.meta_var(
                                        self.kv_offset_cp[kt] + kb * self.page_size * self.head_dim
                                    )
                                    Tx.copy_async(
                                        self.k_smem[
                                            self.idx[0],
                                            kb,
                                            tz,
                                            ty,
                                            kt,
                                            tx_start : tx_start + self.vec_size,
                                        ],
                                        self.kv_cache_global[g_st : g_st + self.vec_size],
                                        dispatch="non-bulk-copy",
                                        vec_len=self.vec_size,
                                    )
                        Tx.ptx.cp_async.commit_group()

                        # calculate softmax(qk)v
                        Tx.ptx.cp_async.wait_group(2 * self.pipe_depth - 1)  # wait for V
                        _sync_blk()
                        for kb in Tx.unroll(self.loop_inner):
                            for kt in Tx.unroll(self.tile_per_bdx * self.bdy):
                                if (
                                    ki * self.bdz + tz
                                ) * self.bdy * self.tile_per_bdx + kt < self.chunk_size[0]:
                                    Tx.cast(
                                        self.v[kb, :],
                                        self.v_smem[
                                            self.idx[0],
                                            kb,
                                            tz,
                                            kt // self.tile_per_bdx,
                                            kt % self.tile_per_bdx,
                                            tx_start : tx_start + self.vec_size,
                                        ],
                                    )
                                    for kv in Tx.unroll(self.vec_size):
                                        self.o[kb, kv] += self.s[kb, kt] * self.v[kb, kv]
                        _sync_blk()

                        # fetch V
                        for kt in Tx.unroll(self.tile_per_bdx):
                            if (
                                ((ki + self.pipe_depth) * self.bdz + tz) * self.bdy + ty
                            ) * self.tile_per_bdx + kt < self.chunk_size[0]:
                                for kb in Tx.unroll(self.loop_inner):
                                    g_st = Tx.meta_var(
                                        self.kv_heads * self.page_size * self.head_dim
                                        + self.kv_offset_cp[kt]
                                        + kb * self.page_size * self.head_dim
                                    )
                                    Tx.copy_async(
                                        self.v_smem[
                                            self.idx[0],
                                            kb,
                                            tz,
                                            ty,
                                            kt,
                                            tx_start : tx_start + self.vec_size,
                                        ],
                                        self.kv_cache_global[g_st : g_st + self.vec_size],
                                        dispatch="non-bulk-copy",
                                        vec_len=self.vec_size,
                                    )
                        Tx.ptx.cp_async.commit_group()
                        self.idx[0] = (self.idx[0] + 1) % self.pipe_depth

                    Tx.ptx.cp_async.wait_group(0)

                    # prepare o,m,d in smem for merging
                    for kb in Tx.unroll(self.loop_inner):
                        for kv in Tx.unroll(self.vec_size):
                            self.epi_o[tz, ty, tx, kb, kv] = self.o[kb, kv]
                    if tx == 0:
                        for kb in Tx.unroll(self.loop_inner):
                            self.epi_md[tz, ty, kb, 0] = self.m[kb, 0]
                            self.epi_md[tz, ty, kb, 1] = self.d[kb, 0]
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.tvm_storage_sync("shared")
                    # merge o through different tz
                    if tz == 0:
                        for kb in Tx.unroll(self.loop_inner):
                            self.m[kb, 0] = Tx.min_value("float32")
                            self.d[kb, 0] = 1.0
                            for kv in Tx.unroll(self.vec_size):
                                self.o[kb, kv] = 0.0
                            for kz in Tx.unroll(self.bdz):
                                if self.tile_per_bdx * self.bdy * kz < self.chunk_size[0]:
                                    self.m_tmp[kb, 0] = self.epi_md[kz, ty, kb, 0]
                                    self.d_tmp[kb, 0] = self.epi_md[kz, ty, kb, 1]
                                    for kv in Tx.unroll(self.vec_size):
                                        self.o_tmp[kb, kv] = self.epi_o[kz, ty, tx, kb, kv]
                                    self.m[kb, 1] = self.m[kb, 0]
                                    self.d[kb, 1] = self.d[kb, 0]
                                    self.m[kb, 0] = Tx.max(self.m[kb, 1], self.m_tmp[kb, 0])
                                    self.d[kb, 0] = self.d[kb, 1] * exp2(
                                        self.m[kb, 1] - self.m[kb, 0]
                                    ) + self.d_tmp[kb, 0] * exp2(self.m_tmp[kb, 0] - self.m[kb, 0])
                                    for kv in Tx.unroll(self.vec_size):
                                        self.o[kb, kv] = self.o[kb, kv] * exp2(
                                            self.m[kb, 1] - self.m[kb, 0]
                                        ) + self.o_tmp[kb, kv] * exp2(
                                            self.m_tmp[kb, 0] - self.m[kb, 0]
                                        )
                            # normalize
                            for kv in Tx.unroll(self.vec_size):
                                self.o[kb, kv] = self.o[kb, kv] / self.d[kb, 0]
                        # store to global mem
                        for kb in Tx.unroll(self.loop_inner):
                            qo_head_id = Tx.meta_var((kv_head_id_beg + kb) * self.bdy + ty)
                            if split_kv:
                                Tx.copy(
                                    self.o_tmp_global[
                                        new_batch_id,
                                        qo_head_id,
                                        tx_start : tx_start + self.vec_size,
                                    ],
                                    self.o[kb, :],
                                )
                                if tx == 0:
                                    self.lse_tmp_global[new_batch_id, qo_head_id] = self.m[
                                        kb, 0
                                    ] + Tx.log2(self.d[kb, 0])
                            else:
                                Tx.cast(self.tmp[kb, :], self.o[kb, :])
                                Tx.copy(
                                    self.o_global[
                                        new_batch_id,
                                        qo_head_id,
                                        tx_start : tx_start + self.vec_size,
                                    ],
                                    self.tmp[kb, :],
                                )
                                if tx == 0:
                                    self.lse_global[new_batch_id, qo_head_id] = self.m[
                                        kb, 0
                                    ] + Tx.log2(self.d[kb, 0])
