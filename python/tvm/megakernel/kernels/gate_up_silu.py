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
from tvm.megakernel.utils.base import SmemManager
from tvm.megakernel.utils.config import KernelConfig, ProfileEventType
from tvm.megakernel.utils.utils import silu
from tvm.script import tirx as Tx
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

from .gemm import GemmTile


class GateUpSiluTile(GemmTile):
    need_init = False

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="exclusive",
        )
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="exclusive",
        )
        self.D_layout = Tx.TileLayout(
            Tx.S[
                (GemmTile.TMEM_PIPE_DEPTH, GemmTile.EPI_TILE, GemmTile.MMA_N // 2) : (
                    GemmTile.EPI_TILE * GemmTile.MMA_N // 2,
                    GemmTile.MMA_N // 2,
                    1,
                )
            ]
        )
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N // 2),
            "float16",
            layout=self.D_layout,
            align=1024,
            method="exclusive",
        )

    def _alloc_local(self, m_idx):
        self.reg = Tx.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local")
        self.reg_fp16 = Tx.alloc_buffer((self.TMEM_LD_SIZE // 2,), "float16", scope="local")
        self.tmem_idx = Tx.local_scalar("int32")
        self.tmem_phase = Tx.local_scalar("int32")
        self.stage = Tx.local_scalar("int32")
        self.wait_complete = Tx.local_scalar("bool")
        self.off = Tx.local_scalar("int32")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @Tx.inline
    def _consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        with Tx.thread():
            tid_in_wg = Tx.thread_id_in_wg([128])
            warp_id = Tx.warp_id_in_wg([KernelConfig.WARP_NUMBER])
            lane_id = Tx.lane_id([32])
            Tx.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
            if warp_id == 0:
                self.smem_manager.wait_specific(lane_id, self.output_smem, 0)
            Tx.ptx.bar.sync(10, 128)
            self.phase[0] = 0
            self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
            self.tmem_phase = self.tile_idx // self.TMEM_PIPE_DEPTH & 1
            self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
            Tx.ptx.tcgen05.fence.after_thread_sync()
            SILU_HANDLE_UNIT = Tx.meta_var(self.TMEM_LD_SIZE // 2)
            self.off = Tx.if_then_else(lane_id < 16, SILU_HANDLE_UNIT, 0)
            for ko in Tx.unroll(self.MMA_M // self.EPI_TILE):
                self.stage = (
                    self.tile_idx * self.MMA_M // self.EPI_TILE + ko
                ) % self.TMEM_PIPE_DEPTH
                if ko >= self.TMEM_PIPE_DEPTH:
                    if (lane_id == 0) & (warp_id == 0):
                        Tx.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                    Tx.cuda.warpgroup_sync(10)
                for ki in Tx.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                    with Tx.warpgroup():
                        reg_wg = self.reg.view(
                            128,
                            self.TMEM_LD_SIZE,
                            layout=TileLayout(
                                S[(128, self.TMEM_LD_SIZE) : (1 @ axis_tid_in_wg, 1)]
                            ),
                        )
                        col_st = Tx.meta_var(
                            self.tmem_idx * self.M_pad_size
                            + ko * self.EPI_TILE
                            + ki * self.TMEM_LD_SIZE
                        )
                        Tx.copy(reg_wg[:, :], self.tmem[:, col_st : col_st + self.TMEM_LD_SIZE])
                    if self.profiler_on:
                        profiler.start(ProfileEventType.SILU_MUL, lane_id == 0)
                    for kv in Tx.unroll(SILU_HANDLE_UNIT):
                        self.reg[self.off + kv] = Tx.tvm_warp_shuffle_xor(
                            4294967295, self.reg[self.off + kv], 16, 32, 32
                        )
                    for kv in Tx.unroll(SILU_HANDLE_UNIT):
                        self.reg[kv] = silu(self.reg[kv]) * self.reg[SILU_HANDLE_UNIT + kv]
                    Tx.cast(self.reg_fp16[:], self.reg[0:SILU_HANDLE_UNIT], vec=SILU_HANDLE_UNIT)
                    st = Tx.meta_var(ki * self.TMEM_LD_SIZE + lane_id // 16 * SILU_HANDLE_UNIT)
                    Tx.copy(
                        self.output_smem[
                            self.stage, st : st + SILU_HANDLE_UNIT, warp_id * 16 + lane_id % 16
                        ],
                        self.reg_fp16[:],
                        vec=SILU_HANDLE_UNIT,
                    )
                    if self.profiler_on:
                        profiler.end(ProfileEventType.SILU_MUL, lane_id == 0)
                if ko == self.MMA_M // self.EPI_TILE - 1:
                    Tx.ptx.tcgen05.fence.before_thread_sync()
                    self.ld2mma_bar.arrive(self.tmem_idx)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.warpgroup_sync(10)
                if tid_in_wg == 0:
                    with Tx.thread():
                        m_st = Tx.meta_var(m_idx * self.M_pad_size + ko * self.EPI_TILE)
                        n_st = Tx.meta_var(n_idx * self.BLK_N // 2)
                        tma_config = Tx.meta_var(
                            {"dispatch": "tma", "cta_group": KernelConfig.CTA_GROUP}
                        )
                        Tx.copy_async(
                            output[m_st : m_st + self.EPI_TILE, n_st : n_st + self.BLK_N // 2],
                            self.output_smem[self.stage, :, :],
                            **tma_config,
                        )
                        Tx.ptx.cp_async.bulk.commit_group()
            if tid_in_wg == 0:
                Tx.ptx.cp_async.bulk.wait_group(0)
            Tx.cuda.warpgroup_sync(10)
            self.tile_idx += 1
            if warp_id == 0:
                self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)
