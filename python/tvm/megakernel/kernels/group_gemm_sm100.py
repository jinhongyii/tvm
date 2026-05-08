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

from typing import Literal

from tvm.megakernel.utils.base import SmemManager
from tvm.megakernel.utils.config import F16_BYTES, F32_BYTES, KernelConfig
from tvm.script import tirx as Tx
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

from .gate_up_silu import GateUpSiluTile
from .gemm import GemmTile

red_f16 = """
__forceinline__ __device__ void red_f16_v4(half* address, half* reg) {
    uint16_t* h_reg = (uint16_t*) reg;
    asm volatile("red.global.v4.f16.add.noftz [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address), "h"(h_reg[0]), "h"(h_reg[1]), "h"(h_reg[2]), "h"(h_reg[3])
                 : "memory");
}
"""

red_f32 = """
__forceinline__ __device__ void red_f32_v4(float* address, float* reg) {
    asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(address), "f"(reg[0]), "f"(reg[1]), "f"(reg[2]), "f"(reg[3])
                 : "memory");
}
"""

cp_async_mbarrier_arrive_noinc = """
__forceinline__ __device__ void tvm_builtin_ptx_cp_async_mbarrier_arrive(void* barrier) {
    unsigned int smem_int_ptr = __cvta_generic_to_shared(barrier);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(smem_int_ptr));
}
"""


########################################################################
# F16-lowM SM100 GroupGEMM Megakernel-Tile
########################################################################


class GroupGEMMTile(GemmTile):
    CP_ASYNC_VEC_LEN = 16 // F16_BYTES
    CP_ASYNC_COMMIT_CHUNK = 32

    def __init__(
        self,
        N,
        K,
        num_experts,
        top_k,
        numel,
        a_type,
        b_type,
        low_batch=True,
        acc_output=False,
        prefetch_on=False,
        profiler_on=False,
        use_cp_async_input=False,
        use_tma_gather4=False,
    ):
        super().__init__(
            N,
            K,
            a_type,
            b_type,
            1,
            BLK_M=-1,  # does not matter because we will set it later
            MMA_M=-1,  # does not matter because we will set it later
            out_type="float16" if not acc_output else "float32",
            low_batch=low_batch,
            prefetch_on=prefetch_on,
            profiler_on=profiler_on,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.numel = numel
        self.acc_output = acc_output
        self.VEC_LEN = 16 // F32_BYTES
        self.BLK_M_candidate = [128, 64, 32]
        self.M_pad_size = max(self.BLK_M_candidate)
        self.use_cp_async_input = use_cp_async_input
        self.use_tma_gather4 = use_tma_gather4
        # cp.async A path uses 32 lane-local noinc-arrives + 1 elected arrive.expect_tx for B TMA.
        self.__class__.TMA2MMA_ARRIVE_COUNT = 33 if self.use_cp_async_input else 1

    def set_moe_info(self, expert_ids, routing_weights, sorted_token_ids):
        self.expert_ids = expert_ids
        self.routing_weights = routing_weights
        self.sorted_token_ids = sorted_token_ids

    @Tx.inline
    def _cp_async_load_A_tile(self, m_idx, ks, stage_k, tid, lane_id, A, mbar):
        vec_len = Tx.meta_var(self.CP_ASYNC_VEC_LEN)
        total_vec = Tx.meta_var(self.BLK_M * self.BLK_K // self.CP_ASYNC_VEC_LEN)
        idx = Tx.alloc_buffer([1], "int32", scope="local")
        commit_count = Tx.alloc_buffer([1], "int32", scope="local")
        row_base = m_idx * self.M_pad_size
        idx[0] = lane_id
        commit_count[0] = 0
        while idx[0] < total_vec:
            offset = idx[0] * vec_len
            row = offset // self.BLK_K
            col = offset % self.BLK_K
            row_global = row_base + row
            row_global_safe = Tx.min(row_global, self.sorted_token_ids.shape[0] - 1)
            row_in_bound = row_global < self.sorted_token_ids.shape[0]
            token_linear = Tx.alloc_buffer([1], "int32", scope="local")
            with Tx.thread():
                tmp = self.sorted_token_ids[row_global_safe]
                val = Tx.if_then_else(row_in_bound, tmp, self.numel)
                token_linear[0] = Tx.min(Tx.max(val, 0), self.numel - 1)
            stage_valid = stage_k + col + vec_len <= self.K
            if stage_valid:
                with Tx.thread():
                    Tx.copy_async(
                        self.A_smem[ks, row, col : col + vec_len],
                        A[
                            (token_linear[0] if self.acc_output else token_linear[0] // self.top_k),
                            stage_k + col : stage_k + col + vec_len,
                        ],
                        dispatch="non-bulk-copy",
                        vec_len=vec_len,
                    )
            else:
                for v in Tx.serial(vec_len):
                    if col + v < self.BLK_K:
                        self.A_smem[ks, row, col + v] = Tx.cast(0, self.a_type)
            idx[0] += 32
            commit_count[0] += 1
            if commit_count[0] == self.CP_ASYNC_COMMIT_CHUNK:
                Tx.ptx.cp_async.commit_group()
                commit_count[0] = 0
        if commit_count[0] > 0:
            Tx.ptx.cp_async.commit_group()
        Tx.cuda.func_call(
            "tvm_builtin_ptx_cp_async_mbarrier_arrive",
            mbar,
            source_code=cp_async_mbarrier_arrive_noinc,
        )

    def _alloc_local(self, m_idx):
        super()._alloc_local(m_idx)
        self.num_tokens_in_block = Tx.local_scalar("int32")
        self.eid = Tx.local_scalar("int32")
        Tx.buffer_store(self.eid.buffer, self.expert_ids[m_idx], 0)

    @Tx.inline
    def _tma(self, ks, buf, buf_name: Literal["A", "B"], mn_st, k_st, tma_config, predicate=True):
        if predicate:
            if buf_name == "A":
                Tx.copy_async(
                    self.A_smem[ks, 0 : self.BLK_M, :],
                    buf[mn_st : mn_st + self.BLK_M, k_st : k_st + self.BLK_K],
                    **tma_config,
                )
            elif buf_name == "B":
                Tx.copy_async(
                    self.B_smem[ks, 0 : self.BLK_N, :],
                    buf[self.eid, mn_st : mn_st + self.BLK_N, k_st : k_st + self.BLK_K],
                    **tma_config,
                )
            else:
                Tx.cuda.trap_when_assert_failed(False)

    @Tx.inline
    def _preload_tma_gather4_A(self, m_idx, lane_id):
        for i in Tx.unroll(0, self.BLK_M // 32):
            idx = i * 32 + lane_id
            row_global = Tx.min(m_idx * self.M_pad_size + idx, self.sorted_token_ids.shape[0] - 1)
            token_linear = self.sorted_token_ids[row_global]
            self.smem_sorted_token_ids_gather4[idx] = Tx.min(
                Tx.max(token_linear, 0), self.numel - 1
            )
        Tx.cuda.warp_sync()

    @Tx.inline
    def _tma_gather4_A(self, ks, A, m_idx, k_st, tma_config, tid, lane_id):
        for rg in Tx.unroll(0, self.BLK_M // 8):
            row_group = Tx.meta_var(rg * 8)
            row_idx = Tx.alloc_buffer([8], "int32", scope="local")
            if self.acc_output:
                for r in Tx.vectorized(4):
                    row_idx[r] = self.smem_sorted_token_ids_gather4[row_group + r]
                for r in Tx.vectorized(4):
                    row_idx[r + 4] = self.smem_sorted_token_ids_gather4[row_group + r + 4]
            else:
                for r in Tx.vectorized(4):
                    row_idx[r] = self.smem_sorted_token_ids_gather4[row_group + r] // self.top_k
                for r in Tx.vectorized(4):
                    row_idx[r + 4] = (
                        self.smem_sorted_token_ids_gather4[row_group + r + 4] // self.top_k
                    )

            Tx.ptx.cp_async.bulk.tensor.g2c_tile_gather4(
                2,
                self.A_smem.ptr_to([ks, row_group, 0]),
                tma_config["mbar"],
                Tx.address_of(tma_config["tensormap"]),
                0,
                tma_config["cta_group"],
                tma_config["cache_hint"],
                k_st,
                row_idx[0],
                row_idx[1],
                row_idx[2],
                row_idx[3],
            )
            Tx.ptx.cp_async.bulk.tensor.g2c_tile_gather4(
                2,
                self.A_smem.ptr_to([ks, row_group + 4, 4 * (2**self.SWIZZLE)]),
                tma_config["mbar"],
                Tx.address_of(tma_config["tensormap"]),
                0,
                tma_config["cta_group"],
                tma_config["cache_hint"],
                k_st,
                row_idx[4],
                row_idx[5],
                row_idx[6],
                row_idx[7],
            )

    @classmethod
    def class_init(cls, smem_manager: SmemManager):
        super().class_init(smem_manager)
        cls.smem_sorted_token_ids = smem_manager.alloc(
            [cls.MAX_BLK_M], "int32", method="persistent"
        )
        cls.smem_sorted_token_ids_gather4 = smem_manager.alloc(
            [cls.MAX_BLK_M], "int32", method="persistent", align=16
        )
        cls.smem_routing_weights = smem_manager.alloc(
            [cls.MAX_BLK_M], "float32", method="persistent"
        )

    @Tx.inline
    def _consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        if not self.acc_output:
            GemmTile._consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler)
        else:
            with Tx.cta():
                tid_in_wg = Tx.thread_id_in_wg([128])
                warp_id = Tx.warp_id_in_wg([KernelConfig.WARP_NUMBER])
                lane_id = Tx.lane_id([32])
                Tx.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
                if tid_in_wg < self.M_pad_size:
                    idx = self.sorted_token_ids[m_idx * self.M_pad_size + tid_in_wg]
                    self.smem_sorted_token_ids[tid_in_wg] = idx
                    self.smem_routing_weights[tid_in_wg] = Tx.if_then_else(
                        idx < self.numel, self.routing_weights[idx], 0.0
                    )
                Tx.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    self.smem_manager.wait_specific(lane_id, self.output_smem, 0)
                Tx.cuda.warpgroup_sync(10)
                self.phase[0] = 0
                self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
                self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

                # flush previous tma
                # wait for the completion of all the mma of the same tile
                self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
                Tx.ptx.tcgen05.fence.after_thread_sync()

                for ko in Tx.unroll(self.MMA_M // self.EPI_TILE):
                    self.stage = (
                        self.tile_idx * self.MMA_M // self.EPI_TILE + ko
                    ) % self.TMEM_PIPE_DEPTH
                    # tmem -> rf (ld) -> smem
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
                        with Tx.thread():
                            st = Tx.meta_var(ki * self.TMEM_LD_SIZE)
                            Tx.copy(
                                self.output_smem[
                                    self.stage, st : st + self.TMEM_LD_SIZE, tid_in_wg
                                ],
                                self.reg[:],
                            )
                    # the tmem can be overwritten
                    if ko == self.MMA_M // self.EPI_TILE - 1:
                        Tx.ptx.tcgen05.fence.before_thread_sync()
                        self.ld2mma_bar.arrive(self.tmem_idx)

                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.warpgroup_sync(10)
                    # smem -> gmem
                    for i in range(self.EPI_TILE * self.BLK_N // (128 * self.VEC_LEN)):
                        row_idx = (i * 128 + tid_in_wg) * self.VEC_LEN // self.BLK_N
                        col_idx = (i * 128 + tid_in_wg) * self.VEC_LEN % self.BLK_N
                        reordered_row_idx = self.smem_sorted_token_ids[ko * self.EPI_TILE + row_idx]
                        if reordered_row_idx >= self.numel:
                            break
                        routing_weight = self.smem_routing_weights[ko * self.EPI_TILE + row_idx]
                        # TODO: vectorize this
                        if output.dtype == "float16":
                            o_reg_f32 = Tx.alloc_buffer([self.VEC_LEN], "float32", scope="local")
                            o_reg_f16 = Tx.alloc_buffer([self.VEC_LEN], "float16", scope="local")
                            for v in range(self.VEC_LEN):
                                o_reg_f32[v] = self.output_smem[self.stage, row_idx, col_idx + v]
                            for v in Tx.unroll(self.VEC_LEN):
                                o_reg_f16[v] = Tx.cast(o_reg_f32[v] * routing_weight, "float16")
                            Tx.cuda.func_call(
                                "red_f16_v4",
                                Tx.address_of(
                                    output[
                                        reordered_row_idx // self.top_k,
                                        n_idx * self.BLK_N + col_idx,
                                    ]
                                ),
                                Tx.address_of(o_reg_f16[0]),
                                source_code=red_f16,
                            )
                        else:
                            o_reg = Tx.alloc_buffer([self.VEC_LEN], "float32", scope="local")
                            for v in range(self.VEC_LEN):
                                o_reg[v] = self.output_smem[self.stage, row_idx, col_idx + v]
                            for v in Tx.unroll(self.VEC_LEN):
                                o_reg[v] = o_reg[v] * routing_weight
                            Tx.cuda.func_call(
                                "red_f32_v4",
                                Tx.address_of(
                                    output[
                                        reordered_row_idx // self.top_k,
                                        n_idx * self.BLK_N + col_idx,
                                    ]
                                ),
                                Tx.address_of(o_reg[0]),
                                source_code=red_f32,
                            )
                Tx.cuda.warpgroup_sync(10)
                self.tile_idx += 1
                if warp_id == 0:
                    self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)

    def set_BLK_M(self, BLK_M):
        assert BLK_M in self.BLK_M_candidate
        self.BLK_M = BLK_M
        self.MMA_M = BLK_M

    @Tx.inline
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        A,
        B,
        output,
        expert_ids,
        routing_weights,
        sorted_token_ids,
        valid_num_tokens,
        A_tensormap=None,
        profiler=None,
    ):
        self.set_moe_info(expert_ids, routing_weights, sorted_token_ids)
        self._alloc_local(m_idx)
        if valid_num_tokens is not None:
            self.num_tokens_in_block = valid_num_tokens[m_idx]
        num_tokens_in_block = Tx.meta_var(
            self.num_tokens_in_block
            if valid_num_tokens is not None
            else 32
            if self.low_batch
            else self.M_pad_size
        )
        with Tx.cta():
            tid = Tx.thread_id([256])
            if num_tokens_in_block <= 32:
                self.set_BLK_M(32)
                GemmTile._run(
                    self, m_idx, n_idx, k_idx, A, B, output, profiler, A_tensormap=A_tensormap
                )
            elif num_tokens_in_block <= 64:
                self.set_BLK_M(64)
                GemmTile._run(
                    self, m_idx, n_idx, k_idx, A, B, output, profiler, A_tensormap=A_tensormap
                )
            else:
                self.set_BLK_M(128)
                GemmTile._run(
                    self, m_idx, n_idx, k_idx, A, B, output, profiler, A_tensormap=A_tensormap
                )
            self.smem_manager.advance()


########################################################################
# F16-lowM SM100 GroupGEMM-Silu Megakernel-Tile
########################################################################


class GroupGEMMSiluTile(GroupGEMMTile, GateUpSiluTile):
    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
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

    @Tx.inline
    def _consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        GateUpSiluTile._consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler)
