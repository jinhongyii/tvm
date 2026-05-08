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
from tvm.megakernel.kernels.gemm import GemmTile
from tvm.megakernel.utils.base import Barriers, SmemManager
from tvm.megakernel.utils.config import F8_BYTES, F16_BYTES, F32_BYTES, F128_BYTES, KernelConfig
from tvm.script import tirx as T
from tvm.script import tirx as Tx
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.layout import S, TileLayout

CTA_GROUP_GLOBAL = 2


@T.inline
def skip():
    pass


@Tx.meta_class
class BarTRANS2MMA(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), cta_id=0, pred=True)


@Tx.meta_class
class BarMMA2LD(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP_GLOBAL, cta_mask=3)


@Tx.meta_class
class BarMMA2TMA(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=CTA_GROUP_GLOBAL, cta_mask=3)


class FP8GemmTile(GemmTile):
    BLK_M, BLK_N, BLK_K = (128, 112, 128)
    MMA_M, MMA_N, MMA_K = (256, 224, 32)
    CTA_GROUP = CTA_GROUP_GLOBAL
    BLK_SFA, BLK_SFB = (128, 256)
    sfa_type = "float8_e8m0fnu"
    sfb_type = "float8_e8m0fnu"
    QUANT_SIZE = BLK_K
    SFA_TMEM_START_COL = GemmTile.TMEM_PIPE_DEPTH * MMA_N
    SFB_TMEM_START_COL = GemmTile.TMEM_PIPE_DEPTH * MMA_N + GemmTile.TMEM_PIPE_DEPTH * BLK_SFA // 32
    SMEM_SIZE = (
        GemmTile.SMEM_PIPE_DEPTH * BLK_M * BLK_K * F8_BYTES
        + GemmTile.SMEM_PIPE_DEPTH * BLK_N * BLK_K * F8_BYTES
        + GemmTile.TMEM_PIPE_DEPTH * BLK_M * GemmTile.EPI_TILE * F16_BYTES
        + GemmTile.SMEM_PIPE_DEPTH * BLK_SFA * F32_BYTES
        + GemmTile.SMEM_PIPE_DEPTH * BLK_SFB * F32_BYTES
        + 1024
    )
    assert SMEM_SIZE <= 232448
    assert GemmTile.TMEM_PIPE_DEPTH * (MMA_N + BLK_SFA // 32 + BLK_SFB // 32) <= 512

    def __init__(
        self,
        N,
        K,
        a_type,
        b_type,
        split_k_factor,
        BLK_M,
        MMA_M,
        out_type=None,
        use_tma_reduce=False,
        low_batch=False,
        prefetch_on=False,
        profiler_on=False,
    ):
        super().__init__(
            N,
            K,
            a_type,
            b_type,
            split_k_factor,
            BLK_M,
            MMA_M,
            out_type,
            use_tma_reduce,
            low_batch,
            prefetch_on,
            profiler_on,
        )
        self.A_layout = T.ComposeLayout(
            T.SwizzleLayout(4, 3, 3, swizzle_inner=True),
            TileLayout(
                S[
                    (self.SMEM_PIPE_DEPTH, self.BLK_M, self.BLK_K) : (
                        self.BLK_M * self.BLK_K,
                        self.BLK_K,
                        1,
                    )
                ]
            ),
        )
        self.B_layout = T.ComposeLayout(
            T.SwizzleLayout(4, 3, 3, swizzle_inner=True),
            TileLayout(
                S[
                    (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K) : (
                        self.BLK_N * self.BLK_K,
                        self.BLK_K,
                        1,
                    )
                ]
            ),
        )
        self.D_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 2, 3, swizzle_inner=True),
            TileLayout(
                S[
                    (self.TMEM_PIPE_DEPTH, self.BLK_M, self.EPI_TILE) : (
                        self.BLK_M * self.EPI_TILE,
                        self.EPI_TILE,
                        1,
                    )
                ]
            ),
        )
        self.SFA_layout = TileLayout(
            S[(self.SMEM_PIPE_DEPTH, self.BLK_SFA // 32, 32) : (self.BLK_SFA, 32, 1)]
        )
        self.SFB_layout = TileLayout(
            S[(self.SMEM_PIPE_DEPTH, self.BLK_SFB // 32, 32) : (self.BLK_SFB, 32, 1)]
        )

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="shared",
        )
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="shared",
        )
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.BLK_M, self.EPI_TILE),
            self.out_type,
            layout=self.D_layout,
            align=1024,
            method="shared",
        )
        self.SFA_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_SFA // 32, 32),
            "uint32",
            layout=self.SFA_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="shared",
        )
        self.SFB_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_SFB // 32, 32),
            "uint32",
            layout=self.SFB_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="shared",
        )
        self.SFA_smem_2d = T.decl_buffer(
            (self.SMEM_PIPE_DEPTH, self.BLK_SFA),
            "uint32",
            data=self.SFA_smem.data,
            elem_offset=self.SFA_smem.elem_offset,
            scope="shared.dyn",
            align=1,
        )
        self.SFB_smem_2d = T.decl_buffer(
            (self.SMEM_PIPE_DEPTH, self.BLK_SFB),
            "uint32",
            data=self.SFB_smem.data,
            elem_offset=self.SFB_smem.elem_offset,
            scope="shared.dyn",
            align=1,
        )

    @Tx.inline
    def _copy_async_B_from_gmem(self, ks, B, n_start, k_start, tma_copy, m_idx):
        """Default dense GEMM: B is (N, K). m_idx is unused."""
        Tx.copy_async(
            self.B_smem[ks, :, :],
            B[n_start : n_start + self.BLK_N, k_start : k_start + self.BLK_K],
            **tma_copy,
        )

    @Tx.inline
    def _copy_async_SFB_from_gmem(self, ks, SFB, n_start_sf, tma_copy, m_idx):
        """Default dense GEMM: SFB is (sfa_rows, N). m_idx is unused."""
        Tx.copy_async(
            self.SFB_smem_2d[ks, 0 : self.BLK_N * self.CTA_GROUP],
            SFB[self.stage // 4, n_start_sf : n_start_sf + self.BLK_N * self.CTA_GROUP],
            **tma_copy,
        )

    def _alloc_local(self, m_idx):
        self.reg = T.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local")
        self.reg_fp16 = T.alloc_buffer((self.BLK_N * self.CTA_GROUP,), self.out_type, scope="local")
        self.stage = T.local_scalar("int32")

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager: SmemManager):
        super()._alloc_buffer_class_member(smem_manager)
        cls.tma2trans_bar = cls.tma2mma_bar
        cls.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        cls.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        cls.trans2mma_bar = BarTRANS2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        cls.tmem_addr = smem_manager.alloc([1], "uint32", method="persistent")

    @classmethod
    @Tx.inline
    def class_init(cls, smem_manager: SmemManager):
        warp_id = Tx.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
        cls._alloc_buffer_class_member(smem_manager)
        cls.tma2mma_bar.init(1)
        cls.trans2mma_bar.init(cls.CTA_GROUP * 32)
        cls.mma2ld_bar.init(1)
        cls.mma2tma_bar.init(1)
        cls.ld2mma_bar.init(cls.CTA_GROUP * 128)
        if T.filter(warp_id, 0, 1):
            with T.warp():
                T.ptx.tcgen05.alloc(
                    T.address_of(cls.tmem_addr[0]), n_cols=cls.N_COLS, cta_group=cls.CTA_GROUP
                )
                T.cuda.warp_sync()
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cluster_sync()
        T.cuda.trap_when_assert_failed(cls.tmem_addr[0] == 0)

    @classmethod
    @Tx.inline
    def class_finalize(cls):
        warp_id = T.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
        T.tvm_storage_sync("shared")
        if T.filter(warp_id, 0, 1):
            with T.warp():
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cls.CTA_GROUP)
                T.ptx.tcgen05.dealloc(cls.tmem_addr[0], n_cols=cls.N_COLS, cta_group=cls.CTA_GROUP)
        T.tvm_storage_sync("shared")

    @Tx.inline
    def _run(
        self,
        m_idx,
        n_idx,
        k_idx,
        cbx,
        A,
        B,
        output,
        tile_scheduler,
        SFA,
        SFB,
        profiler: CudaProfiler,
    ):
        @Tx.inline
        def partitioned_loop(main_loop, epilogue1, epilogue2):
            for ko in T.serial(self.PIPE_CIRCLE_NUM):
                for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                    self.stage = ko * self.SMEM_PIPE_DEPTH + ks
                    main_loop(ks)
                self.phase[0] = self.phase[0] ^ 1
            if self.PIPE_REMAIN_NUM > 0:
                for ks in T.unroll(self.PIPE_REMAIN_NUM):
                    self.stage = self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks
                    main_loop(ks)
                epilogue1()
                for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                    epilogue2(ks)
                self.phase[0] = self.phase[0] ^ 1
            else:
                epilogue1()

        with T.cta():
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id = T.warp_id_in_wg([KernelConfig.WARP_NUMBER])
            lane_id = T.lane_id([32])
            tid_in_wg = T.thread_id_in_wg([128])
            if wg_id == 1:
                with T.warpgroup():
                    if warp_id == 3:
                        with T.warp():
                            self.phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)
                                m_start = T.meta_var((m_idx * self.CTA_GROUP + cbx) * self.BLK_M)
                                n_start = T.meta_var((n_idx * self.CTA_GROUP + cbx) * self.BLK_N)
                                n_start_sf = T.meta_var(n_idx * self.CTA_GROUP * self.BLK_N)
                                k_start = T.meta_var(self.stage * self.BLK_K)

                                @Tx.inline
                                def tma_load(ks):
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    tma_copy = T.meta_var(
                                        {
                                            "dispatch": "tma",
                                            "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                            "cta_group": self.CTA_GROUP,
                                        }
                                    )
                                    if T.ptx.elect_sync():
                                        with T.thread():
                                            Tx.copy_async(
                                                self.A_smem[ks, :, :],
                                                A[
                                                    m_start : m_start + self.BLK_M,
                                                    k_start : k_start + self.BLK_K,
                                                ],
                                                **tma_copy,
                                            )
                                            self._copy_async_B_from_gmem(
                                                ks, B, n_start, k_start, tma_copy, m_idx
                                            )
                                            if self.stage % 4 == 0:
                                                Tx.copy_async(
                                                    self.SFA_smem_2d[ks, :],
                                                    SFA[
                                                        self.stage // 4,
                                                        m_start : m_start + self.BLK_M,
                                                    ],
                                                    **tma_copy,
                                                )
                                                self._copy_async_SFB_from_gmem(
                                                    ks, SFB, n_start_sf, tma_copy, m_idx
                                                )
                                            AB_bytes = T.meta_var(
                                                self.BLK_M * self.BLK_K * F8_BYTES
                                                + self.BLK_N * self.BLK_K * F8_BYTES
                                            )
                                            SFAB_bytes = T.meta_var(
                                                (self.BLK_N * self.CTA_GROUP + self.BLK_M)
                                                * F32_BYTES
                                            )
                                            T.ptx.mbarrier.arrive.expect_tx(
                                                self.tma2trans_bar.mbar.ptr_to([ks]),
                                                T.if_then_else(
                                                    self.stage % 4 == 0,
                                                    AB_bytes + SFAB_bytes,
                                                    AB_bytes,
                                                ),
                                            )

                                @Tx.inline
                                def tma_load_epilogue(ks):
                                    self.mma2tma_bar.wait(ks, self.phase[0])
                                    if T.ptx.elect_sync():
                                        with T.thread():
                                            T.ptx.mbarrier.arrive.expect_tx(
                                                self.tma2trans_bar.mbar.ptr_to([ks]), 0
                                            )

                                partitioned_loop(tma_load, skip, tma_load_epilogue)
                                tile_scheduler.next_tile()
                    if warp_id == 2:
                        with T.warp():
                            self.phase[0] = 0
                            while tile_scheduler.valid():
                                m_idx = T.meta_var(tile_scheduler.m_idx)
                                n_idx = T.meta_var(tile_scheduler.n_idx)

                                @Tx.inline
                                def transpose(ks):
                                    T.ptx.mbarrier.try_wait(
                                        self.tma2trans_bar.mbar.ptr_to([ks]), self.phase[0]
                                    )
                                    if self.stage % 4 == 0:
                                        Tx.permute_dims(self.SFA_smem[ks], [0, 2, 1])
                                        Tx.permute_dims(self.SFB_smem[ks, :4], [0, 2, 1])
                                        Tx.permute_dims(self.SFB_smem[ks, 4:], [0, 2, 1])
                                        T.ptx.fence.proxy_async("shared::cta")
                                    self.trans2mma_bar.arrive(ks)

                                @Tx.inline
                                def transpose_epilogue(ks):
                                    T.ptx.mbarrier.try_wait(
                                        self.tma2trans_bar.mbar.ptr_to([ks]), self.phase[0]
                                    )
                                    self.trans2mma_bar.arrive(ks)

                                partitioned_loop(transpose, skip, transpose_epilogue)
                                tile_scheduler.next_tile()
                    if warp_id == 0:
                        with T.warp():
                            if cbx == 0:
                                descA = T.local_scalar("uint64")
                                descB = T.local_scalar("uint64")
                                descSFA = T.local_scalar("uint64")
                                descSFB = T.local_scalar("uint64")
                                descI = T.local_scalar("uint32")
                                tmem_idx = T.local_scalar("int32")
                                tmem_phase = T.local_scalar("int32")
                                T.ptx.tcgen05.encode_instr_descriptor_block_scaled(
                                    T.address_of(descI),
                                    d_dtype="float32",
                                    a_dtype=self.a_type,
                                    b_dtype=self.b_type,
                                    sfa_dtype=self.sfa_type,
                                    sfb_dtype=self.sfb_type,
                                    sfa_tmem_addr=0,
                                    sfb_tmem_addr=0,
                                    M=self.MMA_M,
                                    N=self.MMA_N,
                                    K=self.MMA_K,
                                    trans_a=False,
                                    trans_b=False,
                                    n_cta_groups=self.CTA_GROUP,
                                )
                                self.phase[0] = 0
                                while tile_scheduler.valid():
                                    m_idx = T.meta_var(tile_scheduler.m_idx)
                                    n_idx = T.meta_var(tile_scheduler.n_idx)
                                    if T.ptx.elect_sync():
                                        with T.thread():
                                            tmem_idx = (
                                                tile_scheduler.tile_idx % self.TMEM_PIPE_DEPTH
                                            )
                                            tmem_phase = (
                                                tile_scheduler.tile_idx // self.TMEM_PIPE_DEPTH & 1
                                            )
                                            self.ld2mma_bar.wait(tmem_idx, tmem_phase)
                                            T.ptx.tcgen05.fence.after_thread_sync()

                                            @Tx.inline
                                            def mma(ks):
                                                self.trans2mma_bar.wait(ks, self.phase[0])
                                                T.ptx.tcgen05.fence.after_thread_sync()
                                                if self.stage % 4 == 0:
                                                    for ki in T.unroll(0, self.BLK_SFA // 128):
                                                        T.ptx.tcgen05.encode_matrix_descriptor(
                                                            T.address_of(descSFA),
                                                            self.SFA_smem.ptr_to([ks, ki * 4, 0]),
                                                            ldo=16,
                                                            sdo=8 * 4 * F32_BYTES // F128_BYTES,
                                                            swizzle=0,
                                                        )
                                                        T.ptx.tcgen05.cp(
                                                            self.SFA_TMEM_START_COL
                                                            + tmem_idx * self.BLK_SFA // 32
                                                            + ki * 4,
                                                            descSFA,
                                                            shape="32x128b",
                                                            cta_group=self.CTA_GROUP,
                                                            multicast="warpx4",
                                                        )
                                                    for ki in T.unroll(0, self.BLK_SFB // 128):
                                                        T.ptx.tcgen05.encode_matrix_descriptor(
                                                            T.address_of(descSFB),
                                                            self.SFB_smem.ptr_to([ks, ki * 4, 0]),
                                                            ldo=16,
                                                            sdo=8 * 4 * F32_BYTES // F128_BYTES,
                                                            swizzle=0,
                                                        )
                                                        T.ptx.tcgen05.cp(
                                                            self.SFB_TMEM_START_COL
                                                            + tmem_idx * self.BLK_SFB // 32
                                                            + ki * 4,
                                                            descSFB,
                                                            shape="32x128b",
                                                            cta_group=self.CTA_GROUP,
                                                            multicast="warpx4",
                                                        )
                                                T.cuda.runtime_instr_desc(
                                                    T.address_of(descI), self.stage % 4
                                                )
                                                for ki in T.unroll(self.BLK_K // self.MMA_K):
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descA),
                                                        self.A_smem.ptr_to(
                                                            [ks, 0, ki * self.MMA_K]
                                                        ),
                                                        ldo=1,
                                                        sdo=8 * self.BLK_K * F8_BYTES // F128_BYTES,
                                                        swizzle=3,
                                                    )
                                                    T.ptx.tcgen05.encode_matrix_descriptor(
                                                        T.address_of(descB),
                                                        self.B_smem.ptr_to(
                                                            [ks, 0, ki * self.MMA_K]
                                                        ),
                                                        ldo=1,
                                                        sdo=8 * self.BLK_K * F8_BYTES // F128_BYTES,
                                                        swizzle=3,
                                                    )
                                                    if self.stage == 0 and ki == 0:
                                                        T.ptx.tcgen05.mma.block_scale(
                                                            tmem_idx * self.MMA_N,
                                                            descA,
                                                            descB,
                                                            self.SFA_TMEM_START_COL
                                                            + tmem_idx * self.BLK_SFA // 32,
                                                            self.SFB_TMEM_START_COL
                                                            + tmem_idx * self.BLK_SFB // 32,
                                                            descI,
                                                            d_dtype="float32",
                                                            a_dtype=self.a_type,
                                                            b_dtype=self.b_type,
                                                            sfa_dtype=self.sfa_type,
                                                            sfb_dtype=self.sfb_type,
                                                            use_a_tmem=False,
                                                            cta_group=self.CTA_GROUP,
                                                            enable_input_d=0,
                                                        )
                                                    else:
                                                        T.ptx.tcgen05.mma.block_scale(
                                                            tmem_idx * self.MMA_N,
                                                            descA,
                                                            descB,
                                                            self.SFA_TMEM_START_COL
                                                            + tmem_idx * self.BLK_SFA // 32,
                                                            self.SFB_TMEM_START_COL
                                                            + tmem_idx * self.BLK_SFB // 32,
                                                            descI,
                                                            d_dtype="float32",
                                                            a_dtype=self.a_type,
                                                            b_dtype=self.b_type,
                                                            sfa_dtype=self.sfa_type,
                                                            sfb_dtype=self.sfb_type,
                                                            use_a_tmem=False,
                                                            cta_group=self.CTA_GROUP,
                                                            enable_input_d=1,
                                                        )
                                                self.mma2tma_bar.arrive(ks)

                                            @Tx.inline
                                            def mma_epilogue1():
                                                self.mma2ld_bar.arrive(tmem_idx)

                                            @Tx.inline
                                            def mma_epilogue2(ks):
                                                self.trans2mma_bar.wait(ks, self.phase[0])
                                                self.mma2tma_bar.arrive(ks)

                                            partitioned_loop(mma, mma_epilogue1, mma_epilogue2)
                                    tile_scheduler.next_tile()
            if wg_id == 0:
                with T.warpgroup():
                    T.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
                    tmem_idx = T.local_scalar("int32")
                    tmem_phase = T.local_scalar("int32")
                    self.phase[0] = 0
                    while tile_scheduler.valid():
                        m_idx = T.meta_var(tile_scheduler.m_idx)
                        n_idx = T.meta_var(tile_scheduler.n_idx)
                        tmem_idx = tile_scheduler.tile_idx % self.TMEM_PIPE_DEPTH
                        tmem_phase = tile_scheduler.tile_idx // self.TMEM_PIPE_DEPTH & 1
                        if tid_in_wg == 0:
                            T.ptx.cp_async.bulk.wait_group(0)
                        T.cuda.warpgroup_sync(10)
                        self.mma2ld_bar.wait(tmem_idx, tmem_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        for ko in T.unroll(self.MMA_N // self.EPI_TILE):
                            stage = (
                                tile_scheduler.tile_idx * self.MMA_N // self.EPI_TILE + ko
                            ) % self.TMEM_PIPE_DEPTH
                            if ko >= self.TMEM_PIPE_DEPTH:
                                if tid_in_wg == 0:
                                    T.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                                T.cuda.warpgroup_sync(10)
                            for ki in T.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                                reg_wg = self.reg.view(
                                    128,
                                    self.TMEM_LD_SIZE,
                                    layout=TileLayout(
                                        S[(128, self.TMEM_LD_SIZE) : (128, self.TMEM_LD_SIZE, 1)]
                                    ),
                                )
                                col_st = T.meta_var(
                                    tmem_idx * self.MMA_N
                                    + ko * self.EPI_TILE
                                    + ki * self.TMEM_LD_SIZE
                                )
                                Tx.copy(
                                    reg_wg[:, :], self.tmem[:, col_st : col_st + self.TMEM_LD_SIZE]
                                )
                                with T.thread():
                                    st = T.meta_var(ki * self.TMEM_LD_SIZE)
                                    Tx.cast(self.reg_fp16[st : st + self.TMEM_LD_SIZE], self.reg[:])
                                    Tx.copy(
                                        self.output_smem[
                                            stage,
                                            warp_id * 32 + lane_id,
                                            st : st + self.TMEM_LD_SIZE,
                                        ],
                                        self.reg_fp16[st : st + self.TMEM_LD_SIZE],
                                    )
                            if ko == self.MMA_N // self.EPI_TILE - 1:
                                T.ptx.tcgen05.fence.before_thread_sync()
                                self.ld2mma_bar.arrive(tmem_idx)
                            T.ptx.fence.proxy_async("shared::cta")
                            T.cuda.warpgroup_sync(10)
                            m_start = (m_idx * self.CTA_GROUP + cbx) * self.BLK_M
                            n_start = n_idx * self.CTA_GROUP * self.BLK_N + ko * self.EPI_TILE
                            if tid_in_wg == 0:
                                with T.thread():
                                    Tx.copy_async(
                                        output[
                                            m_start : m_start + self.BLK_M,
                                            n_start : n_start + self.EPI_TILE,
                                        ],
                                        self.output_smem[stage, :, :],
                                        dispatch="tma",
                                    )
                                    T.ptx.cp_async.bulk.commit_group()
                        tile_scheduler.next_tile()
                    if tid_in_wg == 0:
                        T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

    def run(self, m_idx, n_idx, k_idx, cbx, A, B, output, tile_scheduler, SFA, SFB, profiler=None):
        self._alloc_local(m_idx)
        self._run(m_idx, n_idx, k_idx, cbx, A, B, output, tile_scheduler, SFA, SFB, profiler)
        self.smem_manager.advance()


class FP8GroupGemmTile(FP8GemmTile):
    def set_moe_info(self, expert_ids):
        self.expert_ids = expert_ids

    @Tx.inline
    def _copy_async_B_from_gmem(self, ks, B, n_start, k_start, tma_copy, m_idx):
        eid = self.expert_ids[m_idx]
        Tx.copy_async(
            self.B_smem[ks, :, :],
            B[eid, n_start : n_start + self.BLK_N, k_start : k_start + self.BLK_K],
            **tma_copy,
        )

    @Tx.inline
    def _copy_async_SFB_from_gmem(self, ks, SFB, n_start_sf, tma_copy, m_idx):
        eid = self.expert_ids[m_idx]
        Tx.copy_async(
            self.SFB_smem_2d[ks, 0 : self.BLK_N * self.CTA_GROUP],
            SFB[eid, self.stage // 4, n_start_sf : n_start_sf + self.BLK_N * self.CTA_GROUP],
            **tma_copy,
        )

    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        cbx,
        A,
        B,
        output,
        tile_scheduler,
        SFA,
        SFB,
        expert_ids,
        profiler=None,
    ):
        self.set_moe_info(expert_ids)
        super().run(m_idx, n_idx, k_idx, cbx, A, B, output, tile_scheduler, SFA, SFB, profiler)
