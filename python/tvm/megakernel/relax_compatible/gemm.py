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

from tvm.megakernel.kernels import GateUpSiluTile
from tvm.megakernel.kernels.gemm import BarLD2MMA, BarMMA2LD, BarMMA2TMA, BarTMA2MMA, GemmTile
from tvm.megakernel.utils.base import SmemManager
from tvm.megakernel.utils.config import KernelConfig
from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane


class FuseGemmTile(GemmTile):
    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager):
        # alloc shared memory
        GemmTile.tmem_addr = smem_manager.alloc([1], "uint32", method="persistent")
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        GemmTile.tile_idx = Tx.alloc_scalar("int32", scope="local.persistent")
        GemmTile.phase = Tx.alloc_buffer((1,), "int32", scope="local.persistent")
        GemmTile.tmem = Tx.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem.persistent",
            allocated_addr=0,
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
        )

    @staticmethod
    def get_func(
        N,
        K,
        ab_dtype,
        out_dtype,
        BLK_M,
        prefetch_on=False,
        split_k_factor=1,
        use_tma_reduce=False,
        A_dim=2,
    ):
        symbolic_tile = FuseGemmTile(
            N,
            K,
            ab_dtype,
            ab_dtype,
            split_k_factor,
            BLK_M,
            BLK_M,
            prefetch_on=prefetch_on,
            use_tma_reduce=use_tma_reduce,
            profiler_on=False,
        )
        num_tiles = (1, symbolic_tile.N // symbolic_tile.BLK_N, split_k_factor)
        tile_size = (None, symbolic_tile.BLK_N, symbolic_tile.TILE_K)

        @Tx.inline
        def gemm_body(A, B, output, m_idx, n_idx, k_idx):
            gemm_tile = FuseGemmTile(
                N,
                K,
                ab_dtype,
                ab_dtype,
                split_k_factor,
                BLK_M,
                BLK_M,
                prefetch_on=prefetch_on,
                use_tma_reduce=use_tma_reduce,
                profiler_on=False,
            )
            gemm_tile.host_init()
            with Tx.cta():
                buf = Tx.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = SmemManager(
                    KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True
                )
                smem_manager.set_tile(gemm_tile)
                gemm_tile.init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.persistent.init": True})
                    gemm_tile.class_init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.prefetch": True})
                    if prefetch_on:
                        gemm_tile.prefetch(m_idx, n_idx, k_idx, A, B, output, None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    gemm_tile.run(m_idx, n_idx, k_idx, A, B, output, None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.persistent.finalize": True})
                    gemm_tile.class_finalize()

        if split_k_factor == 1:
            assert A_dim == 2

            @Tx.prim_func(tirx=True, private=True)
            def gemm_func_no_split_k(
                A_ptr: Tx.handle,
                B_ptr: Tx.handle,
                C_ptr: Tx.handle,
                m_idx: Tx.int32,
                n_idx: Tx.int32,
                k_idx: Tx.int32,
            ):
                Tx.func_attr({"megakernel.device_func": "gemm"})
                m = Tx.int32()
                A = Tx.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
                B = Tx.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                C = Tx.match_buffer(C_ptr, (m, N), out_dtype, layout="default")
                gemm_body(A, B, C, m_idx, n_idx, k_idx)

            return gemm_func_no_split_k, num_tiles, tile_size

        else:
            if A_dim == 2:
                if use_tma_reduce:

                    @Tx.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(
                        A_ptr: Tx.handle,
                        B_ptr: Tx.handle,
                        output_ptr: Tx.handle,
                        m_idx: Tx.int32,
                        n_idx: Tx.int32,
                        k_idx: Tx.int32,
                    ):
                        Tx.func_attr({"megakernel.device_func": "gemm"})
                        m = Tx.int32()
                        A = Tx.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
                        B = Tx.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        output = Tx.match_buffer(output_ptr, (m, N), out_dtype, layout="default")
                        gemm_body(A, B, output, m_idx, n_idx, k_idx)

                    return gemm_func_split_k, num_tiles, tile_size
                else:

                    @Tx.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(
                        A_ptr: Tx.handle,
                        B_ptr: Tx.handle,
                        partial_sum_ptr: Tx.handle,
                        m_idx: Tx.int32,
                        n_idx: Tx.int32,
                        k_idx: Tx.int32,
                    ):
                        Tx.func_attr({"megakernel.device_func": "gemm"})
                        m = Tx.int32()
                        A = Tx.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
                        B = Tx.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        partial_sum = Tx.match_buffer(
                            partial_sum_ptr, (split_k_factor, m, N), out_dtype, layout="default"
                        )
                        gemm_body(A, B, partial_sum, m_idx, n_idx, k_idx)

                    return gemm_func_split_k, num_tiles, tile_size
            else:
                assert A_dim == 3
                if use_tma_reduce:

                    @Tx.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(
                        A_ptr: Tx.handle,
                        B_ptr: Tx.handle,
                        output_ptr: Tx.handle,
                        m_idx: Tx.int32,
                        n_idx: Tx.int32,
                        k_idx: Tx.int32,
                    ):
                        Tx.func_attr({"megakernel.device_func": "gemm"})
                        m = Tx.int32()
                        K1 = Tx.int32()
                        K2 = Tx.int32()
                        A = Tx.match_buffer(A_ptr, (m, K1, K2), ab_dtype, layout="default")
                        B = Tx.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        output = Tx.match_buffer(output_ptr, (m, N), out_dtype, layout="default")
                        gemm_body(A.view(m, -1), B, output, m_idx, n_idx, k_idx)

                    return gemm_func_split_k, num_tiles, tile_size
                else:

                    @Tx.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(
                        A_ptr: Tx.handle,
                        B_ptr: Tx.handle,
                        partial_sum_ptr: Tx.handle,
                        m_idx: Tx.int32,
                        n_idx: Tx.int32,
                        k_idx: Tx.int32,
                    ):
                        Tx.func_attr({"megakernel.device_func": "gemm"})
                        m = Tx.int32()
                        K1 = Tx.int32()
                        K2 = Tx.int32()
                        A = Tx.match_buffer(A_ptr, (m, K1, K2), ab_dtype, layout="default")
                        B = Tx.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        partial_sum = Tx.match_buffer(
                            partial_sum_ptr, (split_k_factor, m, N), out_dtype, layout="default"
                        )
                        gemm_body(A.view(m, -1), B, partial_sum, m_idx, n_idx, k_idx)

                    return gemm_func_split_k, num_tiles, tile_size


class FuseGateUpSiluTile(GateUpSiluTile):
    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager):
        # alloc shared memory
        GemmTile.tmem_addr = smem_manager.alloc([1], "uint32", method="persistent")
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        GemmTile.tile_idx = Tx.alloc_scalar("int32", scope="local.persistent")
        GemmTile.phase = Tx.alloc_buffer((1,), "int32", scope="local.persistent")
        GemmTile.tmem = Tx.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem.persistent",
            allocated_addr=0,
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
        )

    @staticmethod
    def get_func(N, K, ab_dtype, out_dtype, BLK_M, prefetch_on=False):
        symbolic_tile = FuseGateUpSiluTile(
            N, K, ab_dtype, ab_dtype, 1, BLK_M, BLK_M, prefetch_on=prefetch_on, profiler_on=False
        )
        num_tiles = (1, symbolic_tile.N // symbolic_tile.BLK_N, 1)
        tile_size = (None, symbolic_tile.BLK_N, symbolic_tile.TILE_K)

        @Tx.prim_func(tirx=True, private=True)
        def gate_up_silu_func(
            A_ptr: Tx.handle,
            B_ptr: Tx.handle,
            C_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "gemm"})
            m = Tx.int32()
            A = Tx.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
            B = Tx.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
            output = Tx.match_buffer(C_ptr, (m, N // 2), out_dtype, layout="default")
            gemm_tile = FuseGateUpSiluTile(
                N,
                K,
                ab_dtype,
                ab_dtype,
                1,
                BLK_M,
                BLK_M,
                prefetch_on=prefetch_on,
                profiler_on=False,
            )
            gemm_tile.host_init()
            with Tx.cta():
                buf = Tx.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = SmemManager(
                    KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True
                )
                smem_manager.set_tile(gemm_tile)
                gemm_tile.init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.persistent.init": True})
                    gemm_tile.class_init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.prefetch": True})
                    if prefetch_on:
                        gemm_tile.prefetch(m_idx, n_idx, k_idx, A, B, output, None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    gemm_tile.run(m_idx, n_idx, k_idx, A, B, output, None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.persistent.finalize": True})
                    gemm_tile.class_finalize()

        return gate_up_silu_func, num_tiles, tile_size
