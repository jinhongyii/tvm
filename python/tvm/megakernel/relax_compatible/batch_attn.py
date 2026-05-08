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

from tvm.megakernel.kernels import BatchAttnTile
from tvm.megakernel.utils.base import SmemManager
from tvm.megakernel.utils.config import KernelConfig
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


class FuseBatchAttnTile(BatchAttnTile):
    @staticmethod
    def get_func(page_size, qo_heads, kv_heads, head_dim, attn_task_num, prefetch_on=False):
        num_tiles = (
            Tx.min(ceildiv(attn_task_num, KernelConfig.WG_NUMBER), KernelConfig.SM_NUMBER),
            1,
            1,
        )
        tile_size = (None, None, None)

        @Tx.prim_func(tirx=True, private=True)
        def batch_attn_func(
            q_ptr: Tx.handle,
            kv_ptr: Tx.handle,
            q_indptr_ptr: Tx.handle,
            kv_indptr_ptr: Tx.handle,
            partial_indptr_ptr: Tx.handle,
            kv_indices_ptr: Tx.handle,
            q_len_ptr: Tx.handle,
            kv_len_ptr: Tx.handle,
            q_start_ptr: Tx.handle,
            kv_start_ptr: Tx.handle,
            kv_end_ptr: Tx.handle,
            kv_head_idx_ptr: Tx.handle,
            work_indptr_ptr: Tx.handle,
            len_kv_chunk_ptr: Tx.handle,
            o_ptr: Tx.handle,
            partial_o_ptr: Tx.handle,
            partial_lse_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "batch_attn"})
            batch_size = Tx.int32()
            max_page_num = Tx.int32()
            total_page_num = Tx.int32()
            q = Tx.match_buffer(
                q_ptr, (batch_size, qo_heads + 2 * kv_heads, head_dim), "float16", scope="global"
            )
            kv = Tx.match_buffer(
                kv_ptr, (max_page_num, 2, kv_heads, page_size, head_dim), "float16", scope="global"
            )
            q_indptr = Tx.match_buffer(
                q_indptr_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            kv_indptr = Tx.match_buffer(
                kv_indptr_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            partial_indptr = Tx.match_buffer(
                partial_indptr_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            kv_indices = Tx.match_buffer(
                kv_indices_ptr, (total_page_num), "int32", scope="global", offset_factor=1
            )
            q_len = Tx.match_buffer(
                q_len_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            kv_len = Tx.match_buffer(
                kv_len_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            q_start = Tx.match_buffer(
                q_start_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            kv_start = Tx.match_buffer(
                kv_start_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            kv_end = Tx.match_buffer(
                kv_end_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            kv_head_idx = Tx.match_buffer(
                kv_head_idx_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            work_indptr = Tx.match_buffer(
                work_indptr_ptr,
                (FuseBatchAttnTile.max_total_num_workers),
                "int32",
                scope="global",
                offset_factor=1,
            )
            len_kv_chunk = Tx.match_buffer(
                len_kv_chunk_ptr, (2), "int32", scope="global", offset_factor=1
            )
            o = Tx.match_buffer(o_ptr, (batch_size, qo_heads, head_dim), "float16", scope="global")
            partial_o = Tx.match_buffer(
                partial_o_ptr,
                (FuseBatchAttnTile.max_num_kv_splits * kv_heads * head_dim),
                "float32",
                scope="global",
            )
            partial_lse = Tx.match_buffer(
                partial_lse_ptr,
                (FuseBatchAttnTile.max_num_kv_splits * kv_heads),
                "float32",
                scope="global",
            )
            attn_tile = FuseBatchAttnTile(
                page_size, qo_heads, kv_heads, head_dim, prefetch_on=prefetch_on, profiler_on=False
            )
            with Tx.cta():
                buf = Tx.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = SmemManager(
                    KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True
                )
                smem_manager.set_tile(attn_tile)
                attn_tile.init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.prefetch": True})
                    attn_tile.prefetch(
                        m_idx,
                        n_idx,
                        k_idx,
                        q,
                        kv,
                        q_indptr,
                        kv_indptr,
                        partial_indptr,
                        kv_indices,
                        q_len,
                        kv_len,
                        q_start,
                        kv_start,
                        kv_end,
                        kv_head_idx,
                        work_indptr,
                        len_kv_chunk,
                        o,
                        partial_o,
                        partial_lse,
                        None,
                    )
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    attn_tile.run(
                        m_idx,
                        n_idx,
                        k_idx,
                        q,
                        kv,
                        q_indptr,
                        kv_indptr,
                        partial_indptr,
                        kv_indices,
                        q_len,
                        kv_len,
                        q_start,
                        kv_start,
                        kv_end,
                        kv_head_idx,
                        work_indptr,
                        len_kv_chunk,
                        o,
                        partial_o,
                        partial_lse,
                        None,
                    )

        return batch_attn_func, num_tiles, tile_size
