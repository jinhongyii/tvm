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

from tvm.megakernel.kernels import BatchMergeTile
from tvm.megakernel.utils.base import SmemManager
from tvm.megakernel.utils.config import KernelConfig
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


class FuseBatchMergeTile(BatchMergeTile):
    @staticmethod
    def get_func(head_dim, attn_task_num, kv_heads, qo_heads, batch_size):
        num_tiles = (
            Tx.if_then_else(
                attn_task_num > kv_heads * batch_size,
                ceildiv(batch_size * qo_heads, KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER),
                0,
            ),
            1,
            1,
        )
        tile_size = (None, None, None)

        @Tx.prim_func(tirx=True, private=True)
        def batch_merge_func(
            partial_o_ptr: Tx.handle,
            partial_lse_ptr: Tx.handle,
            num_qo_len_ptr: Tx.handle,
            merge_indptr_ptr: Tx.handle,
            merge_o_indices_ptr: Tx.handle,
            final_o_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "batch_merge"})
            batch_size = Tx.int32()
            partial_o = Tx.match_buffer(
                partial_o_ptr,
                (FuseBatchMergeTile.max_num_kv_splits * kv_heads * head_dim),
                "float32",
                scope="global",
            )
            final_o = Tx.match_buffer(
                final_o_ptr, (batch_size, qo_heads, head_dim), "float16", scope="global"
            )
            partial_lse = Tx.match_buffer(
                partial_lse_ptr,
                (FuseBatchMergeTile.max_num_kv_splits * kv_heads),
                "float32",
                scope="global",
            )
            num_qo_len = Tx.match_buffer(
                num_qo_len_ptr, (1), "int32", scope="global", offset_factor=1
            )
            merge_indptr = Tx.match_buffer(
                merge_indptr_ptr,
                (FuseBatchMergeTile.max_num_kv_splits),
                "int32",
                scope="global",
                offset_factor=1,
            )
            merge_o_indices = Tx.match_buffer(
                merge_o_indices_ptr,
                (FuseBatchMergeTile.max_num_kv_splits),
                "int32",
                scope="global",
                offset_factor=1,
            )
            batch_merge_tile = FuseBatchMergeTile(head_dim, kv_heads, qo_heads)
            with Tx.cta():
                buf = Tx.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = SmemManager(
                    KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True
                )
                smem_manager.set_tile(batch_merge_tile)
                batch_merge_tile.init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    batch_merge_tile.run(
                        m_idx,
                        n_idx,
                        k_idx,
                        partial_o,
                        final_o,
                        partial_lse,
                        num_qo_len,
                        merge_indptr,
                        merge_o_indices,
                    )

        return batch_merge_func, num_tiles, tile_size
