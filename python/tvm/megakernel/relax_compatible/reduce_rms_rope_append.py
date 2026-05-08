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

from tvm.megakernel.kernels import (
    SplitKReduceAppendVTile,
    SplitKReduceRMSnormRopeAppendKTile,
    SplitKReduceRMSnormRopeQTile,
)
from tvm.script import tirx as Tx


class FuseSplitKReduceRMSnormRopeQTile(SplitKReduceRMSnormRopeQTile):
    @staticmethod
    def get_func(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor):
        symbolic_tile = FuseSplitKReduceRMSnormRopeQTile(
            batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor
        )
        num_tiles = (symbolic_tile.m_split, qo_heads, 1)
        tile_size = (symbolic_tile.m_tile, 1, None)

        @Tx.prim_func(tirx=True, private=True)
        def split_k_reduce_rms_rope_func(
            partial_ptr: Tx.handle,
            q_weight_ptr: Tx.handle,
            rope_pos_ptr: Tx.handle,
            cos_sin_cache_ptr: Tx.handle,
            qkv_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "reduce_rms_rope"})
            batch_size = Tx.int32()
            cos_sin_cache_len = Tx.int32()
            partial = Tx.match_buffer(
                partial_ptr,
                [split_k_factor, batch_size, (qo_heads + 2 * kv_heads) * head_dim],
                "float32",
                scope="global",
            )
            qkv = Tx.match_buffer(
                qkv_ptr, [batch_size, qo_heads + 2 * kv_heads, head_dim], "float16", scope="global"
            )
            q_rms_weight = Tx.match_buffer(q_weight_ptr, [head_dim], "float16", scope="global")
            rope_pos = Tx.match_buffer(
                rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1
            )
            cos_sin_cache = Tx.match_buffer(
                cos_sin_cache_ptr, [cos_sin_cache_len, head_dim], "float32", scope="global"
            )
            reduce_rms_rope_tile = FuseSplitKReduceRMSnormRopeQTile(
                batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor
            )
            with Tx.cta():
                reduce_rms_rope_tile.init(None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    reduce_rms_rope_tile.run(
                        m_idx, n_idx, k_idx, partial, qkv, q_rms_weight, rope_pos, cos_sin_cache
                    )

        return split_k_reduce_rms_rope_func, num_tiles, tile_size


class FuseSplitKReduceRMSnormRopeAppendKTile(SplitKReduceRMSnormRopeAppendKTile):
    @staticmethod
    def get_func(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, page_size):
        symbolic_tile = FuseSplitKReduceRMSnormRopeAppendKTile(
            batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, page_size
        )
        num_tiles = (symbolic_tile.m_split, kv_heads, 1)
        tile_size = (symbolic_tile.m_tile, 1, None)

        @Tx.prim_func(tirx=True, private=True)
        def split_k_reduce_rms_rope_append_func(
            partial_ptr: Tx.handle,
            k_weight_ptr: Tx.handle,
            rope_pos_ptr: Tx.handle,
            cos_sin_cache_ptr: Tx.handle,
            append_pos_ptr: Tx.handle,
            kv_cache_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "reduce_rms_rope_append"})
            max_page_num = Tx.int32()
            batch_size = Tx.int32()
            cos_sin_cache_len = Tx.int32()
            partial = Tx.match_buffer(
                partial_ptr,
                [split_k_factor, batch_size, (qo_heads + 2 * kv_heads) * head_dim],
                "float32",
                scope="global",
            )
            k_rms_weight = Tx.match_buffer(k_weight_ptr, [head_dim], "float16", scope="global")
            rope_pos = Tx.match_buffer(
                rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1
            )
            cos_sin_cache = Tx.match_buffer(
                cos_sin_cache_ptr, [cos_sin_cache_len, head_dim], "float32", scope="global"
            )
            append_pos = Tx.match_buffer(
                append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1
            )
            kv_cache = Tx.match_buffer(
                kv_cache_ptr,
                [max_page_num, 2, kv_heads, page_size, head_dim],
                "float16",
                scope="global",
            )
            reduce_rms_rope_append_tile = FuseSplitKReduceRMSnormRopeAppendKTile(
                batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, page_size
            )
            with Tx.cta():
                reduce_rms_rope_append_tile.init(None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    reduce_rms_rope_append_tile.run(
                        m_idx,
                        n_idx,
                        k_idx,
                        partial,
                        k_rms_weight,
                        rope_pos,
                        cos_sin_cache,
                        append_pos,
                        kv_cache,
                    )

        return split_k_reduce_rms_rope_append_func, num_tiles, tile_size


class FuseSplitKReduceAppendVTile(SplitKReduceAppendVTile):
    @staticmethod
    def get_func(batch_size, qo_heads, kv_heads, head_dim, split_k_factor, page_size):
        symbolic_tile = FuseSplitKReduceAppendVTile(
            batch_size, kv_heads, qo_heads, head_dim, split_k_factor, page_size
        )
        num_tiles = (symbolic_tile.m_split, kv_heads, 1)
        tile_size = (symbolic_tile.m_tile, 1, None)

        @Tx.prim_func(tirx=True, private=True)
        def split_k_reduce_append_func(
            partial_ptr: Tx.handle,
            kv_cache_ptr: Tx.handle,
            append_pos_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "reduce_append"})
            max_page_num = Tx.int32()
            batch_size = Tx.int32()
            partial = Tx.match_buffer(
                partial_ptr,
                [split_k_factor, batch_size, (qo_heads + 2 * kv_heads) * head_dim],
                "float32",
                scope="global",
            )
            append_pos = Tx.match_buffer(
                append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1
            )
            kv_cache = Tx.match_buffer(
                kv_cache_ptr,
                [max_page_num, 2, kv_heads, page_size, head_dim],
                "float16",
                scope="global",
            )
            reduce_rms_rope_append_tile = FuseSplitKReduceAppendVTile(
                batch_size, kv_heads, qo_heads, head_dim, split_k_factor, page_size
            )
            with Tx.cta():
                reduce_rms_rope_append_tile.init(None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    reduce_rms_rope_append_tile.run(
                        m_idx, n_idx, k_idx, partial, kv_cache, append_pos
                    )

        return split_k_reduce_append_func, num_tiles, tile_size
