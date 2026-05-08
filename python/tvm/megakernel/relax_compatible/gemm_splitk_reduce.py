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

from tvm.megakernel.kernels import SplitKReduceTile
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


class FuseSplitKReduceTile(SplitKReduceTile):
    @staticmethod
    def get_func(batch_size, N, dtype, split_k_factor):
        symbolic_tile = FuseSplitKReduceTile(batch_size, N, dtype, split_k_factor)
        num_tiles = (symbolic_tile.M_split, ceildiv(N, symbolic_tile.N_TILE), 1)
        tile_size = (symbolic_tile.M_TILE, symbolic_tile.N_TILE, 1)

        @Tx.prim_func(tirx=True, private=True)
        def split_k_reduce_func(
            input_ptr: Tx.handle,
            output_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "split_k_reduce"})
            batch_size = Tx.int32()
            input = Tx.match_buffer(
                input_ptr, (split_k_factor, batch_size, N), "float32", layout="default"
            )
            output = Tx.match_buffer(output_ptr, (batch_size, N), dtype, layout="default")
            split_k_reduce_tile = FuseSplitKReduceTile(batch_size, N, dtype, split_k_factor)
            with Tx.cta():
                split_k_reduce_tile.init(None)
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    split_k_reduce_tile.run(m_idx, n_idx, k_idx, input, output)

        return split_k_reduce_func, num_tiles, tile_size
