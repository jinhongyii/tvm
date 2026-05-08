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

from tvm.megakernel.kernels import AddRMSNormTile, RMSNormTile
from tvm.megakernel.utils.base import SmemManager
from tvm.megakernel.utils.config import KernelConfig
from tvm.script import tirx as Tx


class FuseAddRMSNormTile(AddRMSNormTile):
    @staticmethod
    def get_func(batch_size, rms_norm_eps, hidden_size):
        num_tiles = (batch_size, 1, 1)
        tile_size = (1, None, None)

        @Tx.prim_func(tirx=True, private=True)
        def add_rmsnorm_func(
            input_ptr: Tx.handle,
            residual_ptr: Tx.handle,
            weight_ptr: Tx.handle,
            output_ptr: Tx.handle,
            out_residual_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "add_rmsnorm"})
            seq_len = Tx.int32()
            input = Tx.match_buffer(input_ptr, (seq_len, hidden_size), "float16", scope="global")
            residual = Tx.match_buffer(
                residual_ptr, (seq_len, hidden_size), "float16", scope="global"
            )
            weight = Tx.match_buffer(weight_ptr, (hidden_size), "float16", scope="global")
            output = Tx.match_buffer(output_ptr, (seq_len, hidden_size), "float16", scope="global")
            out_residual = Tx.match_buffer(
                out_residual_ptr, (seq_len, hidden_size), "float16", scope="global"
            )
            add_rmsnorm_tile = FuseAddRMSNormTile(rms_norm_eps, hidden_size)
            with Tx.cta():
                buf = Tx.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = SmemManager(
                    KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True
                )
                smem_manager.set_tile(add_rmsnorm_tile)
                add_rmsnorm_tile.init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    add_rmsnorm_tile.run(
                        m_idx, n_idx, k_idx, input, residual, weight, output, out_residual
                    )

        return add_rmsnorm_func, num_tiles, tile_size


class FuseRMSNormTile(RMSNormTile):
    @staticmethod
    def get_func(batch_size, rms_norm_eps, hidden_size):
        num_tiles = (batch_size, 1, 1)
        tile_size = (1, None, None)

        @Tx.prim_func(tirx=True, private=True)
        def rmsnorm_func(
            input_ptr: Tx.handle,
            weight_ptr: Tx.handle,
            output_ptr: Tx.handle,
            m_idx: Tx.int32,
            n_idx: Tx.int32,
            k_idx: Tx.int32,
        ):
            Tx.func_attr({"megakernel.device_func": "rmsnorm"})
            seq_len = Tx.int32()
            output = Tx.match_buffer(output_ptr, (seq_len, hidden_size), "float16", scope="global")
            input = Tx.match_buffer(input_ptr, (seq_len, hidden_size), "float32", scope="global")
            weight = Tx.match_buffer(weight_ptr, (hidden_size), "float16", scope="global")
            rmsnorm_tile = FuseRMSNormTile(rms_norm_eps, hidden_size)
            with Tx.cta():
                buf = Tx.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = SmemManager(
                    KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True
                )
                smem_manager.set_tile(rmsnorm_tile)
                rmsnorm_tile.init(smem_manager)
                with Tx.cta():
                    Tx.attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with Tx.cta():
                    Tx.attr({"tirx.tile_class.run": True})
                    Tx.cuda.thread_fence()
                    rmsnorm_tile.run(m_idx, n_idx, k_idx, output, input, weight)

        return rmsnorm_func, num_tiles, tile_size
