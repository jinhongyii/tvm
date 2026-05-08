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

from tvm.megakernel.utils.base import KernelConfig, Tile
from tvm.script import tirx as Tx

ld_reduce_8xfp16 = """
__forceinline__ __device__ void ld_reduce_8_fp16(void* src_addr, void* dst_addr) {
    int4* source = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, src_addr);
    int4* dest = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, dst_addr);
    constexpr int UNROLL = 1;
    union {
        uint32_t u4[4 * UNROLL];
        uint16_t u2[8 * UNROLL];
    };
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        asm("multimem.ld_reduce.acquire.sys.global.add.acc::f32.v8.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=h"(u2[8 * u]), "=h"(u2[8 * u + 1]), "=h"(u2[8 * u + 2]), "=h"(u2[8 * u + 3]), "=h"(u2[8 * u + 4]), "=h"(u2[8 * u + 5]), "=h"(u2[8 * u + 6]), "=h"(u2[8 * u + 7])
            : "l"(source));
    }
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        asm("multimem.st.release.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dest), "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]), "r"(u4[4 * u + 3]): "memory");
    }
}
"""  # noqa: E501


class AllreduceTile(Tile):
    M_TILE = 16
    N_TILE = 128

    def __init__(self, world_size):
        super().__init__()
        self.world_size = world_size

    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, input, output):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            rank: Tx.let = Tx.nvshmem.my_pe()
            # restrict tile size so that this will not be iterated over
            if (
                tid < self.M_TILE * self.N_TILE // 8
                and (m_idx * self.M_TILE + (tid // (self.N_TILE // 8))) < input.shape[0]
            ):
                m_start = Tx.meta_var(m_idx * self.M_TILE + (tid // (self.N_TILE // 8)))
                n_start = Tx.meta_var(
                    n_idx * self.N_TILE * self.world_size
                    + rank * self.N_TILE
                    + tid % (self.N_TILE // 8) * 8
                )
                Tx.cuda.func_call(
                    "ld_reduce_8_fp16",
                    input.ptr_to([m_start, n_start]),
                    output.ptr_to([m_start, n_start]),
                    source_code=ld_reduce_8xfp16,
                )
