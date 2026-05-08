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

import tvm
from tvm.megakernel.utils.base import SmemManager, Tile
from tvm.megakernel.utils.config import F16_BYTES, F32_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


def upcast_size(dtype):
    return 128 // tvm.DataType(dtype).bits


def int_var(val=None):
    buf = Tx.alloc_local([1], "int32")
    if val is not None:
        Tx.buffer_store(buf, val, 0)
    return buf


def float_var(val=None):
    buf = Tx.alloc_local([1], "float32")
    if val is not None:
        Tx.buffer_store(buf, val, 0)
    return buf


def size_of(dtype):
    return tvm.DataType(dtype).bits // 8


def ptx_exp2(x):
    func_name = "tvm_builtin_ptx_exp2"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def ptx_log2(x):
    func_name = "tvm_builtin_ptx_log2"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def ptx_rcp(x):
    func_name = "tvm_builtin_ptx_rcp"
    source_code = f"""
__forceinline__ __device__ float {func_name}(float x) {{
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def half_to_float(x):
    func_name = "tvm_builtin_half_to_float"
    source_code = f"""
__device__ __forceinline__ float {func_name}(half x) {{
  return __half2float(x);
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


def fdivdef(x, y):
    func_name = "tvm_builtin_fdivdef"
    source_code = f"""
__device__ __forceinline__ float {func_name}(float x, float y) {{
  return __fdividef(x, y);
}}
"""
    return Tx.cuda.func_call(func_name, x, y, source_code=source_code, return_type="float32")


@Tx.inline
def cast_load(v, vec_len, buf, *indices):
    with Tx.thread():
        v_tmp = Tx.alloc_local([vec_len], buf.dtype)
        for i in Tx.vectorized(vec_len):
            buffer_load = Tx.meta_var(Tx.BufferLoad(buf, (*indices[:-1], indices[-1] + i)))
            v_tmp[i] = buffer_load
        Tx.cast(v[:], v_tmp[:])


@Tx.inline
def cast_store(v, vec_len, buf, *indices):
    with Tx.thread():
        v_tmp = Tx.alloc_local([vec_len], buf.dtype)
        Tx.cast(v_tmp[:], v[:])
        for i in Tx.vectorized(vec_len):
            Tx.buffer_store(buf, v_tmp[i], (*indices[:-1], indices[-1] + i))


class BatchMergeTile(Tile):
    num_warps = KernelConfig.NUM_THREADS // 32
    num_smem_stages = 1
    num_workers = num_warps * KernelConfig.SM_NUMBER
    max_num_kv_splits = 4 * KernelConfig.SM_NUMBER * 2 * 16

    class State:
        def __init__(self, vec_size):
            self.vec_size = vec_size
            self.o = Tx.alloc_local([vec_size], "float32")
            self.m = Tx.alloc_local([1], "float32")
            self.d = Tx.alloc_local([1], "float32")
            self.INF = 5e4

        @Tx.inline
        def init(self):
            with Tx.thread():
                for i in Tx.unroll(self.vec_size):
                    self.o[i] = 0.0
                self.m[0] = -self.INF
                self.d[0] = 1.0

        # fmt: off
        @Tx.inline
        def merge(self, other_o, other_m, other_d):
            with Tx.thread():
                m_prev = float_var(val=self.m[0])
                d_prev = float_var(val=self.d[0])
                self.m[0] = Tx.max(m_prev[0], other_m)
                self.d[0] = d_prev[0] * ptx_exp2(m_prev[0] - self.m[0]) + other_d * ptx_exp2(other_m - self.m[0])  # noqa: E501
                for i in Tx.unroll(self.vec_size):
                    self.o[i] = self.o[i] * ptx_exp2(m_prev[0] - self.m[0]) + other_o[i] * ptx_exp2(other_m - self.m[0])  # noqa: E501
        # fmt: on

        # fmt: off
        @Tx.inline
        def normalize(self):
            with Tx.thread():
                for i in Tx.unroll(self.vec_size):
                    self.o[i] = fdivdef(self.o[i], self.d[0])
        # fmt: on

        def get_lse(self):
            return self.m[0] + ptx_log2(self.d[0])

    def _warp_sync(self):
        func_name = "tvm_builtin_warp_sync"
        source_code = f"""
__device__ __forceinline__ void {func_name}() {{
__syncwarp();
}}
"""
        return Tx.cuda.func_call(func_name, source_code=source_code)

    def __init__(self, head_dim, num_key_value_heads, num_attention_heads):
        super().__init__()
        self.head_dim = head_dim
        self.kv_heads = num_key_value_heads
        self.qo_heads = num_attention_heads
        self.gqa_group_size = self.qo_heads // self.kv_heads
        self.vec_size = max(16 // F16_BYTES, self.head_dim // 32)
        self.bdx = self.head_dim // self.vec_size
        assert KernelConfig.NUM_THREADS % self.bdx == 0
        self.bdy = 32 // self.bdx
        self.smem_size = (
            self.num_warps * self.num_smem_stages * self.bdy * self.head_dim * F16_BYTES
            + KernelConfig.NUM_THREADS * F32_BYTES
        )

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.v_smem = smem_manager.alloc(
            [self.num_warps, self.num_smem_stages, self.bdy, self.head_dim], "float16"
        )
        self.s_smem = smem_manager.alloc([self.num_warps, 32], "float32")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @Tx.inline
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        partial_o_tvm,
        final_o_tvm,
        partial_lse_tvm,
        num_qo_len_tvm,
        merge_indptr_tvm,
        merge_o_indices_tvm,
    ):
        with Tx.cta():
            warp_id = Tx.warp_id([self.num_warps])
            lane_id = Tx.lane_id([32])

            with Tx.thread():
                self.smem_manager.wait_all("cta")

                tx = int_var(val=lane_id % self.bdx)
                ty = int_var(val=lane_id // self.bdx)
                worker_id = int_var(val=m_idx * self.num_warps + warp_id)
                num_qo_len_local = int_var(val=num_qo_len_tvm[0])
                if worker_id[0] < num_qo_len_local[0] * self.kv_heads:
                    with Tx.thread():
                        self._warp_sync()
                        packed_qo_idx = int_var(val=Tx.floormod(worker_id[0], num_qo_len_local[0]))
                        kv_head_idx = int_var(val=Tx.floordiv(worker_id[0], num_qo_len_local[0]))
                        qo_head_idx = int_var(
                            val=Tx.floormod(packed_qo_idx[0], self.gqa_group_size)
                        )

                        partial_idx_to_offset = Tx.meta_var(
                            lambda off: (
                                (merge_indptr_tvm[packed_qo_idx[0]] + off) * self.kv_heads
                                + kv_head_idx[0]
                            )
                        )
                        merge_idx_to_offset = Tx.meta_var(
                            (merge_o_indices_tvm[packed_qo_idx[0]] * self.kv_heads + kv_head_idx[0])
                            * self.gqa_group_size
                            + qo_head_idx[0]
                        )

                        state = Tx.meta_var(self.State(self.vec_size))
                        state.init()

                        num_index_sets = int_var(
                            val=merge_indptr_tvm[packed_qo_idx[0] + 1]
                            - merge_indptr_tvm[packed_qo_idx[0]]
                        )

                        # prelogue
                        for it in range(self.num_smem_stages):
                            with Tx.thread():
                                Tx.ptx.cp_async(
                                    self.v_smem.ptr_to([warp_id, it, ty[0], tx[0] * self.vec_size]),
                                    partial_o_tvm.ptr_to(
                                        [
                                            partial_idx_to_offset(it * self.bdy + ty[0])
                                            * self.head_dim
                                            + tx[0] * self.vec_size
                                        ]
                                    ),
                                    cp_size=16,
                                    prefetch_size=128,
                                    predicate=it * self.bdy + ty[0] < num_index_sets[0],
                                )
                                Tx.ptx.cp_async.commit_group()

                        for it in Tx.serial(ceildiv(num_index_sets[0], self.bdy)):
                            with Tx.thread():
                                if it % self.bdx == 0:
                                    self.s_smem[warp_id, ty[0] * self.bdx + tx[0]] = (
                                        Tx.if_then_else(
                                            it * self.bdy + (ty[0] * self.bdx + tx[0])
                                            < num_index_sets[0],
                                            partial_lse_tvm[
                                                partial_idx_to_offset(
                                                    it * self.bdy + ty[0] * self.bdx + tx[0]
                                                )
                                            ],
                                            Tx.float32(0),
                                        )
                                    )
                                    self._warp_sync()
                                Tx.ptx.cp_async.wait_group(self.num_smem_stages - 1)
                                self._warp_sync()

                                v = Tx.alloc_local([self.vec_size], "float32")
                                cast_load(
                                    v,
                                    self.vec_size,
                                    self.v_smem,
                                    warp_id,
                                    it % self.num_smem_stages,
                                    ty[0],
                                    tx[0] * self.vec_size,
                                )

                                if it * self.bdy + ty[0] < num_index_sets[0]:
                                    s = float_var(
                                        val=self.s_smem[warp_id, (it % self.bdx) * self.bdy + ty[0]]
                                    )
                                    state.merge(v, s[0], 1)
                                self._warp_sync()

                                Tx.ptx.cp_async(
                                    self.v_smem.ptr_to(
                                        [
                                            warp_id,
                                            (it % self.num_smem_stages),
                                            ty[0],
                                            tx[0] * self.vec_size,
                                        ]
                                    ),
                                    partial_o_tvm.ptr_to(
                                        [
                                            partial_idx_to_offset(
                                                (it + self.num_smem_stages) * self.bdy + ty[0]
                                            )
                                            * self.head_dim
                                            + tx[0] * self.vec_size
                                        ]
                                    ),
                                    cp_size=16,
                                    prefetch_size=128,
                                    predicate=(it + self.num_smem_stages) * self.bdy + ty[0]
                                    < num_index_sets[0],
                                )
                                Tx.ptx.cp_async.commit_group()

                        Tx.ptx.cp_async.wait_group(0)
                        self._warp_sync()

                        @Tx.inline
                        def warp_sync_state():
                            cast_store(
                                state.o,
                                self.vec_size,
                                self.v_smem,
                                warp_id,
                                0,
                                ty[0],
                                tx[0] * self.vec_size,
                            )
                            self.s_smem[warp_id, ty[0]] = state.get_lse()
                            state.init()
                            self._warp_sync()

                            for it in Tx.unroll(self.bdy):
                                with Tx.thread():
                                    s = float_var(val=self.s_smem[warp_id, it])
                                    v = Tx.alloc_local([self.vec_size], "float32")
                                    cast_load(
                                        v,
                                        self.vec_size,
                                        self.v_smem,
                                        warp_id,
                                        0,
                                        it,
                                        tx[0] * self.vec_size,
                                    )
                                    state.merge(v, s[0], 1)

                        state.normalize()
                        if self.bdy > 1:
                            warp_sync_state()
                            state.normalize()

                        final_o_buf_2d = final_o_tvm.view(-1, self.head_dim)
                        cast_store(
                            state.o,
                            self.vec_size,
                            final_o_buf_2d,
                            merge_idx_to_offset,
                            tx[0] * self.vec_size,
                        )

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
