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
from tvm.megakernel.utils.config import KernelConfig, ProfileEventType
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx
from tvm.tirx.bench import CudaProfiler


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


class BatchAttnTile(Tile):
    inf = 5e4

    num_mma_q = 1
    num_mma_kv = 2

    num_warps_q = 1
    num_warps_kv = 4
    num_warps = num_warps_q * num_warps_kv
    cta_tile_q = 16
    cta_tile_kv = 128
    kv_thr_layout_col = 8
    kv_thr_layout_row = 4
    num_stages = 1
    assert KernelConfig.WARP_NUMBER == 4
    assert num_mma_kv * 4 % num_warps_q == 0
    assert (
        num_mma_kv * kv_thr_layout_col // 2 // num_warps_q
        == cta_tile_kv // 4 // kv_thr_layout_row
        == num_mma_kv * 4 // num_warps_q
        == num_mma_kv * num_warps_kv
    )
    assert num_warps_kv * cta_tile_q * 2 == num_warps * num_mma_q * 16 * 2
    max_total_num_workers = 1025
    max_num_kv_splits = 4 * KernelConfig.SM_NUMBER * 2 * 16

    def get_permuted_offset(self, stride, i, j):
        return i * stride + (j ^ (i % 8))

    def get_warp_idx_q(self, tid):
        if self.num_warps_q == 1:
            return 0
        else:
            return tid[1]

    def get_warp_idx_kv(self, tid):
        if self.num_warps_kv == 1:
            return 0
        else:
            return tid[2]

    def advance_offset_by_column(self, step_size: int, offset, step_idx: int):
        if not (step_size == 2 or step_size == 4 or step_size % 8 == 0):
            raise ValueError(f"Unsupported step_size {step_size} for K128B mode")

        if step_size == 2:
            return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + ((step_idx % 4 == 3) * 8)
        elif step_size == 4:
            return (offset ^ 0x4) + ((step_idx % 2 == 1) * 8)
        else:  # This condition implies step_size % 8 == 0
            return offset + step_size

    def advance_offset_by_row(self, step_size: int, row_stride: int, offset):
        if not (step_size == 4 or step_size % 8 == 0):
            raise ValueError(f"Unsupported step_size: {step_size}. Must be 4 or a multiple of 8.")
        if step_size % 8 == 0:
            return offset + step_size * row_stride
        return (offset ^ 0x4) + step_size * row_stride

    def scope_sync(self, wg_id):
        return Tx.ptx.bar.sync(6 + wg_id, 128)
        # return Tx.tvm_storage_sync("shared")

    def m16k16_row_sum_f16f16f32(self, C_ptr, A_ptr):
        func_name = "m16k16_rowsum_f16f16f32"
        source_code = f"""
__device__ __forceinline__ void {func_name}(float* d, half* s) {{
uint32_t* s_u32 = (uint32_t*)(s);
asm volatile(
"{{\\n"
"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
"{{%0,  _,  %1,  _}}, "
"{{%2,  %3,  %4,  %5}}, "
"{{%6,  %7}}, "
"{{%8,  0.,  %9,  0.}};\\n"
"}}\\n"
: "=f"(d[0]), "=f"(d[1])
: "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(0x3C003C00),
    "r"(0x3C003C00), "f"(d[0]), "f"(d[1]));
}}
"""
        return Tx.cuda.func_call(func_name, C_ptr, A_ptr, source_code=source_code)

    def store_128b(self, dst_ptr, src_ptr):
        func_name = "store_128b"
        source_code = f"""
__device__ __forceinline__ void {func_name}(void* dst_ptr, void* src_ptr) {{
using b128_t = uint4;
b128_t* dst_ptr_b128 = reinterpret_cast<b128_t*>(dst_ptr);
b128_t* src_ptr_b128 = reinterpret_cast<b128_t*>(src_ptr);
*dst_ptr_b128 = *src_ptr_b128;
}}
"""
        return Tx.cuda.func_call(func_name, dst_ptr, src_ptr, source_code=source_code)

    def get_sm_scale(self):
        func_name = "get_sm_scale"
        source_code = f"""
__device__ __forceinline__ float {func_name}() {{
return 1.44269504088896340736 * 1 / sqrtf({self.head_dim});
}}
"""
        return Tx.cuda.func_call(func_name, source_code=source_code, return_type="float32")

    def __init__(
        self, page_size, qo_heads, kv_heads, head_dim, prefetch_on=False, profiler_on=False
    ):
        super().__init__()
        self.page_size = page_size
        self.qo_heads = qo_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.gqa_group_size = self.qo_heads // self.kv_heads
        assert self.qo_heads % self.kv_heads == 0
        self.num_mma_d_qk = self.head_dim // 16
        self.num_mma_d_vo = self.head_dim // 16
        self.upcast_stride_q = self.head_dim // upcast_size("float16")
        self.upcast_stride_k = self.head_dim // upcast_size("float16")
        self.upcast_stride_v = self.head_dim // upcast_size("float16")
        self.upcast_stride_o = self.head_dim // upcast_size("float16")
        assert (
            self.num_warps_kv * self.cta_tile_q * self.head_dim
            == self.num_warps * self.num_mma_q * self.num_mma_d_vo * 32 * 8
        )
        self.prefetch_on = prefetch_on
        self.profiler_on = profiler_on

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # allocate smem
        self.q_smem = smem_manager.alloc(
            [KernelConfig.WG_NUMBER * self.cta_tile_q * self.head_dim], "float16", align=16
        )
        self.k_smem = smem_manager.alloc(
            [KernelConfig.WG_NUMBER * self.cta_tile_kv * self.head_dim], "float16", align=16
        )
        self.v_smem = smem_manager.alloc(
            [KernelConfig.WG_NUMBER * self.cta_tile_kv * self.head_dim], "float16", align=16
        )
        self.cta_sync_o_smem = smem_manager.alloc(
            [KernelConfig.WG_NUMBER, 1]
            if self.num_warps_kv == 1
            else [KernelConfig.WG_NUMBER, self.num_warps, self.num_mma_q, self.num_mma_d_vo, 32, 8],
            "float32",
            align=16,
        )
        self.cta_sync_md_smem = smem_manager.alloc(
            [KernelConfig.WG_NUMBER, 1]
            if self.num_warps_kv == 1
            else [KernelConfig.WG_NUMBER, self.num_warps, self.num_mma_q, 16, 2],
            "float32",
            align=16,
        )
        self.smem_o = smem_manager.alloc(
            [KernelConfig.WG_NUMBER * self.cta_tile_q * self.head_dim], "float16", align=16
        )

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @Tx.inline
    def page_produce_kv(
        self, produce_v: bool, kv_idx_base_in, smem_offset, smem, kv_tvm, warp_id, lane_id
    ):
        v_offset = self.kv_heads * self.page_size * self.head_dim if produce_v else 0
        fill_mode = Tx.meta_var("zero" if produce_v else "")
        NUM_MMA_D = Tx.meta_var(self.num_mma_d_qk if produce_v else self.num_mma_d_vo)
        UPCAST_STRIDE = Tx.meta_var(self.upcast_stride_v if produce_v else self.upcast_stride_k)
        with Tx.thread():
            kv_idx_base = int_var(val=kv_idx_base_in)
            kv_idx = int_var(val=kv_idx_base[0] + warp_id * 4 + lane_id // 8)
            kv_buf_1d = kv_tvm.view(-1)
            # unroll
            for i in Tx.unroll(self.num_mma_kv * 4 // self.num_warps_q):
                for j in Tx.unroll(NUM_MMA_D // (8 // size_of("float16"))):
                    Tx.ptx.cp_async(
                        smem.ptr_to([smem_offset[0] * upcast_size("float16")]),
                        # TODO: optimize the addr computation here
                        kv_buf_1d.ptr_to(
                            [
                                v_offset
                                + self.thr_local_kv_offset[i]
                                + 8 * j * upcast_size("float16")
                            ]
                        ),
                        cp_size=16,
                        prefetch_size=128,
                        fill_mode=fill_mode,
                        predicate=kv_idx[0] < self.kv_len[0],
                    )
                    smem_offset[0] = self.advance_offset_by_column(8, smem_offset[0], j)
                kv_idx[0] += self.num_warps * 4
                smem_offset[0] = (
                    self.advance_offset_by_row(self.num_warps * 4, UPCAST_STRIDE, smem_offset[0])
                    - size_of("float16") * NUM_MMA_D
                )
            smem_offset[0] -= self.cta_tile_kv * UPCAST_STRIDE

    def _alloc_prelogue(self):
        # allocate register
        self.s_frag = Tx.alloc_local([self.num_mma_q, self.num_mma_kv, 8], "float32", align=0)
        self.o_frag = Tx.alloc_local([self.num_mma_q, self.num_mma_d_vo, 8], "float32", align=16)
        self.m = Tx.alloc_local([self.num_mma_q, 2], "float32")
        self.d = Tx.alloc_local([self.num_mma_q, 2], "float32")
        self.tid = Tx.alloc_local([3], "int32")
        self.q_smem_offset_r = int_var()
        self.k_smem_offset_r = int_var()
        self.v_smem_offset_r = int_var()
        self.k_smem_offset_w = int_var()
        self.v_smem_offset_w = int_var()
        self.thr_local_kv_offset = Tx.alloc_local(
            [self.num_mma_kv * self.kv_thr_layout_col // 2 // self.num_warps_q], "int64"
        )
        self.work_idx = int_var()

        self.q_indptr = int_var()
        self.kv_indptr = int_var()
        self.o_indptr = int_var()
        self.q_len = int_var()
        self.kv_len = int_var()
        self.packed_qo_start = int_var()
        self.kv_start = int_var()
        self.kv_end = int_var()
        self.kv_head_idx = int_var()
        self.len_kv_chunk = int_var()

        self.kv_chunk_idx = int_var()
        self.num_kv_chunks = int_var()
        self.qo_packed_idx_base = int_var()
        self.qo_upperbound = int_var()
        self.kv_tile_idx = int_var()
        self.mast_tile_idx = int_var()
        self.block_iter_base = int_var()
        self.packed_kv_bound = int_var()

    @Tx.inline
    def _init_state(
        self,
        m_idx,
        n_idx,
        k_idx,
        q_tvm,
        kv_tvm,
        q_indptr_tvm,
        kv_indptr_tvm,
        partial_indptr_tvm,
        kv_indices_tvm,
        q_len_tvm,
        kv_len_tvm,
        q_start_tvm,
        kv_start_tvm,
        kv_end_tvm,
        kv_head_idx_tvm,
        work_indptr_tvm,
        len_kv_chunk_tvm,
        o_tvm,
        partial_o_tvm,
        partial_lse_tvm,
    ):
        with Tx.cta():
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id = Tx.warp_id_in_wg([self.num_warps_q * self.num_warps_kv])
            lane_id = Tx.lane_id([32])
            self.q_smem_offset_r = self.get_permuted_offset(
                self.upcast_stride_q,
                wg_id * self.cta_tile_q
                + self.get_warp_idx_q(self.tid) * self.num_mma_q * 16
                + lane_id % 16,
                lane_id // 16,
            )
            self.k_smem_offset_r = self.get_permuted_offset(
                self.upcast_stride_k,
                wg_id * self.cta_tile_kv
                + self.get_warp_idx_kv(self.tid) * self.num_mma_kv * 16
                + 8 * (lane_id // 16)
                + lane_id % 8,
                (lane_id % 16 // 8),
            )
            self.v_smem_offset_r = self.get_permuted_offset(
                self.upcast_stride_v,
                wg_id * self.cta_tile_kv
                + self.get_warp_idx_kv(self.tid) * self.num_mma_kv * 16
                + lane_id % 16,
                lane_id // 16,
            )
            self.k_smem_offset_w = self.get_permuted_offset(
                self.upcast_stride_k,
                wg_id * self.cta_tile_kv
                + warp_id * self.kv_thr_layout_row
                + lane_id // self.kv_thr_layout_col,
                lane_id % self.kv_thr_layout_col,
            )
            self.v_smem_offset_w = self.get_permuted_offset(
                self.upcast_stride_v,
                wg_id * self.cta_tile_kv
                + warp_id * self.kv_thr_layout_row
                + lane_id // self.kv_thr_layout_col,
                lane_id % self.kv_thr_layout_col,
            )
            # get_block_coord
            self.q_indptr = q_indptr_tvm[self.work_idx[0]]
            self.kv_indptr = kv_indptr_tvm[self.work_idx[0]]
            self.o_indptr = partial_indptr_tvm[self.work_idx[0]]
            self.q_len = q_len_tvm[self.work_idx[0]]
            self.kv_len = kv_len_tvm[self.work_idx[0]]
            self.packed_qo_start = q_start_tvm[self.work_idx[0]]
            self.kv_start = kv_start_tvm[self.work_idx[0]]
            self.kv_end = kv_end_tvm[self.work_idx[0]]
            self.kv_head_idx = kv_head_idx_tvm[self.work_idx[0]]
            self.len_kv_chunk = len_kv_chunk_tvm[1]
            self.kv_chunk_idx = ceildiv(self.kv_start[0], self.len_kv_chunk[0])
            self.num_kv_chunks = ceildiv(self.kv_len[0], self.len_kv_chunk[0])
            self.qo_packed_idx_base = (
                self.packed_qo_start[0] + self.get_warp_idx_q(self.tid) * self.num_mma_q * 16
            )
            self.qo_upperbound = Tx.min(
                self.q_len[0],
                ceildiv(self.qo_packed_idx_base[0] + self.cta_tile_q, self.gqa_group_size),
            )
            for i0, i1, i2 in Tx.grid(self.num_mma_q, self.num_mma_d_vo, 8):
                self.o_frag[i0, i1, i2] = Tx.float32(0)
            for i0, i1 in Tx.grid(self.num_mma_q, 2):
                self.m[i0, i1] = Tx.float32(-self.inf)
                self.d[i0, i1] = Tx.float32(0)
            self.kv_tile_idx = (
                ceildiv(self.kv_end[0], self.cta_tile_kv)
                - 1
                - (self.kv_start[0] // self.cta_tile_kv)
            )
            self.mast_tile_idx = self.kv_end[0] // self.cta_tile_kv - (
                self.kv_start[0] // self.cta_tile_kv
            )
            self.block_iter_base = self.kv_indptr[0] * self.page_size + self.kv_start[0]
            self.packed_kv_bound = self.kv_indptr[0] * self.page_size + self.kv_len[0]

    @Tx.inline
    def prefetch(
        self,
        m_idx,
        n_idx,
        k_idx,
        q_tvm,
        kv_tvm,
        q_indptr_tvm,
        kv_indptr_tvm,
        partial_indptr_tvm,
        kv_indices_tvm,
        q_len_tvm,
        kv_len_tvm,
        q_start_tvm,
        kv_start_tvm,
        kv_end_tvm,
        kv_head_idx_tvm,
        work_indptr_tvm,
        len_kv_chunk_tvm,
        o_tvm,
        partial_o_tvm,
        partial_lse_tvm,
        profiler: CudaProfiler = None,
    ):
        with Tx.cta():
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id = Tx.warp_id_in_wg([self.num_warps_q * self.num_warps_kv])
            lane_id = Tx.lane_id([32])

            @Tx.inline
            def prefetch_offset(packed_block_iter_base_in):
                with Tx.thread():
                    packed_block_iter_base = int_var(val=packed_block_iter_base_in)
                    for i in Tx.unroll(self.num_mma_kv * 4 // self.num_warps_q):
                        packed_block_iter = int_var(
                            val=packed_block_iter_base[0]
                            + warp_id * self.kv_thr_layout_row
                            + lane_id // self.kv_thr_layout_col
                            + self.kv_thr_layout_row * self.num_warps_q * self.num_warps_kv * i
                        )
                        page_iter = int_var(val=Tx.floordiv(packed_block_iter[0], self.page_size))
                        entry_idx = int_var(val=Tx.floormod(packed_block_iter[0], self.page_size))
                        mapped_page = Tx.meta_var(
                            Tx.if_then_else(
                                packed_block_iter[0] < self.packed_kv_bound[0],
                                kv_indices_tvm[page_iter[0]],
                                0,
                            )
                        )
                        self.thr_local_kv_offset[i] = kv_tvm.elem_offset_of(
                            [
                                mapped_page,
                                0,
                                self.kv_head_idx[0],
                                entry_idx[0],
                                (lane_id % self.kv_thr_layout_col) * upcast_size("float16"),
                            ]
                        )

            with Tx.thread():
                self._alloc_prelogue()
                self.tid[0] = lane_id
                self.tid[1] = warp_id % self.num_warps_q
                self.tid[2] = warp_id // self.num_warps_q

                attn_task_num = work_indptr_tvm[KernelConfig.WG_NUMBER * KernelConfig.SM_NUMBER]
                self.work_idx[0] = m_idx * KernelConfig.WG_NUMBER + wg_id
                self.smem_manager.wait_all("cta")
                if self.work_idx[0] < attn_task_num:
                    if self.profiler_on:
                        profiler.start(ProfileEventType.ATTN_INIT, lane_id == 0)
                    self._init_state(
                        m_idx,
                        n_idx,
                        k_idx,
                        q_tvm,
                        kv_tvm,
                        q_indptr_tvm,
                        kv_indptr_tvm,
                        partial_indptr_tvm,
                        kv_indices_tvm,
                        q_len_tvm,
                        kv_len_tvm,
                        q_start_tvm,
                        kv_start_tvm,
                        kv_end_tvm,
                        kv_head_idx_tvm,
                        work_indptr_tvm,
                        len_kv_chunk_tvm,
                        o_tvm,
                        partial_o_tvm,
                        partial_lse_tvm,
                    )
                    if self.profiler_on:
                        profiler.end(ProfileEventType.ATTN_INIT, lane_id == 0)
                    prefetch_offset(
                        self.block_iter_base[0] + self.kv_tile_idx[0] * self.cta_tile_kv
                    )

    @Tx.inline
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        q_tvm,
        kv_tvm,
        q_indptr_tvm,
        kv_indptr_tvm,
        partial_indptr_tvm,
        kv_indices_tvm,
        q_len_tvm,
        kv_len_tvm,
        q_start_tvm,
        kv_start_tvm,
        kv_end_tvm,
        kv_head_idx_tvm,
        work_indptr_tvm,
        len_kv_chunk_tvm,
        o_tvm,
        partial_o_tvm,
        partial_lse_tvm,
        profiler: CudaProfiler = None,
    ):
        with Tx.cta():
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id = Tx.warp_id_in_wg([self.num_warps_q * self.num_warps_kv])
            lane_id = Tx.lane_id([32])

            with Tx.thread():
                if not self.prefetch_on:
                    self._alloc_prelogue()
                    self.tid[0] = lane_id
                    self.tid[1] = warp_id % self.num_warps_q
                    self.tid[2] = warp_id // self.num_warps_q

                @Tx.inline
                def prefetch_offset(packed_block_iter_base_in):
                    with Tx.thread():
                        packed_block_iter_base = int_var(val=packed_block_iter_base_in)
                        for i in Tx.unroll(self.num_mma_kv * 4 // self.num_warps_q):
                            packed_block_iter = int_var(
                                val=packed_block_iter_base[0]
                                + warp_id * self.kv_thr_layout_row
                                + lane_id // self.kv_thr_layout_col
                                + self.kv_thr_layout_row * self.num_warps_q * self.num_warps_kv * i
                            )
                            page_iter = int_var(
                                val=Tx.floordiv(packed_block_iter[0], self.page_size)
                            )
                            entry_idx = int_var(
                                val=Tx.floormod(packed_block_iter[0], self.page_size)
                            )
                            mapped_page = Tx.meta_var(
                                Tx.if_then_else(
                                    packed_block_iter[0] < self.packed_kv_bound[0],
                                    kv_indices_tvm[page_iter[0]],
                                    0,
                                )
                            )
                            self.thr_local_kv_offset[i] = kv_tvm.elem_offset_of(
                                [
                                    mapped_page,
                                    0,
                                    self.kv_head_idx[0],
                                    entry_idx[0],
                                    (lane_id % self.kv_thr_layout_col) * upcast_size("float16"),
                                ]
                            )

                @Tx.inline
                def handle_single_task(has_prefetched):
                    @Tx.inline
                    def load_q_global_smem():
                        if self.get_warp_idx_kv(self.tid) == 0:
                            q_smem_offset_w = int_var(
                                val=self.get_permuted_offset(
                                    self.upcast_stride_q,
                                    wg_id * self.cta_tile_q
                                    + self.get_warp_idx_q(self.tid) * self.num_mma_q * 16
                                    + lane_id // 8,
                                    lane_id % 8,
                                )
                            )
                            # unroll
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for j in Tx.unroll(4):
                                    with Tx.thread():
                                        qo_packed_id = Tx.meta_var(
                                            self.qo_packed_idx_base[0]
                                            + lane_id // 8
                                            + mma_q * 16
                                            + j * 4
                                        )
                                        q = int_var(
                                            val=Tx.floordiv(qo_packed_id, self.gqa_group_size)
                                        )
                                        r = int_var(
                                            val=Tx.floormod(qo_packed_id, self.gqa_group_size)
                                        )
                                        for mma_do in Tx.unroll(self.num_mma_d_qk // 4):
                                            Tx.ptx.cp_async(
                                                self.q_smem.ptr_to(
                                                    [q_smem_offset_w[0] * upcast_size("float16")]
                                                ),
                                                # TODO: optimize the addr computation here
                                                q_tvm.ptr_to(
                                                    [
                                                        self.q_indptr[0] + q[0],
                                                        self.kv_head_idx[0] * self.gqa_group_size
                                                        + r[0],
                                                        (lane_id % 8 + mma_do * 8)
                                                        * upcast_size("float16"),
                                                    ]
                                                ),
                                                cp_size=16,
                                                prefetch_size=128,
                                                predicate=q[0] < self.qo_upperbound[0],
                                            )
                                            q_smem_offset_w[0] = self.advance_offset_by_column(
                                                8, q_smem_offset_w[0], mma_do
                                            )
                                    q_smem_offset_w[0] = (
                                        self.advance_offset_by_row(
                                            4, self.upcast_stride_q, q_smem_offset_w[0]
                                        )
                                        - 2 * self.num_mma_d_qk
                                    )

                    if self.profiler_on:
                        profiler.start(ProfileEventType.ATTN_LOAD_Q, lane_id == 0)
                    load_q_global_smem()
                    if self.profiler_on:
                        profiler.end(ProfileEventType.ATTN_LOAD_Q, lane_id == 0)
                    self.scope_sync(wg_id)

                    @Tx.inline
                    def mma_sync_m16n16k16_row_col_f16f16f32(
                        C_in, c_offset, A_in, a_offset, B_in, b_offset, init: bool
                    ):
                        with Tx.thread():
                            C_mma = Tx.decl_buffer(
                                [8], dtype="float32", data=C_in.data, byte_offset=c_offset
                            )
                            A_mma = Tx.decl_buffer(
                                [4], dtype="uint32", data=A_in.data, byte_offset=a_offset
                            )
                            B_mma = Tx.decl_buffer(
                                [4], dtype="uint32", data=B_in.data, byte_offset=b_offset
                            )
                            if init:
                                Tx.ptx.mma(
                                    "m16n8k16",
                                    "row",
                                    "col",
                                    "float32",
                                    "float16",
                                    "float16",
                                    "float32",
                                    C_mma.ptr_to([0]),
                                    A_mma.ptr_to([0]),
                                    B_mma.ptr_to([0]),
                                )
                                Tx.ptx.mma(
                                    "m16n8k16",
                                    "row",
                                    "col",
                                    "float32",
                                    "float16",
                                    "float16",
                                    "float32",
                                    C_mma.ptr_to([4]),
                                    A_mma.ptr_to([0]),
                                    B_mma.ptr_to([2]),
                                )
                            else:
                                Tx.ptx.mma(
                                    "m16n8k16",
                                    "row",
                                    "col",
                                    "float32",
                                    "float16",
                                    "float16",
                                    "float32",
                                    C_mma.ptr_to([0]),
                                    A_mma.ptr_to([0]),
                                    B_mma.ptr_to([0]),
                                    C_mma.ptr_to([0]),
                                )
                                Tx.ptx.mma(
                                    "m16n8k16",
                                    "row",
                                    "col",
                                    "float32",
                                    "float16",
                                    "float16",
                                    "float32",
                                    C_mma.ptr_to([4]),
                                    A_mma.ptr_to([0]),
                                    B_mma.ptr_to([2]),
                                    C_mma.ptr_to([4]),
                                )

                    @Tx.inline
                    def compute_qk():
                        with Tx.thread():
                            a_frag = Tx.alloc_local([self.num_mma_q, 4], "uint32")
                            b_frag = Tx.alloc_local([4], "uint32")

                            # unroll
                            for mma_d in Tx.unroll(self.num_mma_d_qk):
                                for mma_q in Tx.unroll(self.num_mma_q):
                                    Tx.ptx.ldmatrix(
                                        False,
                                        4,
                                        ".b16",
                                        a_frag.ptr_to([mma_q, 0]),
                                        self.q_smem.ptr_to(
                                            [self.q_smem_offset_r[0] * upcast_size("float16")]
                                        ),
                                    )
                                    self.q_smem_offset_r[0] = self.advance_offset_by_row(
                                        16, self.upcast_stride_q, self.q_smem_offset_r[0]
                                    )
                                self.q_smem_offset_r[0] = (
                                    self.advance_offset_by_column(2, self.q_smem_offset_r[0], mma_d)
                                    - self.num_mma_q * 16 * self.upcast_stride_q
                                )
                                for mma_kv in Tx.unroll(self.num_mma_kv):
                                    Tx.ptx.ldmatrix(
                                        False,
                                        4,
                                        ".b16",
                                        b_frag.ptr_to([0]),
                                        self.k_smem.ptr_to(
                                            [self.k_smem_offset_r[0] * upcast_size("float16")]
                                        ),
                                    )
                                    self.k_smem_offset_r[0] = self.advance_offset_by_row(
                                        16, self.upcast_stride_k, self.k_smem_offset_r[0]
                                    )
                                    for mma_q in Tx.unroll(self.num_mma_q):
                                        if mma_d == 0:
                                            mma_sync_m16n16k16_row_col_f16f16f32(
                                                self.s_frag,
                                                self.s_frag.byte_offset_of([mma_q, mma_kv, 0]),
                                                a_frag,
                                                a_frag.byte_offset_of([mma_q, 0]),
                                                b_frag,
                                                b_frag.byte_offset_of([0]),
                                                True,
                                            )
                                        else:
                                            mma_sync_m16n16k16_row_col_f16f16f32(
                                                self.s_frag,
                                                self.s_frag.byte_offset_of([mma_q, mma_kv, 0]),
                                                a_frag,
                                                a_frag.byte_offset_of([mma_q, 0]),
                                                b_frag,
                                                b_frag.byte_offset_of([0]),
                                                False,
                                            )
                                self.k_smem_offset_r[0] = (
                                    self.advance_offset_by_column(2, self.k_smem_offset_r[0], mma_d)
                                    - self.num_mma_kv * 16 * self.upcast_stride_k
                                )
                            self.q_smem_offset_r[0] -= self.num_mma_d_qk * 2
                            self.k_smem_offset_r[0] -= self.num_mma_d_qk * size_of("float16")

                    @Tx.inline
                    def logits_mask():
                        chunk_end = Tx.meta_var(self.kv_end[0])
                        with Tx.thread():
                            # unroll
                            kv_idx_base = int_var(
                                val=self.kv_start[0]
                                + (
                                    self.kv_tile_idx[0] * self.num_warps_kv
                                    + self.get_warp_idx_kv(self.tid)
                                )
                                * self.num_mma_kv
                                * 16
                            )
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for mma_kv in Tx.unroll(self.num_mma_kv):
                                    for reg_id in Tx.unroll(8):
                                        with Tx.thread():
                                            kv_idx = int_var(
                                                val=kv_idx_base[0]
                                                + mma_kv * 16
                                                + 2 * (lane_id % 4)
                                                + 8 * (reg_id // 4)
                                                + reg_id % 2
                                            )
                                            self.s_frag[mma_q, mma_kv, reg_id] = Tx.if_then_else(
                                                Tx.Not(kv_idx[0] >= chunk_end),
                                                self.s_frag[mma_q, mma_kv, reg_id],
                                                Tx.float32(-self.inf),
                                            )

                    @Tx.inline
                    def update_mdo_states():
                        WARP_MASK = Tx.meta_var(0xFFFFFFFF)
                        with Tx.thread():
                            sm_scale = float_var(val=self.get_sm_scale())
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for j in Tx.unroll(2):
                                    m_prev = float_var(val=self.m[mma_q, j])
                                    for mma_kv in Tx.unroll(self.num_mma_kv):
                                        m_local = float_var(
                                            val=Tx.max(
                                                Tx.max(
                                                    self.s_frag[mma_q, mma_kv, j * 2 + 0],
                                                    self.s_frag[mma_q, mma_kv, j * 2 + 1],
                                                ),
                                                Tx.max(
                                                    self.s_frag[mma_q, mma_kv, j * 2 + 4],
                                                    self.s_frag[mma_q, mma_kv, j * 2 + 5],
                                                ),
                                            )
                                        )
                                        self.m[mma_q, j] = Tx.max(self.m[mma_q, j], m_local[0])
                                    self.m[mma_q, j] = Tx.max(
                                        self.m[mma_q, j],
                                        Tx.tvm_warp_shuffle_xor(
                                            WARP_MASK, self.m[mma_q, j], 0x2, 32, 32
                                        ),
                                    )
                                    self.m[mma_q, j] = Tx.max(
                                        self.m[mma_q, j],
                                        Tx.tvm_warp_shuffle_xor(
                                            WARP_MASK, self.m[mma_q, j], 0x1, 32, 32
                                        ),
                                    )

                                    o_scale = float_var(
                                        val=ptx_exp2(
                                            m_prev[0] * sm_scale[0] - self.m[mma_q, j] * sm_scale[0]
                                        )
                                    )
                                    self.d[mma_q, j] *= o_scale[0]
                                    # unroll
                                    for mma_d in Tx.unroll(self.num_mma_d_vo):
                                        self.o_frag[mma_q, mma_d, j * 2 + 0] *= o_scale[0]
                                        self.o_frag[mma_q, mma_d, j * 2 + 1] *= o_scale[0]
                                        self.o_frag[mma_q, mma_d, j * 2 + 4] *= o_scale[0]
                                        self.o_frag[mma_q, mma_d, j * 2 + 5] *= o_scale[0]
                                    # unroll
                                    for mma_kv in Tx.unroll(self.num_mma_kv):
                                        self.s_frag[mma_q, mma_kv, j * 2 + 0] = ptx_exp2(
                                            self.s_frag[mma_q, mma_kv, j * 2 + 0] * sm_scale[0]
                                            - self.m[mma_q, j] * sm_scale[0]
                                        )
                                        self.s_frag[mma_q, mma_kv, j * 2 + 1] = ptx_exp2(
                                            self.s_frag[mma_q, mma_kv, j * 2 + 1] * sm_scale[0]
                                            - self.m[mma_q, j] * sm_scale[0]
                                        )
                                        self.s_frag[mma_q, mma_kv, j * 2 + 4] = ptx_exp2(
                                            self.s_frag[mma_q, mma_kv, j * 2 + 4] * sm_scale[0]
                                            - self.m[mma_q, j] * sm_scale[0]
                                        )
                                        self.s_frag[mma_q, mma_kv, j * 2 + 5] = ptx_exp2(
                                            self.s_frag[mma_q, mma_kv, j * 2 + 5] * sm_scale[0]
                                            - self.m[mma_q, j] * sm_scale[0]
                                        )

                    @Tx.inline
                    def compute_sfm_v():
                        with Tx.thread():
                            s_frag_f16 = Tx.alloc_local(
                                [self.num_mma_q, self.num_mma_kv, 8], "float16"
                            )
                            Tx.cast(s_frag_f16[:, :, :], self.s_frag[:, :, :])
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for mma_kv in Tx.unroll(self.num_mma_kv):
                                    self.m16k16_row_sum_f16f16f32(
                                        self.d.ptr_to([mma_q, 0]),
                                        s_frag_f16.ptr_to([mma_q, mma_kv, 0]),
                                    )
                            for mma_kv in Tx.unroll(self.num_mma_kv):
                                for mma_d in Tx.unroll(self.num_mma_d_vo):
                                    with Tx.thread():
                                        b_frag = Tx.alloc_local([4], "uint32")
                                        Tx.ptx.ldmatrix(
                                            True,
                                            4,
                                            ".b16",
                                            b_frag.ptr_to([0]),
                                            self.v_smem.ptr_to(
                                                [self.v_smem_offset_r[0] * upcast_size("float16")]
                                            ),
                                        )
                                        for mma_q in Tx.unroll(self.num_mma_q):
                                            mma_sync_m16n16k16_row_col_f16f16f32(
                                                self.o_frag,
                                                self.o_frag.byte_offset_of([mma_q, mma_d, 0]),
                                                s_frag_f16,
                                                s_frag_f16.byte_offset_of([mma_q, mma_kv, 0]),
                                                b_frag,
                                                b_frag.byte_offset_of([0]),
                                                False,
                                            )
                                        self.v_smem_offset_r[0] = self.advance_offset_by_column(
                                            2, self.v_smem_offset_r[0], mma_d
                                        )
                                self.v_smem_offset_r[0] = (
                                    self.advance_offset_by_row(
                                        16, self.upcast_stride_v, self.v_smem_offset_r[0]
                                    )
                                    - size_of("float16") * self.num_mma_d_vo
                                )
                            self.v_smem_offset_r[0] -= 16 * self.num_mma_kv * self.upcast_stride_v

                    @Tx.inline
                    def loop_body(WITH_MASK: bool):
                        prefetch_offset(
                            self.block_iter_base[0] + (self.kv_tile_idx[0] - 1) * self.cta_tile_kv
                        )
                        Tx.ptx.cp_async.wait_group(1)
                        self.scope_sync(wg_id)

                        compute_qk()
                        if WITH_MASK:
                            logits_mask()
                        update_mdo_states()

                        self.scope_sync(wg_id)
                        self.page_produce_kv(
                            False,
                            self.kv_start[0] + (self.kv_tile_idx[0] - 1) * self.cta_tile_kv,
                            self.k_smem_offset_w,
                            self.k_smem,
                            kv_tvm,
                            warp_id,
                            lane_id,
                        )
                        Tx.ptx.cp_async.commit_group()
                        Tx.ptx.cp_async.wait_group(1)

                        self.scope_sync(wg_id)
                        compute_sfm_v()
                        self.scope_sync(wg_id)

                        self.page_produce_kv(
                            True,
                            self.kv_start[0] + (self.kv_tile_idx[0] - 1) * self.cta_tile_kv,
                            self.v_smem_offset_w,
                            self.v_smem,
                            kv_tvm,
                            warp_id,
                            lane_id,
                        )
                        Tx.ptx.cp_async.commit_group()

                    if not has_prefetched:
                        prefetch_offset(
                            self.block_iter_base[0] + self.kv_tile_idx[0] * self.cta_tile_kv
                        )
                    self.page_produce_kv(
                        False,
                        self.kv_start[0] + self.kv_tile_idx[0] * self.cta_tile_kv,
                        self.k_smem_offset_w,
                        self.k_smem,
                        kv_tvm,
                        warp_id,
                        lane_id,
                    )
                    Tx.ptx.cp_async.commit_group()
                    self.page_produce_kv(
                        True,
                        self.kv_start[0] + self.kv_tile_idx[0] * self.cta_tile_kv,
                        self.v_smem_offset_w,
                        self.v_smem,
                        kv_tvm,
                        warp_id,
                        lane_id,
                    )
                    Tx.ptx.cp_async.commit_group()

                    if self.profiler_on:
                        profiler.start(ProfileEventType.ATTN_LOOP_BODY, lane_id == 0)
                    while self.kv_tile_idx[0] >= self.mast_tile_idx[0] and self.kv_tile_idx[0] > 0:
                        loop_body(True)
                        self.kv_tile_idx[0] -= 1
                    while self.kv_tile_idx[0] + 1 > self.num_stages:
                        loop_body(False)
                        self.kv_tile_idx[0] -= 1
                    if self.profiler_on:
                        profiler.end(ProfileEventType.ATTN_LOOP_BODY, lane_id == 0)

                    Tx.ptx.cp_async.wait_group(0)
                    self.scope_sync(wg_id)

                    if self.profiler_on:
                        profiler.start(ProfileEventType.ATTN_COMPUTE_QKV, lane_id == 0)
                    while self.kv_tile_idx[0] >= 0:
                        compute_qk()
                        logits_mask()
                        update_mdo_states()
                        compute_sfm_v()
                        self.kv_tile_idx[0] -= 1
                    if self.profiler_on:
                        profiler.end(ProfileEventType.ATTN_COMPUTE_QKV, lane_id == 0)

                    self.scope_sync(wg_id)

                    @Tx.inline
                    def finalize_m():
                        with Tx.thread():
                            sm_scale = float_var(val=self.get_sm_scale())
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for j in Tx.unroll(2):
                                    if self.m[mma_q, j] != -self.inf:
                                        self.m[mma_q, j] *= sm_scale[0]

                    @Tx.inline
                    def threadblock_sync_mdo_states():
                        for mma_q in Tx.unroll(self.num_mma_q):
                            for mma_d in Tx.unroll(self.num_mma_d_vo):
                                Tx.copy(
                                    self.cta_sync_o_smem[wg_id, warp_id, mma_q, mma_d, lane_id, :],
                                    self.o_frag[mma_q, mma_d, :],
                                )

                        for mma_q in Tx.unroll(self.num_mma_q):
                            for j in Tx.unroll(2):
                                with Tx.thread():
                                    md = Tx.alloc_local([2], "float32")
                                    md[0] = self.m[mma_q, j]
                                    md[1] = self.d[mma_q, j]
                                    Tx.copy(
                                        self.cta_sync_md_smem[
                                            wg_id, warp_id, mma_q, j * 8 + lane_id // 4, :
                                        ],
                                        md[:],
                                    )
                        self.scope_sync(wg_id)

                        for mma_q in Tx.unroll(self.num_mma_q):
                            with Tx.thread():
                                o_scale = Tx.alloc_local([2, self.num_warps_kv], "float32")
                                for j in Tx.unroll(2):
                                    with Tx.thread():
                                        m_new = float_var(val=-self.inf)
                                        d_new = float_var(val=1.0)
                                        for i in Tx.unroll(self.num_warps_kv):
                                            with Tx.thread():
                                                md = Tx.alloc_local([2], "float32")
                                                Tx.copy(
                                                    md[:],
                                                    self.cta_sync_md_smem[
                                                        wg_id,
                                                        i * self.num_warps_q
                                                        + self.get_warp_idx_q(self.tid),
                                                        mma_q,
                                                        j * 8 + lane_id // 4,
                                                        :,
                                                    ],
                                                )
                                                m_prev = float_var(val=m_new[0])
                                                d_prev = float_var(val=d_new[0])
                                                m_new[0] = Tx.max(m_new[0], md[0])
                                                d_new[0] = d_prev[0] * ptx_exp2(
                                                    m_prev[0] - m_new[0]
                                                ) + md[1] * ptx_exp2(md[0] - m_new[0])
                                        for i in Tx.unroll(self.num_warps_kv):
                                            with Tx.thread():
                                                md = Tx.alloc_local([2], "float32")
                                                Tx.copy(
                                                    md[:],
                                                    self.cta_sync_md_smem[
                                                        wg_id,
                                                        i * self.num_warps_q
                                                        + self.get_warp_idx_q(self.tid),
                                                        mma_q,
                                                        j * 8 + lane_id // 4,
                                                        :,
                                                    ],
                                                )
                                                o_scale[j, i] = ptx_exp2(md[0] - m_new[0])
                                        self.m[mma_q, j] = m_new[0]
                                        self.d[mma_q, j] = d_new[0]
                                for mma_d in Tx.unroll(self.num_mma_d_vo):
                                    with Tx.thread():
                                        o_new = Tx.alloc_local([8], "float32")
                                        for i in Tx.unroll(8):
                                            o_new[i] = 0.0
                                        for i in Tx.unroll(self.num_warps_kv):
                                            with Tx.thread():
                                                o_i = Tx.alloc_local([8], "float32")
                                                Tx.copy(
                                                    o_i[:],
                                                    self.cta_sync_o_smem[
                                                        wg_id,
                                                        i * self.num_warps_q
                                                        + self.get_warp_idx_q(self.tid),
                                                        mma_q,
                                                        mma_d,
                                                        lane_id,
                                                        :,
                                                    ],
                                                )
                                                for reg_id in Tx.unroll(8):
                                                    o_new[reg_id] += (
                                                        o_i[reg_id] * o_scale[(reg_id % 4) // 2, i]
                                                    )
                                        Tx.copy(self.o_frag[mma_q, mma_d, :], o_new[:])

                    @Tx.inline
                    def normalize_d():
                        with Tx.thread():
                            d_rcp = Tx.alloc_local([self.num_mma_q, 2], "float32")
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for j in Tx.unroll(2):
                                    d_rcp[mma_q, j] = Tx.if_then_else(
                                        self.m[mma_q, j] != -self.inf,
                                        ptx_rcp(self.d[mma_q, j]),
                                        0.0,
                                    )
                            for mma_q in Tx.unroll(self.num_mma_q):
                                for mma_d in Tx.unroll(self.num_mma_d_vo):
                                    for reg_id in Tx.unroll(8):
                                        self.o_frag[mma_q, mma_d, reg_id] *= d_rcp[
                                            mma_q, (reg_id >> 1) & 1
                                        ]

                    @Tx.inline
                    def store_o_to_smem(o_smem):
                        for mma_q in Tx.unroll(self.num_mma_q):
                            for mma_d in Tx.unroll(self.num_mma_d_vo):
                                with Tx.thread():
                                    o_frag_f16 = Tx.alloc_local([8], "float16")
                                    Tx.cast(o_frag_f16[:], self.o_frag[mma_q, mma_d, :])
                                    o_smem_offset_w = int_var(
                                        val=self.get_permuted_offset(
                                            self.upcast_stride_o,
                                            wg_id * self.cta_tile_q
                                            + (
                                                self.get_warp_idx_q(self.tid) * self.num_mma_q
                                                + mma_q
                                            )
                                            * 16
                                            + lane_id % 16,
                                            mma_d * 2 + lane_id // 16,
                                        )
                                    )
                                    Tx.ptx.stmatrix(
                                        o_smem.ptr_to(
                                            [o_smem_offset_w[0] * upcast_size("float16")]
                                        ),
                                        o_frag_f16.ptr_to([0]),
                                        num=4,
                                    )

                    @Tx.inline
                    def write_partial_o(o_ptr_base_offset, o_stride_n):
                        o_packed_idx_base_warp = Tx.meta_var(self.qo_packed_idx_base[0])
                        o_packed_idx_base_cta = Tx.meta_var(self.packed_qo_start[0])
                        o_smem = Tx.meta_var(self.smem_o)
                        warp_id_x = int_var(val=self.get_warp_idx_q(self.tid))
                        warp_id_z = int_var(val=self.get_warp_idx_kv(self.tid))

                        if warp_id_z[0] == 0:
                            with Tx.thread():
                                store_o_to_smem(o_smem)
                                o_smem_offset_w = int_var(
                                    val=self.get_permuted_offset(
                                        self.upcast_stride_o,
                                        wg_id * self.cta_tile_q
                                        + warp_id_x[0] * self.num_mma_q * 16
                                        + lane_id // 8,
                                        lane_id % 8,
                                    )
                                )
                                for mma_q in Tx.unroll(self.num_mma_q):
                                    for j in Tx.unroll(4):
                                        with Tx.thread():
                                            o_packed_idx = int_var(
                                                val=o_packed_idx_base_warp
                                                + lane_id // 8
                                                + mma_q * 16
                                                + j * 4
                                            )
                                            q = int_var(
                                                val=Tx.floordiv(
                                                    o_packed_idx[0], self.gqa_group_size
                                                )
                                            )
                                            int_var(
                                                val=Tx.floormod(
                                                    o_packed_idx[0], self.gqa_group_size
                                                )
                                            )
                                            o_ptr_offset = int_var(
                                                val=o_ptr_base_offset
                                                + (o_packed_idx[0] - o_packed_idx_base_cta)
                                                * o_stride_n
                                                + (lane_id % 8) * upcast_size("float16")
                                            )
                                            for mma_do in Tx.unroll(self.num_mma_d_vo // 4):
                                                if q[0] < self.qo_upperbound[0]:
                                                    self.store_128b(
                                                        partial_o_tvm.ptr_to([o_ptr_offset[0]]),
                                                        o_smem.ptr_to(
                                                            [
                                                                o_smem_offset_w[0]
                                                                * upcast_size("float16")
                                                            ]
                                                        ),
                                                    )
                                                o_ptr_offset[0] += 8 * upcast_size("float16")
                                                o_smem_offset_w[0] = self.advance_offset_by_column(
                                                    8, o_smem_offset_w[0], mma_do
                                                )
                                            o_smem_offset_w[0] = (
                                                self.advance_offset_by_row(
                                                    4, self.upcast_stride_o, o_smem_offset_w[0]
                                                )
                                                - 2 * self.num_mma_d_vo
                                            )

                    @Tx.inline
                    def write_final_o(o_ptr_base_offset):
                        o_packed_idx_base = Tx.meta_var(self.qo_packed_idx_base[0])
                        o_smem = Tx.meta_var(self.smem_o)
                        warp_id_x = int_var(val=self.get_warp_idx_q(self.tid))
                        warp_id_z = int_var(val=self.get_warp_idx_kv(self.tid))

                        if warp_id_z[0] == 0:
                            with Tx.thread():
                                store_o_to_smem(o_smem)
                                o_smem_offset_w = int_var(
                                    val=self.get_permuted_offset(
                                        self.upcast_stride_o,
                                        wg_id * self.cta_tile_q
                                        + warp_id_x[0] * self.num_mma_q * 16
                                        + lane_id // 8,
                                        lane_id % 8,
                                    )
                                )
                                for mma_q in Tx.unroll(self.num_mma_q):
                                    for j in Tx.unroll(4):
                                        with Tx.thread():
                                            o_packed_idx = int_var(
                                                val=o_packed_idx_base
                                                + lane_id // 8
                                                + mma_q * 16
                                                + j * 4
                                            )
                                            q = int_var(
                                                val=Tx.floordiv(
                                                    o_packed_idx[0], self.gqa_group_size
                                                )
                                            )
                                            r = int_var(
                                                val=Tx.floormod(
                                                    o_packed_idx[0], self.gqa_group_size
                                                )
                                            )
                                            o_ptr_offset = int_var(
                                                val=o_ptr_base_offset
                                                + o_tvm.elem_offset_of(
                                                    [
                                                        q[0],
                                                        r[0],
                                                        (lane_id % 8) * upcast_size("float16"),
                                                    ]
                                                )
                                            )
                                            for mma_do in Tx.unroll(self.num_mma_d_vo // 4):
                                                if q[0] < self.qo_upperbound[0]:
                                                    o_buf_1d = o_tvm.view(-1)
                                                    self.store_128b(
                                                        o_buf_1d.ptr_to([o_ptr_offset[0]]),
                                                        o_smem.ptr_to(
                                                            [
                                                                o_smem_offset_w[0]
                                                                * upcast_size("float16")
                                                            ]
                                                        ),
                                                    )
                                                o_ptr_offset[0] += 8 * upcast_size("float16")
                                                o_smem_offset_w[0] = self.advance_offset_by_column(
                                                    8, o_smem_offset_w[0], mma_do
                                                )
                                            o_smem_offset_w[0] = (
                                                self.advance_offset_by_row(
                                                    4, self.upcast_stride_o, o_smem_offset_w[0]
                                                )
                                                - 2 * self.num_mma_d_vo
                                            )

                    @Tx.inline
                    def write_partial_lse():
                        if self.num_kv_chunks[0] > 1:
                            if self.get_warp_idx_kv(self.tid) == 0:
                                for mma_q in Tx.unroll(self.num_mma_q):
                                    for j in Tx.unroll(2):
                                        with Tx.thread():
                                            packed_qo_idx = int_var(
                                                val=self.qo_packed_idx_base[0]
                                                + lane_id // 4
                                                + j * 8
                                                + mma_q * 16
                                            )
                                            q = int_var(
                                                val=Tx.floordiv(
                                                    packed_qo_idx[0], self.gqa_group_size
                                                )
                                            )
                                            int_var(
                                                val=Tx.floormod(
                                                    packed_qo_idx[0], self.gqa_group_size
                                                )
                                            )
                                            if q[0] < self.qo_upperbound[0]:
                                                partial_lse_buf_offset = Tx.meta_var(
                                                    (
                                                        self.o_indptr[0]
                                                        + (
                                                            packed_qo_idx[0]
                                                            - self.packed_qo_start[0]
                                                        )
                                                        * self.num_kv_chunks[0]
                                                        + self.kv_chunk_idx[0]
                                                    )
                                                    * self.kv_heads
                                                    + self.kv_head_idx[0]
                                                )
                                                partial_lse_tvm[partial_lse_buf_offset] = ptx_log2(
                                                    self.d[mma_q, j]
                                                ) + Tx.cast(self.m[mma_q, j], "float32")

                    finalize_m()
                    threadblock_sync_mdo_states()
                    normalize_d()

                    if self.profiler_on:
                        profiler.start(ProfileEventType.ATTN_WRITE_BACK, lane_id == 0)
                    if self.num_kv_chunks[0] > 1:
                        # reuse q, k, v's smem
                        with Tx.thread():
                            o_ptr_base_offset = int_var(
                                val=(
                                    (self.o_indptr[0] + self.kv_chunk_idx[0]) * self.kv_heads
                                    + self.kv_head_idx[0]
                                )
                                * self.head_dim
                            )
                            write_partial_o(
                                o_ptr_base_offset[0],
                                self.num_kv_chunks[0] * self.kv_heads * self.head_dim,
                            )
                    else:
                        with Tx.thread():
                            o_ptr_base_offset = int_var(
                                val=o_tvm.elem_offset_of(
                                    [self.q_indptr[0], self.kv_head_idx[0] * self.gqa_group_size, 0]
                                )
                            )
                            write_final_o(o_ptr_base_offset[0])

                    write_partial_lse()
                    self.scope_sync(wg_id)

                    if self.profiler_on:
                        profiler.end(ProfileEventType.ATTN_WRITE_BACK, lane_id == 0)

                attn_task_num = work_indptr_tvm[KernelConfig.WG_NUMBER * KernelConfig.SM_NUMBER]
                if not self.prefetch_on:
                    self.smem_manager.wait_all("cta")
                self.work_idx[0] = m_idx * KernelConfig.WG_NUMBER + wg_id
                while self.work_idx[0] < attn_task_num:
                    if not self.prefetch_on:
                        if self.profiler_on:
                            profiler.start(ProfileEventType.ATTN_INIT, lane_id == 0)
                        self._init_state(
                            m_idx,
                            n_idx,
                            k_idx,
                            q_tvm,
                            kv_tvm,
                            q_indptr_tvm,
                            kv_indptr_tvm,
                            partial_indptr_tvm,
                            kv_indices_tvm,
                            q_len_tvm,
                            kv_len_tvm,
                            q_start_tvm,
                            kv_start_tvm,
                            kv_end_tvm,
                            kv_head_idx_tvm,
                            work_indptr_tvm,
                            len_kv_chunk_tvm,
                            o_tvm,
                            partial_o_tvm,
                            partial_lse_tvm,
                        )
                        if self.profiler_on:
                            profiler.end(ProfileEventType.ATTN_INIT, lane_id == 0)
                        prefetch_offset(
                            self.block_iter_base[0] + self.kv_tile_idx[0] * self.cta_tile_kv
                        )
                    handle_single_task(self.prefetch_on)
                    self.work_idx[0] += KernelConfig.SM_NUMBER * KernelConfig.WG_NUMBER
                    if self.work_idx[0] < attn_task_num:
                        if self.profiler_on:
                            profiler.start(ProfileEventType.ATTN_INIT, lane_id == 0)
                        self._init_state(
                            m_idx,
                            n_idx,
                            k_idx,
                            q_tvm,
                            kv_tvm,
                            q_indptr_tvm,
                            kv_indptr_tvm,
                            partial_indptr_tvm,
                            kv_indices_tvm,
                            q_len_tvm,
                            kv_len_tvm,
                            q_start_tvm,
                            kv_start_tvm,
                            kv_end_tvm,
                            kv_head_idx_tvm,
                            work_indptr_tvm,
                            len_kv_chunk_tvm,
                            o_tvm,
                            partial_o_tvm,
                            partial_lse_tvm,
                        )
                        if self.profiler_on:
                            profiler.end(ProfileEventType.ATTN_INIT, lane_id == 0)
                        prefetch_offset(
                            self.block_iter_base[0] + self.kv_tile_idx[0] * self.cta_tile_kv
                        )
                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
