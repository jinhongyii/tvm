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

from tvm.megakernel.utils.base import SmemManager, Tile
from tvm.megakernel.utils.config import F16_BYTES, KernelConfig
from tvm.megakernel.utils.utils import ceildiv
from tvm.script import tirx as Tx


def int_cell(value):
    buf = Tx.local_scalar("int32")
    if value is not None:
        Tx.buffer_store(buf.buffer, value, 0)
    return buf


def get_permuted_offset(stride, i, j):
    return i * stride + (j ^ (i % 8))


def advance_offset_by_column(step_size: int, offset, step_idx: int):
    if not (step_size == 2 or step_size == 4 or step_size % 8 == 0):
        raise ValueError(f"Unsupported step_size {step_size} for K128B mode")

    if step_size == 2:
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + ((step_idx % 4 == 3) * 8)
    elif step_size == 4:
        return (offset ^ 0x4) + ((step_idx % 2 == 1) * 8)
    else:  # This condition implies step_size % 8 == 0
        return offset + step_size


def advance_offset_by_row(step_size: int, row_stride: int, offset):
    if not (step_size == 4 or step_size % 8 == 0):
        raise ValueError(f"Unsupported step_size: {step_size}. Must be 4 or a multiple of 8.")
    if step_size % 8 == 0:
        return offset + step_size * row_stride
    return (offset ^ 0x4) + step_size * row_stride


def half_to_float(x):
    func_name = "tvm_builtin_half_to_float"
    source_code = f"""
__device__ __forceinline__ float {func_name}(half x) {{
return __half2float(x);
}}
"""
    return Tx.cuda.func_call(func_name, x, source_code=source_code, return_type="float32")


# fmt: off
@Tx.inline
def mma_sync_m16n16k16_row_col_f16f16f32(C_in, c_offset, A_in, a_offset, B_in, b_offset, init: bool):  # noqa: E501
    with Tx.thread():
        C_mma = Tx.decl_buffer([8], dtype="float32", data=C_in.data, byte_offset=c_offset)
        A_mma = Tx.decl_buffer([4], dtype="uint32", data=A_in.data, byte_offset=a_offset)
        B_mma = Tx.decl_buffer([4], dtype="uint32", data=B_in.data, byte_offset=b_offset)
        if init:
            Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                    C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]))
            Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                    C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]))
        else:
            Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                    C_mma.ptr_to([0]), A_mma.ptr_to([0]), B_mma.ptr_to([0]), C_mma.ptr_to([0]))
            Tx.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
                    C_mma.ptr_to([4]), A_mma.ptr_to([0]), B_mma.ptr_to([2]), C_mma.ptr_to([4]))
# fmt: on


class GroupGEMMTile(Tile):
    VEC_LEN = 128 // (F16_BYTES * 8)

    def __init__(self, N, K, BLK_M, BLK_N, BLK_K, num_stages):
        self.BLK_M = BLK_M
        self.BLK_N = BLK_N
        self.BLK_K = BLK_K
        self.num_stages = num_stages
        if BLK_M == 16:
            self.NUM_WARPS_M = 1
            self.NUM_WARPS_N = 4
        else:
            self.NUM_WARPS_M = 2
            self.NUM_WARPS_N = 2
        assert self.BLK_M % 16 == 0
        assert self.BLK_N % 16 == 0
        assert self.BLK_K % 16 == 0
        assert N % self.BLK_N == 0
        assert K % self.BLK_K == 0
        assert KernelConfig.WARP_NUMBER == self.NUM_WARPS_M * self.NUM_WARPS_N

        self.MMA_M = self.BLK_M // 16
        self.MMA_N = self.BLK_N // 16
        self.MMA_K = self.BLK_K // 16
        assert self.MMA_M % self.NUM_WARPS_M == 0
        assert self.MMA_N % self.NUM_WARPS_N == 0
        self.NUM_MMA_M = self.MMA_M // self.NUM_WARPS_M
        self.NUM_MMA_N = self.MMA_N // self.NUM_WARPS_N

        self.N_TILE_CNT = ceildiv(N, self.BLK_N)
        self.K_TILE_CNT = ceildiv(K, self.BLK_K)

        self.AB_THR_LAYOUT_COL = min(32, self.BLK_K // self.VEC_LEN)
        self.AB_THR_LAYOUT_ROW = 32 // self.AB_THR_LAYOUT_COL
        assert 32 % self.AB_THR_LAYOUT_COL == 0
        assert self.AB_THR_LAYOUT_COL >= 8

        self.C_THR_LAYOUT_COL = min(32, self.BLK_N // self.VEC_LEN)
        self.C_THR_LAYOUT_ROW = 32 // self.C_THR_LAYOUT_COL
        assert 32 % self.C_THR_LAYOUT_COL == 0
        assert self.C_THR_LAYOUT_COL >= 8

        self.UPCAST_STRIDE_K = self.BLK_K // self.VEC_LEN
        self.UPCAST_STRIDE_N = self.BLK_N // self.VEC_LEN

        # A load
        assert self.BLK_K % (self.AB_THR_LAYOUT_COL * self.VEC_LEN) == 0
        assert self.BLK_M % (self.AB_THR_LAYOUT_ROW * KernelConfig.WARP_NUMBER) == 0

        # B load
        assert self.BLK_K % (self.AB_THR_LAYOUT_COL * self.VEC_LEN) == 0
        assert self.BLK_N % (self.AB_THR_LAYOUT_ROW * KernelConfig.WARP_NUMBER) == 0

        # C store
        assert self.BLK_N % (self.C_THR_LAYOUT_COL * self.VEC_LEN) == 0
        assert self.BLK_M % (self.C_THR_LAYOUT_ROW * KernelConfig.WARP_NUMBER) == 0

        SMEM_SIZE = max(
            num_stages * BLK_M * BLK_K * F16_BYTES + num_stages * BLK_N * BLK_K * F16_BYTES,
            BLK_K * BLK_K * F16_BYTES,
        )
        print(f"SMEM_SIZE: {SMEM_SIZE}")
        assert SMEM_SIZE * KernelConfig.WG_NUMBER <= 232448
        assert SMEM_SIZE % self.VEC_LEN == 0

    def _alloc_buffer(self, smem_manager: SmemManager):
        start_offset = smem_manager.pool_allocator.offset
        smem_manager.alloc([self.num_stages * self.BLK_M * self.BLK_K], "float16", align=16)
        smem_manager.alloc([self.num_stages * self.BLK_N * self.BLK_K], "float16", align=16)
        smem_manager.pool_allocator.move_base_to(start_offset)
        smem_manager.alloc([self.BLK_M * self.BLK_N], "float16", align=16)

    @Tx.inline
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        A,
        B,
        C,
        topk_weights,
        topk_ids,
        sorted_tok_ids,
        expert_ids,
        num_tokens_post_pad,
    ):
        with Tx.cta():
            wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id_in_wg = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
