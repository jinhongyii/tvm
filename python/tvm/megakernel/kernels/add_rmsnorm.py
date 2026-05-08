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
from tvm.megakernel.utils.utils import ceildiv, find_power_of_two, rsqrt
from tvm.script import tirx as Tx


class AddRMSNormTile(Tile):
    vec_size = 16 // F16_BYTES
    loop_inner = 1
    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx

    # inplace add rms norm
    def __init__(self, rms_norm_eps, hidden_size):
        super().__init__()
        self.EPS = rms_norm_eps
        self.hidden_size = hidden_size

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.x_smem = smem_manager.alloc([self.loop_inner * self.hidden_size], dtype="float32")
        self.sum_sq_smem = smem_manager.alloc([self.loop_inner, self.bdy], dtype="float32")

    def _alloc_local(self):
        # alloc local memory
        self.input_vec = Tx.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.residual_vec = Tx.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.weight_vec = Tx.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.input_vec_f32 = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.residual_vec_f32 = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.weight_vec_f32 = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.x_vec = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.x_tmp = Tx.alloc_local([self.loop_inner, 1], "float32")
        self.sum_sq = Tx.alloc_local([self.loop_inner, 1], "float32")
        self.rms_norm = Tx.alloc_local([1], "float32")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    # fmt: off
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, input, residual, weight, output=None, out_residual=None):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            warp_id_in_cta = Tx.warp_id([self.bdy])
            lane_id = Tx.lane_id([self.bdx])
            output_buf = Tx.meta_var(input if output is None else output)
            out_residual_buf = Tx.meta_var(residual if out_residual is None else out_residual)
            # add & sum square
            self._alloc_local()
            with Tx.thread():
                self.smem_manager.wait_all("cta")
                Tx.fill(self.sum_sq[:, 0], 0.0)
                self.rms_norm[0] = 0.0
                vs, nt = self.vec_size, KernelConfig.NUM_THREADS
                for ki in Tx.serial(ceildiv(self.hidden_size, vs * nt * self.loop_inner)):
                    Tx.fill(self.input_vec[:, :], 0.0)
                    Tx.fill(self.residual_vec[:, :], 0.0)
                    Tx.fill(self.x_vec[:, :], 0.0)
                    st = Tx.meta_var((ki * nt + tid) * vs * self.loop_inner)

                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(self.input_vec[kl, :], input[m_idx, lo : lo + vs], vec_len=vs)
                            Tx.copy(self.residual_vec[kl, :],
                                   residual[m_idx, lo : lo + vs], vec_len=vs)

                    for kl in Tx.unroll(self.loop_inner):
                        Tx.cast(self.input_vec_f32[kl, :], self.input_vec[kl, :])
                        Tx.cast(self.residual_vec_f32[kl, :], self.residual_vec[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        for kv in Tx.unroll(self.vec_size):
                            self.x_tmp[kl, 0] = (
                                self.input_vec_f32[kl, kv] + self.residual_vec_f32[kl, kv]
                            )
                            self.sum_sq[kl, 0] += self.x_tmp[kl, 0] * self.x_tmp[kl, 0]
                            self.residual_vec[kl, kv] = Tx.cast(self.x_tmp[kl, 0], "float16")
                            self.x_vec[kl, kv] = self.x_tmp[kl, 0]
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(out_residual_buf[m_idx, lo : lo + vs],
                                   self.residual_vec[kl, :], vec_len=vs)
                            Tx.copy(self.x_smem[lo : lo + vs], self.x_vec[kl, :], vec_len=vs)

                # warp reduce sum
                for kl in Tx.unroll(self.loop_inner):
                    for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                        self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + Tx.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                        )
                Tx.copy(self.sum_sq_smem[:, warp_id_in_cta], self.sum_sq[:, 0])
                Tx.tvm_storage_sync("shared")

                # reduce sum through different warps
                if warp_id_in_cta == 0:
                    for kl in Tx.unroll(self.loop_inner):
                        if lane_id < self.bdy:
                            self.sum_sq[kl, 0] = self.sum_sq_smem[kl, lane_id]
                        else:
                            self.sum_sq[kl, 0] = 0.0
                    for kl in Tx.unroll(self.loop_inner):
                        for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + Tx.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                            )
                    Tx.copy(self.sum_sq_smem[:, 0], self.sum_sq[:, 0])
                Tx.tvm_storage_sync("shared")

                # rms norm
                for kl in Tx.unroll(self.loop_inner):
                    self.rms_norm[0] += self.sum_sq_smem[kl, 0]
                self.rms_norm[0] = rsqrt(self.rms_norm[0] / self.hidden_size + self.EPS)

                # handle the weight
                for ki in Tx.serial(ceildiv(self.hidden_size, vs * nt)):
                    Tx.fill(self.input_vec[:, :], 0.0)
                    Tx.fill(self.weight_vec_f32[:, :], 0.0)
                    Tx.fill(self.x_vec[:, :], 0.0)
                    st = Tx.meta_var((ki * nt + tid) * vs * self.loop_inner)
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(self.weight_vec[kl, :], weight[lo : lo + vs], vec_len=vs)
                            Tx.copy(self.x_vec[kl, :], self.x_smem[lo : lo + vs], vec_len=vs)
                    for kl in Tx.unroll(self.loop_inner):
                        Tx.cast(self.weight_vec_f32[kl, :], self.weight_vec[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        Tx.mul(self.weight_vec_f32[kl, :],
                               self.weight_vec_f32[kl, :], self.rms_norm[0:1])
                        Tx.mul(self.input_vec_f32[kl, :],
                               self.x_vec[kl, :], self.weight_vec_f32[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        Tx.cast(self.input_vec[kl, :], self.input_vec_f32[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(output_buf[m_idx, lo : lo + vs],
                                   self.input_vec[kl, :], vec_len=vs)

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on


class RMSNormTile(Tile):
    vec_size = 16 // F16_BYTES
    loop_inner = 1
    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx

    # inplace add rms norm
    def __init__(self, rms_norm_eps, hidden_size):
        super().__init__()
        self.EPS = rms_norm_eps
        self.hidden_size = hidden_size

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.x_smem = smem_manager.alloc([self.loop_inner * self.hidden_size], "float32")
        self.sum_sq_smem = smem_manager.alloc([self.loop_inner, self.bdy], "float32")

    def _alloc_local(self):
        # alloc local memory
        self.input_vec = Tx.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.weight_vec = Tx.alloc_local([self.loop_inner, self.vec_size], "float16")
        self.input_vec_f32 = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.weight_vec_f32 = Tx.alloc_local([self.loop_inner, self.vec_size], "float32")
        self.sum_sq = Tx.alloc_local([self.loop_inner, 1], "float32")
        self.rms_norm = Tx.alloc_local([1], "float32")

    @Tx.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    # fmt: off
    @Tx.inline
    def run(self, m_idx, n_idx, k_idx, output, input, weight):
        with Tx.cta():
            tid = Tx.thread_id([KernelConfig.NUM_THREADS])
            warp_id_in_cta = Tx.warp_id([self.bdy])
            lane_id = Tx.lane_id([self.bdx])
            # add & sum square
            self._alloc_local()
            with Tx.thread():
                self.smem_manager.wait_all("cta")
                Tx.fill(self.sum_sq[:, 0], 0.0)
                self.rms_norm[0] = 0.0
                vs, nt = self.vec_size, KernelConfig.NUM_THREADS
                for ki in Tx.serial(ceildiv(self.hidden_size, vs * nt * self.loop_inner)):
                    Tx.fill(self.input_vec_f32[:, :], 0.0)
                    st = Tx.meta_var((ki * nt + tid) * vs * self.loop_inner)
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(self.input_vec_f32[kl, :],
                                   input[m_idx, lo : lo + vs], vec_len=vs)
                    for kl in Tx.unroll(self.loop_inner):
                        for kv in Tx.unroll(vs):
                            self.sum_sq[kl, 0] += (
                                self.input_vec_f32[kl, kv] * self.input_vec_f32[kl, kv]
                            )
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(self.x_smem[lo : lo + vs],
                                   self.input_vec_f32[kl, :], vec_len=vs)

                # warp reduce sum
                for kl in Tx.unroll(self.loop_inner):
                    for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                        self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + Tx.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                        )
                Tx.copy(self.sum_sq_smem[:, warp_id_in_cta], self.sum_sq[:, 0])
                Tx.tvm_storage_sync("shared")

                # reduce sum through different warps
                if warp_id_in_cta == 0:
                    for kl in Tx.unroll(self.loop_inner):
                        if lane_id < self.bdy:
                            self.sum_sq[kl, 0] = self.sum_sq_smem[kl, lane_id]
                        else:
                            self.sum_sq[kl, 0] = 0.0
                    for kl in Tx.unroll(self.loop_inner):
                        for kr in Tx.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + Tx.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                            )
                    Tx.copy(self.sum_sq_smem[:, 0], self.sum_sq[:, 0])
                Tx.tvm_storage_sync("shared")

                # rms norm
                for kl in Tx.unroll(self.loop_inner):
                    self.rms_norm[0] += self.sum_sq_smem[kl, 0]
                self.rms_norm[0] = rsqrt(self.rms_norm[0] / self.hidden_size + self.EPS)

                # handle the weight
                for ki in Tx.serial(ceildiv(self.hidden_size, vs * nt)):
                    Tx.fill(self.input_vec_f32[:, :], 0.0)
                    st = Tx.meta_var((ki * nt + tid) * vs * self.loop_inner)
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(self.weight_vec[kl, :], weight[lo : lo + vs], vec_len=vs)
                            Tx.copy(self.input_vec_f32[kl, :],
                                   self.x_smem[lo : lo + vs], vec_len=vs)
                    for kl in Tx.unroll(self.loop_inner):
                        Tx.cast(self.weight_vec_f32[kl, :], self.weight_vec[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        Tx.mul(self.weight_vec_f32[kl, :],
                               self.weight_vec_f32[kl, :], self.rms_norm[0:1])
                        Tx.mul(self.input_vec_f32[kl, :],
                               self.input_vec_f32[kl, :], self.weight_vec_f32[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        Tx.cast(self.input_vec[kl, :], self.input_vec_f32[kl, :])
                    for kl in Tx.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * vs:
                            lo = st + kl * vs
                            Tx.copy(output[m_idx, lo : lo + vs], self.input_vec[kl, :], vec_len=vs)

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on
