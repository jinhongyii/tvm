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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.bench import ProtonContext, bench


@tvm.testing.requires_cuda_compute_version(8)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_layernorm(dtype):
    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    if dtype == "bfloat16":
        np_dtype = "bfloat16"
        const_func = Tx.bfloat16
    else:
        assert dtype == "float16"
        np_dtype = np.float16
        const_func = Tx.float16

    ATTN_B, ATTN_N, ATTN_D = 4, 1024, 1024  # attention batch size, seq length, feature dim
    PIPELINE_DEPTH = 2
    NUM_WORKERS = 2
    N_PER_TILE = 2

    # NOTE 1: this kernel is applied on attention output:
    # Step 1. compute dropout(inp) - omitted in inference
    # Step 2. compute x = dropout(inp) + inp_resid (residual connection)
    #         write x to out_resid
    # Step 3. compute layernorm(x)
    #         layernorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    #         write result to out

    # NOTE 2: this kernel follows CTA-level prog style, where we assume all warps
    #         in a CTA participate in each op.

    # fmt: off
    @Tx.prim_func(tirx=True)
    def layernorm(inp_ptr: Tx.handle, inp_resid_ptr: Tx.handle, norm_weight_ptr: Tx.handle, norm_bias_ptr: Tx.handle,  # noqa: E501
                  out_ptr: Tx.handle, out_resid_ptr: Tx.handle) -> None:
        inp = Tx.match_buffer(inp_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=Tx.TileLayout(Tx.S[ATTN_B, 1, ATTN_N, ATTN_D]))  # noqa: E501
        out = Tx.match_buffer(out_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=Tx.TileLayout(Tx.S[ATTN_B, 1, ATTN_N, ATTN_D]))  # noqa: E501
        inp_resid = Tx.match_buffer(inp_resid_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=Tx.TileLayout(Tx.S[ATTN_B, 1, ATTN_N, ATTN_D]))  # noqa: E501
        out_resid = Tx.match_buffer(out_resid_ptr, (ATTN_B, 1, ATTN_N, ATTN_D), dtype, scope="global", layout=Tx.TileLayout(Tx.S[ATTN_B, 1, ATTN_N, ATTN_D]))  # noqa: E501
        norm_weight = Tx.match_buffer(norm_weight_ptr, (ATTN_D,), dtype, scope="global", layout=Tx.TileLayout(Tx.S[ATTN_D])) # gamma  # noqa: E501
        norm_bias = Tx.match_buffer(norm_bias_ptr, (ATTN_D,), dtype, scope="global", layout=Tx.TileLayout(Tx.S[ATTN_D])) # beta  # noqa: E501

        with Tx.kernel():
            bx, by = Tx.cta_id([Tx.ceildiv(ATTN_N, N_PER_TILE), ATTN_B])
            tid = Tx.thread_id([NUM_WORKERS * 32])
            warp_id = Tx.warp_id([NUM_WORKERS])
            lane_id = Tx.lane_id([32])
            with Tx.cta():
                x_smem = Tx.alloc_buffer([NUM_WORKERS, PIPELINE_DEPTH, ATTN_D], dtype, scope="shared", layout=Tx.TileLayout(Tx.S[NUM_WORKERS, PIPELINE_DEPTH, ATTN_D]))  # noqa: E501
                resid_smem = Tx.alloc_buffer([NUM_WORKERS, PIPELINE_DEPTH, ATTN_D], dtype, scope="shared", layout=Tx.TileLayout(Tx.S[NUM_WORKERS, PIPELINE_DEPTH, ATTN_D]))  # noqa: E501
                norm_weight_smem = Tx.alloc_buffer([ATTN_D,], dtype, scope="shared", layout=Tx.TileLayout(Tx.S[ATTN_D]))  # noqa: E501
                norm_bias_smem = Tx.alloc_buffer([ATTN_D,], dtype, scope="shared", layout=Tx.TileLayout(Tx.S[ATTN_D]))  # noqa: E501
                mean = Tx.alloc_buffer([NUM_WORKERS, 1, 1], dtype, scope="shared", layout=Tx.TileLayout(Tx.S[NUM_WORKERS, 1, 1]), align=8)  # noqa: E501
                var = Tx.alloc_buffer([NUM_WORKERS, 1, 1], dtype, scope="shared", layout=Tx.TileLayout(Tx.S[NUM_WORKERS, 1, 1]), align=8)  # noqa: E501

                Tx.copy(norm_bias_smem[:], norm_bias[:])
                Tx.copy(norm_weight_smem[:], norm_weight[:])

                # two cp.async
                non_bulk_copy = Tx.meta_var({"dispatch": "non-bulk-copy"})
                Tx.copy_async(x_smem[:, 0, :], inp[by, 0, slice(bx * NUM_WORKERS, (bx + 1) * NUM_WORKERS), :], **non_bulk_copy)  # noqa: E501
                Tx.copy_async(resid_smem[:, 0, :], inp_resid[by, 0, slice(bx * NUM_WORKERS, (bx + 1) * NUM_WORKERS), :], **non_bulk_copy)  # noqa: E501
                Tx.ptx.cp_async.commit_group()
                Tx.cuda.cta_sync()

                # main loop
                n_loops = Tx.meta_var(Tx.ceildiv(N_PER_TILE, NUM_WORKERS))
                for k in Tx.serial(n_loops):
                    curr_idx = Tx.meta_var((k + 0) * NUM_WORKERS)
                    next_idx = Tx.meta_var((k + 1) * NUM_WORKERS)

                    if k < n_loops - 1:
                        # TODO(@kathy): support pipeline token flip when pipeline is long
                        Tx.copy_async(x_smem[:, 1, :], inp[by, 0, slice(bx * NUM_WORKERS + next_idx, (bx + 1) * NUM_WORKERS + next_idx), :], **non_bulk_copy)  # noqa: E501
                        Tx.copy_async(resid_smem[:, 1, :], inp_resid[by, 0, slice(bx * NUM_WORKERS + next_idx, (bx + 1) * NUM_WORKERS + next_idx), :], **non_bulk_copy)  # noqa: E501
                        Tx.ptx.cp_async.commit_group()
                    Tx.ptx.cp_async.wait_group(0)
                    Tx.cuda.cta_sync()

                    # store residual
                    Tx.add(resid_smem[:, 0, :], resid_smem[:, 0, :], x_smem[:, 0, :])
                    Tx.copy(out_resid[by, 0, slice(bx * NUM_WORKERS + curr_idx, (bx + 1) * NUM_WORKERS + curr_idx), :], resid_smem[:, 0, :])  # noqa: E501
                    # numerator
                    Tx.sum(mean[:, 0, 0], resid_smem[:, 0, :])
                    Tx.fdiv(mean[:, 0, 0], mean[:, 0, 0], const_func(ATTN_D))
                    Tx.sub(resid_smem[:, 0, :], resid_smem[:, 0, :], mean[:, 0, 0])
                    # denominator
                    Tx.mul(x_smem[:, 0, :], resid_smem[:, 0, :], resid_smem[:, 0, :])
                    Tx.sum(var[:, 0, 0], x_smem[:, 0, :])
                    Tx.fdiv(var[:, 0, 0], var[:, 0, 0], const_func(ATTN_D))
                    Tx.add(var[:, 0, 0], var[:, 0, 0], const_func(1e-5))
                    Tx.sqrt(var[:, 0, 0], var[:, 0, 0])
                    # layernorm
                    Tx.fdiv(resid_smem[:, 0, :], resid_smem[:, 0, :], var[:, 0, 0])
                    Tx.mul(resid_smem[:, 0, :], resid_smem[:, 0, :], norm_weight_smem[:])
                    Tx.add(resid_smem[:, 0, :], resid_smem[:, 0, :], norm_bias_smem[:])
                    # store result
                    Tx.copy(out[by, 0, slice(bx * NUM_WORKERS + curr_idx, (bx + 1) * NUM_WORKERS + curr_idx), :], resid_smem[:, 0, :])  # noqa: E501
    # fmt: on

    import torch
    from torch import nn

    # get inputs
    inp_np = np.random.randn(ATTN_B, 1, ATTN_N, ATTN_D).astype(np_dtype)
    inp_resid_np = np.random.randn(ATTN_B, 1, ATTN_N, ATTN_D).astype(np_dtype)
    norm_weight_np = np.random.randn(ATTN_D).astype(np_dtype)
    norm_bias_np = np.random.randn(ATTN_D).astype(np_dtype)
    out_np = np.zeros((ATTN_B, 1, ATTN_N, ATTN_D)).astype(np_dtype)
    out_resid_np = np.zeros((ATTN_B, 1, ATTN_N, ATTN_D)).astype(np_dtype)

    inp_tvm = tvm.runtime.tensor(inp_np, DEV)
    inp_resid_tvm = tvm.runtime.tensor(inp_resid_np, DEV)
    norm_weight_tvm = tvm.runtime.tensor(norm_weight_np, DEV)
    norm_bias_tvm = tvm.runtime.tensor(norm_bias_np, DEV)
    out_tvm = tvm.runtime.tensor(out_np, DEV)
    out_resid_tvm = tvm.runtime.tensor(out_resid_np, DEV)

    def tir_layernorm():
        with target:
            mod = tvm.IRModule({"main": layernorm})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            def func():
                return mod(
                    inp_tvm, inp_resid_tvm, norm_weight_tvm, norm_bias_tvm, out_tvm, out_resid_tvm
                )

            bench(func, warmup=2, repeat=10, proton_name="tir")

        return torch.from_dlpack(out_tvm).cpu(), torch.from_dlpack(out_resid_tvm).cpu()

    def torch_layernorm():
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu

        torch_dev = torch.device("cuda")
        inp_torch = torch.from_dlpack(inp_tvm)
        inp_resid_torch = torch.from_dlpack(inp_resid_tvm)
        norm_weight_torch = torch.from_dlpack(norm_weight_tvm)
        norm_bias_torch = torch.from_dlpack(norm_bias_tvm)
        out_torch = torch.zeros((ATTN_B, 1, ATTN_N, ATTN_D), device=torch_dev)
        out_resid_torch = torch.zeros((ATTN_B, 1, ATTN_N, ATTN_D), device=torch_dev)
        ln_func = nn.LayerNorm([ATTN_D], device=torch_dev, dtype=torch.float16)
        ln_func.weight = nn.Parameter(norm_weight_torch)
        ln_func.bias = nn.Parameter(norm_bias_torch)

        def func():
            return ln_func(inp_torch + inp_resid_torch)

        bench(func, warmup=2, repeat=10, proton_name="torch")
        out_torch = func().detach().cpu()
        out_resid_torch = (inp_torch + inp_resid_torch).detach().cpu()
        return out_torch, out_resid_torch

    with ProtonContext("layernorm"):
        out_tvm, out_resid_tvm = tir_layernorm()
        out_torch, out_resid_torch = torch_layernorm()

    torch.testing.assert_close(out_resid_tvm, out_resid_torch, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_tvm, out_torch, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    tvm.testing.main()
