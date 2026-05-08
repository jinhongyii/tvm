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


import pytest
import torch
import torch.nn.functional as F
from triton.testing import do_bench

import tvm
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as Tx

F16_BYTES = 2
F32_BYTES = 4
SM_COUNT = 132
SMEM_CAPACITY = 228 * 1024


def ceildiv(a, b):
    return (a + b - 1) // b


def get_qk_norm_kernel(head_dim, dtype="float16"):
    """
    Single-pass fused RMS norm kernel for Q and K vectors.

    Key optimizations:
    - Single pass: data stays in registers, no SMEM for intermediate values
    - No SMEM for weights: load directly to registers when needed
    - Multi-row per warp: for small head_dim, each warp processes multiple rows
    - fp16 register storage: only convert to fp32 during computation
    - Non-persistent: one CTA per batch of rows, hardware handles scheduling

    Thread/warp configuration:
    - head_dim=64:  8 threads/row, 4 rows/warp (32 threads)
    - head_dim=128: 16 threads/row, 2 rows/warp (32 threads)
    - head_dim=256: 32 threads/row, 1 row/warp (32 threads)
    """
    assert head_dim in [64, 128, 256], f"head_dim {head_dim} is not supported"
    assert dtype in ["float16", "bfloat16"], f"dtype {dtype} is not supported"

    vec_size = 8  # 128-bit loads (8 x fp16)

    # Configuration based on head_dim
    # Each thread handles vec_size elements, we need head_dim/vec_size threads per row
    threads_per_row = head_dim // vec_size  # 8, 16, or 32
    rows_per_warp = 32 // threads_per_row  # 4, 2, or 1

    # Thread block: multiple warps for higher occupancy
    NUM_WARPS = 8
    bdx = threads_per_row  # threads handling one row
    bdy = rows_per_warp * NUM_WARPS  # total rows per CTA
    rows_per_cta = bdy  # each CTA processes this many rows

    # Reduction shuffle steps based on threads_per_row
    # head_dim=64: threads_per_row=8, need 3 steps (4,2,1)
    # head_dim=128: threads_per_row=16, need 4 steps (8,4,2,1)
    # head_dim=256: threads_per_row=32, need 5 steps (16,8,4,2,1)
    shuffle_steps = threads_per_row.bit_length() - 1

    # fmt: off
    @Tx.prim_func(tirx=True)
    def tir_qk_norm(q_ptr: Tx.handle, k_ptr: Tx.handle, q_weight_ptr: Tx.handle, k_weight_ptr: Tx.handle, eps_ptr: Tx.handle, weight_bias_ptr: Tx.handle, bound_m_ptr: Tx.handle):  # noqa: E501
        num_tokens = Tx.int32()
        qo_heads = Tx.int32()
        kv_heads = Tx.int32()

        q = Tx.match_buffer(q_ptr, [num_tokens, qo_heads, head_dim], dtype, scope="global")
        k = Tx.match_buffer(k_ptr, [num_tokens, kv_heads, head_dim], dtype, scope="global")
        q_weight = Tx.match_buffer(q_weight_ptr, [head_dim], dtype, scope="global")
        k_weight = Tx.match_buffer(k_weight_ptr, [head_dim], dtype, scope="global")
        eps_global = Tx.match_buffer(eps_ptr, [1], "float32", scope="global")
        weight_bias_global = Tx.match_buffer(weight_bias_ptr, [1], "float32", scope="global")
        bound_m_global = Tx.match_buffer(bound_m_ptr, [1], "int32", scope="global")

        cta_count: Tx.let = ceildiv(num_tokens * (qo_heads + kv_heads), rows_per_cta)

        with Tx.kernel():
            bx = Tx.cta_id([cta_count])
            tx, ty = Tx.thread_id([bdx, bdy])

            with Tx.thread():
                # Load scalar parameters
                eps = Tx.alloc_local([1], "float32")
                weight_bias = Tx.alloc_local([1], "float32")
                bound_m = Tx.alloc_local([1], "int32")
                q_job_cnt = Tx.alloc_local([1], "int32")
                total_jobs = Tx.alloc_local([1], "int32")

                eps[0] = eps_global[0]
                weight_bias[0] = weight_bias_global[0]
                bound_m[0] = bound_m_global[0]
                q_job_cnt[0] = Tx.min(num_tokens, bound_m[0]) * qo_heads
                total_jobs[0] = q_job_cnt[0] + Tx.min(num_tokens, bound_m[0]) * kv_heads

                # Register allocation for single-pass algorithm
                x_reg = Tx.alloc_local([vec_size], dtype)
                x_reg_f32 = Tx.alloc_local([vec_size], "float32")
                weight_reg = Tx.alloc_local([vec_size], dtype)
                weight_reg_f32 = Tx.alloc_local([vec_size], "float32")
                out_reg = Tx.alloc_local([vec_size], dtype)
                sum_sq = Tx.alloc_local([1], "float32")
                rms_inv = Tx.alloc_local([1], "float32")
                row_idx = Tx.alloc_local([1], "int32")
                actual_row = Tx.alloc_local([1], "int32")

                # Non-persistent: each CTA handles one batch of rows
                row_idx[0] = bx * rows_per_cta + ty

                if row_idx[0] < total_jobs[0]:
                        # Determine Q or K
                        is_q = Tx.meta_var(row_idx[0] < q_job_cnt[0])
                        qk_ptr: Tx.let[Tx.Var(name="qk_ptr", dtype=PointerType(PrimType(dtype)))] = Tx.if_then_else(is_q, q.data, k.data)  # noqa: E501
                        weight_ptr: Tx.let[Tx.Var(name="weight_ptr", dtype=PointerType(PrimType(dtype)))] = Tx.if_then_else(is_q, q_weight.data, k_weight.data)  # noqa: E501
                        batch_size = Tx.meta_var(Tx.if_then_else(is_q, num_tokens * qo_heads, num_tokens * kv_heads))  # noqa: E501

                        if is_q:
                            actual_row[0] = row_idx[0]
                        else:
                            actual_row[0] = row_idx[0] - q_job_cnt[0]

                        qk = Tx.decl_buffer(data=qk_ptr, shape=[batch_size, head_dim], dtype=dtype)
                        weight = Tx.decl_buffer(data=weight_ptr, shape=[head_dim], dtype=dtype)

                        # ============ SINGLE PASS: Load + Sum Squares ============
                        col_offset = Tx.meta_var(tx * vec_size)

                        Tx.copy(x_reg[:], qk[actual_row[0], col_offset:col_offset + vec_size])
                        Tx.cast(x_reg_f32[:], x_reg[:])

                        sum_sq[0] = 0.0
                        for vi in Tx.unroll(vec_size):
                            sum_sq[0] += x_reg_f32[vi] * x_reg_f32[vi]

                        # ============ Warp Reduction for sum_sq ============
                        # Reduce within threads_per_row threads handling the same row
                        for kr in Tx.unroll(shuffle_steps):
                            mask = Tx.meta_var(threads_per_row >> (kr + 1))
                            if mask > 0:
                                sum_sq[0] = sum_sq[0] + Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, sum_sq[0], mask, 32, 32)  # noqa: E501

                        # Compute RMS inverse
                        rms_inv[0] = Tx.rsqrt(sum_sq[0] / head_dim + eps[0])

                        # ============ SINGLE PASS: Normalize + Write Back ============
                        # Load weight directly from global memory to registers
                        Tx.copy(weight_reg[:], weight[col_offset:col_offset + vec_size])
                        Tx.cast(weight_reg_f32[:], weight_reg[:])

                        for vi in Tx.unroll(vec_size):
                            weight_reg_f32[vi] = weight_reg_f32[vi] + weight_bias[0]
                            x_reg_f32[vi] = x_reg_f32[vi] * rms_inv[0] * weight_reg_f32[vi]

                        Tx.cast(out_reg[:], x_reg_f32[:])
                        Tx.copy(qk[actual_row[0], col_offset:col_offset + vec_size], out_reg[:])

    # fmt: on

    with tvm.target.Target("cuda"):
        mod = tvm.IRModule({"main": tir_qk_norm})
        mod = tvm.compile(mod, target="cuda", tir_pipeline="tirx")
        return mod


def qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    weight_bias: float | None = None,
    bound_m: int | None = None,
) -> None:
    head_dim = q.shape[-1]
    q_shape = q.shape
    k_shape = k.shape

    if bound_m is None:
        bound_m = torch.tensor([max(q_shape[0], k_shape[0])], dtype=torch.int32, device=q.device)
    else:
        bound_m = torch.tensor([bound_m], dtype=torch.int32, device=q.device)

    weight_bias = torch.tensor(
        [weight_bias if weight_bias is not None else 0.0], dtype=torch.float32, device=q.device
    )
    eps = torch.tensor([eps], dtype=torch.float32, device=q.device)

    mod = get_qk_norm_kernel(head_dim)
    mod(q, k, q_weight, k_weight, eps, weight_bias, bound_m)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    ("head_dim", "num_qo_heads", "num_kv_heads", "num_tokens", "bound_tokens", "weight_bias"),
    [
        (64, 1, 1, 1, None, 0.02),
        (64, 1, 1, 128, 1, None),
        (64, 32, 8, 1024, None, 0.02),
        (128, 16, 16, 32, 16, 0.02),
        (128, 16, 16, 32, None, None),
        (128, 32, 8, 32, 16, None),
        (128, 32, 8, 32, None, 0.02),
        (128, 16, 1, 1024, None, 0.02),
        (256, 16, 16, 32, None, 0.02),
        (256, 32, 8, 512, None, None),
    ],
)
def test_qk_norm(
    dtype: torch.dtype,
    head_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    bound_tokens: int | None,
    weight_bias: float | None,
) -> None:
    """Test QK norm kernel with bound_m parameter to process partial tokens."""

    device = torch.device("cuda")

    eps = 1e-6

    gen = torch.Generator(device)
    gen.manual_seed(0xDEADBEEF)

    q = torch.rand((num_tokens, num_qo_heads, head_dim), dtype=dtype, device=device, generator=gen)

    k = torch.rand((num_tokens, num_kv_heads, head_dim), dtype=dtype, device=device, generator=gen)

    q_weight = torch.rand((head_dim,), dtype=dtype, device=device, generator=gen)
    k_weight = torch.rand((head_dim,), dtype=dtype, device=device, generator=gen)

    q_test = q.clone()
    k_test = k.clone()
    bound_m = (
        torch.tensor([bound_tokens], dtype=torch.int64, device=device)
        if bound_tokens is not None
        else None
    )

    qk_norm(
        q=q_test,
        k=k_test,
        q_weight=q_weight,
        k_weight=k_weight,
        weight_bias=weight_bias,
        eps=eps,
        bound_m=bound_m,
    )

    biased_q_weight = q_weight if weight_bias is None else q_weight + weight_bias
    biased_k_weight = k_weight if weight_bias is None else k_weight + weight_bias

    bound_m = bound_tokens if bound_tokens is not None else num_tokens

    expected_q = F.rms_norm(
        q[:bound_m].to(torch.float32), (head_dim,), biased_q_weight.to(torch.float32), eps
    )
    expected_k = F.rms_norm(
        k[:bound_m].to(torch.float32), (head_dim,), biased_k_weight.to(torch.float32), eps
    )

    torch.testing.assert_close(expected_q, q_test[:bound_m].to(torch.float32), atol=1e-2, rtol=1e-4)
    torch.testing.assert_close(expected_k, k_test[:bound_m].to(torch.float32), atol=1e-2, rtol=1e-4)


def bench_qk_norm_my():
    def compute_bandwidth(
        ms: float,
        num_tokens: int,
        qo_heads: int,
        kv_heads: int,
        head_dim: int,
        bound_m: int | None = None,
    ) -> float:
        """Returns achieved bandwidth in GB/s"""
        effective_tokens = min(num_tokens, bound_m) if bound_m else num_tokens
        bytes_transferred = effective_tokens * (qo_heads + kv_heads) * head_dim * 4
        return bytes_transferred / (ms * 1e6)

    num_tokens, qo_heads, kv_heads, head_dim, eps, weight_bias = (1024, 32 * 16, 1, 128, 1e-6, 0)
    q = torch.rand(
        (num_tokens, qo_heads, head_dim), dtype=torch.float16, device=torch.device("cuda")
    )
    k = torch.rand(
        (num_tokens, kv_heads, head_dim), dtype=torch.float16, device=torch.device("cuda")
    )
    q_weight = torch.rand((head_dim,), dtype=torch.float16, device=torch.device("cuda"))
    k_weight = torch.rand((head_dim,), dtype=torch.float16, device=torch.device("cuda"))
    eps = torch.tensor([eps], dtype=torch.float32, device=torch.device("cuda"))
    weight_bias = torch.tensor([weight_bias], dtype=torch.float32, device=torch.device("cuda"))
    bound_m = torch.tensor([num_tokens], dtype=torch.int32, device=torch.device("cuda"))

    mod = get_qk_norm_kernel(head_dim)

    def func():
        return mod(q, k, q_weight, k_weight, eps, weight_bias, bound_m)

    for i in range(10):
        func()


def benchmark_qk_norm() -> None:
    """Benchmark for QK norm kernels."""

    device = torch.device("cuda")
    gen = torch.Generator(device)
    gen.manual_seed(0xDEADBEEF)

    head_dim = 128
    num_qo_heads = 16
    num_kv_heads = 1
    dtype = torch.bfloat16

    def bench_qk_norm(m: int) -> None:
        q = torch.rand((m, num_qo_heads, head_dim), dtype=dtype, device=device, generator=gen)
        k = torch.rand((m, num_kv_heads, head_dim), dtype=dtype, device=device, generator=gen)
        q_weight = torch.rand((head_dim,), dtype=dtype, device=device, generator=gen)
        k_weight = torch.rand((head_dim,), dtype=dtype, device=device, generator=gen)
        num_bytes = dtype.itemsize * m * (num_qo_heads + num_kv_heads) * head_dim

        tir_qk_norm_kernel = get_qk_norm_kernel(head_dim, "bfloat16")
        eps = torch.tensor([1e-6], dtype=torch.float32, device=device)
        bound_m = torch.tensor([m], dtype=torch.int32, device=device)
        weight_bias = torch.tensor([0.0], dtype=torch.float32, device=device)

        def qk_norm_fn() -> None:
            tir_qk_norm_kernel(q, k, q_weight, k_weight, eps, weight_bias, bound_m)

        time = do_bench(qk_norm_fn, warmup=10, rep=100)

        print(
            f"QK norm: M={m:6d} time={time * 1e3:.2f}us, bw={(num_bytes / 2**30) / (time * 1e-3):.2f}GB/s"  # noqa: E501
        )

    for m in [1, 4, 8, 32, 128, 1024, 12 * 1024, 16 * 1024, 32 * 1024]:
        # for m in [32 * 1024]:
        bench_qk_norm(m)


if __name__ == "__main__":
    bench_qk_norm_my()
    benchmark_qk_norm()
