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

from pathlib import Path

import numpy as np
import tvm_ffi
from mlc_llm.compiler_pass.blas_dispatch import BLASDispatch
from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation
from mlc_llm.compiler_pass.fuse_add_norm import FuseAddRMSNorm
from mlc_llm.compiler_pass.pipeline import _DebugDump
from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
from mlc_llm.nn.kv_cache import PagedKVCache
from tqdm import tqdm

import tvm
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.runtime import ShapeTuple
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as Tx

from .test_append_paged_kv_cache import get_append_paged_kv_cache_kernel
from .test_batch_decode import PlanInfo, decode_attn_plan, get_decode_kernel
from .test_fused_add_rms_norm import get_fused_add_rmsnorm_kernel
from .test_fused_split_silu_multiply import get_fused_split_silu_multiply_kernel_cp_async
from .test_hgemm_1consumer_1cta_swap_splitk import get_hgemm_kernel
from .test_rmsnorm import get_rmsnorm_kernel
from .test_rope import get_cos_sin_cache_kernel, get_rope_kernel

dev = tvm.cuda()
target = tvm.target.Target("cuda")

config = Qwen3Config(
    hidden_act="silu",
    hidden_size=5120,
    intermediate_size=25600,
    attention_bias=False,
    num_attention_heads=64,
    num_hidden_layers=1,  # 64,
    num_key_value_heads=8,
    rms_norm_eps=1e-06,
    rope_theta=1000000,
    vocab_size=151936,
    tie_word_embeddings=False,
    context_window_size=40960,
    prefill_chunk_size=2048,
    tensor_parallel_shards=1,
    head_dim=128,
    dtype="float32",
    max_batch_size=128,
    weight_block_size=None,
    kwargs={},
)


def get_default_spec(model):
    mod_spec = {
        "embed": {
            "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        },
        "batch_decode": {
            "input_embeds": nn.spec.Tensor(["batch_size", 1, model.hidden_size], model.dtype),
            "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        },
        "create_paged_kv_cache": {
            "max_batch_size": int,
            "max_total_seq_len": int,
            "prefill_chunk_size": int,
            "page_size": int,
            "support_sliding_window": int,
            "$": {"param_mode": "none", "effect_mode": "none"},
        },
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, model)


def _craft_pipeline(ext_mods: list[nn.ExternModule], dump_file_prefix: Path):
    ext_mods = ext_mods or []
    debug_dir = Path("/home/guanjiew/qwen3-mg-debug")

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                _DebugDump(f"{dump_file_prefix}-phase0.py", debug_dir, show_meta=False),
                tvm.tirx.transform.BindTarget(target),
                DispatchKVCacheCreation(target, True, dict()),
                BLASDispatch(target),
                FuseAddRMSNorm(target=target),
                # Phase 1. Passes on high-level operator graph
                # We can enable cublas for further optimization
                relax.transform.FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                relax.transform.DeadCodeElimination(),
                _DebugDump(f"{dump_file_prefix}-phase1.py", debug_dir, show_meta=False),
                # Phase 4. Low-level Optimizations
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                # Phase 5. Lowering to VM bytecode
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                _DebugDump(f"{dump_file_prefix}-phase2.py", debug_dir, show_meta=False),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                relax.transform.AttachExternModules(ext_mods),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline


@register_pipeline("opt_llm_1")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: list[nn.ExternModule] | None = None,
):
    return _craft_pipeline(ext_mods, "opt_llm_1")


@register_pipeline("opt_llm_2")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: list[nn.ExternModule] | None = None,
):
    return _craft_pipeline(ext_mods, "opt_llm_2")


@register_pipeline("opt_llm_mg")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: list[nn.ExternModule] | None = None,
):
    return _craft_pipeline(ext_mods, "opt_llm_mg")


tvm_ffi.register_global_func("megakernel.decode_attn_plan", decode_attn_plan, override=True)


def get_params(named_params):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    result = list()
    for k, param in tqdm(named_params):
        torch_tensor = torch.empty(tuple(param.shape)).to(getattr(torch, param.dtype)).to("cuda")
        if k.endswith("norm.weight"):
            nn.init.ones_(torch_tensor)
        elif k.endswith("embed_tokens.weight"):
            nn.init.normal_(torch_tensor, mean=0.0, std=0.02)
        else:
            nn.init.xavier_uniform_(torch_tensor, gain=1.0)
        torch_tensor = torch_tensor
        result.append(tvm.runtime.from_dlpack(torch.to_dlpack(torch_tensor)))
    return result


def sample_token(logits):
    return np.argmax(logits, axis=-1)


model = Qwen3LMHeadModel(config)
model.to("float16")
mod, named_params = model.export_tvm(get_default_spec(model))
params = get_params(named_params)
with target:
    ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline("opt_llm_1"))
    vm = relax.VirtualMachine(ex, dev)

MAX_BATCH_SIZE = 128
MAX_SEQ_LEN = 4096
MAX_TOTAL_SEQ_LEN = MAX_BATCH_SIZE * MAX_SEQ_LEN
PAGE_SIZE = 16
ROPE_THETA = 1000000


def test_qwen3_layer(batch_decode_func, is_megakernel=False, cos_sin_cache_func=None):
    kv_cache = vm["create_flashinfer_paged_kv_cache"](
        ShapeTuple([MAX_BATCH_SIZE]),  # max_batch_size
        ShapeTuple([MAX_TOTAL_SEQ_LEN]),  # max_total_seq_len
        ShapeTuple([MAX_SEQ_LEN]),  # prefill_chunk_size
        ShapeTuple([PAGE_SIZE]),  # page_size
        ShapeTuple([0]),  # support_sliding_window
    )

    if is_megakernel:
        cos_sin_cache = tvm.runtime.tensor(
            np.zeros((MAX_SEQ_LEN, 128), dtype="float32"), device=dev
        )
        cos_sin_cache_func(cos_sin_cache)

    nd_view_func = tvm.get_global_func("vm.builtin.reshape")

    def embed(tokens, params):
        _embed = vm["embed"](tokens, params)
        _embed = nd_view_func(_embed, ShapeTuple([_embed.shape[0], 1, _embed.shape[1]]))
        return _embed

    add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    batch_size = 32
    seq_len = 16
    seq_ids = []

    for i in range(batch_size):
        seq_ids.append(i)
        add_sequence_func(kv_cache, i)

    np.random.seed(0)
    logits_arr = list()
    last_tokens = np.random.randint(0, 100, size=(batch_size,))
    for i in tqdm(range(seq_len)):
        tokens = tvm.runtime.tensor(last_tokens.astype("int32"), device=dev)
        hidden_states = embed(tokens, params)
        begin_forward_func(kv_cache, ShapeTuple(seq_ids), ShapeTuple([1] * batch_size))
        if is_megakernel:
            logits = batch_decode_func(hidden_states, kv_cache, cos_sin_cache, params)
        else:
            logits, kv_cache = batch_decode_func(hidden_states, kv_cache, params)

        end_forward_func(kv_cache)
        last_tokens = sample_token(logits.numpy()).flatten()
        logits_arr.append(logits)

    return logits_arr


# res0 = test_qwen3_layer(vm["batch_decode"])


def attach_attr(func, name):
    func = func.without_attr("global_symbol")
    func = func.with_attr("global_symbol", name)
    func = func.with_attr("tir.is_scheduled", True)
    func = func.with_attr("tir.noalias", True)
    return func


def get_qwen3_megakernel_mod():
    rms_norm = get_rmsnorm_kernel(5120)
    fused_add_rmsnorm = get_fused_add_rmsnorm_kernel(5120)
    rms_norm1 = get_rmsnorm_kernel(128)
    fused_split_silu_multiply = get_fused_split_silu_multiply_kernel_cp_async(25600)
    hgemm1, reduce1, tile_k_num1 = get_hgemm_kernel(dim_n=10240, dim_k=5120)
    hgemm2, reduce2, tile_k_num2 = get_hgemm_kernel(dim_n=5120, dim_k=8192)
    hgemm3, reduce3, tile_k_num3 = get_hgemm_kernel(dim_n=51200, dim_k=5120)
    hgemm4, reduce4, tile_k_num4 = get_hgemm_kernel(dim_n=5120, dim_k=25600)
    hgemm5, reduce5, tile_k_num5 = get_hgemm_kernel(dim_n=151936, dim_k=5120)
    rope = get_rope_kernel(128)
    append_paged_kv_cache = get_append_paged_kv_cache_kernel(8, 1, 128)
    decode, merge = get_decode_kernel(PlanInfo(64, 8, 128, enforce_no_split_kv=True), PAGE_SIZE)
    cos_sin_cache = get_cos_sin_cache_kernel(128, ROPE_THETA)

    # fmt: off
    @I.ir_module
    class Module:
        @Tx.prim_func(private=True)
        def rms_norm(input_ptr: Tx.handle, weight_ptr: Tx.handle, out_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def rms_norm1(input_ptr: Tx.handle, weight_ptr: Tx.handle, out_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def fused_add_rmsnorm(input_ptr: Tx.handle, residual_ptr: Tx.handle, weight_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def fused_split_silu_multiply(input_ptr: Tx.handle, output_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def split(var_reshape: Tx.handle, var_T_split: Tx.handle, var_T_split_1: Tx.handle, var_T_split_2: Tx.handle):
            Tx.func_attr({"tir.noalias": True})
            batch_size = Tx.int64()
            reshape = Tx.match_buffer(var_reshape, (batch_size, Tx.int64(1), Tx.int64(80), Tx.int64(128)), "float16")
            T_split = Tx.match_buffer(var_T_split, (batch_size, Tx.int64(1), Tx.int64(64), Tx.int64(128)), "float16")
            T_split_1 = Tx.match_buffer(var_T_split_1, (batch_size, Tx.int64(1), Tx.int64(8), Tx.int64(128)), "float16")
            T_split_2 = Tx.match_buffer(var_T_split_2, (batch_size, Tx.int64(1), Tx.int64(8), Tx.int64(128)), "float16")
            # with Tx.sblock("root"):
            for ax0, ax1, ax2, ax3 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(64), Tx.int64(128)):
                with Tx.sblock("T_split"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Tx.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Tx.reads(reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    Tx.writes(T_split[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_split[v_ax0, v_ax1, v_ax2, v_ax3] = reshape[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(8), Tx.int64(128)):
                with Tx.sblock("T_split_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Tx.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Tx.reads(reshape[v_ax0, v_ax1, v_ax2 + Tx.int64(64), v_ax3])
                    Tx.writes(T_split_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_split_1[v_ax0, v_ax1, v_ax2, v_ax3] = reshape[v_ax0, v_ax1, v_ax2 + Tx.int64(64), v_ax3]
            for ax0, ax1, ax2, ax3 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(8), Tx.int64(128)):
                with Tx.sblock("T_split_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Tx.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Tx.reads(reshape[v_ax0, v_ax1, v_ax2 + Tx.int64(72), v_ax3])
                    Tx.writes(T_split_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_split_2[v_ax0, v_ax1, v_ax2, v_ax3] = reshape[v_ax0, v_ax1, v_ax2 + Tx.int64(72), v_ax3]

        @Tx.prim_func(private=True)
        def fused_concatenate(p_split_2: Tx.handle, p_rms_norm1: Tx.handle, p_rms_norm2: Tx.handle, p_output0: Tx.handle):
            Tx.func_attr({"tir.noalias": True})
            batch_size = Tx.int64()
            split_2 = Tx.match_buffer(p_split_2, (batch_size, Tx.int64(1), Tx.int64(8), Tx.int64(128)), "float16")
            rms_norm1 = Tx.match_buffer(p_rms_norm1, (batch_size, Tx.int64(1), Tx.int64(64), Tx.int64(128)), "float16")
            rms_norm2 = Tx.match_buffer(p_rms_norm2, (batch_size, Tx.int64(1), Tx.int64(8), Tx.int64(128)), "float16")
            T_concat_intermediate = Tx.match_buffer(p_output0, (batch_size, Tx.int64(1), Tx.int64(80), Tx.int64(128)), "float16")
            # with Tx.sblock("root"):
            for ax0, ax1, ax2, ax3 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(80), Tx.int64(128)):
                with Tx.sblock("T_concat"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Tx.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Tx.reads(split_2[v_ax0, v_ax1, v_ax2 - Tx.int64(72), v_ax3], rms_norm2[v_ax0, v_ax1, v_ax2 - Tx.int64(64), v_ax3], rms_norm1[v_ax0, v_ax1, v_ax2, v_ax3])
                    Tx.writes(T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = Tx.if_then_else(Tx.int64(72) <= v_ax2, split_2[v_ax0, v_ax1, v_ax2 - Tx.int64(72), v_ax3], Tx.if_then_else(Tx.int64(64) <= v_ax2, rms_norm2[v_ax0, v_ax1, v_ax2 - Tx.int64(64), v_ax3], rms_norm1[v_ax0, v_ax1, v_ax2, v_ax3]))

        @Tx.prim_func(private=True)
        def cast(var_lv4: Tx.handle, var_compute: Tx.handle):
            Tx.func_attr({"op_pattern": 0, "tir.noalias": True})
            batch_size = Tx.int64()
            lv4 = Tx.match_buffer(var_lv4, (batch_size, Tx.int64(1), Tx.int64(151936)), "float16")
            compute = Tx.match_buffer(var_compute, (batch_size, Tx.int64(1), Tx.int64(151936)))
            # with Tx.sblock("root"):
            for i0, i1, i2 in Tx.grid(batch_size, Tx.int64(1), Tx.int64(151936)):
                with Tx.sblock("compute"):
                    v_i0, v_i1, v_i2 = Tx.axis.remap("SSS", [i0, i1, i2])
                    Tx.reads(lv4[v_i0, v_i1, v_i2])
                    Tx.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = Tx.Cast("float32", lv4[v_i0, v_i1, v_i2])

        @Tx.prim_func(private=True)
        def hgemm1(A_ptr: Tx.handle, b_ptr: Tx.handle, partial_sum_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def reduce1(partial_sum_ptr: Tx.handle, D_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def hgemm2(A_ptr: Tx.handle, b_ptr: Tx.handle, partial_sum_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def reduce2(partial_sum_ptr: Tx.handle, D_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def hgemm3(A_ptr: Tx.handle, b_ptr: Tx.handle, partial_sum_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def reduce3(partial_sum_ptr: Tx.handle, D_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def hgemm4(A_ptr: Tx.handle, b_ptr: Tx.handle, partial_sum_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def reduce4(partial_sum_ptr: Tx.handle, D_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def hgemm5(A_ptr: Tx.handle, b_ptr: Tx.handle, partial_sum_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def reduce5(partial_sum_ptr: Tx.handle, D_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def rope(q: Tx.handle, cos_sin_cache: Tx.handle, pos_ids: Tx.handle, q_rope: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def append_paged_kv_cache(cache_ptr: Tx.handle, k_ptr: Tx.handle, v_ptr: Tx.handle, pos_map_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def decode(q: Tx.handle, kv: Tx.handle, lse: Tx.handle, kv_indptr: Tx.handle, kv_last_page_len: Tx.handle, kv_indices: Tx.handle, request_indices: Tx.handle, kv_tile_indices: Tx.handle, max_chunk_size: Tx.handle, o: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def cos_sin_cache(cos_sin_cache: Tx.handle):
            pass

        @R.function
        def cos_sin_cache_func(cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32")):
            cls = Module
            with R.dataflow():
                cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32") = R.call_tir_inplace(cls.cos_sin_cache, (cache), [0], out_sinfo=R.Tensor((MAX_SEQ_LEN, 128), dtype="float32") )
                R.output(cache)
            return cache

        @R.function
        def batch_decode(
            input_embeds: R.Tensor(("batch_size", 1, 5120), dtype="float16"),
            paged_kv_cache: R.Object,
            cos_sin_cache: R.Tensor((MAX_SEQ_LEN, 128), dtype="float32"),
            packed_params: R.Tuple(
                R.Tensor((151936, 5120), dtype="float16"),
                R.Tensor((10240, 5120), dtype="float16"),
                R.Tensor((5120, 8192), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((51200, 5120), dtype="float16"),
                R.Tensor((5120, 25600), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((151936, 5120), dtype="float16"),
            ),
        ):
            batch_size = Tx.int64()
            new_batch_size = Tx.int64()
            total_page_num = Tx.int64()
            max_page_num = Tx.int64()
            page_size = Tx.int64()

            R.func_attr({"num_input": 2})
            cls = Module
            with R.dataflow():
                res0 = R.call_pure_packed(
                    "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                    paged_kv_cache, R.prim_value(1),
                    sinfo_args=[
                        R.Tuple([R.Tensor(None, dtype="float16")]),
                        R.Tensor((batch_size + 1,), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                    ],
                )
                kv_data_, kv_indptr, kv_indices_, kv_last_page_len, append_position_map, q_rope_position_map = (
                    res0[0],
                    res0[1],
                    res0[3],
                    res0[4],
                    res0[5],
                    res0[6],
                )
                kv_data = R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, 8, page_size, 128), dtype="float16")]))
                kv_indices = R.match_cast(kv_indices_, R.Tensor((total_page_num,), dtype="int32"))
                res1 = R.call_pure_packed(
                    "megakernel.decode_attn_plan",
                    R.prim_value(64),
                    R.prim_value(8),
                    R.prim_value(128),
                    R.prim_value(batch_size),
                    kv_indptr,
                    R.prim_value(page_size),
                    R.prim_value(max_page_num),
                    sinfo_args=[
                        R.Tensor(None, dtype="float32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                    ],
                )
                plan_lse_tvm_, plan_request_indices_, plan_kv_tile_indices_, plan_max_chunk_size = res1[0], res1[1], res1[2], res1[3]
                plan_lse_tvm = R.match_cast(plan_lse_tvm_, R.Tensor((new_batch_size, 64), dtype="float32"))
                plan_request_indices = R.match_cast(plan_request_indices_, R.Tensor((new_batch_size,), dtype="int32"))
                plan_kv_tile_indices = R.match_cast(plan_kv_tile_indices_, R.Tensor((new_batch_size,), dtype="int32"))

                model_layers_0_self_attn_c_attn_weight1: R.Tensor((10240, 5120), dtype="float16") = packed_params[1]
                model_layers_0_self_attn_o_proj_weight1: R.Tensor((5120, 8192), dtype="float16") = packed_params[2]
                model_layers_0_self_attn_q_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[3]
                model_layers_0_self_attn_k_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[4]
                model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((51200, 5120), dtype="float16") = packed_params[5]
                model_layers_0_mlp_down_proj_weight1: R.Tensor((5120, 25600), dtype="float16") = packed_params[6]
                model_layers_0_input_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[7]
                model_layers_0_post_attention_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8]
                model_norm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[9]
                lm_head_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[10]

                rs0 = R.reshape(input_embeds, (batch_size, 5120))
                rms_norm = R.call_tir(cls.rms_norm, (rs0, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

                # permute_dims0 = R.permute_dims(model_layers_0_self_attn_c_attn_weight1, axes=None)
                # lv = R.matmul(rms_norm, permute_dims0, out_dtype="void")
                ps0 = R.call_tir(cls.hgemm1, (rms_norm, model_layers_0_self_attn_c_attn_weight1), out_sinfo=R.Tensor((tile_k_num1, batch_size, 10240), dtype="float32"))
                lv = R.call_tir(cls.reduce1, (ps0,), out_sinfo=R.Tensor((batch_size, 10240), dtype="float16"))

                reshape = R.reshape(lv, (batch_size, 1, 80, 128))
                split = R.call_tir(cls.split, (reshape,), out_sinfo=[R.Tensor((batch_size, 1, 64, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16")])
                split_0: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = split[0]
                split_1: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[1]
                rs1 = R.reshape(split_0, (batch_size * 64, 128))
                rs2 = R.reshape(split_1, (batch_size * 8, 128))
                rms_norm1 = R.call_tir(cls.rms_norm1, (rs1, model_layers_0_self_attn_q_norm_weight1), out_sinfo=R.Tensor((batch_size * 64, 128), dtype="float16"))
                rms_norm2 = R.call_tir(cls.rms_norm1, (rs2, model_layers_0_self_attn_k_norm_weight1), out_sinfo=R.Tensor((batch_size * 8, 128), dtype="float16"))
                rms_norm1_rs = R.reshape(rms_norm1, (batch_size, 1, 64, 128)) # q
                rms_norm2_rs = R.reshape(rms_norm2, (batch_size, 1, 8, 128)) # k
                v: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[2] # v

                ######################################################### Attention #########################################################
                #########################################################
                # lv1 = R.call_tir(cls.fused_concatenate, (v, rms_norm1_rs, rms_norm2_rs), out_sinfo=R.Tensor((batch_size, 1, 80, 128), dtype="float16"))
                # reshape1 = R.reshape(lv1, (batch_size, 80, 128))
                # lv_2 = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(Tx.float32(0.088388347648318447)), reshape1), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
                # reshape2 = R.reshape(lv_2, (batch_size, 1, 64, 128))
                # reshape3 = R.reshape(reshape2, (batch_size, 8192))
                #########################################################
                # rope
                q_rs = R.reshape(rms_norm1_rs, (batch_size, 64, 128))
                q_rope = R.call_tir(cls.rope, (q_rs, cos_sin_cache, q_rope_position_map), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
                k_rs = R.reshape(rms_norm2_rs, (batch_size, 8, 128))
                k = R.call_tir(cls.rope, (k_rs, cos_sin_cache, q_rope_position_map), out_sinfo=R.Tensor((batch_size, 8, 128), dtype="float16"))
                k_rope = R.reshape(k, (batch_size, 1, 8, 128))
                # append
                append_position_map_rs = R.reshape(append_position_map, (batch_size, 1))
                new_kv_data = R.call_tir_inplace(cls.append_paged_kv_cache, (kv_data[0], k_rope, v, append_position_map_rs), [0],
                                                 out_sinfo=R.Tensor((max_page_num, 2, 8, page_size, 128), dtype="float16"))
                # attention
                o = R.call_tir(cls.decode,
                               (q_rope, new_kv_data, plan_lse_tvm, kv_indptr, kv_last_page_len, kv_indices, plan_request_indices, plan_kv_tile_indices, plan_max_chunk_size), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
                reshape3 = R.reshape(o, (batch_size, 8192))
                #########################################################
                ######################################################### Attention #########################################################

                # permute_dims1 = R.permute_dims(model_layers_0_self_attn_o_proj_weight1, axes=None)
                # lv1_1: R.Tensor((batch_size, 5120), dtype="float16") = R.matmul(reshape3, permute_dims1, out_dtype="void")
                rs1 = R.call_tir(cls.hgemm2, (reshape3, model_layers_0_self_attn_o_proj_weight1), out_sinfo=R.Tensor((tile_k_num2, batch_size, 5120), dtype="float32"))
                lv1_1 = R.call_tir(cls.reduce2, (rs1,), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

                lv_3 = R.call_tir_inplace(cls.fused_add_rmsnorm, (lv1_1, rs0, model_layers_0_post_attention_layernorm_weight1), [0, 1], out_sinfo=[R.Tensor((batch_size, 5120), dtype="float16"), R.Tensor((batch_size, 5120), dtype="float16")])
                lv1_2: R.Tensor((batch_size, 5120), dtype="float16") = lv_3[1]
                rms_norm3: R.Tensor((batch_size, 5120), dtype="float16") = lv_3[0]

                # permute_dims2 = R.permute_dims(model_layers_0_mlp_gate_up_proj_weight1, axes=None)
                # lv2: R.Tensor((batch_size, 51200), dtype="float16") = R.matmul(rms_norm3, permute_dims2, out_dtype="void")
                rs2 = R.call_tir(cls.hgemm3, (rms_norm3, model_layers_0_mlp_gate_up_proj_weight1), out_sinfo=R.Tensor((tile_k_num3, batch_size, 51200), dtype="float32"))
                lv2 = R.call_tir(cls.reduce3, (rs2,), out_sinfo=R.Tensor((batch_size, 51200), dtype="float16"))

                lv2_1 = R.call_tir(cls.fused_split_silu_multiply, (lv2,), out_sinfo=R.Tensor((batch_size, 25600), dtype="float16"))

                # permute_dims3 = R.permute_dims(model_layers_0_mlp_down_proj_weight1, axes=None)
                # lv3: R.Tensor((batch_size, 5120), dtype="float16") = R.matmul(lv2_1, permute_dims3, out_dtype="void")
                rs3 = R.call_tir(cls.hgemm4, (lv2_1, model_layers_0_mlp_down_proj_weight1), out_sinfo=R.Tensor((tile_k_num4, batch_size, 5120), dtype="float32"))
                lv3 = R.call_tir(cls.reduce4, (rs3,), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

                lv2_2 = R.call_tir_inplace(cls.fused_add_rmsnorm, (lv3, lv1_2, model_norm_weight1), [0, 1], out_sinfo=[R.Tensor((batch_size, 5120), dtype="float16"), R.Tensor((batch_size, 5120), dtype="float16")])
                rms_norm4: R.Tensor((batch_size, 5120), dtype="float16") = lv2_2[0]

                # permute_dims4 = R.permute_dims(lm_head_weight1, axes=None)
                # lv4: R.Tensor((batch_size, 151936), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")
                rs4 = R.call_tir(cls.hgemm5, (rms_norm4, lm_head_weight1), out_sinfo=R.Tensor((tile_k_num5, batch_size, 151936), dtype="float32"))
                lv4 = R.call_tir(cls.reduce5, (rs4,), out_sinfo=R.Tensor((batch_size, 151936), dtype="float16"))

                lv4_rs = R.reshape(lv4, (batch_size, 1, 151936))
                astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, 151936), dtype="float32"))
                R.output(astype)
            return astype
    # fmt: on
    mod = Module

    mod.update_func(mod.get_global_var("rms_norm"), attach_attr(rms_norm, "rms_norm"))
    mod.update_func(
        mod.get_global_var("fused_add_rmsnorm"), attach_attr(fused_add_rmsnorm, "fused_add_rmsnorm")
    )
    mod.update_func(mod.get_global_var("rms_norm1"), attach_attr(rms_norm1, "rms_norm1"))
    mod.update_func(
        mod.get_global_var("fused_split_silu_multiply"),
        attach_attr(fused_split_silu_multiply, "fused_split_silu_multiply"),
    )
    mod.update_func(mod.get_global_var("hgemm1"), attach_attr(hgemm1, "hgemm1"))
    mod.update_func(mod.get_global_var("reduce1"), attach_attr(reduce1, "reduce1"))
    mod.update_func(mod.get_global_var("hgemm2"), attach_attr(hgemm2, "hgemm2"))
    mod.update_func(mod.get_global_var("reduce2"), attach_attr(reduce2, "reduce2"))
    mod.update_func(mod.get_global_var("hgemm3"), attach_attr(hgemm3, "hgemm3"))
    mod.update_func(mod.get_global_var("reduce3"), attach_attr(reduce3, "reduce3"))
    mod.update_func(mod.get_global_var("hgemm4"), attach_attr(hgemm4, "hgemm4"))
    mod.update_func(mod.get_global_var("reduce4"), attach_attr(reduce4, "reduce4"))
    mod.update_func(mod.get_global_var("hgemm5"), attach_attr(hgemm5, "hgemm5"))
    mod.update_func(mod.get_global_var("reduce5"), attach_attr(reduce5, "reduce5"))
    mod.update_func(mod.get_global_var("rope"), attach_attr(rope, "rope"))
    mod.update_func(
        mod.get_global_var("append_paged_kv_cache"),
        attach_attr(append_paged_kv_cache, "append_paged_kv_cache"),
    )
    mod.update_func(mod.get_global_var("decode"), attach_attr(decode, "decode"))
    mod.update_func(
        mod.get_global_var("cos_sin_cache"), attach_attr(cos_sin_cache, "cos_sin_cache")
    )
    return mod


def get_qwen3_megakernel_batch_decode_func():
    mg_model = get_qwen3_megakernel_mod()

    with target:
        ex = tvm.compile(
            mg_model, target, relax_pipeline=relax.get_pipeline("opt_llm_mg"), tir_pipeline="tirx"
        )
        vm = relax.VirtualMachine(ex, dev)
    return vm["batch_decode"], vm["cos_sin_cache_func"]


batch_decode_func, cos_sin_cache_func = get_qwen3_megakernel_batch_decode_func()
res1 = test_qwen3_layer(
    batch_decode_func, is_megakernel=True, cos_sin_cache_func=cos_sin_cache_func
)


# fmt: off
@I.ir_module
class Module2:
    @R.function
    def batch_decode(input_embeds: R.Tensor(("batch_size", 1, 5120), dtype="float16"), paged_kv_cache: R.Object, packed_params: R.Tuple(R.Tensor((151936, 5120), dtype="float16"), R.Tensor((10240, 5120), dtype="float16"), R.Tensor((5120, 8192), dtype="float16"), R.Tensor((128,), dtype="float16"), R.Tensor((128,), dtype="float16"), R.Tensor((51200, 5120), dtype="float16"), R.Tensor((5120, 25600), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((151936, 5120), dtype="float16"))) -> R.Tuple(R.Tensor(("batch_size", 1, 151936), dtype="float32"), R.Object):
        batch_size = Tx.int64()
        R.func_attr({"num_input": 2})
        with R.dataflow():
            packed_params[0]
            model_layers_0_self_attn_c_attn_weight1: R.Tensor((10240, 5120), dtype="float16") = packed_params[1]
            model_layers_0_self_attn_o_proj_weight1: R.Tensor((5120, 8192), dtype="float16") = packed_params[2]
            model_layers_0_self_attn_q_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[3]
            model_layers_0_self_attn_k_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[4]
            model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((51200, 5120), dtype="float16") = packed_params[5]
            model_layers_0_mlp_down_proj_weight1: R.Tensor((5120, 25600), dtype="float16") = packed_params[6]
            model_layers_0_input_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[7]
            model_layers_0_post_attention_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8]
            model_norm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[9]
            lm_head_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[10]
            rms_norm: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.nn.rms_norm(input_embeds, model_layers_0_input_layernorm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            permute_dims: R.Tensor((5120, 10240), dtype="float16") = R.permute_dims(model_layers_0_self_attn_c_attn_weight1, axes=None)
            matmul: R.Tensor((batch_size, 1, 10240), dtype="float16") = R.matmul(rms_norm, permute_dims, out_dtype="void")
            reshape: R.Tensor((batch_size, 1, 80, 128), dtype="float16") = R.reshape(matmul, R.shape([batch_size, 1, 80, 128]))
            split: R.Tuple(R.Tensor((batch_size, 1, 64, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16"), R.Tensor((batch_size, 1, 8, 128), dtype="float16")) = R.split(reshape, indices_or_sections=[64, 72], axis=2)
            split_0: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = split[0]
            split_1: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[1]
            split_2: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = split[2]
            rms_norm1: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = R.nn.rms_norm(split_0, model_layers_0_self_attn_q_norm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            rms_norm2: R.Tensor((batch_size, 1, 8, 128), dtype="float16") = R.nn.rms_norm(split_1, model_layers_0_self_attn_k_norm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            concat: R.Tensor((batch_size, 1, 80, 128), dtype="float16") = R.concat((rms_norm1, rms_norm2, split_2), axis=2)
            reshape1: R.Tensor((batch_size, 80, 128), dtype="float16") = R.reshape(concat, R.shape([batch_size, 80, 128]))
            lv = R.call_dps_packed("vm.builtin.attention_kv_cache_attention_with_fused_qkv", (paged_kv_cache, R.prim_value(0), R.prim_value(Tx.float32(0.088388347648318447)), reshape1), out_sinfo=R.Tensor((batch_size, 64, 128), dtype="float16"))
            reshape2: R.Tensor((batch_size, 1, 64, 128), dtype="float16") = R.reshape(lv, R.shape([batch_size, 1, 64, 128]))
            reshape3: R.Tensor((batch_size, 1, 8192), dtype="float16") = R.reshape(reshape2, R.shape([batch_size, 1, 8192]))
            permute_dims1: R.Tensor((8192, 5120), dtype="float16") = R.permute_dims(model_layers_0_self_attn_o_proj_weight1, axes=None)
            matmul1: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.matmul(reshape3, permute_dims1, out_dtype="void")
            add: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.add(matmul1, input_embeds)
            rms_norm3: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.nn.rms_norm(add, model_layers_0_post_attention_layernorm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            permute_dims2: R.Tensor((5120, 51200), dtype="float16") = R.permute_dims(model_layers_0_mlp_gate_up_proj_weight1, axes=None)
            matmul2: R.Tensor((batch_size, 1, 51200), dtype="float16") = R.matmul(rms_norm3, permute_dims2, out_dtype="void")
            split1: R.Tuple(R.Tensor((batch_size, 1, 25600), dtype="float16"), R.Tensor((batch_size, 1, 25600), dtype="float16")) = R.split(matmul2, indices_or_sections=2, axis=-1)
            split_01: R.Tensor((batch_size, 1, 25600), dtype="float16") = split1[0]
            split_11: R.Tensor((batch_size, 1, 25600), dtype="float16") = split1[1]
            silu: R.Tensor((batch_size, 1, 25600), dtype="float16") = R.nn.silu(split_01)
            mul: R.Tensor((batch_size, 1, 25600), dtype="float16") = R.multiply(silu, split_11)
            permute_dims3: R.Tensor((25600, 5120), dtype="float16") = R.permute_dims(model_layers_0_mlp_down_proj_weight1, axes=None)
            matmul3: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.matmul(mul, permute_dims3, out_dtype="void")
            add1: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.add(matmul3, add)
            rms_norm4: R.Tensor((batch_size, 1, 5120), dtype="float16") = R.nn.rms_norm(add1, model_norm_weight1, axes=[-1], epsilon=9.9999999999999995e-07)
            permute_dims4: R.Tensor((5120, 151936), dtype="float16") = R.permute_dims(lm_head_weight1, axes=None)
            matmul4: R.Tensor((batch_size, 1, 151936), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")
            astype: R.Tensor((batch_size, 1, 151936), dtype="float32") = R.astype(matmul4, dtype="float32")
            gv1: R.Tuple(R.Tensor((batch_size, 1, 151936), dtype="float32"), R.Object) = astype, paged_kv_cache
            R.output(gv1)
        return gv1
# fmt: on


def get_ref_batch_decode_func():
    with target:
        ex = tvm.compile(Module2, target, relax_pipeline=relax.get_pipeline("opt_llm_2"))
        vm = relax.VirtualMachine(ex, dev)
    return vm["batch_decode"]


res2 = test_qwen3_layer(get_ref_batch_decode_func())


for i, (mg, ref) in enumerate(zip(res1, res2)):
    print(f"batch {i}")
    mg = mg.numpy()
    ref = ref.numpy().reshape(mg.shape)
    tokens_mg = sample_token(mg)
    tokens_ref = sample_token(ref)
    try:
        np.testing.assert_equal(tokens_mg, tokens_ref)
    except Exception as e:
        print(e)
