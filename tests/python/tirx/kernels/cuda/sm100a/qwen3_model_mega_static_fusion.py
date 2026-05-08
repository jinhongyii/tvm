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

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from mlc_llm.compiler_pass.attach_support_info import AttachMemoryPlanAttr, AttachVariableBounds
from mlc_llm.compiler_pass.blas_dispatch import BLASDispatch
from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation
from mlc_llm.compiler_pass.fuse_add_norm import FuseAddRMSNorm
from mlc_llm.compiler_pass.pipeline import _DebugDump
from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
from mlc_llm.nn.kv_cache import PagedKVCache
from tirx_kernels.megakernel.layer import (
    MAX_NUM_KV_SPLITS,
    MAX_TOTAL_NUM_WORKERS,
    PROFILER_BUFFER_SIZE,
)
from tirx_kernels.megakernel.static_fused_layer import get_qwen3_layer
from tqdm import tqdm

import tvm
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as Tx

from .test_hgemm_1consumer_1cta_swap_splitk import get_hgemm_kernel
from .test_rmsnorm import get_rmsnorm_kernel
from .test_rope import get_cos_sin_cache_kernel

# pyright: reportInvalidTypeForm=false

parser = ArgumentParser()
parser.add_argument("--tp-size", type=int, default=1, choices=[1])
args = parser.parse_args()

dev = tvm.cuda()
target = tvm.target.Target("cuda")

TP_SIZE = args.tp_size
NUM_HIDDEN_LAYERS = 64
LOAD_WEIGHTS = "/raid/catalyst/models/Qwen3-32B-q0f16-MLC"
MODEL_LIB_PATH = f"/raid/catalyst/ruihang-shared/latest/Qwen3-32B-q0f16-tp{TP_SIZE}.so"
MEGA_LIB_PATH = (
    f"/home/hongyij/qwen3-mg-debug/mega_layer_lib_tp{TP_SIZE}.so"  # NOTE: update this path
)
DEBUG_PATH = "/home/hongyij/qwen3-mg-debug"  # NOTE: update this path
# LOAD_WEIGHTS = None  # generate weights
MAX_BATCH_SIZE = 32
MAX_SEQ_LEN = 1024
MAX_TOTAL_SEQ_LEN = MAX_BATCH_SIZE * MAX_SEQ_LEN
PAGE_SIZE = 16
ROPE_THETA = 1000000

config_tp1 = Qwen3Config(
    hidden_act="silu",
    hidden_size=5120,
    intermediate_size=25600,
    attention_bias=False,
    num_attention_heads=64,
    num_hidden_layers=NUM_HIDDEN_LAYERS,  # 64,
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
    max_batch_size=256,
    weight_block_size=None,
    kwargs={},
)


def init_disco_session():
    if TP_SIZE == 1:
        return None

    devices = [i for i in range(TP_SIZE)]
    sess = di.ProcessSession(num_workers=len(devices), entrypoint="mlc_llm.cli.worker")
    sess.init_ccl("nccl", *devices)
    return sess


disco_sess = init_disco_session()


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
    debug_dir = Path(DEBUG_PATH)

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                AttachVariableBounds({"batch_size": config_tp1.max_batch_size}),
                AttachMemoryPlanAttr(),
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
                # tvm.transform.PrintIR(),
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


def get_params(named_params, vm):
    if LOAD_WEIGHTS is None:
        print("Generating weights")
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        result = list()
        for k, param in tqdm(named_params):
            torch_tensor = (
                torch.empty(tuple(param.shape)).to(getattr(torch, param.dtype)).to("cuda")
            )
            if k.endswith("norm.weight"):
                nn.init.ones_(torch_tensor)
            elif k.endswith("embed_tokens.weight"):
                nn.init.normal_(torch_tensor, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(torch_tensor, gain=1.0)
            torch_tensor = torch_tensor
            result.append(tvm.runtime.ndarray.from_dlpack(torch.to_dlpack(torch_tensor)))
    else:
        from tvm.contrib import tvmjs

        print("Loading weights from", LOAD_WEIGHTS)
        if TP_SIZE == 1:
            params, _ = tvmjs.load_ndarray_cache(LOAD_WEIGHTS, device=dev)
            print("Loaded", len(params), "weights")
            result = [params[k] for k, v in named_params]
        else:
            loader = disco_sess.get_global_func("mlc.multi_gpu.LoadMultiGPU")
            result = disco_sess.call_packed(
                loader, LOAD_WEIGHTS, vm, json.dumps({"vocab_size": config_tp1.vocab_size})
            )
    return result


def sample_token(logits):
    return np.argmax(logits, axis=-1)


model = Qwen3LMHeadModel(config_tp1)
model.to("float16")
mod, named_params = model.export_tvm(get_default_spec(model))

get_global_func = (
    tvm.get_global_func
    if TP_SIZE == 1
    else lambda name: lambda *args: disco_sess.call_packed(disco_sess.get_global_func(name), *args)
)


def load_reference_model_lib():
    if TP_SIZE == 1:
        ex = tvm.runtime.load_module(MODEL_LIB_PATH)
        vm = relax.VirtualMachine(ex, dev)
        batch_decode_func = vm["batch_decode"]
        kv_cache_create_func = vm["create_flashinfer_paged_kv_cache"]
        embed_func = vm["embed"]
    else:
        vm = get_global_func("runtime.disco.load_vm_module")(MODEL_LIB_PATH, None)
        mod_get_func = get_global_func("runtime.ModuleGetFunction")
        batch_decode_func_ = mod_get_func(vm, "batch_decode", True)
        kv_cache_create_func_ = mod_get_func(vm, "create_flashinfer_paged_kv_cache", True)
        embed_func_ = mod_get_func(vm, "embed", True)

        def batch_decode_func(*args):
            return disco_sess.call_packed(batch_decode_func_, *args)

        def kv_cache_create_func(*args):
            return disco_sess.call_packed(kv_cache_create_func_, *args)

        def embed_func(*args):
            return disco_sess.call_packed(embed_func_, *args)

    return vm, batch_decode_func, kv_cache_create_func, embed_func


vm, batch_decode_func, kv_cache_create_func, embed_func = load_reference_model_lib()
params = get_params(named_params, vm)


def test_qwen3_model(
    get_global_func,
    batch_decode_func,
    kv_cache_create_func,
    embed_func,
    is_megakernel=False,
    cos_sin_cache_func=None,
):
    kv_cache = kv_cache_create_func(
        ShapeTuple([MAX_BATCH_SIZE]),  # max_batch_size
        ShapeTuple([MAX_TOTAL_SEQ_LEN]),  # max_total_seq_len
        ShapeTuple([MAX_SEQ_LEN]),  # prefill_chunk_size
        ShapeTuple([PAGE_SIZE]),  # page_size
        ShapeTuple([2 if is_megakernel else 0]),  # megakernel_mode
    )

    if is_megakernel:
        cos_sin_cache = cos_sin_cache_func(ShapeTuple([MAX_SEQ_LEN]))
        # cos_sin_cache = prepare_cos_sin_cache(128, MAX_SEQ_LEN, ROPE_THETA)
        # cos_sin_cache = tvm.nd.array.from_dlpack(cos_sin_cache.to_dlpack())

    nd_view_func = get_global_func("vm.builtin.reshape")

    def embed(tokens, params, batch_size):
        _embed = embed_func(tokens, params)
        _embed = nd_view_func(_embed, ShapeTuple([batch_size, 1, config_tp1.hidden_size]))
        return _embed

    add_sequence_func = get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward_func = get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward_func = get_global_func("vm.builtin.kv_state_end_forward")

    batch_size = 31
    seq_len = 16
    seq_ids = []

    for i in range(batch_size):
        seq_ids.append(i)
        add_sequence_func(kv_cache, i)

    np.random.seed(0)
    logits_arr = list()
    last_tokens = np.random.randint(0, 100, size=(batch_size,))
    for i in tqdm(range(seq_len)):
        tokens = tvm.nd.array(last_tokens.astype("int32"), device=dev)
        if TP_SIZE > 1:
            tokens_d = get_global_func("runtime.disco.empty")(
                ShapeTuple(list(last_tokens.shape)), "int32", None, False, False
            )
            disco_sess.copy_to_worker_0(tokens, tokens_d)
            tokens = tokens_d
        hidden_states = embed(tokens, params, batch_size)
        begin_forward_func(
            kv_cache, ShapeTuple(seq_ids), ShapeTuple([1] * batch_size), None, is_megakernel
        )
        if is_megakernel:
            results = batch_decode_func(hidden_states, kv_cache, cos_sin_cache, params)
        else:
            results = batch_decode_func(hidden_states, kv_cache, params)
        if TP_SIZE == 1:
            logits, _ = results
        else:
            logits, _ = list(results.debug_get_from_remote(0))

        end_forward_func(kv_cache)
        last_tokens = sample_token(logits.numpy()).flatten()
        logits_arr.append(logits)

    return logits_arr


res0 = test_qwen3_model(get_global_func, batch_decode_func, kv_cache_create_func, embed_func)


def attach_attr(func, name):
    func = func.without_attr("global_symbol")
    func = func.with_attr("global_symbol", name)
    func = func.with_attr("tir.is_scheduled", True)
    func = func.with_attr("tir.noalias", True)
    return func


def get_qwen3_megakernel_mod():
    rms_norm = get_rmsnorm_kernel(5120)
    layer_megakernel = get_qwen3_layer(max_batch_size=MAX_BATCH_SIZE)
    hgemm, reduce, tile_k_num = get_hgemm_kernel(dim_n=151936, dim_k=5120)
    cos_sin_cache = get_cos_sin_cache_kernel(128, ROPE_THETA)

    # fmt: off
    @R.macro(hygienic=False)
    def call_qwen3_layer(input0, input1, layer_id):
        with R.dataflow():
            # 8i+1, 8i+2, 8i+3, 8i+4, 8i+5, 8i+6, 8i+8, 8i+15 if i<num_hidden_layers-1 else 8i+9
            model_layers_0_self_attn_c_attn_weight1: R.Tensor((10240 // TP_SIZE, 5120), dtype="float16") = packed_params[8*layer_id+1]
            model_layers_0_self_attn_o_proj_weight1: R.Tensor((5120, 8192 // TP_SIZE), dtype="float16") = packed_params[8*layer_id+2]
            model_layers_0_self_attn_q_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[8*layer_id+3]
            model_layers_0_self_attn_k_norm_weight1: R.Tensor((128,), dtype="float16") = packed_params[8*layer_id+4]
            model_layers_0_mlp_gate_up_proj_weight1: R.Tensor((51200 // TP_SIZE, 5120), dtype="float16") = packed_params[8*layer_id+5]
            model_layers_0_mlp_down_proj_weight1: R.Tensor((5120, 25600 // TP_SIZE), dtype="float16") = packed_params[8*layer_id+6]
            model_layers_0_post_attention_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8*layer_id+8]
            model_norm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[8*layer_id+15 if layer_id < NUM_HIDDEN_LAYERS-1 else 8*layer_id+9]

            etensors_on_layer = R.call_pure_packed(
                "megakernel.get_event_tensors_on_layer",
                etensors,
                R.prim_value(layer_id),
                sinfo_args=[
                    R.Tuple([R.Tensor(ndim=1, dtype="int32")] + [R.Tensor(ndim=2, dtype="int32")] * 2 + [R.Tensor(ndim=1, dtype="int32")] * 12 + [R.Tensor(ndim=2, dtype="int32")]),
                ]
            )
            (
                etensor_qkv_partial,
                etensor_qkv_reduce,
                etensor_attn,
                etensor_o_partial,
                _etensor_o_allreduce,
                etensor_attn_add_rms_norm,
                etensor_attn_mlp,
                _etensor_gate_up_proj_reduce,
                etensor_gate_up_proj,
                etensor_down_proj_reduce,
                _etensor_down_proj_allreduce,
                etensor_mlp_add_rms_norm,
                _etensor_end,
                etensor_o_proj,
                etensor_down_proj,
                etensor_attn_merge,
            ) = (
                etensors_on_layer[0],
                etensors_on_layer[1],
                etensors_on_layer[2],
                etensors_on_layer[3],
                etensors_on_layer[4],
                etensors_on_layer[5],
                etensors_on_layer[6],
                etensors_on_layer[7],
                etensors_on_layer[8],
                etensors_on_layer[9],
                etensors_on_layer[10],
                etensors_on_layer[11],
                etensors_on_layer[12],
                etensors_on_layer[13],
                etensors_on_layer[14],
                etensors_on_layer[15],
            )
            R.builtin.alloc_tensor(R.shape([PROFILER_BUFFER_SIZE]), dtype="uint64", runtime_device_index=0)

            tuple_1 = (
                    kv_data[layer_id],
                    page_kv_indptr,
                    page_kv_indices,
                    page_kv_last_page_len,
                    append_pos,
                    rope_pos,
                    len_kv_chunk,
                    merge_indptr,
                    merge_o_indices,
                    num_qo_len,
                    q_indptr,
                    kv_indptr,
                    partial_indptr,
                    q_len,
                    kv_len,
                    q_start,
                    kv_start,
                    kv_end,
                    kv_head_idx,
                    work_indptr,
                    cos_sin_cache,
                    attn_task_num,
                )
            tuple_2 = (
                    model_layers_0_self_attn_c_attn_weight1,
                    model_layers_0_self_attn_q_norm_weight1, model_layers_0_self_attn_k_norm_weight1,
                    model_layers_0_self_attn_o_proj_weight1,
                    model_layers_0_post_attention_layernorm_weight1,
                    model_layers_0_mlp_gate_up_proj_weight1, model_layers_0_mlp_down_proj_weight1,
                    model_norm_weight1,
                )
            tuple_3 = (
                etensor_qkv_partial,
                etensor_qkv_reduce,
                etensor_attn,
                etensor_attn_merge,
                etensor_o_proj,
                etensor_o_partial,
                etensor_attn_add_rms_norm,
                etensor_attn_mlp,
                etensor_gate_up_proj,
                etensor_down_proj,
                etensor_down_proj_reduce,
                etensor_mlp_add_rms_norm,
            )
            layer_res = cls.megakernel(
                input0,
                input1,
                tuple_1,
                tuple_2,
                tuple_3,
            )
            R.output(layer_res)
        return layer_res


    @I.ir_module
    class Module:
        @Tx.prim_func
        def rms_norm(input_ptr: Tx.handle, weight_ptr: Tx.handle, out_ptr: Tx.handle):
            pass

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
        def hgemm(A_ptr: Tx.handle, b_ptr: Tx.handle, partial_sum_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def reduce(partial_sum_ptr: Tx.handle, D_ptr: Tx.handle):
            pass

        @Tx.prim_func(private=True)
        def cos_sin_cache(cos_sin_cache: Tx.handle):
            pass

        @R.function
        def cos_sin_cache_func(max_seq_len_: R.Shape(["max_seq_len"])):
            max_seq_len = Tx.int64()
            cls = Module
            with R.dataflow():
                cache: R.Tensor((max_seq_len, 128), dtype="float32") = R.call_tir(cls.cos_sin_cache, [], out_sinfo=R.Tensor((max_seq_len, 128), dtype="float32") )
                R.output(cache)
            return cache

        @R.function
        def megakernel(x: R.Tensor, residual: R.Tensor, packed_kv_cache_info: R.Tuple(*([R.Tensor] * 21 + [R.Shape()])), packed_weights: R.Tuple(*([R.Tensor] * 8)), packed_events: R.Tuple(*([R.Tensor] * 12))):
            return x, residual

        @R.function
        def batch_decode(
            input_embeds: R.Tensor(("batch_size", 1, 5120), dtype="float16"),
            paged_kv_cache: R.Object,
            # rope
            cos_sin_cache: R.Tensor(("max_seq_len", 128), dtype="float32"),
            packed_params: R.Tuple(
                R.Tensor((151936, 5120), dtype="float16"),
                #
                *([R.Tensor((10240 // TP_SIZE, 5120), dtype="float16"),
                R.Tensor((5120, 8192 // TP_SIZE), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((128,), dtype="float16"),
                R.Tensor((51200 // TP_SIZE, 5120), dtype="float16"),
                R.Tensor((5120, 25600 // TP_SIZE), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((5120,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                R.Tensor((5120,), dtype="float16"),
                R.Tensor((151936, 5120), dtype="float16"),
            ),
        ):
            batch_size = Tx.int64()
            total_page_num = Tx.int64()
            max_page_num = Tx.int64()
            page_size = Tx.int64()

            cls = Module
            with R.dataflow():
                res0 = R.call_pure_packed(
                    "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                    paged_kv_cache, R.prim_value(1),
                    sinfo_args=[
                        R.Tuple([R.Tensor(None, dtype="float16")] * NUM_HIDDEN_LAYERS),
                        R.Tensor((batch_size + 1,), dtype="int32"),
                        R.Tensor(None, dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tensor((batch_size,), dtype="int32"),
                        R.Tuple([R.Prim("int64")] * 2 + [R.Tuple([R.Tensor(None, dtype="int32")] * 13)] * 2 + [R.Tensor(None, dtype="int32")] * 5),
                        R.Tuple([R.Tensor(None, dtype="int32")] * 18),
                        R.Shape(),
                    ],
                )
                kv_data_, _page_kv_indptr, page_kv_indices_, _page_kv_last_page_len, _append_pos, _rope_pos, attn_plan_results, _etensors, _attn_task_num = (
                    res0[0],
                    res0[1],
                    res0[2],
                    res0[3],
                    res0[4],
                    res0[5],
                    res0[6],
                    res0[7],
                    res0[8],
                )
                R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, 8 // TP_SIZE, page_size, 128), dtype="float16")] * NUM_HIDDEN_LAYERS))
                R.match_cast(page_kv_indices_, R.Tensor((total_page_num,), dtype="int32"))
                task, len_kv_chunk_, merge_indptr_, merge_o_indices_, num_qo_len_ = attn_plan_results[3], attn_plan_results[4], attn_plan_results[5], attn_plan_results[6], attn_plan_results[7]
                q_indptr_, kv_indptr_, partial_indptr_, q_len_, kv_len_, q_start_, kv_start_, kv_end_, kv_head_idx_, work_indptr_ = task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7], task[8], task[9]
                R.match_cast(len_kv_chunk_, R.Tensor((2,), dtype="int32"))
                R.match_cast(merge_indptr_, R.Tensor((MAX_NUM_KV_SPLITS,), dtype="int32"))
                R.match_cast(merge_o_indices_, R.Tensor((MAX_NUM_KV_SPLITS,), dtype="int32"))
                R.match_cast(num_qo_len_, R.Tensor((1,), dtype="int32"))
                R.match_cast(q_indptr_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(kv_indptr_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(partial_indptr_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(q_len_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(kv_len_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(q_start_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(kv_start_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(kv_end_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(kv_head_idx_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                R.match_cast(work_indptr_, R.Tensor((MAX_TOTAL_NUM_WORKERS,), dtype="int32"))

                model_layers_0_input_layernorm_weight1: R.Tensor((5120,), dtype="float16") = packed_params[7]
                lm_head_weight1: R.Tensor((151936, 5120), dtype="float16") = packed_params[NUM_HIDDEN_LAYERS*8+2] # num_hidden_layers*8+2

                rs0 = R.reshape(input_embeds, (batch_size, 5120))
                rms_norm = R.call_tir(cls.rms_norm, (rs0, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, 5120), dtype="float16"))

                o_layer0 = call_qwen3_layer(rms_norm, rs0, 0)
                o_layer1 = call_qwen3_layer(o_layer0[0], o_layer0[1], 1)
                o_layer2 = call_qwen3_layer(o_layer1[0], o_layer1[1], 2)
                o_layer3 = call_qwen3_layer(o_layer2[0], o_layer2[1], 3)
                o_layer4 = call_qwen3_layer(o_layer3[0], o_layer3[1], 4)
                o_layer5 = call_qwen3_layer(o_layer4[0], o_layer4[1], 5)
                o_layer6 = call_qwen3_layer(o_layer5[0], o_layer5[1], 6)
                o_layer7 = call_qwen3_layer(o_layer6[0], o_layer6[1], 7)
                o_layer8 = call_qwen3_layer(o_layer7[0], o_layer7[1], 8)
                o_layer9 = call_qwen3_layer(o_layer8[0], o_layer8[1], 9)
                o_layer10 = call_qwen3_layer(o_layer9[0], o_layer9[1], 10)
                o_layer11 = call_qwen3_layer(o_layer10[0], o_layer10[1], 11)
                o_layer12 = call_qwen3_layer(o_layer11[0], o_layer11[1], 12)
                o_layer13 = call_qwen3_layer(o_layer12[0], o_layer12[1], 13)
                o_layer14 = call_qwen3_layer(o_layer13[0], o_layer13[1], 14)
                o_layer15 = call_qwen3_layer(o_layer14[0], o_layer14[1], 15)
                o_layer16 = call_qwen3_layer(o_layer15[0], o_layer15[1], 16)
                o_layer17 = call_qwen3_layer(o_layer16[0], o_layer16[1], 17)
                o_layer18 = call_qwen3_layer(o_layer17[0], o_layer17[1], 18)
                o_layer19 = call_qwen3_layer(o_layer18[0], o_layer18[1], 19)
                o_layer20 = call_qwen3_layer(o_layer19[0], o_layer19[1], 20)
                o_layer21 = call_qwen3_layer(o_layer20[0], o_layer20[1], 21)
                o_layer22 = call_qwen3_layer(o_layer21[0], o_layer21[1], 22)
                o_layer23 = call_qwen3_layer(o_layer22[0], o_layer22[1], 23)
                o_layer24 = call_qwen3_layer(o_layer23[0], o_layer23[1], 24)
                o_layer25 = call_qwen3_layer(o_layer24[0], o_layer24[1], 25)
                o_layer26 = call_qwen3_layer(o_layer25[0], o_layer25[1], 26)
                o_layer27 = call_qwen3_layer(o_layer26[0], o_layer26[1], 27)
                o_layer28 = call_qwen3_layer(o_layer27[0], o_layer27[1], 28)
                o_layer29 = call_qwen3_layer(o_layer28[0], o_layer28[1], 29)
                o_layer30 = call_qwen3_layer(o_layer29[0], o_layer29[1], 30)
                o_layer31 = call_qwen3_layer(o_layer30[0], o_layer30[1], 31)
                o_layer32 = call_qwen3_layer(o_layer31[0], o_layer31[1], 32)
                o_layer33 = call_qwen3_layer(o_layer32[0], o_layer32[1], 33)
                o_layer34 = call_qwen3_layer(o_layer33[0], o_layer33[1], 34)
                o_layer35 = call_qwen3_layer(o_layer34[0], o_layer34[1], 35)
                o_layer36 = call_qwen3_layer(o_layer35[0], o_layer35[1], 36)
                o_layer37 = call_qwen3_layer(o_layer36[0], o_layer36[1], 37)
                o_layer38 = call_qwen3_layer(o_layer37[0], o_layer37[1], 38)
                o_layer39 = call_qwen3_layer(o_layer38[0], o_layer38[1], 39)
                o_layer40 = call_qwen3_layer(o_layer39[0], o_layer39[1], 40)
                o_layer41 = call_qwen3_layer(o_layer40[0], o_layer40[1], 41)
                o_layer42 = call_qwen3_layer(o_layer41[0], o_layer41[1], 42)
                o_layer43 = call_qwen3_layer(o_layer42[0], o_layer42[1], 43)
                o_layer44 = call_qwen3_layer(o_layer43[0], o_layer43[1], 44)
                o_layer45 = call_qwen3_layer(o_layer44[0], o_layer44[1], 45)
                o_layer46 = call_qwen3_layer(o_layer45[0], o_layer45[1], 46)
                o_layer47 = call_qwen3_layer(o_layer46[0], o_layer46[1], 47)
                o_layer48 = call_qwen3_layer(o_layer47[0], o_layer47[1], 48)
                o_layer49 = call_qwen3_layer(o_layer48[0], o_layer48[1], 49)
                o_layer50 = call_qwen3_layer(o_layer49[0], o_layer49[1], 50)
                o_layer51 = call_qwen3_layer(o_layer50[0], o_layer50[1], 51)
                o_layer52 = call_qwen3_layer(o_layer51[0], o_layer51[1], 52)
                o_layer53 = call_qwen3_layer(o_layer52[0], o_layer52[1], 53)
                o_layer54 = call_qwen3_layer(o_layer53[0], o_layer53[1], 54)
                o_layer55 = call_qwen3_layer(o_layer54[0], o_layer54[1], 55)
                o_layer56 = call_qwen3_layer(o_layer55[0], o_layer55[1], 56)
                o_layer57 = call_qwen3_layer(o_layer56[0], o_layer56[1], 57)
                o_layer58 = call_qwen3_layer(o_layer57[0], o_layer57[1], 58)
                o_layer59 = call_qwen3_layer(o_layer58[0], o_layer58[1], 59)
                o_layer60 = call_qwen3_layer(o_layer59[0], o_layer59[1], 60)
                o_layer61 = call_qwen3_layer(o_layer60[0], o_layer60[1], 61)
                o_layer62 = call_qwen3_layer(o_layer61[0], o_layer61[1], 62)
                o_layer63 = call_qwen3_layer(o_layer62[0], o_layer62[1], 63)

                # permute_dims4 = R.permute_dims(lm_head_weight1, axes=None)
                # lv4: R.Tensor((batch_size, 151936), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")
                rs4 = R.call_tir(cls.hgemm, (o_layer63[0], lm_head_weight1), out_sinfo=R.Tensor((tile_k_num, batch_size, 151936), dtype="float32"))
                lv4 = R.call_tir(cls.reduce, (rs4,), out_sinfo=R.Tensor((batch_size, 151936), dtype="float16"))

                lv4_rs = R.reshape(lv4, (batch_size, 1, 151936))
                astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, 151936), dtype="float32"))

                gv1 = astype, paged_kv_cache
                R.output(gv1)
            return gv1
    # fmt: on
    mod = Module

    mod.update_func(mod.get_global_var("rms_norm"), attach_attr(rms_norm, "rms_norm"))
    mod.update_func(mod.get_global_var("hgemm"), attach_attr(hgemm, "hgemm"))
    mod.update_func(mod.get_global_var("reduce"), attach_attr(reduce, "reduce"))
    for gv, func in layer_megakernel.functions_items():
        if gv.name_hint == "megakernel":
            mod.update_func(mod.get_global_var("megakernel"), attach_attr(func, "megakernel"))
        else:
            mod.update_func(gv, attach_attr(func, gv.name_hint))
    mod.update_func(
        mod.get_global_var("cos_sin_cache"), attach_attr(cos_sin_cache, "cos_sin_cache")
    )
    return mod


def get_qwen3_megakernel_batch_decode_func():
    mg_model = get_qwen3_megakernel_mod()
    mg_model = relax.transform.StaticHorizontalFusion("megakernel")(mg_model)
    mg_model.update_func(
        mg_model.get_global_var("gen_exec_queue"),
        attach_attr(mg_model["gen_exec_queue"], "gen_exec_queue"),
    )
    mg_model.update_func(
        mg_model.get_global_var("persistent_kernel_megakernel"),
        attach_attr(mg_model["persistent_kernel_megakernel"], "persistent_kernel_megakernel"),
    )
    with target:
        ex = tvm.compile(
            mg_model, target, relax_pipeline=relax.get_pipeline("opt_llm_mg"), tir_pipeline="tirx"
        )
        ex.export_library(MEGA_LIB_PATH)
    if TP_SIZE == 1:
        vm = relax.VirtualMachine(ex, dev)
        batch_decode_func = vm["batch_decode"]
        cos_sin_cache_func = vm["cos_sin_cache_func"]
    else:
        vm = get_global_func("runtime.disco.load_vm_module")(MEGA_LIB_PATH, None)
        mod_get_func = get_global_func("runtime.ModuleGetFunction")
        batch_decode_func_ = mod_get_func(vm, "batch_decode", True)
        cos_sin_cache_func_ = mod_get_func(vm, "cos_sin_cache_func", True)

        def batch_decode_func(*args):
            return disco_sess.call_packed(batch_decode_func_, *args)

        def cos_sin_cache_func(*args):
            return disco_sess.call_packed(cos_sin_cache_func_, *args)

    return batch_decode_func, cos_sin_cache_func


batch_decode_func, cos_sin_cache_func = get_qwen3_megakernel_batch_decode_func()
res1 = test_qwen3_model(
    get_global_func,
    batch_decode_func,
    kv_cache_create_func,
    embed_func,
    is_megakernel=True,
    cos_sin_cache_func=cos_sin_cache_func,
)


for i, (ref, mg) in enumerate(zip(res0, res1)):
    print(f"batch {i}")
    mg = mg.numpy()
    ref = ref.numpy().reshape(mg.shape)
    mg_token = sample_token(mg)
    ref_token = sample_token(ref)
    print(mg.flatten()[:50])
    print(ref.flatten()[:50])
    print("--------------------------------")
    try:
        np.testing.assert_allclose(ref, mg, atol=1e-2, rtol=1e-2)
    except Exception as e:
        print(e)
    try:
        np.testing.assert_allclose(ref_token, mg_token, atol=1e-2, rtol=1e-2)
    except Exception as e:
        print(e)
