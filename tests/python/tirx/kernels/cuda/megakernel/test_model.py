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
# ruff: noqa: E402

import functools
import json
import random
from argparse import ArgumentParser
from pathlib import Path

import pytest

pytest.skip(
    "Skipping pytest for tests/python/tirx/kernels/cuda/megakernel/test_model.py.",
    allow_module_level=True,
)

import numpy as np
from mlc_llm.compiler_pass.attach_support_info import AttachMemoryPlanAttr, AttachVariableBounds
from mlc_llm.compiler_pass.blas_dispatch import BLASDispatch
from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation
from mlc_llm.compiler_pass.fuse_add_norm import FuseAddRMSNorm
from mlc_llm.compiler_pass.pipeline import _DebugDump
from mlc_llm.model.llama.llama_model import LlamaConfig, LlamaForCausalLM
from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
from mlc_llm.model.qwen3_moe.qwen3_moe_model import Qwen3MoeConfig, Qwen3MoeForCausalLM
from mlc_llm.nn.kv_cache import PagedKVCache
from tirx_kernels.megakernel.layer import MegaKernelDenseLayer
from tirx_kernels.megakernel.lm_head import LMHeadLayer
from tirx_kernels.megakernel.moe_full_layer import MegaKernelMOEFullLayer
from tqdm import tqdm

import tvm
from tvm import dlight, relax
from tvm.megakernel.model.llama3_1b import get_llama3_megakernel_relax_mod
from tvm.megakernel.model.qwen3_30b_a3b import get_qwen3_30b_a3b_megakernel_relax_mod
from tvm.megakernel.model.qwen3_32b import get_qwen3_megakernel_relax_mod
from tvm.megakernel.utils.config import llama3_1b_config, qwen3_30b_a3b_config, qwen3_32b_config
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di

from ..sm100a.test_rmsnorm import get_rmsnorm_kernel
from ..sm100a.test_rope import get_cos_sin_cache_kernel

# pyright: reportInvalidTypeForm=false


def test(args):
    dev = tvm.cuda()
    target = tvm.target.Target("cuda")

    # profiler
    # for detailed config, modify in /python/tvm/megakernel/utils/support.py
    PROFILER_ON = args.profiler_on
    MAX_BATCH_SIZE = 128
    # model config
    if args.model == "Qwen3-32B":
        mk_config = qwen3_32b_config
        mlc_config_tp1 = Qwen3Config(
            hidden_act="silu",
            hidden_size=mk_config["HIDDEN_SIZE"],
            intermediate_size=mk_config["INTERMEDIATE_SIZE"],
            attention_bias=False,
            num_attention_heads=mk_config["NUM_ATTENTION_HEADS"],
            num_hidden_layers=mk_config["NUM_HIDDEN_LAYERS"],  # 64,
            num_key_value_heads=mk_config["NUM_KEY_VALUE_HEADS"],
            rms_norm_eps=mk_config["RMS_NORM_EPS"],
            rope_theta=mk_config["ROPE_THETA"],
            vocab_size=mk_config["VOCAB_SIZE"],
            tie_word_embeddings=mk_config["TIE_WORD_EMBEDDINGS"],
            context_window_size=mk_config["MAX_POSITION_EMBEDDINGS"],
            prefill_chunk_size=2048,
            tensor_parallel_shards=1,
            head_dim=mk_config["HEAD_DIM"],
            dtype="float32",
            max_batch_size=MAX_BATCH_SIZE,
            weight_block_size=None,
            kwargs={"megakernel": True},
        )
        mlc_model_func = Qwen3LMHeadModel
        mk_wrapper_class = MegaKernelDenseLayer
        relax_mod_func = get_qwen3_megakernel_relax_mod
        model_type = 0  # 0: dense, 1: moe
    elif args.model == "Qwen3-30B-A3B":
        mk_config = qwen3_30b_a3b_config
        mlc_config_tp1 = Qwen3MoeConfig(
            hidden_act="silu",
            hidden_size=mk_config["HIDDEN_SIZE"],
            intermediate_size=mk_config["INTERMEDIATE_SIZE"],
            attention_bias=False,
            num_attention_heads=mk_config["NUM_ATTENTION_HEADS"],
            num_hidden_layers=mk_config["NUM_HIDDEN_LAYERS"],  # 48,
            num_key_value_heads=mk_config["NUM_KEY_VALUE_HEADS"],
            rms_norm_eps=mk_config["RMS_NORM_EPS"],
            rope_theta=mk_config["ROPE_THETA"],
            vocab_size=mk_config["VOCAB_SIZE"],
            tie_word_embeddings=mk_config["TIE_WORD_EMBEDDINGS"],
            context_window_size=mk_config["MAX_POSITION_EMBEDDINGS"],
            prefill_chunk_size=2048,
            tensor_parallel_shards=1,
            head_dim=mk_config["HEAD_DIM"],
            dtype="float32",
            max_batch_size=MAX_BATCH_SIZE,
            weight_block_size=None,
            kwargs={"megakernel": True},
            moe_intermediate_size=mk_config["INTERMEDIATE_SIZE"],
            num_experts_per_tok=mk_config["NUM_EXPERTS_PER_TOK"],
            num_experts=mk_config["NUM_EXPERTS"],
            decoder_sparse_step=1,
            norm_topk_prob=True,
        )
        mlc_model_func = Qwen3MoeForCausalLM
        mk_wrapper_class = MegaKernelMOEFullLayer
        relax_mod_func = functools.partial(
            get_qwen3_30b_a3b_megakernel_relax_mod, max_batch_size=MAX_BATCH_SIZE
        )
        model_type = 1  # 0: dense, 1: moe
    elif args.model == "Llama3-1B":
        mk_config = llama3_1b_config
        mlc_config_tp1 = LlamaConfig(
            hidden_size=mk_config["HIDDEN_SIZE"],
            intermediate_size=mk_config["INTERMEDIATE_SIZE"],
            num_attention_heads=mk_config["NUM_ATTENTION_HEADS"],
            num_hidden_layers=mk_config["NUM_HIDDEN_LAYERS"],  # 16,
            num_key_value_heads=mk_config["NUM_KEY_VALUE_HEADS"],
            rms_norm_eps=mk_config["RMS_NORM_EPS"],
            vocab_size=mk_config["VOCAB_SIZE"],
            tie_word_embeddings=mk_config["TIE_WORD_EMBEDDINGS"],
            context_window_size=mk_config["MAX_POSITION_EMBEDDINGS"],
            prefill_chunk_size=2048,
            tensor_parallel_shards=1,
            head_dim=mk_config["HEAD_DIM"],
            max_batch_size=MAX_BATCH_SIZE,
            rope_scaling=mk_config["ROPE_SCALING"],
            kwargs={"megakernel": True, "position_embedding_base": mk_config["ROPE_THETA"]},
        )
        mlc_model_func = LlamaForCausalLM
        mk_wrapper_class = MegaKernelDenseLayer
        relax_mod_func = get_llama3_megakernel_relax_mod
        model_type = 3  # 3 for llama3_1b
    else:
        raise ValueError(f"Invalid model: {args.model}")

    TP_SIZE = args.tp_size
    if args.model == "Qwen3-30B-A3B":
        assert TP_SIZE == 1
    # notes: "/raid/catalyst/models/Qwen3-32B-q0f16-MLC" is the weights converted directly from huggingface
    #        "/raid/catalyst/models/Qwen3-32B-q0f16-MLC-mega" is the weights converted with interwoven gate_up_weight
    use_mega_weights = (
        mk_config["GATE_UP_PROJ_SPLIT_K_FACTOR_DICT"][args.tp_size] == 1
        if "GATE_UP_PROJ_SPLIT_K_FACTOR_DICT" in mk_config
        else model_type == 1
    )
    LOAD_WEIGHTS = (
        f"/raid/catalyst/models/{args.model}-q0f16-MLC-mega"
        if use_mega_weights
        else f"/raid/catalyst/models/{args.model}-q0f16-MLC"
    )
    MODEL_LIB_PATH = f"/raid/catalyst/ruihang-shared/latest/{args.model}-q0f16-tp{TP_SIZE}.so"
    MEGA_LIB_PATH = f"{Path('~/megalib').expanduser()}/{args.model}-q0f16-MLC-{args.scheduler}-tp{TP_SIZE}-profiler{'on' if PROFILER_ON else 'off'}.so"  # NOTE: update this path
    DEBUG_PATH = Path("~/qwen3-mg-debug").expanduser()  # NOTE: update this path

    # LOAD_WEIGHTS = None  # generate weights
    MAX_SEQ_LEN = 1024
    MAX_TOTAL_SEQ_LEN = MAX_BATCH_SIZE * MAX_SEQ_LEN
    PAGE_SIZE = 16

    nvshmem_initialized = False

    # problem config
    BATCH_SIZE = args.init_batch_size
    SEQ_LEN = args.final_seq_len
    assert BATCH_SIZE <= MAX_BATCH_SIZE
    assert SEQ_LEN <= MAX_SEQ_LEN

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
                    AttachVariableBounds({"batch_size": mlc_config_tp1.max_batch_size}),
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
                ]
            )
            mod = seq(mod)
            return mod

        return _pipeline

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

            torch.manual_seed(1)
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
                result.append(tvm.runtime.from_dlpack(torch.to_dlpack(torch_tensor)))
        else:
            from tvm.contrib import tvmjs

            print("Loading weights from", LOAD_WEIGHTS)
            if TP_SIZE == 1:
                params, _ = tvmjs.load_tensor_cache(LOAD_WEIGHTS, device=dev)
                print("Loaded", len(params), "weights")
                result = [params[k] for k, v in named_params]
            else:
                loader = disco_sess.get_global_func("mlc.multi_gpu.LoadMultiGPU")
                result = disco_sess.call_packed(
                    loader, LOAD_WEIGHTS, vm, json.dumps({"vocab_size": mk_config["VOCAB_SIZE"]})
                )
        return result

    def sample_token(logits):
        return np.argmax(logits, axis=-1)

    def load_reference_model_lib():
        if TP_SIZE == 1:
            ex = tvm.runtime.load_module(MODEL_LIB_PATH)
            vm = relax.VirtualMachine(ex, dev)
            batch_decode_func = vm["batch_decode"]
            kv_cache_create_func = vm["create_flashinfer_paged_kv_cache"]
            embed_func = vm["embed"]
        else:
            vm = get_global_func("runtime.disco.load_vm_module")(MODEL_LIB_PATH, None)
            mod_get_func = get_global_func("ffi.ModuleGetFunction")
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

    model = mlc_model_func(mlc_config_tp1)
    model.to("float16")
    _, named_params = model.export_tvm(get_default_spec(model))

    get_global_func = (
        tvm.get_global_func
        if TP_SIZE == 1
        else lambda name: (
            lambda *args: disco_sess.call_packed(disco_sess.get_global_func(name), *args)
        )
    )

    vm, batch_decode_func, kv_cache_create_func, embed_func = load_reference_model_lib()
    params = get_params(named_params, vm)
    decrease_bs = set(random.sample(range(1, SEQ_LEN + 1), BATCH_SIZE))

    def test_qwen3_model(
        get_global_func,
        batch_decode_func,
        kv_cache_create_func,
        embed_func,
        is_megakernel=False,
        cos_sin_cache_func=None,
    ):
        nonlocal nvshmem_initialized
        if not nvshmem_initialized and TP_SIZE > 1:
            print("start to initialize nvshmem")
            f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
            uid = f_init_nvshmem_uid()
            f_init_nvshmem = get_global_func("runtime.disco.nvshmem.init_nvshmem")
            f_init_nvshmem(uid, TP_SIZE, 0)
            nvshmem_initialized = True
            print("nvshmem initialized")

        kv_cache = kv_cache_create_func(
            ShapeTuple([MAX_BATCH_SIZE]),  # max_batch_size
            ShapeTuple([MAX_TOTAL_SEQ_LEN]),  # max_total_seq_len
            ShapeTuple([MAX_SEQ_LEN]),  # prefill_chunk_size
            ShapeTuple([PAGE_SIZE]),  # page_size
            ShapeTuple([bool(is_megakernel)]),  # may_use_megakernel
            ShapeTuple([model_type]),  # model_type, see megakernel_utils.h for details
        )

        if is_megakernel:
            cos_sin_cache = cos_sin_cache_func(ShapeTuple([MAX_SEQ_LEN]))
            # cos_sin_cache = prepare_cos_sin_cache(128, MAX_SEQ_LEN, ROPE_THETA)
            # cos_sin_cache = tvm.nd.array.from_dlpack(cos_sin_cache.to_dlpack())

        nd_view_func = get_global_func("vm.builtin.reshape")

        def embed(tokens, params, batch_size):
            _embed = embed_func(tokens, params)
            _embed = nd_view_func(_embed, ShapeTuple([batch_size, 1, mlc_config_tp1.hidden_size]))
            return _embed

        add_sequence_func = get_global_func("vm.builtin.kv_state_add_sequence")
        begin_forward_func = get_global_func("vm.builtin.kv_state_begin_forward")
        end_forward_func = get_global_func("vm.builtin.kv_state_end_forward")

        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
        seq_ids = []

        for i in range(batch_size):
            seq_ids.append(i)
            add_sequence_func(kv_cache, i)

        np.random.seed(1)
        logits_arr = list()
        last_tokens = np.random.randint(0, 100, size=(batch_size,))
        for i in tqdm(range(seq_len)):
            if i in decrease_bs and batch_size > 1:
                batch_size -= 1
                last_tokens = last_tokens[:batch_size]
                seq_ids = seq_ids[:batch_size]
            print(f"cur batch_size: {batch_size}", flush=True)
            tokens = tvm.runtime.tensor(last_tokens.astype("int32"), device=dev)
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

    def attach_attr(func, name):
        func = func.without_attr("global_symbol")
        func = func.with_attr("global_symbol", name)
        func = func.with_attr("tir.is_scheduled", True)
        func = func.with_attr("tir.noalias", True)
        return func

    def get_qwen3_megakernel_mod():
        rms_norm = get_rmsnorm_kernel(mk_config["HIDDEN_SIZE"])
        mk = mk_wrapper_class(mk_config, world_size=TP_SIZE, profiler_on=PROFILER_ON)
        layer_kernel = mk.get_func(args.scheduler)
        hgemm = LMHeadLayer(N=mk_config["VOCAB_SIZE"], K=mk_config["HIDDEN_SIZE"]).get_func()
        cos_sin_cache = get_cos_sin_cache_kernel(mk_config["HEAD_DIM"], mk_config["ROPE_THETA"])
        relax_mod = relax_mod_func(mk, args.scheduler, TP_SIZE, PROFILER_ON)
        mod = relax_mod
        mod.update_func(mod.get_global_var("rms_norm"), attach_attr(rms_norm, "rms_norm"))
        mod.update_func(mod.get_global_var("hgemm"), attach_attr(hgemm, "hgemm"))
        mod.update_func(
            mod.get_global_var("layer_kernel"), attach_attr(layer_kernel, "layer_kernel")
        )
        mod.update_func(
            mod.get_global_var("cos_sin_cache"), attach_attr(cos_sin_cache, "cos_sin_cache")
        )
        mod = mod.with_attr("relax.skip_shape_check", True)
        return mod

    def get_qwen3_megakernel_batch_decode_func():
        mg_model = get_qwen3_megakernel_mod()

        with target:
            ex = tvm.compile(
                mg_model,
                target,
                relax_pipeline=relax.get_pipeline("opt_llm_mg"),
                tir_pipeline="tirx",
            )
            ex.export_library(MEGA_LIB_PATH)
        if TP_SIZE == 1:
            vm = relax.VirtualMachine(ex, dev)
            batch_decode_func = vm["batch_decode"]
            cos_sin_cache_func = vm["cos_sin_cache_func"]
        else:
            vm = get_global_func("runtime.disco.load_vm_module")(MEGA_LIB_PATH, None)
            mod_get_func = get_global_func("ffi.ModuleGetFunction")
            batch_decode_func_ = mod_get_func(vm, "batch_decode", True)
            cos_sin_cache_func_ = mod_get_func(vm, "cos_sin_cache_func", True)

            def batch_decode_func(*args):
                return disco_sess.call_packed(batch_decode_func_, *args)

            def cos_sin_cache_func(*args):
                return disco_sess.call_packed(cos_sin_cache_func_, *args)

        return batch_decode_func, cos_sin_cache_func

    print("start to get qwen3 megakernel batch decode func", flush=True)
    batch_decode_func_mega, cos_sin_cache_func_mega = get_qwen3_megakernel_batch_decode_func()

    res0 = test_qwen3_model(get_global_func, batch_decode_func, kv_cache_create_func, embed_func)
    res1 = test_qwen3_model(
        get_global_func,
        batch_decode_func_mega,
        kv_cache_create_func,
        embed_func,
        is_megakernel=True,
        cos_sin_cache_func=cos_sin_cache_func_mega,
    )

    for i, (ref, mg) in enumerate(zip(res0, res1)):
        print(f"seq {i}", flush=True)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-32B",
        choices=["Qwen3-32B", "Qwen3-30B-A3B", "Llama3-1B"],
    )
    parser.add_argument("--tp-size", type=int, default=1, choices=[1, 4, 8])
    parser.add_argument("--profiler-on", action="store_true", default=False)
    parser.add_argument("--scheduler", type=str, default="static", choices=["static", "dynamic"])
    parser.add_argument("--init-batch-size", type=int, default=128)
    parser.add_argument("--final-seq-len", type=int, default=300)
    args = parser.parse_args()
    test(args)
