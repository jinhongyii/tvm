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
import os
import json
import argparse
import logging
from typing import Dict
import numpy as np  # type: ignore

import tvm
from tvm import relay, relax, runtime, transform
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator
from tvm.target.target import Target

import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torchvision
import tvm
import tvm.testing

from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tvm import topi, relax, te
from tvm.script import tir as T
weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=False,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=False,
    )
    args.add_argument(
        "--target",
        type=str,
        required=False,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        default=None,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        default=None,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        default=None,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=None,
    )
    parsed = args.parse_args()
    parsed.target = parsed.target or os.environ.get("TVM_TARGET")
    parsed.target = Target(parsed.target)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    parsed.rpc_host = parsed.rpc_host or os.environ.get("TVM_RPC_HOST")
    parsed.rpc_port = parsed.rpc_port or int(os.environ.get("TVM_RPC_PORT"))
    parsed.rpc_key = parsed.rpc_key or os.environ.get("TVM_RPC_KEY")
    if parsed.rpc_host and parsed.rpc_port and parsed.rpc_key:
        parsed.rpc_config = ms.runner.RPCConfig(
            tracker_host=parsed.rpc_host,
            tracker_port=parsed.rpc_port,
            tracker_key=parsed.rpc_key,
            session_timeout_sec=180,
        )
        parsed.workers = parsed.rpc_config.count_num_servers(
            allow_missing=False)
    else:
        # check all rpc configs are None
        assert (
            (parsed.rpc_host is None) and (
                parsed.rpc_port is None) and (parsed.rpc_key is None)
        ), "Please set all 'rpc_host', 'rpc_port' and 'rpc_key' to use PRC server"
        parsed.rpc_config = None
        parsed.workers = 1
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def apply_opt_before_tuning(
    relax_mod: IRModule, target: Target
):
    with transform.PassContext(opt_level=3):
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        # relax_mod = relax.transform.BindParams()(relax_mod, params)
        relax_mod = relax.transform.AnnotateLayoutFreeBuffers()(relax_mod)
    return relax_mod


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input_data):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        repeat=5,
        min_repeat_ms=500,
    )
    print(evaluator(*input_data))


def get_runner():
    runner_config = {
        "evaluator_config": ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        ),
        "alloc_repeat": ARGS.alloc_repeat,
    }
    if ARGS.rpc_config:
        runner = ms.runner.RPCRunner(
            rpc_config=ARGS.rpc_config, max_workers=ARGS.workers, **runner_config
        )
    else:
        runner = ms.runner.LocalRunner(**runner_config)

    return runner

@tvm.script.ir_module
class Module:
    @T.prim_func
    def gemm_1(a_1: T.Buffer[(1024, 1024), "int8"], param_0_1: T.Buffer[(1024, 1024), "int8"], T_matmul_NT_1: T.Buffer[(1024, 1024), "int32"]) -> None:
        # function attr dict
        T.func_attr(
            {"tir.noalias": True, "global_symbol": "gemm1"})
        # body
        # with T.block("root")
        for i0_4, i1_4, i2 in T.grid(1024, 1024, 1024):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0_4, i1_4, i2])
                T.reads(a_1[i, k], param_0_1[j, k])
                T.writes(T_matmul_NT_1[i, j])
                with T.init():
                    T_matmul_NT_1[i, j] = 0
                T_matmul_NT_1[i, j] = T_matmul_NT_1[i, j] + \
                    T.cast(a_1[i, k], "int32") * \
                    T.cast(param_0_1[j, k], "int32")
    @T.prim_func
    def gemm_2(a_1: T.Buffer[(1024, 1024), "int8"], param_0_1: T.Buffer[(1024, 1024), "int8"], T_matmul_NT_1: T.Buffer[(1024, 1024), "int32"]) -> None:
        # function attr dict
        T.func_attr(
            {"tir.noalias": True, "global_symbol": "gemm2"})
        # body
        # with T.block("root")
        for i0_4, i1_4, i2 in T.grid(1024, 1024, 1024):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0_4, i1_4, i2])
                T.reads(a_1[i, k], param_0_1[j, k])
                T.writes(T_matmul_NT_1[i, j])
                with T.init():
                    T_matmul_NT_1[i, j] = 0
                T_matmul_NT_1[i, j] = T_matmul_NT_1[i, j] + \
                    T.cast(a_1[i, k], "int32") * \
                    T.cast(param_0_1[j, k], "int32")
    def cast(a: T.buffer[(1024, 1024), "int8"], b: T.buffer[(1024, 1024), "int32"]) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "cast"})
        # body
        # with T.block("root")
        for i0, i1 in T.grid(1024, 1024):
            with T.block("root"):
                T.reads(a[i0, i1])
                T.writes(b[i0, i1])
                b[i0, i1] = T.cast(a[i0, i1], "int32")
    @relax.function            
    def main(x: Tensor(()))

def main():


    input_shape = (1024, 1024)




    def create_model_via_emit_te():
        bb = relax.BlockBuilder()
        x = relax.Var("x", input_shape, relax.DynTensorType(2, "int8"))
        weight1 = relax.Var("weight1", (1024, 1024), relax.DynTensorType(2, "int8"))
        bias1 = relax.Var("bias1", (1024,), relax.DynTensorType(1, "int8"))
        weight2 = relax.Var("weight2", (1024, 1024), relax.DynTensorType(2, "int8"))
        bias2 = relax.Var("bias2", (1024,), relax.DynTensorType(1, "int8"))
        
        
        with bb.function("main", [x, weight1, weight2, bias1, bias2]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.dense, x, weight1)
                lv1 = bb.emit_te(topi.add, lv0, bias1)
                lv2 = bb.emit_te(topi.nn.relu, lv1)
                cast = bb.emit_te(topi.cast, lv2, "int32")
                lv3 = bb.emit_te(topi.nn.dense, cast, weight2)
                lv4 = bb.emit_te(topi.add, lv3, bias2)
                gv = bb.emit_output(lv4)
            bb.emit_func_output(gv)

        return bb.get()

    # translate the ResNet model from Relay to Relax
    mod = create_model_via_emit_te()
    weight1 = tvm.nd.array(np.random.rand(1024, 1024).astype(np.int8))
    weight2 = tvm.nd.array(np.random.rand(1024, 1024).astype(np.int8))
    bias1 = tvm.nd.array(np.random.rand(1024).astype(np.int8))
    bias2 = tvm.nd.array(np.random.rand(1024).astype(np.int8))
    mod = relax.transform.BindParams("main", {"weight1": weight1, "weight2": weight2, "bias1": bias1, "bias2": bias2})(mod)

    relax_mod = apply_opt_before_tuning(mod, target=ARGS.target)
    print(relax_mod.script())

    assert isinstance(relax_mod, tvm.IRModule)

    executable = ms.tune_relax(
        mod=relax_mod,
        target=ARGS.target,
        config=ms.TuneConfig(
            strategy="evolutionary",
            num_trials_per_iter=64,
            max_trials_per_task=ARGS.num_trials,
            max_trials_global=ARGS.num_trials,
        ),
        runner=get_runner(),
        work_dir=ARGS.work_dir,
    )

    # input_dtype = "float32"
    # if input_dtype.startswith("float"):
    #     input_data = [np.random.uniform(size=input_shape).astype(input_dtype)]
    # else:
    #     input_data = [np.random.randint(
    #         low=0, high=10000, size=input_shape, dtype=input_dtype)]

    # if ARGS.rpc_config:
    #     run_module_via_rpc(
    #         rpc_config=ARGS.rpc_config,
    #         lib=executable.mod,
    #         dev_type=ARGS.target.kind.name,
    #         args=input_data,
    #         continuation=f_measurement,
    #     )
    # else:
    #     dev = tvm.device(ARGS.target.kind.name)
    #     input_data = [runtime.ndarray.array(arg, dev) for arg in input_data]
    #     f_measurement(executable.mod, dev, *input_data)


if __name__ == "__main__":
    main()
