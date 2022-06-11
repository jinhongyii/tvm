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
# pylint: disable=missing-docstring
import logging
import tempfile
import numpy as np
import sys

import pytest
import tvm

from tvm import meta_schedule as ms
from tvm.meta_schedule import TuneConfig, tune_tir
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_with_layout_free_attr(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    T.func_attr({"layout_free_buffers": [1]})
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=no-member,invalid-name,unused-variable
@T.prim_func
def fused_dense_multiply_subtract_add_fixed_point_multiply_add1_clip_cast1(a_1: T.Buffer[(1024, 1024), "int8"], param_0_1: T.Buffer[(1024, 1024), "int8"], param_1_1: T.Buffer[(), "int32"], lv2_1: T.Buffer[(1024, 1), "int32"], param_2_1: T.Buffer[(1, 1024), "int32"], param_3_1: T.Buffer[(), "int32"], T_cast_1: T.Buffer[(1024, 1024), "int8"]) -> None:
    # function attr dict
    T.func_attr({"T.noalias": True, "global_symbol": "fused_dense_multiply_subtract_add_fixed_point_multiply_add1_clip_cast1"})
    # body
    # with T.block("root")
    T_matmul_NT_1 = T.alloc_buffer([1024, 1024], dtype="int32")
    T_multiply_1 = T.alloc_buffer([1024, 1], dtype="int32")
    T_subtract_1 = T.alloc_buffer([1024, 1024], dtype="int32")
    T_add_2 = T.alloc_buffer([1024, 1024], dtype="int32")
    compute_2 = T.alloc_buffer([1024, 1024], dtype="int32")
    T_add_3 = T.alloc_buffer([1024, 1024], dtype="int32")
    compute_3 = T.alloc_buffer([1024, 1024], dtype="int32")
    for i0_4, i1_4, i2 in T.grid(1024, 1024, 1024):
        with T.block("T_matmul_NT"):
            i, j, k = T.axis.remap("SSR", [i0_4, i1_4, i2])
            T.reads(a_1[i, k], param_0_1[j, k])
            T.writes(T_matmul_NT_1[i, j])
            with T.init():
                T_matmul_NT_1[i, j] = 0
            T_matmul_NT_1[i, j] = T_matmul_NT_1[i, j] + T.cast(a_1[i, k], "int32") * T.cast(param_0_1[j, k], "int32")
    for i0_5, i1_5 in T.grid(1024, 1):
        with T.block("T_multiply"):
            ax0, ax1 = T.axis.remap("SS", [i0_5, i1_5])
            T.reads(param_1_1[()], lv2_1[ax0, 0])
            T.writes(T_multiply_1[ax0, ax1])
            T_multiply_1[ax0, ax1] = param_1_1[()] * lv2_1[ax0, 0]
    for i0_6, i1_6 in T.grid(1024, 1024):
        with T.block("T_subtract"):
            ax0, ax1 = T.axis.remap("SS", [i0_6, i1_6])
            T.reads(T_matmul_NT_1[ax0, ax1], T_multiply_1[ax0, 0])
            T.writes(T_subtract_1[ax0, ax1])
            T_subtract_1[ax0, ax1] = T_matmul_NT_1[ax0, ax1] - T_multiply_1[ax0, 0]
    for i0_7, i1_7 in T.grid(1024, 1024):
        with T.block("T_add"):
            ax0, ax1 = T.axis.remap("SS", [i0_7, i1_7])
            T.reads(T_subtract_1[ax0, ax1], param_2_1[0, ax1])
            T.writes(T_add_2[ax0, ax1])
            T_add_2[ax0, ax1] = T_subtract_1[ax0, ax1] + param_2_1[0, ax1]
    for i0_8, i1_8 in T.grid(1024, 1024):
        with T.block("compute"):
            i0_9, i1_9 = T.axis.remap("SS", [i0_8, i1_8])
            T.reads(T_add_2[i0_9, i1_9])
            T.writes(compute_2[i0_9, i1_9])
            compute_2[i0_9, i1_9] = T.q_multiply_shift(T_add_2[i0_9, i1_9], 1340867788, 31, -27, dtype="int32")
    for i0_10, i1_10 in T.grid(1024, 1024):
        with T.block("T_add_1"):
            ax0, ax1 = T.axis.remap("SS", [i0_10, i1_10])
            T.reads(param_3_1[()], compute_2[ax0, ax1])
            T.writes(T_add_3[ax0, ax1])
            T_add_3[ax0, ax1] = param_3_1[()] + compute_2[ax0, ax1]
    for i0_11, i1_11 in T.grid(1024, 1024):
        with T.block("compute_1"):
            i0_12, i1_12 = T.axis.remap("SS", [i0_11, i1_11])
            T.reads(T_add_3[i0_12, i1_12])
            T.writes(compute_3[i0_12, i1_12])
            compute_3[i0_12, i1_12] = T.max(T.min(T_add_3[i0_12, i1_12], 127), -128)
    for i0_13, i1_13 in T.grid(1024, 1024):
        with T.block("T_cast"):
            ax0, ax1 = T.axis.remap("SS", [i0_13, i1_13])
            T.reads(compute_3[ax0, ax1])
            T.writes(T_cast_1[ax0, ax1])
            T_cast_1[ax0, ax1] = T.cast(compute_3[ax0, ax1], "int8")

@pytest.mark.skip("Integration test")
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=fused_dense_multiply_subtract_add_fixed_point_multiply_add1_clip_cast1,
            target=Target("llvm --num-cores=16"),
            config=TuneConfig(
                strategy="replay_trace",
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


@pytest.mark.skip("Integration test")
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=matmul,
            target=Target("nvidia/geforce-rtx-3070"),
            config=TuneConfig(
                strategy="replay_trace",
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


@pytest.mark.skip("Integration test")
def test_tune_run_module_via_rpc():
    target = tvm.target.Target("llvm")
    rt_mod = tvm.build(matmul, target)

    # construct the input
    input_data = {}
    input_shape = (128, 128)
    input_dtype = "float32"
    a_np = np.random.uniform(size=input_shape).astype(input_dtype)
    b_np = np.random.uniform(size=input_shape).astype(input_dtype)
    c_np = np.zeros(input_shape).astype(input_dtype)
    for i in range(128):
        for j in range(128):
            for k in range(128):
                c_np[i, j] = c_np[i, j] + a_np[i, k] * b_np[j, k]
    input_data["a"] = a_np
    input_data["b"] = b_np
    input_data["c"] = np.zeros(input_shape).astype(input_dtype)

    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )

        def f_timer(rt_mod, dev, input_data):
            rt_mod(input_data["a"], input_data["b"], input_data["c"])
            return input_data["c"]

        result = run_module_via_rpc(
            rpc_config=rpc_config,
            lib=rt_mod,
            dev_type=target.kind.name,
            args=input_data,
            continuation=f_timer,
        )
        tvm.testing.assert_allclose(result.numpy(), c_np, rtol=1e-3)

@pytest.mark.skip("Integration test")
@pytest.mark.parametrize("strategy", ["replay_trace", "replay_func", "evolutionary"])
def test_tune_matmul_layout_rewrite(strategy):
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=matmul_with_layout_free_attr,
            target=Target("llvm --num-cores=16"),
            config=TuneConfig(
                strategy=strategy,
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


if __name__ == """__main__""":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
