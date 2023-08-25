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

#  type: ignore
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
import tvm
from tvm import relax
from tvm.ir import assert_structural_equal
import tvm.testing


@I.ir_module
class MLP:
    I.module_attrs({"device_num": 10})
    I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]})
    @T.prim_func(private=True)
    def gelu(A: T.Buffer((T.int64(128), T.int64(128)), "float32"), T_multiply: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_multiply_1 = T.alloc_buffer((T.int64(128), T.int64(128)))
        compute = T.alloc_buffer((T.int64(128), T.int64(128)))
        T_multiply_2 = T.alloc_buffer((T.int64(128), T.int64(128)))
        T_add = T.alloc_buffer((T.int64(128), T.int64(128)))
        for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_multiply_1[v_ax0, v_ax1])
                T_multiply_1[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T.float32(0.70710678118654757)
        for i0, i1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_multiply_1[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
        for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(compute[v_ax0, v_ax1])
                T.writes(T_multiply_2[v_ax0, v_ax1])
                T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
        for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_multiply_2[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def matmul(A: T.Buffer((T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(128), T.int64(128)), "float32"), matmul_1: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_k], B[v_k, v_i1])
                T.writes(matmul_1[v_i0, v_i1])
                with T.init():
                    matmul_1[v_i0, v_i1] = T.float32(0)
                matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    @R.function
    def foo(x: R.DTensor((128, 128), "float32", "mesh[0]", "R"), weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"), weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]")) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
        cls = MLP
        lv0 = R.dist.call_tir(cls.matmul, (x, weight1), out_sinfo=R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"))
        lv1 = R.dist.call_tir(cls.gelu, (lv0,), out_sinfo=R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"))
        lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv1
        lv3 = R.dist.call_tir(cls.matmul, (lv2, weight2), out_sinfo=R.DTensor((128, 128), "float32", "mesh[0]", "R"))
        return lv3

def test_mlp():
    mod = MLP
    mod = relax.distributed.transform.LowerGlobalViewToLocalView()(mod)
    print(mod)
    
if __name__ == "__main__":
    test_mlp()
