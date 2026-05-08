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

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as Tx


def test_call_tir_device():
    @I.ir_module(tirx=True)
    class Before:
        @Tx.prim_func(tirx=True, private=True)
        def test1(A: Tx.Buffer((2048,), "float32"), m: Tx.int32):
            with Tx.cta():
                tid = Tx.thread_id([128])
                Tx.fill(A[m * 128 : m * 128 + 128], 0.0)

        @Tx.prim_func(tirx=True, private=True)
        def test2(A: Tx.Buffer((2048, 2048), "float32"), m: Tx.int32, n: Tx.int32):
            with Tx.cta():
                tid = Tx.thread_id([128])
                Tx.fill(A[m * 128 : m * 128 + 128, n * 128 : n * 128 + 128], 0.0)

        @R.function
        def main():  # type: ignore
            cls = Before
            with R.dataflow():
                x = R.call_tir_device(cls.test1, [], R.Tensor((2048,), "float32"), 0, (16,))
                y = R.call_tir_device(cls.test2, [], R.Tensor((2048, 2048), "float32"), 0, (16, 16))
                R.output(x, y)
            return x, y

    @I.ir_module(tirx=True)
    class After:
        @Tx.prim_func(tirx=True, private=True)
        def test1_kernel(A: Tx.Buffer((2048,), "float32")):
            with Tx.kernel():
                m = Tx.cta_id([16])
                with Tx.cta():
                    tid = Tx.thread_id([128])
                    Tx.fill(A[m * 128 : m * 128 + 128], 0.0)

        @Tx.prim_func(tirx=True, private=True)
        def test2_kernel(A: Tx.Buffer((2048, 2048), "float32")):
            with Tx.kernel():
                m, n = Tx.cta_id([16, 16])
                with Tx.cta():
                    tid = Tx.thread_id([128])
                    Tx.fill(A[m * 128 : m * 128 + 128, n * 128 : n * 128 + 128], 0.0)

        @R.function
        def main():  # type: ignore
            cls = After
            with R.dataflow():
                x = R.call_tir(cls.test1_kernel, [], R.Tensor((2048,), "float32"))
                y = R.call_tir(cls.test2_kernel, [], R.Tensor((2048, 2048), "float32"))
                R.output(x, y)
            return x, y

    mod = Before
    mod = relax.transform.LowerCallTIRDevice()(mod)
    tvm.ir.assert_structural_equal(mod, After)


if __name__ == "__main__":
    test_call_tir_device()
