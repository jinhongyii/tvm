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
from tvm import te
import tvm.testing
from tvm.script import tir as T, ir as I


@tvm.testing.requires_cuda
def test_split_host_device_func_attr():
    m = te.size_var("m")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")

    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], factor=8)
    s[A2].bind(xo, te.thread_axis("blockIdx.x"))
    s[A1].compute_at(s[A2], xo)
    s[A1].set_scope("shared")

    mod = tvm.lower(s, [A, A2])

    cuda_target = tvm.target.Target("cuda", host="llvm")
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target})
    )(mod)

    mod = tvm.ir.transform.Sequential(
        [
            tvm.tir.transform.AnnotateDeviceRegions(),
            tvm.tir.transform.SplitHostDevice(),
            tvm.tir.transform.MakePackedAPI(),
            tvm.tir.transform.LowerDeviceKernelLaunch(),
        ]
    )(mod)

    fdevice = mod["test_kernel"]

    assert fdevice.attrs["global_symbol"] == "test_kernel"
    assert fdevice.attrs["calling_conv"].value == 2
    assert str(fdevice.attrs["target"]) == str(tvm.target.Target("cuda"))
    assert fdevice.attrs["tir.is_global_func"].value


def test_ssa_across_entire_module():
    """The host and device functions should not share TIR vars

    Any arguments that are passed from the host to the device should
    be in terms of independent TIR variables.
    """

    @I.ir_module
    class before:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
            for i in range(16):
                T.attr(0, "device_scope", 0)
                for j in range(16):
                    T.evaluate(i)

    after = tvm.ir.transform.Sequential(
        [
            tvm.tir.transform.AnnotateDeviceRegions(),
            tvm.tir.transform.SplitHostDevice(),
            tvm.tir.transform.LowerDeviceKernelLaunch(),
        ]
    )(before)
    loop_var = after["main"].body.loop_var
    param_var = after["main_kernel"].params[0]

    assert not loop_var.same_as(param_var)


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.SplitHostDevice()


class TestSplitHostDevice(BaseCompare):
    """SplitHostDevice divides a function at the "target" attribute"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm -opt-level=0")})
                T.attr(T.target("cuda"), "target", 0)
                T.evaluate(n)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm -opt-level=0")})
                mod.main_kernel(n)

            @T.prim_func(private=True)
            def main_kernel(n: T.int32):
                T.func_attr(
                    {
                        "target": T.target("cuda"),
                        "tir.noalias": T.bool(True),
                        "tir.is_global_func": True,
                    }
                )
                T.evaluate(n)

        return mod


class TestSplitHostDeviceOnCPU(BaseCompare):
    """A kernel running on the CPU may return an error code"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm -opt-level=0")})
                T.attr(T.target("llvm"), "target", 0)
                T.evaluate(n)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm -opt-level=0")})
                err = mod.main_kernel(n)
                assert err == 0, "Error executing compute kernel"

            @T.prim_func(private=True)
            def main_kernel(n: T.int32) -> T.int32:
                T.func_attr(
                    {
                        "target": T.target("llvm"),
                        "tir.noalias": T.bool(True),
                        "tir.is_global_func": True,
                    }
                )
                T.evaluate(n)
                T.ret(0)

        return mod


class TestSplitHostDeviceWithoutFuncHostAttribute(BaseCompare):
    """Like TestSplitHostDevice, but no host specified in the host's target

    The `T.attr` specifying the device still requires splitting out
    the kernel.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("llvm")})
                T.attr(T.target("cuda"), "target", 0)
                T.evaluate(n)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("llvm")})
                mod.main_kernel(n)

            @T.prim_func(private=True)
            def main_kernel(n: T.int32):
                T.func_attr(
                    {
                        "target": T.target("cuda"),
                        "tir.noalias": T.bool(True),
                        "tir.is_global_func": True,
                    }
                )
                T.evaluate(n)

        return mod


class TestSplitHostDeviceWithoutDeviceRegion(BaseCompare):
    """Like TestSplitHostDevice, but no device regions to extract

    Because MakePackedAPI/MakeUnpackedAPI still require both the
    device and host, SplitHostDevice does not modify the "target"
    attribute.
    """

    def before():
        T.func_attr({"target": T.target("ext_dev", host="llvm")})
        T.evaluate(0)

    expected = before


class TestSplitHostDeviceNameCollision(BaseCompare):
    """Like TestSplitHostDevice, but with the default name already taken

    The default name is generated as `func.name + "_kernel"`.  If this
    name is already taken by another function in the IRModule, then
    SplitHostDevice should select a different name.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm -opt-level=0")})
                T.attr(T.target("cuda"), "target", 0)
                T.evaluate(n)

            @T.prim_func
            def main_kernel():
                T.func_attr({"target": T.target("llvm")})
                T.evaluate(0)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(n: T.int32):
                T.func_attr({"target": T.target("cuda", host="llvm -opt-level=0")})
                mod.main_kernel_1(n)

            @T.prim_func(private=True)
            def main_kernel_1(n: T.int32):
                T.func_attr(
                    {
                        "target": T.target("cuda"),
                        "tir.noalias": T.bool(True),
                        "tir.is_global_func": True,
                    }
                )
                T.evaluate(n)

            @T.prim_func
            def main_kernel():
                T.func_attr({"target": T.target("llvm")})
                T.evaluate(0)

        return mod


def test_dynamic_launch_thread():
    """Dynamic T.launch_thread

    A dynamic parameter for `T.launch_thread` must be computable
    within the parent scope.
    """

    @I.ir_module
    class before:
        I.module_attrs({"runtime": None})

        @T.prim_func
        def default_function(var_A: T.handle, var_T_add: T.handle, seq_len: T.int32):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "tir.is_entry_func": T.bool(True),
                    "tir.is_scheduled": 1,
                    "tir.noalias": T.bool(True),
                }
            )
            A = T.match_buffer(var_A, (seq_len * 8,), "int32")
            T_add = T.match_buffer(var_T_add, (seq_len * 8,), "int32")
            cse_var_1: T.int32 = (seq_len + 127) // 128
            T_expand_dims = T.allocate([seq_len * 8], "int32", "global")
            output_buf = T.allocate([T.Cast("int64", seq_len) * T.int64(8)], "int32", "global")
            T_expand_dims_1 = T.Buffer(
                (
                    T.min(
                        seq_len * 8,
                        T.max(
                            seq_len * 8,
                            T.Select(not T.bool(False), seq_len * 8, seq_len * -8),
                        )
                        - T.min(0, 1 - T.Select(not T.bool(False), seq_len * 8, seq_len * -8)),
                    ),
                ),
                "int32",
                data=T_expand_dims,
            )
            A_1 = T.Buffer((seq_len * 8,), "int32", data=A.data)
            with T.attr(T.target("cuda"), "target", 0):
                blockIdx_x = T.launch_thread("blockIdx.x", cse_var_1)
                threadIdx_x = T.launch_thread("threadIdx.x", 1024)
                if blockIdx_x * 1024 + threadIdx_x < seq_len * 8:
                    T_expand_dims_1[blockIdx_x * 1024 + threadIdx_x] = A_1[
                        blockIdx_x * 1024 + threadIdx_x
                    ]
            output_buf_1 = T.Buffer(
                (T.Cast("int64", seq_len) * T.int64(8),), "int32", data=output_buf, align=8
            )
            if 0 < seq_len:
                cse_var_3: T.int32 = seq_len * 8
                cse_var_2: T.float32 = T.Cast("float32", cse_var_3)
                with T.attr(T.target("cuda"), "target", 0):
                    threadIdx_x = T.launch_thread("threadIdx.x", 256)
                    blockIdx_x = T.launch_thread("blockIdx.x", (seq_len + 31) // 32)
                    blockIdx_y = T.launch_thread("blockIdx.y", 1)
                    if blockIdx_x * 256 + threadIdx_x < cse_var_3:
                        output_buf_1[
                            T.Cast("int64", (blockIdx_x * 256 + threadIdx_x) // cse_var_3)
                            * T.Cast("int64", seq_len)
                            * T.int64(8)
                            + T.Cast("int64", (blockIdx_x * 256 + threadIdx_x) % cse_var_3)
                        ] = T_expand_dims_1[blockIdx_x * 256 + threadIdx_x]
                for i in range(T.Cast("int32", T.ceil(T.log2(cse_var_2)))):
                    cse_var_4: T.int64 = T.Cast("int64", seq_len) * T.int64(8)
                    T.attr(T.target("cuda"), "target", 0)
                    threadIdx_x = T.launch_thread("threadIdx.x", 256)
                    start_1 = T.allocate([1], "int64", "local")
                    middle_1 = T.allocate([1], "int64", "local")
                    end_1 = T.allocate([1], "int64", "local")
                    blockIdx_x = T.launch_thread(
                        "blockIdx.x", (cse_var_3 - 1) // (T.shift_left(2, i) * 256) + 1
                    )
                    blockIdx_y = T.launch_thread("blockIdx.y", 1)
                    start_1_1 = T.Buffer((1,), "int64", data=start_1, scope="local")
                    start_1_1[0] = T.Cast("int64", T.shift_left(2, i)) * (
                        T.Cast("int64", blockIdx_x) * T.int64(256) + T.Cast("int64", threadIdx_x)
                    )
                    if start_1_1[0] < cse_var_4:
                        middle_1_1 = T.Buffer((1,), "int64", data=middle_1, scope="local")
                        middle_1_1[0] = (
                            T.Cast("int64", T.shift_left(2, i)) // T.int64(2) + start_1_1[0]
                        )
                        end_1_1 = T.Buffer((1,), "int64", data=end_1, scope="local")
                        end_1_1[0] = T.min(
                            start_1_1[0] + T.Cast("int64", T.shift_left(2, i)), cse_var_4
                        )
                        if middle_1_1[0] < cse_var_4:
                            output_buf_1[end_1_1[0] - T.int64(1)] = (
                                output_buf_1[end_1_1[0] - T.int64(1)]
                                + output_buf_1[middle_1_1[0] - T.int64(1)]
                            )
                with T.attr(T.target("cuda"), "target", 0):
                    blockIdx_x = T.launch_thread("blockIdx.x", 1)
                    output_buf_1[
                        (T.Cast("int64", -1 // cse_var_3) + T.int64(1))
                        * T.Cast("int64", seq_len)
                        * T.int64(8)
                        + T.Cast("int64", (cse_var_3 - 1) % cse_var_3)
                    ] = 0
                for j in range(T.Cast("int32", T.ceil(T.log2(cse_var_2)))):
                    cse_var_6: T.int64 = T.Cast("int64", j)
                    cse_var_5: T.int64 = T.Cast("int64", seq_len) * T.int64(8)
                    T.attr(T.target("cuda"), "target", 0)
                    threadIdx_x = T.launch_thread("threadIdx.x", 256)
                    start_1 = T.allocate([1], "int64", "local")
                    middle_1 = T.allocate([1], "int64", "local")
                    end_2 = T.allocate([1], "int64", "local")
                    end_3 = T.allocate([1], "int32", "local")
                    blockIdx_x = T.launch_thread(
                        "blockIdx.x",
                        T.max(
                            1,
                            T.Cast(
                                "int32",
                                (
                                    T.shift_left(
                                        T.int64(2),
                                        T.Cast("int64", T.ceil(T.log2(cse_var_2)))
                                        - cse_var_6
                                        - T.int64(1),
                                    )
                                    * T.int64(256)
                                    + cse_var_5
                                    - T.int64(1)
                                )
                                // (
                                    T.shift_left(
                                        T.int64(2),
                                        T.Cast("int64", T.ceil(T.log2(cse_var_2)))
                                        - cse_var_6
                                        - T.int64(1),
                                    )
                                    * T.int64(256)
                                ),
                            ),
                        ),
                    )
                    blockIdx_y = T.launch_thread("blockIdx.y", 1)
                    start_1_1 = T.Buffer((1,), "int64", data=start_1, scope="local")
                    start_1_1[0] = T.shift_left(
                        T.int64(2),
                        T.Cast("int64", T.ceil(T.log2(cse_var_2))) - cse_var_6 - T.int64(1),
                    ) * (T.Cast("int64", blockIdx_x) * T.int64(256) + T.Cast("int64", threadIdx_x))
                    if start_1_1[0] < cse_var_5:
                        middle_1_1 = T.Buffer((1,), "int64", data=middle_1, scope="local")
                        middle_1_1[0] = (
                            T.shift_left(
                                T.int64(2),
                                T.Cast("int64", T.ceil(T.log2(cse_var_2))) - cse_var_6 - T.int64(1),
                            )
                            // T.int64(2)
                            + start_1_1[0]
                        )
                        end_2_1 = T.Buffer((1,), "int64", data=end_2, scope="local")
                        end_2_1[0] = T.min(
                            start_1_1[0]
                            + T.shift_left(
                                T.int64(2),
                                T.Cast("int64", T.ceil(T.log2(cse_var_2))) - cse_var_6 - T.int64(1),
                            ),
                            cse_var_5,
                        )
                        if middle_1_1[0] < cse_var_5:
                            end_3_1 = T.Buffer((1,), "int32", data=end_3, scope="local")
                            end_3_1[0] = output_buf_1[middle_1_1[0] - T.int64(1)]
                            output_buf_1[middle_1_1[0] - T.int64(1)] = output_buf_1[
                                end_2_1[0] - T.int64(1)
                            ]
                            output_buf_1[end_2_1[0] - T.int64(1)] = (
                                output_buf_1[end_2_1[0] - T.int64(1)] + end_3_1[0]
                            )
            T_expand_dims_2 = T.Buffer((seq_len * 8,), "int32", data=T_expand_dims)
            with T.attr(T.target("cuda"), "target", 0):
                blockIdx_x = T.launch_thread("blockIdx.x", cse_var_1)
                threadIdx_x = T.launch_thread("threadIdx.x", 1024)
                if blockIdx_x * 1024 + threadIdx_x < seq_len * 8:
                    T_expand_dims_2[blockIdx_x * 1024 + threadIdx_x] = output_buf_1[
                        T.Cast("int64", blockIdx_x) * T.int64(1024) + T.Cast("int64", threadIdx_x)
                    ]
            T.attr(T.target("cuda"), "target", 0)
            blockIdx_x = T.launch_thread("blockIdx.x", cse_var_1)
            threadIdx_x = T.launch_thread("threadIdx.x", 1024)
            if blockIdx_x * 1024 + threadIdx_x < seq_len * 8:
                T_add_1 = T.Buffer((seq_len * 8,), "int32", data=T_add.data)
                T_add_1[blockIdx_x * 1024 + threadIdx_x] = (
                    A_1[blockIdx_x * 1024 + threadIdx_x]
                    + T_expand_dims_2[blockIdx_x * 1024 + threadIdx_x]
                )

    after = tvm.tir.transform.SplitHostDevice()(before)
    tvm.tir.analysis.verify_well_formed(after)


if __name__ == "__main__":
    tvm.testing.main()
