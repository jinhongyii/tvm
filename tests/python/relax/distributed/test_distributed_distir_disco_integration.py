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
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.runtime.relax_vm import VirtualMachine
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm import get_global_func
from tvm import relax, tir

_all_session_kinds = [di.ThreadedSession, di.ProcessSession]
_ccl = [get_global_func("runtime.disco.compiled_ccl")()]

def create_device_target(ccl):
    if ccl == "nccl":
        dev = tvm.cuda(0)
    else:
        dev = tvm.rocm(0)
    target = tvm.target.Target.from_device(dev)
    return (dev, target)

@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_mlp(session_kind, ccl):  # pylint: disable=too-many-locals
    devices = [0, 1]
    sess = session_kind(num_workers=len(devices))
    sess.init_ccl(ccl, *devices)

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class MLP:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 128), "float32"),
            W2: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((128, 128), "float32") = R.matmul(x, W1)
                lv1: R.Tensor((128, 128), "float32") = R.nn.gelu(lv0)
                lv2: R.Tensor((128, 128), "float32") = R.matmul(lv1, W2)
                R.output(lv2)
            return lv2

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class AnnotatedMLP:  # pylint: disable=too-few-public-methods
        I.module_attrs({"device_num": 2})
        I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 128), "float32"),
            W2: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main", "num_input": 1})
            with R.dataflow():
                lv0: R.Tensor((128, 128), "float32") = R.matmul(x, W1)
                lv1: R.Tensor((128, 128), "float32") = R.dist.annotate_sharding(lv0, "mesh[0]", "S[1]")
                lv2: R.Tensor((128, 128), "float32") = R.nn.gelu(lv1)
                lv3: R.Tensor((128, 128), "float32") = R.matmul(lv2, W2)
                R.output(lv3)
            return lv3

    # pylint: enable=invalid-name
    dev, target = create_device_target(ccl)

    def relax_build(mod, target):
        with target:
            mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            return rx.build(mod, target=target)

    # pylint: disable=invalid-name
    X = np.random.randn(128, 128).astype("float32")
    W1 = np.random.randn(128, 128).astype("float32")
    W2 = np.random.randn(128, 128).astype("float32")
    Y_expected = VirtualMachine(relax_build(MLP, target), device=dev)["main"](
        tvm.nd.array(X, device=dev),
        tvm.nd.array(W1, device=dev),
        tvm.nd.array(W2, device=dev),
    ).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        sharded_mod = AnnotatedMLP
        sharded_mod = relax.transform.LegalizeOps()(sharded_mod)
        sharded_mod = relax.distributed.transform.PropagateSharding()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerGlobalViewToLocalView()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerDistIR()(sharded_mod)
        print(sharded_mod)

        # print(sharded_mod)
        relax_build(sharded_mod, target).export_library(path)

        mod = sess.load_vm_module(path)

        d_X = sess.empty((128, 128), "float32")
        d_W1 = sess.empty((128, 64), "float32")
        d_W2 = sess.empty((64, 128), "float32")

        d_X.debug_copy_from(0, X)
        d_W1.debug_copy_from(0, W1[:, :64])
        d_W1.debug_copy_from(1, W1[:, 64:])
        d_W2.debug_copy_from(0, W2[:64, :])
        d_W2.debug_copy_from(1, W2[64:, :])
        d_Y = mod["main"](d_X, d_W1, d_W2)
        Y_result = tvm.nd.empty((128, 128), "float32", device=dev)
        sess.copy_from_worker_0(Y_result, d_Y)
        sess.sync_worker_0()
        sess._sync_worker(1)
        Y_result = Y_result.numpy()
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-4, atol=1e-4)
    
@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_attention(session_kind, ccl):  # pylint: disable=too-many-locals,too-many-statements
    # devices = [0, 1]
    # sess = session_kind(num_workers=len(devices))
    # sess.init_ccl(ccl, *devices)

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class Attention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 512), "float32"),
            Wk: R.Tensor((128, 512), "float32"),
            Wv: R.Tensor((128, 512), "float32"),
            Wo: R.Tensor((512, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                # q
                lv0: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wq)
                lv1: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv0, [1, 10, 8, 64])
                lv2: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wk)
                lv4: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv3, [1, 10, 8, 64])
                lv5: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wv)
                lv7: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv6, [1, 10, 8, 64])
                lv8: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 8, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 8, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 8, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 8, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 8, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 8, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 512), "float32") = R.reshape(lv14, [1, 10, 512])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16
    @tvm.script.ir_module
    class AnnotatedAttention:  # pylint: disable=too-few-public-methods
        I.module_attrs({"device_num": 2})
        I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 512), "float32"),
            Wk: R.Tensor((128, 512), "float32"),
            Wv: R.Tensor((128, 512), "float32"),
            Wo: R.Tensor((512, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main", "num_input": 1})
            with R.dataflow():
                # q
                lv0: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wq)
                lv1: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv0, [1, 10, 8, 64])
                lv1 = R.dist.annotate_sharding(lv1, "mesh[0]", "S[2]")
                lv2: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wk)
                lv4: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv3, [1, 10, 8, 64])
                lv4 = R.dist.annotate_sharding(lv4, "mesh[0]", "S[2]")
                lv5: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wv)
                lv7: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv6, [1, 10, 8, 64])
                lv7 = R.dist.annotate_sharding(lv7, "mesh[0]", "S[2]")
                lv8: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 8, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 8, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 8, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 8, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 8, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 8, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 512), "float32") = R.reshape(lv14, [1, 10, 512])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16

    # pylint: enable=invalid-name
    dev, target = create_device_target(ccl)

    def relax_build(mod, target):
        with target:
            mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            return rx.build(mod, target=target)

    # pylint: disable=invalid-name
    X = np.random.randn(1, 10, 128).astype("float32")
    Wq = np.random.randn(128, 512).astype("float32")
    Wk = np.random.randn(128, 512).astype("float32")
    Wv = np.random.randn(128, 512).astype("float32")
    Wo = np.random.randn(512, 128).astype("float32")
    Y_expected = VirtualMachine(relax_build(Attention, target), device=dev)["main"](
        tvm.nd.array(X, device=dev),
        tvm.nd.array(Wq, device=dev),
        tvm.nd.array(Wk, device=dev),
        tvm.nd.array(Wv, device=dev),
        tvm.nd.array(Wo, device=dev),
    ).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        sharded_mod = AnnotatedAttention
        sharded_mod = relax.transform.LegalizeOps({"relax.reshape": (lambda bb, call: call)})(sharded_mod)
        sharded_mod = relax.distributed.transform.PropagateSharding()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerGlobalViewToLocalView()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerDistIR()(sharded_mod)
        print(sharded_mod)
        relax_build(sharded_mod, target).export_library(path)

        mod = sess.load_vm_module(path)

        d_X = sess.empty((1, 10, 128), "float32")
        d_Wq = sess.empty((128, 256), "float32")
        d_Wk = sess.empty((128, 256), "float32")
        d_Wv = sess.empty((128, 256), "float32")
        d_Wo = sess.empty((256, 128), "float32")

        d_X.debug_copy_from(0, X)
        d_Wq.debug_copy_from(0, Wq[:, :256])
        d_Wq.debug_copy_from(1, Wq[:, 256:])
        d_Wk.debug_copy_from(0, Wk[:, :256])
        d_Wk.debug_copy_from(1, Wk[:, 256:])
        d_Wv.debug_copy_from(0, Wv[:, :256])
        d_Wv.debug_copy_from(1, Wv[:, 256:])
        d_Wo.debug_copy_from(0, Wo[:256, :])
        d_Wo.debug_copy_from(1, Wo[256:, :])
        d_Y = mod["main"](d_X, d_Wq, d_Wk, d_Wv, d_Wo)
        Y_result = tvm.nd.empty((1, 10, 128), "float32", device=dev)
        sess.copy_from_worker_0(Y_result, d_Y)
        sess.sync_worker_0()
        Y_result = Y_result.numpy()
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-3, atol=1e-3)
    
    
@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("ccl", _ccl)
def test_attention_combine_qkv(session_kind, ccl):  # pylint: disable=too-many-locals,too-many-statements
    # devices = [0, 1]
    # sess = session_kind(num_workers=len(devices))
    # sess.init_ccl(ccl, *devices)

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class Attention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 8192), "float32"),
            Wqkv: R.Tensor((8192, 10240), "float32"),
            Wo: R.Tensor((8192, 8192), "float32"),
        ) -> R.Tensor((1, 10, 8192), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                # q
                lv0 = R.matmul(x, Wqkv)
                lv_split = R.split(lv0, indices_or_sections=[8192, 9216], axis=-1)
                lv1 = lv_split[0]
                lv1 = R.reshape(lv1, [1, 10, 64, 128])
                lv2 = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3 = lv_split[1]
                lv4 = R.reshape(lv3, [1, 10, 8, 128])
                lv4 = R.repeat(lv4, repeats=8, axis=2)
                lv5 = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6 = lv_split[2]
                lv7 = R.reshape(lv6, [1, 10, 8, 128])
                lv7 = R.repeat(lv7, repeats=8, axis=2)
                lv8 = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9= R.permute_dims(lv5, [0, 1, 3, 2])
                lv10 = R.matmul(lv2, lv9)
                lv11 = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12= R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13 = R.matmul(lv12, lv8)
                lv14 = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15 = R.reshape(lv14, [1, 10, 8192])
                # attn_output @ o
                lv16 = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16

    @tvm.script.ir_module
    class AttentionStep1:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 8192), "float32"),
            Wqkv: R.Tensor((8192, 10240), "float32"),
            Wo: R.Tensor((8192, 8192), "float32"),
        ) -> R.Tensor((1, 10, 8192), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                # q
                Wqkv_split = R.split(Wqkv, indices_or_sections=[8192, 9216], axis=-1)
                Wq = Wqkv_split[0]
                Wq = R.reshape(Wq, [8192, 2, 4096])
                Wk = Wqkv_split[1]
                Wk = R.reshape(Wk, [8192, 2, 512])
                Wv = Wqkv_split[2]
                Wv = R.reshape(Wv, [8192, 2, 512])
                Wqkv = R.concat([Wq, Wk, Wv], axis=2)
                lv0 = R.einsum([x, Wqkv], subscripts="abc,cde->abde")
                lv_split = R.split(lv0, indices_or_sections=[4096, 4608], axis=-1)
                lv1 = lv_split[0]
                lv1 = R.reshape(lv1, [1, 10, 64, 128])
                lv2 = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3 = lv_split[1]
                lv4 = R.reshape(lv3, [1, 10, 8, 128])
                lv4 = R.repeat(lv4, repeats=8, axis=2)
                lv5 = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6 = lv_split[2]
                lv7 = R.reshape(lv6, [1, 10, 8, 128])
                lv7 = R.repeat(lv7, repeats=8, axis=2)
                lv8 = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9= R.permute_dims(lv5, [0, 1, 3, 2])
                lv10 = R.matmul(lv2, lv9)
                lv11 = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12= R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13 = R.matmul(lv12, lv8)
                lv14 = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15 = R.reshape(lv14, [1, 10, 8192])
                # attn_output @ o
                lv16 = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16
    @tvm.script.ir_module
    class AttentionStep2:  # pylint: disable=too-few-public-methods
        I.module_attrs({"device_num": 2})
        I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 8192), "float32"),
            Wqkv: R.Tensor((8192, 10240), "float32"),
            Wo: R.Tensor((8192, 8192), "float32"),
        ) -> R.Tensor((1, 10, 8192), "float32"):
            R.func_attr({"global_symbol": "main", "num_input": 1})
            with R.dataflow():
                # q
                Wqkv_split = R.split(Wqkv, indices_or_sections=[8192, 9216], axis=-1)
                Wq = Wqkv_split[0]
                Wq = R.reshape(Wq, [8192, 2, 4096])
                Wk = Wqkv_split[1]
                Wk = R.reshape(Wk, [8192, 2, 512])
                Wv = Wqkv_split[2]
                Wv = R.reshape(Wv, [8192, 2, 512])
                Wqkv = R.concat([Wq, Wk, Wv], axis=2)
                lv0 = R.einsum([x, Wqkv], subscripts="abc,cde->abde")
                lv_split = R.split(lv0, indices_or_sections=[4096, 4608], axis=-1)
                lv1 = lv_split[0]
                lv1 = R.reshape(lv1, [1, 10, 64, 128])
                lv1 = R.dist.annotate_sharding(lv1, "mesh[0]", "S[2]")
                lv2 = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3 = lv_split[1]
                lv4 = R.reshape(lv3, [1, 10, 8, 128])
                lv4 = R.dist.annotate_sharding(lv4, "mesh[0]", "S[2]")
                lv4 = R.repeat(lv4, repeats=8, axis=2)
                lv5 = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6 = lv_split[2]
                lv7 = R.reshape(lv6, [1, 10, 8, 128])
                lv7 = R.dist.annotate_sharding(lv7, "mesh[0]", "S[2]")
                lv7 = R.repeat(lv7, repeats=8, axis=2)
                lv8 = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9= R.permute_dims(lv5, [0, 1, 3, 2])
                lv10 = R.matmul(lv2, lv9)
                lv11 = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12= R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13 = R.matmul(lv12, lv8)
                lv14 = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15 = R.reshape(lv14, [1, 10, 8192])
                # attn_output @ o
                lv16 = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16
    
    # pylint: enable=invalid-name
    dev, target = create_device_target(ccl)

    def relax_build(mod, target):
        with target:
            mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            return rx.build(mod, target=target)

    # pylint: disable=invalid-name
    # X = np.random.randn(1, 10, 128).astype("float32")
    # Wq = np.random.randn(128, 512).astype("float32")
    # Wk = np.random.randn(128, 512).astype("float32")
    # Wv = np.random.randn(128, 512).astype("float32")
    # Wo = np.random.randn(512, 128).astype("float32")
    # Y_expected = VirtualMachine(relax_build(Attention, target), device=dev)["main"](
    #     tvm.nd.array(X, device=dev),
    #     tvm.nd.array(Wq, device=dev),
    #     tvm.nd.array(Wk, device=dev),
    #     tvm.nd.array(Wv, device=dev),
    #     tvm.nd.array(Wo, device=dev),
    # ).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        sharded_mod = AttentionStep2
        sharded_mod = relax.transform.LegalizeOps()(sharded_mod)
        sharded_mod = tir.transform.Simplify()(sharded_mod)
        # print(sharded_mod)
        sharded_mod = relax.distributed.transform.PropagateSharding()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerGlobalViewToLocalView()(sharded_mod)
        sharded_mod = relax.distributed.transform.LegalizeRedistribute()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerDistIR()(sharded_mod)
        sharded_mod = relax.transform.LiftTransformParams()(sharded_mod)
        print(sharded_mod)

    #     relax_build(sharded_mod, target).export_library(path)

    #     mod = sess.load_vm_module(path)

    #     d_X = sess.empty((1, 10, 128), "float32")
    #     d_Wq = sess.empty((128, 256), "float32")
    #     d_Wk = sess.empty((128, 256), "float32")
    #     d_Wv = sess.empty((128, 256), "float32")
    #     d_Wo = sess.empty((256, 128), "float32")

    #     d_X.debug_copy_from(0, X)
    #     d_Wq.debug_copy_from(0, Wq[:, :256])
    #     d_Wq.debug_copy_from(1, Wq[:, 256:])
    #     d_Wk.debug_copy_from(0, Wk[:, :256])
    #     d_Wk.debug_copy_from(1, Wk[:, 256:])
    #     d_Wv.debug_copy_from(0, Wv[:, :256])
    #     d_Wv.debug_copy_from(1, Wv[:, 256:])
    #     d_Wo.debug_copy_from(0, Wo[:256, :])
    #     d_Wo.debug_copy_from(1, Wo[256:, :])
    #     d_Y = mod["main"](d_X, d_Wq, d_Wk, d_Wv, d_Wo)
    #     Y_result = tvm.nd.empty((1, 10, 128), "float32", device=dev)
    #     sess.copy_from_worker_0(Y_result, d_Y)
    #     sess.sync_worker_0()
    #     Y_result = Y_result.numpy()
    # # pylint: enable=invalid-name
    # np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-3, atol=1e-3)
    
if __name__ == "__main__":
    test_attention_combine_qkv(di.ProcessSession, "rccl")
