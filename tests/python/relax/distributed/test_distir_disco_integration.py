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
from tvm import relax

def create_device_target(ccl):
    if ccl == "nccl":
        dev = tvm.cuda(0)
    else:
        dev = tvm.rocm(0)
    target = tvm.target.Target.from_device(dev)
    return (dev, target)

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
            R.func_attr({"global_symbol": "main"})
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
        sharded_mod = relax.distributed.transform.PropagateSharding()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerGlobalViewToLocalView()(sharded_mod)
        sharded_mod = relax.distributed.transform.LowerDistIR()(sharded_mod)
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
        Y_result = Y_result.numpy()
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-4, atol=1e-4)
    
if __name__ == "__main__":
    test_mlp(di.ProcessSession, "rccl" )
