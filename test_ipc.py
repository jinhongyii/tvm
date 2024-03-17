import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.runtime.relax_vm import VirtualMachine
from tvm.script import relax as R, tir as T

devices = [0, 1]
sess = di.ProcessSession(num_workers=len(devices))
sess.init_ccl("nccl", *devices)

@tvm.script.ir_module
class AllReduce:  # pylint: disable=too-few-public-methods
    @T.prim_func
    def add_one(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        T_add_one: T.buffer((T.int64(128), T.int64(128)), "float32"),
    ):
        for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("T_add_one"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1])
                T.writes(T_add_one[v_ax0, v_ax1])
                T_add_one[v_ax0, v_ax1] = A[v_ax0, v_ax1] + 1
    
    
    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        R.func_attr({"global_symbol": "main"})
        cls = AllReduce
        with R.dataflow():
            x: R.Tensor((128, 128), "float32") = R.ccl.broadcast_from_worker0(x)
            send = R.call_packed("cuda_ipc.alloc_storage", R.shape((128, 128)), "float32", sinfo_args=R.Object())
            send = R.call_packed("vm.builtin.alloc_tensor", send, 0, R.shape((128, 128)), "float32", sinfo_args=R.Tensor((128, 128), "float32"))
            send = R.call_tir_inplace(cls.add_one, (x, send), inplace_indices=1, out_sinfo=R.Tensor((128, 128), "float32"))
            out = R.call_dps_packed("cuda_ipc.custom_allreduce", (send, 1), out_sinfo=R.Tensor((128, 128), "float32"))
            R.output(out)
        return out

def create_device_target(ccl):
    if ccl == "nccl":
        dev = tvm.cuda(0)
    else:
        dev = tvm.rocm(0)
    target = tvm.target.Target.from_device(dev)
    return (dev, target)

    
finit_ipc = sess.get_global_func("cuda_ipc.init_ipc_allocator")
sess.call_packed(finit_ipc)
dev, target = create_device_target("nccl")
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
    
with tempfile.TemporaryDirectory() as tmpdir:
    path = tmpdir + "/test.so"
    relax_build(AllReduce, target).export_library(path)

    mod = sess.load_vm_module(path)
    X = np.random.randn(128, 128).astype("float32")
    d_X = sess.empty((128, 128), "float32")

    d_X.debug_copy_from(0, X)
    d_Y = mod["main"](d_X)
    Y_result = tvm.nd.empty((128, 128), "float32", device=dev)
    sess.copy_from_worker_0(Y_result, d_Y)
    sess.sync_worker_0()
    Y_result = Y_result.numpy()
    
    tvm.testing.assert_allclose(Y_result, 2*(X + 1), rtol=1e-5, atol=1e-5)
    
    