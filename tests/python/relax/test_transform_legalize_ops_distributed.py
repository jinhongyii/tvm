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
from tvm.relax.transform import LegalizeOps
from tvm.script import ir as I
from tvm.script import relax as R

def test_redistribute_replica_to_shard():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((10, 5), "float32"):
            gv0 = R.dist.redistribute_replica_to_shard(x, num_workers=2, axis=1)
            return gv0

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            gv0: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([0])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv1: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([1])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv2: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([2])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv3: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([3])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            gv4: R.Tensor((10, 10), dtype="float32") = R.call_dps_packed("runtime.disco.allreduce", [x, R.shape([4])], out_sinfo=R.Tensor((10, 10), dtype="float32"))
            return x
    # fmt: on

    mod = LegalizeOps()(Before)
    print(mod)
    relax.build(mod, target="llvm")


if __name__ == "__main__":
    test_redistribute_replica_to_shard()
