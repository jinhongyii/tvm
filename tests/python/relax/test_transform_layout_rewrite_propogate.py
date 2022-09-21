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

from __future__ import annotations
import pytest
import tvm
import sys
import tvm.testing
from tvm import relax, topi
import tvm.script
from tvm.script import tir as T, relax as R

def test_two_matmul():

    def fn_schedule(sch, func_name):
        b0 = sch.get_block("T_matmul_NT", func_name)
        block_read = sch.cache_read(b0, 0, "global")
        sch.transform_layout(b0, ("read", 0), lambda i, j: (i//16, j//16, i%16, j%16))
        sch.annotate(block_read, "meta_schedule.layout_rewrite_preproc", True)
        block_write = sch.cache_write(b0, 0, "global")
        sch.transform_layout(b0, ("write", 0), lambda i, j: (i//16, j//16, i%16, j%16))
        sch.annotate(block_write, "meta_schedule.layout_rewrite_postproc", True)

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", [128, 64], relax.DynTensorType(2, "float32"))
        y = relax.Var("y", [128, 64], relax.DynTensorType(2, "float32"))
        z = relax.Var("z", [128, 128], relax.DynTensorType(2, "float32"))
        with bb.function("two_matmul", [x, y, z], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.dense, x, y)
                lv1 = bb.emit_te(topi.nn.dense, lv0, z)
                gv = bb.emit_output(lv1)
            bb.emit_func_output(gv)
        mod = bb.get()
        sch = tvm.tir.Schedule(mod)
        fn_schedule(sch, "dense")
        fn_schedule(sch, "dense1")
        mod = sch.mod
        mod = relax.transform.SplitLayoutRewritePreproc()(mod)
        mod = relax.transform.LayoutRewritePropogate()(mod)
        
        return mod
        # fn_schedule(mod[])

    print(before().script())
    

if __name__ == "__main__":
    # sys.exit(pytest.main([__file__] + sys.argv[1:]))
    test_two_matmul()
