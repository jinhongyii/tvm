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
"""Lower the call_tir_device to call_tir
The pass is written in Python for experiment, fast development.
"""

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.relax.expr import Expr
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.script import tirx as Tx
from tvm.script.parser import tirx as T
from tvm.tirx import ExecScopeStmt, PrimFunc
from tvm.tirx.analysis import verify_tirx_well_formed
from tvm.tirx.stmt_functor import StmtExprVisitor
from tvm.tirx.transform.common import BufferReplacer, seek_kernel_replace_point


@tvm.transform.module_pass(opt_level=0, name="LowerCallTIRDevice")
class LowerCallTIRDevice:
    """Lower the call_tir_device to call_tir."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        mod = _Rewriter(mod).transform()
        mod = relax.transform.DeadCodeElimination()(mod)
        return mod


class DeviceFuncToKernel(StmtExprVisitor):
    def __init__(self, tile_num: relax.ShapeExpr):
        self.tile_num = tile_num
        self.exec_scope = None
        super().__init__()

    @staticmethod
    def rewrite(func, tile_num: relax.ShapeExpr):
        mutator = DeviceFuncToKernel(tile_num)
        mutator.visit_stmt(func.body)
        assert mutator.exec_scope is not None, "no root scope found"
        # TODO: support dynamic tile num
        for n in tile_num:
            if not isinstance(n, T.IntImm):
                raise NotImplementedError("only support constant tile num for now")
        const_tile_num = [T.int32(n) for n in tile_num]

        def _get_scope_id():
            T.scope_id(const_tile_num, "kernel", mutator.exec_scope.name)

        @T.prim_func(tirx=True)
        def kernel_func():
            with T.kernel():
                _get_scope_id()
                Tx.tvm_kernel_replace_point()

        new_body = seek_kernel_replace_point(kernel_func.body, func.body)
        # Navigate through ExecScopeStmt to find the exec_scope with scope_id_def
        if isinstance(new_body, ExecScopeStmt):
            exec_scope = new_body.exec_scope
        else:
            raise ValueError("Expected ExecScopeStmt wrapping kernel body")
        added_vars = exec_scope.scope_id_def[0].def_ids
        var_replace_map = {
            func.params[i + len(func.params) - len(tile_num)]: v for i, v in enumerate(added_vars)
        }
        new_body = BufferReplacer(var_map=var_replace_map).visit_stmt(new_body)
        new_params = func.params[: -len(tile_num)]
        return PrimFunc(new_params, new_body, func.ret_type, func.buffer_map, func.attrs)

    def visit_exec_scope_stmt_(self, stmt: ExecScopeStmt):
        self.exec_scope = stmt.exec_scope
        super().visit_exec_scope_stmt_(stmt)


@mutator
class _Rewriter(PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod = mod
        self.call_tir_device_op = tvm.ir.Op.get("relax.call_tir_device")

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                updated_func = self.visit_expr(func)
                self.builder_.update_func(g_var, updated_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed
        if call.op == self.call_tir_device_op:
            return self.rewrite_call_tir_device(call)
        else:
            return call

    def rewrite_call_tir_device(self, call: relax.Call) -> relax.Call:
        tile_num = call.args[2]
        tir_gvar = call.args[0]
        prim_func = self.mod[tir_gvar]
        verify_tirx_well_formed(prim_func, device_func=True)
        new_prim_func = DeviceFuncToKernel.rewrite(prim_func, tile_num)
        new_gvar = self.builder_.add_func(new_prim_func, tir_gvar.name_hint + "_kernel")
        new_call = relax.call_tir(new_gvar, call.args[1], call.sinfo_args[0])
        return new_call
