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
"""Static horizontal fusion
The pass is written in Python for experiment, fast development.
"""

import functools
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Literal

import tvm
from tvm import relax
from tvm.ir import load_json, save_json
from tvm.ir.module import IRModule
from tvm.megakernel.utils.base import SemaphoreBase, SmemManager, TileSchedulerBase
from tvm.megakernel.utils.config import JobType, KernelConfig, map_job_type_to_profile_event_type
from tvm.megakernel.utils.utils import any_sync, pack_into_32bit
from tvm.relax.expr import Expr, ShapeExpr
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator
from tvm.relax.struct_info import StructInfo, TensorStructInfo, TupleStructInfo
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.tirx import (
    AttrStmt,
    Bind,
    Buffer,
    BufferStore,
    Evaluate,
    ExecScopeStmt,
    IntImm,
    PrimExpr,
    PrimFunc,
    SBlockRealize,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tirx.analysis import verify_tirx_well_formed
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.operator import KernelReplacePoint
from tvm.tirx.stmt_functor import StmtExprMutator, StmtExprVisitor
from tvm.tirx.transform.common import BufferReplacer, seek_kernel_replace_point


def _get_exec_scope_body_attr(op: ExecScopeStmt, key: str):
    """Check if the ExecScopeStmt body has an AttrStmt with the given key.

    Returns the AttrStmt value if found, None otherwise.
    Annotations on ExecScopeStmt are now represented as AttrStmt wrapping the body.
    """
    body = op.body
    if isinstance(body, AttrStmt) and body.attr_key == key:
        return body.value
    return None


def _strip_exec_scope_body_attr(body: Stmt, key: str):
    """Strip an AttrStmt with the given key from the outermost level of body.

    Returns the inner body with the AttrStmt removed. If the key is not found,
    returns the original body unchanged.
    """
    if isinstance(body, AttrStmt) and body.attr_key == key:
        return body.body
    return body


def _find_outermost_exec_scope(stmt):
    """Find the outermost ExecScope from a statement tree."""
    if isinstance(stmt, ExecScopeStmt):
        return stmt.exec_scope
    raise ValueError("No ExecScopeStmt found in device function body")


def _as_seq(stmts: list[Stmt], span=None) -> Stmt:
    """Build a SeqStmt when needed, otherwise return the single stmt."""
    if not stmts:
        return Evaluate(0)
    if len(stmts) == 1:
        return stmts[0]
    return SeqStmt(stmts, span)


def _prepend_stmt(body: Stmt, prefix: Stmt, span=None) -> Stmt:
    """Insert one statement before a body while preserving sequence shape."""
    if isinstance(body, SeqStmt):
        return SeqStmt([prefix, *body.seq], body.span)
    return SeqStmt([prefix, body], span)


def _prepend_bind(body: Stmt, var: Var, value: PrimExpr, span=None) -> Stmt:
    return _prepend_stmt(body, Bind(var=var, value=value, span=span), span)


# FIXME: add decl_buffer for all newly generated buffers


@tvm.transform.module_pass(opt_level=0, name="StaticHorizontalFusion")
class StaticHorizontalFusion:
    """Static horizontal fusion."""

    def __init__(
        self,
        func_name: str | list[str],
        strategy: Literal["static", "dynamic"],
        tile_scheduler_class: type[TileSchedulerBase],
        semaphore_class: type[SemaphoreBase],
        fusion_prefix: str = "megakernel_",
        profiler_on: bool = False,
    ):
        self.func_name = func_name if isinstance(func_name, list) else [func_name]
        self.strategy = strategy
        self.tile_scheduler_class = tile_scheduler_class
        self.semaphore_class = semaphore_class
        self.fusion_prefix = fusion_prefix
        self.profiler_on = profiler_on
        self.gvar_to_remove = []

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        rewriter = _Rewriter(
            mod,
            self.strategy,
            self.func_name,
            self.fusion_prefix,
            self.tile_scheduler_class,
            self.semaphore_class,
            self.profiler_on,
        )
        mod = rewriter.transform()
        for gvar in rewriter.gvar_to_remove:
            del mod[gvar]
        mod = relax.transform.DeadCodeElimination()(mod)
        return mod


@dataclass
class EventHandleEntry:
    idx: int
    event: Expr
    scope: Literal["thread", "warp", "warpgroup", "cta"]
    scope_id: int
    push_level: Literal["thread", "warp", "warpgroup", "cta", "none"]
    push_job_id: int
    freeze: bool


@dataclass
class EventHandleConfigs:
    func_type: str
    job_id: int
    wait_scope: Literal["warp", "cta"]
    notify_scope: Literal["thread", "warp", "warpgroup", "cta"]
    notify_scope_id: int
    push_scope: Literal["thread", "warp", "warpgroup", "cta"]
    push_level: Literal["thread", "warp", "warpgroup", "cta", "none"]
    push_scope_id: int
    freeze: bool


@dataclass(frozen=True)
class VarEntry:
    relax_var: Expr
    tir_var: Var
    tir_buffer: Buffer


@dataclass
class DeviceFuncInfo:
    func_body: Stmt
    in_var_entries: list[VarEntry]
    out_var_entries: list[VarEntry]
    tile_idx: list[PrimExpr]
    in_event_tensors: list[Buffer]
    out_event_tensors: list[Buffer]
    sym_vars: dict[Var, Var]
    in_info_list: list[callable]
    out_info_list: list[callable]
    push_info_list: list[tuple[callable, list[callable]]]
    job_type_id: int
    evt_handle_config: EventHandleConfigs


@dataclass
class AllocEventInfo:
    workspace: Expr
    event_tensor: Expr | None
    shape: ShapeExpr
    f_init: PrimFunc
    extra_args: list[Expr | PrimExpr]


@relax.expr_functor.visitor
class EventHandleHelper(PyExprVisitor):
    def __init__(self, mod: IRModule, strategy: Literal["static", "dynamic"]):
        super().__init__()
        self.mod = mod
        self.strategy = strategy
        self.call_tir_device_op = tvm.ir.Op.get("relax.call_tir_device")
        self.call_alloc_evt_op = tvm.ir.Op.get("relax.alloc_event_tensor")
        self.event_handle_configs: dict[relax.Call, EventHandleConfigs] = {}
        self.in_evt: dict[Expr, set[tuple[int, relax.Call]]] = defaultdict(set)
        self.out_evt: dict[Expr, set[tuple[int, relax.Call]]] = defaultdict(set)
        self.prev: dict[relax.Call, set[relax.Call]] = defaultdict(set)
        self.next: dict[relax.Call, set[relax.Call]] = defaultdict(set)
        self.topological_order: list[relax.Call] = []
        self.entry_points: set[relax.Call] = set()
        self.end_points: set[relax.Call] = set()
        self.alloc_evt_infos: dict[Expr, AllocEventInfo] = {}

    def clear_state(self):
        self.event_handle_configs = {}
        self.in_evt = defaultdict(set)
        self.out_evt = defaultdict(set)
        self.prev = defaultdict(set)
        self.next = defaultdict(set)
        self.topological_order = []
        self.entry_points = set()
        self.end_points = set()
        self.alloc_evt_infos = {}

    def visit_call_(self, call: relax.Call) -> None:
        if call.op == self.call_tir_device_op:
            tir_gvar = call.args[0]
            job_id = call.attrs.job_id
            in_events = call.attrs.in_events
            out_events = call.attrs.out_events
            handle_configs = call.attrs.handle_config
            prim_func = load_json(save_json(self.mod[tir_gvar]))
            func_type = prim_func.attrs.get("megakernel.device_func")
            # set the default handle config
            default_config_pool = {
                "gemm": {
                    "wait_scope": "warp",
                    "notify_scope": "warpgroup",
                    "notify_scope_id": 0,
                    "push_level": "warp",
                    "push_scope": "warpgroup",
                    "push_scope_id": 0,
                },
                "default": {
                    "wait_scope": "cta",
                    "notify_scope": "cta",
                    "notify_scope_id": 0,
                    "push_level": "warp",
                    "push_scope": "cta",
                    "push_scope_id": 0,
                },
            }
            default_config = default_config_pool.get(func_type, default_config_pool["default"])
            self.event_handle_configs[call] = EventHandleConfigs(
                func_type, job_id, **default_config, freeze=False
            )
            if handle_configs:
                self.event_handle_configs[call].wait_scope = handle_configs.get(
                    "wait_scope", self.event_handle_configs[call].wait_scope
                )
                self.event_handle_configs[call].notify_scope = handle_configs.get(
                    "notify_scope", self.event_handle_configs[call].notify_scope
                )
                self.event_handle_configs[call].notify_scope_id = handle_configs.get(
                    "notify_scope_id", self.event_handle_configs[call].notify_scope_id
                )
                self.event_handle_configs[call].push_level = handle_configs.get(
                    "push_level", self.event_handle_configs[call].push_level
                )
                self.event_handle_configs[call].push_scope = handle_configs.get(
                    "push_scope", self.event_handle_configs[call].push_scope
                )
                self.event_handle_configs[call].push_scope_id = handle_configs.get(
                    "push_scope_id", self.event_handle_configs[call].push_scope_id
                )
                self.event_handle_configs[call].freeze = True
            for idx, evt in enumerate(in_events):
                self.in_evt[evt].add((idx, call))
            for idx, evt in enumerate(out_events):
                self.out_evt[evt].add((idx, call))
            return super().visit_call_(call)
        elif call.op == self.call_alloc_evt_op:
            return super().visit_call_(call)
        else:
            raise ValueError(
                "StaticHorizontalFusion can only be applied to relax function with only call_tir_device"  # noqa: E501
            )

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        ret = super().visit_var_binding_(binding)
        if isinstance(binding.value, relax.Call) and binding.value.op == self.call_alloc_evt_op:
            call = binding.value
            self.alloc_evt_infos[binding.var] = AllocEventInfo(
                call.args[0],
                binding.var,
                call.args[1],
                load_json(save_json(self.mod[call.args[2]])),
                call.attrs.extra_args,
            )
        return ret

    def infer(self, func: relax.Function):
        self.clear_state()
        self.visit_expr(func)
        # for dynamic scheduling, one task can only rely on one event
        if self.strategy == "dynamic":
            for call in self.event_handle_configs.keys():
                assert len(call.attrs.in_events) <= 1, (
                    "Dynamic scheduling only supports one event wait per task"
                )
        # check whether the dep graph is a DAG and find entry nodes
        for call, config in self.event_handle_configs.items():
            for evt in call.attrs.in_events:
                self.prev[call].update({call for _, call in self.out_evt[evt]})
            for evt in call.attrs.out_events:
                self.next[call].update({call for _, call in self.in_evt[evt]})
        # topological sort
        visited = set()
        stack = set()

        def dfs(cur_call):
            if cur_call in stack:
                raise ValueError("The event dependency graph is not a DAG")
            if cur_call in visited:
                return
            visited.add(cur_call)
            stack.add(cur_call)
            for next_call in self.next[cur_call]:
                dfs(next_call)
            stack.remove(cur_call)
            self.topological_order.append(cur_call)

        for call in self.event_handle_configs.keys():
            dfs(call)
            if len(self.prev[call]) == 0:
                self.entry_points.add(call)
            if len(self.next[call]) == 0:
                self.end_points.add(call)
        self.topological_order = self.topological_order[::-1]
        # check the end points
        final_evt = None
        for end_call in self.end_points:
            if len(end_call.attrs.out_events) != 1:
                raise ValueError("The end point tasks call must have exactly one out event")
            if final_evt is not None and final_evt != end_call.attrs.out_events[0]:
                raise ValueError("All end point tasks must share the same out event")
            final_evt = end_call.attrs.out_events[0]
        final_evt_shape = [s for s in final_evt.struct_info.shape.values]
        if (
            len(final_evt_shape) != 1
            or not isinstance(final_evt_shape[0], tvm.tirx.IntImm)
            or final_evt_shape[0].value != 1
        ):
            raise ValueError("The final event must have shape (1,)")
        # propagate the inv dependencies to previous calls (push)
        if self.strategy == "dynamic":
            for call, config in self.event_handle_configs.items():
                if not config.freeze:
                    for prev_call in self.prev[call]:
                        if (
                            self.event_handle_configs[call].func_type != "gemm"
                            and self.event_handle_configs[prev_call].func_type == "gemm"
                        ):
                            config.push_level = "warp"
                            config.push_scope = "warpgroup"
                            config.push_scope_id = 1


class EventPrimFuncHelper(StmtExprMutator):
    # This class is used to post process the primfunc body generated for event operations
    # Will aggregates all statements immediately following a marked Block into that Block's body.
    # The purpose is to fix the scope of vars declared in input dependency callbacks.
    def __init__(self):
        super().__init__()
        self.stmt_list = []

    @staticmethod
    def _is_mark(stmt: Stmt, mark: str) -> bool:
        return (
            isinstance(stmt, AttrStmt)
            and stmt.attr_key == "marks"
            and isinstance(stmt.value, tvm.tirx.StringImm)
            and stmt.value.value == mark
        )

    @staticmethod
    def _extract_entry_payload(
        stmt: Stmt,
    ) -> tuple[list[Stmt], list[tuple[Var, PrimExpr, object]]] | None:
        if EventPrimFuncHelper._is_mark(stmt, "entry_mark"):
            return ([], [])
        if isinstance(stmt, SeqStmt) and len(stmt.seq) >= 1:
            if not EventPrimFuncHelper._is_mark(stmt.seq[-1], "entry_mark"):
                return None
            seq = list(stmt.seq[:-1])
            binds = []
            while len(seq) > 0 and isinstance(seq[-1], Bind):
                bind_stmt = seq.pop()
                binds.append((bind_stmt.var, bind_stmt.value, bind_stmt.span))
            binds.reverse()
            return (seq, binds)
        return None

    def visit_seqstmt_(self, op):
        new_seq = []
        seq = [stmt for stmt in op.seq]
        if seq and self._is_mark(seq[-1], "entry_mark"):
            stmt_list = self.stmt_list
            self.stmt_list = []
            seq = seq[:-1] + stmt_list
        for stmt in reversed(seq):
            if self._is_mark(stmt, "beg_mark"):
                entry_payload = self._extract_entry_payload(stmt.body)
                if entry_payload is None:
                    raise ValueError(
                        "Expected beg_mark body to end with entry_mark and optional binds, "
                        f"but got {type(stmt.body)}"
                    )
                prefix_stmts, entry_binds = entry_payload
                if len(prefix_stmts) == 0 and len(entry_binds) == 0:
                    continue
                captured = new_seq
                bound_body = _as_seq(captured)
                for var, value, span in entry_binds:
                    bound_body = _prepend_bind(bound_body, var, value, span)
                if len(prefix_stmts) > 0:
                    prefix_seq = list(prefix_stmts)
                    if isinstance(bound_body, SeqStmt):
                        prefix_seq.extend(bound_body.seq)
                    else:
                        prefix_seq.append(bound_body)
                    bound_body = _as_seq(prefix_seq)
                new_seq = [bound_body]
            else:
                new_seq.insert(0, self.visit_stmt(stmt))
        return _as_seq(new_seq, op.span)


class EventOpInserter(StmtExprMutator):
    def __init__(
        self,
        strategy: Literal["static", "dynamic"],
        func_info: DeviceFuncInfo,
        tile_scheduler: TileSchedulerBase,
        semaphore_class: type[SemaphoreBase],
    ):
        super().__init__()
        self.strategy = strategy
        self.func_info = func_info
        self.tile_scheduler = tile_scheduler
        self.semaphore_class = semaphore_class

    @staticmethod
    def rewrite(
        strategy: Literal["static", "dynamic"],
        func_info: DeviceFuncInfo,
        tile_scheduler: TileSchedulerBase,
        semaphore_class: type[SemaphoreBase],
    ) -> Stmt:
        inserter = EventOpInserter(strategy, func_info, tile_scheduler, semaphore_class)
        return inserter.visit_stmt(func_info.func_body)

    def visit_exec_scope_stmt_(self, op: ExecScopeStmt):
        if _get_exec_scope_body_attr(op, "tirx.tile_class.prefetch") is not None:
            return op
        elif _get_exec_scope_body_attr(op, "tirx.tile_class.run") is not None:

            def get_wait(idx):
                @T.prim_func(tirx=True, check_well_formed=False)
                def wait():
                    sem = T.meta_var(self.semaphore_class(self.func_info.in_event_tensors[idx]))
                    wait_func = T.meta_var(self.func_info.in_info_list[idx])
                    self.tile_scheduler.wait(
                        sem,
                        *(
                            wait_func(T.int32(0))[2:]
                        ),  # notes: here (num, rank, *coords) = wait_func()
                        wait_level=self.func_info.evt_handle_config.wait_scope,
                    )

                return EventPrimFuncHelper().visit_stmt(wait.body)

            def get_notify(idx):
                @T.prim_func(tirx=True, check_well_formed=False)
                def notify():
                    sem = T.meta_var(self.semaphore_class(self.func_info.out_event_tensors[idx]))
                    notify_func = T.meta_var(self.func_info.out_info_list[idx])
                    self.tile_scheduler.notify(
                        sem,
                        notify_func,
                        scope=self.func_info.evt_handle_config.notify_scope,
                        scope_id=self.func_info.evt_handle_config.notify_scope_id,
                    )

                return EventPrimFuncHelper().visit_stmt(notify.body)

            if self.strategy == "dynamic":

                def get_pre_notify_and_push(idx):
                    if len(self.func_info.push_info_list[idx][1]) == 0:
                        return None

                    @T.prim_func(tirx=True, check_well_formed=False)
                    def pre_notify_and_push():
                        sem = T.meta_var(
                            self.semaphore_class(self.func_info.out_event_tensors[idx])
                        )
                        notify_func = T.meta_var(self.func_info.push_info_list[idx][0])
                        push_func_list = T.meta_var(self.func_info.push_info_list[idx][1])
                        self.tile_scheduler.pre_notify_and_push(
                            sem,
                            notify_func,
                            push_func_list,
                            push_level=self.func_info.evt_handle_config.push_level,
                            scope=self.func_info.evt_handle_config.push_scope,
                            scope_id=self.func_info.evt_handle_config.push_scope_id,
                        )

                    return EventPrimFuncHelper().visit_stmt(pre_notify_and_push.body)

            # insert wait and notify
            waits = [get_wait(idx) for idx in range(len(self.func_info.in_info_list))]
            commits = [get_notify(idx) for idx in range(len(self.func_info.out_info_list))]
            pushes = []
            if self.strategy == "dynamic":
                for idx in range(len(self.func_info.out_info_list)):
                    pre_notify_and_push = get_pre_notify_and_push(idx)
                    if pre_notify_and_push is not None:
                        pushes.append(pre_notify_and_push)

            # Strip the tirx.tile_class.run AttrStmt to get the actual body
            inner_body = _strip_exec_scope_body_attr(op.body, "tirx.tile_class.run")
            body_list = pushes + waits + [inner_body] + commits
            new_body = SeqStmt(body_list) if len(body_list) > 1 else body_list[0]
            # Re-wrap with the tirx.tile_class.run AttrStmt
            new_body = AttrStmt(
                IntImm("int32", 0), "tirx.tile_class.run", IntImm("bool", 1), new_body
            )
            # Reconstruct the ExecScopeStmt with the new body.
            return ExecScopeStmt(op.exec_scope, new_body, op.span)
        else:
            return super().visit_exec_scope_stmt_(op)


@dataclass
class PersistentVarInfo:
    init_body: Stmt
    finalize_body: Stmt
    vars: dict[str, list[Buffer]]


class PersistentVarCollector(StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.persistent_vars_tile_class = defaultdict(list)
        self.persistent_var_tile_class_init = None
        self.persistent_var_tile_class_finalize = None
        self.persistent_vars_megakernel = defaultdict(list)
        self.persistent_var_megakernel_init = None
        self.persistent_var_megakernel_finalize = None
        self.visit_tile_class = False

    @staticmethod
    def rewrite(stmt: Stmt) -> tuple[Stmt, PersistentVarInfo, PersistentVarInfo]:
        collector = PersistentVarCollector()
        return (
            collector.visit_stmt(stmt),
            PersistentVarInfo(
                collector.persistent_var_tile_class_init,
                collector.persistent_var_tile_class_finalize,
                collector.persistent_vars_tile_class,
            ),
            PersistentVarInfo(
                collector.persistent_var_megakernel_init,
                collector.persistent_var_megakernel_finalize,
                collector.persistent_vars_megakernel,
            ),
        )

    def visit_alloc_buffer_(self, op):
        op = super().visit_alloc_buffer_(op)
        if "persistent" in op.buffer.scope():
            # notes: here using buffer.name to distinguish different persistent buffers
            #        it's not a robust way, need to be improved in the future
            cur_persistent_vars = (
                self.persistent_vars_tile_class
                if self.visit_tile_class
                else self.persistent_vars_megakernel
            )
            cur_persistent_vars[op.buffer.name].append(op.buffer)
            return Evaluate(0)
        return op

    def visit_decl_buffer_(self, op):
        op = super().visit_decl_buffer_(op)
        if "persistent" in op.buffer.scope():
            cur_persistent_vars = (
                self.persistent_vars_tile_class
                if self.visit_tile_class
                else self.persistent_vars_megakernel
            )
            cur_persistent_vars[op.buffer.name].append(op.buffer)
            return Evaluate(0)
        return op

    def visit_allocate_(self, op):
        # here buffer with shared.persistent scope will be declared with None data pointer
        # so we need to remove the extra allocate node due to this
        op = super().visit_allocate_(op)
        if "persistent" in op.buffer_var.type_annotation.storage_scope:
            return op.body
        return op

    def visit_exec_scope_stmt_(self, op: ExecScopeStmt):
        if any(
            _get_exec_scope_body_attr(op, key) is not None
            for key in ["tirx.tile_class.persistent.init", "tirx.tile_class.persistent.finalize"]
        ):
            self.visit_tile_class = True
        elif any(
            _get_exec_scope_body_attr(op, key) is not None
            for key in ["tirx.megakernel.persistent.init", "tirx.megakernel.persistent.finalize"]
        ):
            self.visit_tile_class = False
        op = super().visit_exec_scope_stmt_(op)
        annotation_to_var = {
            "tirx.tile_class.persistent.init": "persistent_var_tile_class_init",
            "tirx.tile_class.persistent.finalize": "persistent_var_tile_class_finalize",
            "tirx.megakernel.persistent.init": "persistent_var_megakernel_init",
            "tirx.megakernel.persistent.finalize": "persistent_var_megakernel_finalize",
        }
        for annotation, var_name in annotation_to_var.items():
            if _get_exec_scope_body_attr(op, annotation) is not None:
                # The actual body is inside the AttrStmt
                inner_body = _strip_exec_scope_body_attr(op.body, annotation)
                setattr(self, var_name, inner_body)
                break
        if not any(_get_exec_scope_body_attr(op, key) for key in annotation_to_var.keys()):
            return op
        return ExecScopeStmt(op.exec_scope, Evaluate(0), op.span)


class HostStmtCollector(StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.uppermost_block = None

    def visit_exec_scope_stmt_(self, op):
        """Capture the outermost ExecScopeStmt.

        In the new IR, ExecScopeStmt replaces the old outermost SBlockRealize
        as the top-level scope container for device function bodies.
        """
        if self.uppermost_block is None:
            self.uppermost_block = op
            return KernelReplacePoint()
        return super().visit_exec_scope_stmt_(op)

    def visit_block_realize_(self, op: SBlockRealize):
        # Fallback: capture SBlockRealize if no ExecScopeStmt was found first
        if self.uppermost_block is None:
            self.uppermost_block = op
            return KernelReplacePoint()
        return super().visit_block_realize_(op)

    @staticmethod
    def rewrite(stmt: Stmt) -> tuple[Stmt, Stmt]:
        collector = HostStmtCollector()
        host_replace_template = collector.visit_stmt(stmt)
        return collector.uppermost_block, host_replace_template


class VarCollector(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.vars = set()

    def visit_var_(self, op):
        self.vars.add(op)
        return super().visit_var_(op)


class DuplicateVarReplacer(StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.vars = {}

    def visit_bind_(self, op):
        assert op.var not in self.vars
        self.vars[op.var] = T.Var("var", dtype=op.value.dtype, span=op.var.span)
        value = self.visit_expr(op.value)
        return Bind(self.vars[op.var], value, op.span)

    def visit_for_(self, op):
        assert op.loop_var not in self.vars
        self.vars[op.loop_var] = T.Var("var", dtype=op.loop_var.dtype, span=op.loop_var.span)
        min_val = self.visit_expr(op.min)
        extent = self.visit_expr(op.extent)
        body = self.visit_stmt(op.body)
        return tvm.tirx.For(
            self.vars[op.loop_var],
            min_val,
            extent,
            op.kind,
            body,
            op.thread_binding,
            op.annotations,
            op.span,
        )

    def visit_allocate_(self, op):
        assert op.buffer_var not in self.vars
        self.vars[op.buffer_var] = T.Var("var", dtype=op.buffer_var.dtype, span=op.buffer_var.span)
        extents = [self.visit_expr(extent) for extent in op.extents]
        body = self.visit_stmt(op.body)
        condition = self.visit_expr(op.condition)
        return tvm.tirx.Allocate(
            self.vars[op.buffer_var], op.dtype, extents, condition, body, op.annotations, op.span
        )

    def visit_allocate_const_(self, op):
        assert op.buffer_var not in self.vars
        self.vars[op.buffer_var] = T.Var("var", dtype=op.buffer_var.dtype, span=op.buffer_var.span)
        extents = [self.visit_expr(extent) for extent in op.extents]
        body = self.visit_stmt(op.body)
        if op.data is not None:
            data_or_idx = op.data
        elif op.irmod_storage_idx is not None:
            data_or_idx = op.irmod_storage_idx
        else:
            data_or_idx = None
        return tvm.tirx.AllocateConst(
            self.vars[op.buffer_var], op.dtype, extents, data_or_idx, body, op.annotations, op.span
        )

    def visit_var_(self, op):
        if op in self.vars:
            return self.vars[op]
        else:
            return super().visit_var_(op)


class DynSharedMemCollector(StmtExprMutator):
    def __init__(self):
        super().__init__()
        self.dyn_shared_mem_buffer = None

    def visit_alloc_buffer_(self, op):
        op = super().visit_alloc_buffer_(op)
        if op.buffer.scope() == "shared.dyn":
            if self.dyn_shared_mem_buffer is None:
                self.dyn_shared_mem_buffer = op.buffer
                return Evaluate(0)
            else:
                raise ValueError(
                    f"Multiple dynamic shared memory buffers found: {self.dyn_shared_mem_buffer} and {op.buffer}"  # noqa: E501
                )
        return op

    @staticmethod
    def rewrite(stmt: Stmt) -> tuple[Buffer, Stmt]:
        collector = DynSharedMemCollector()
        stmt = collector.visit_stmt(stmt)
        return collector.dyn_shared_mem_buffer, stmt


class DisjointSet:
    """A compact Disjoint Set Union (DSU) data structure with path compression and union-by-size."""

    def __init__(self):
        self.parent: dict[object, object] = {}
        self.size: dict[object, int] = {}

    def add_element(self, element: object):
        if element not in self.parent:
            self.parent[element] = element
            self.size[element] = 1

    def find(self, element: object) -> object:
        if self.parent[element] == element:
            return element
        self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def join(self, a: object, b: object):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.size[root_a] < self.size[root_b]:
                root_a, root_b = root_b, root_a
            self.parent[root_b] = root_a
            self.size[root_a] += self.size[root_b]

    def is_connected(self, a: object, b: object) -> bool:
        return self.find(a) == self.find(b)

    def get_size(self, a: object) -> int:
        return self.size[self.find(a)]


class DependencyHandler(StmtExprMutator):
    def __init__(
        self,
        tile_idx: list[PrimExpr],
        in_var_entries: list[VarEntry],
        sym_vars: dict[Var, Var],
        relax_var_to_entry: dict[Var, VarEntry],
        builder_: relax.BlockBuilder,
        var_entry_mapping: DisjointSet,
    ):
        super().__init__()
        self.tile_idx = tile_idx
        self.in_var_entries = in_var_entries
        self.sym_vars = sym_vars
        self.relax_var_to_entry = relax_var_to_entry
        self.builder_ = builder_
        self.var_entry_mapping = var_entry_mapping

    def _get_func_template(self, job_id: int):
        if job_id == -1:

            def _func_template(
                idx: T.Var,
                buffer_replace_map,
                var_replace_map,
                func,
                out_buf_list,
                input_coord_dim,
                job_id,
                emit=True,
            ):
                replacer = BufferReplacer(
                    buffer_replace_map, var_replace_map | {func.params[input_coord_dim]: idx}
                )
                func_body = replacer.visit_stmt(func.body)
                stmt, output = self.rewrite(func_body, buffer_replace_map, out_buf_list)
                if emit:
                    T.add_to_parent(stmt)
                return output
        else:

            def _func_template(
                idx: T.Var,
                *args,
                buffer_replace_map,
                var_replace_map,
                func,
                out_buf_list,
                input_coord_dim,
                job_id,
                emit=True,
            ):
                replacer = BufferReplacer(
                    buffer_replace_map,
                    var_replace_map
                    | {func.params[i]: args[i] for i in range(input_coord_dim)}
                    | {func.params[input_coord_dim]: idx},
                )
                func_body = replacer.visit_stmt(func.body)
                stmt, output = self.rewrite(func_body, buffer_replace_map, out_buf_list)
                if emit:
                    T.add_to_parent(stmt)
                return [T.int32(job_id), *output]

        return _func_template

    def _create_func(
        self,
        job_id: int,
        buffer_replace_map,
        var_replace_map,
        func: PrimFunc,
        out_buf_list: list[Buffer],
        input_coord_dim: int,
    ):
        func_template = self._get_func_template(job_id)
        return partial(
            func_template,
            buffer_replace_map=buffer_replace_map,
            var_replace_map=var_replace_map,
            func=func,
            out_buf_list=out_buf_list,
            input_coord_dim=input_coord_dim,
            job_id=job_id,
        )

    def handle(
        self,
        func: PrimFunc,
        extra_args: list[Expr | PrimExpr],
        input_dim: int,
        output_dim: int,
        job_id: int = -1,
    ):
        # notes: here we don't directly capture the outer variables in the closure to avoid wrongly using values when executing rather than compiling  # noqa: E501
        var_replace_map = {}
        buffer_replace_map = {}

        def _replace_var(src: Var, dst: Var):
            var_replace_map[src] = dst

        def _replace_buffer(src: Buffer, dst: Buffer):
            buffer_replace_map[src] = dst
            for i in range(len(src.shape)):
                if isinstance(src.shape[i], Var):
                    var_replace_map[src.shape[i]] = dst.shape[i]

        # handle the tile_idx
        if job_id == -1:
            for i, var in enumerate(self.tile_idx):
                _replace_var(func.params[i], var)
        # handle extra tensors and tir sym vars
        extra_args_offset = input_dim - len(extra_args)
        for i, var in enumerate(extra_args):
            if isinstance(var, Var | PrimExpr):
                if var not in self.sym_vars:
                    if isinstance(var, IntImm):
                        self.sym_vars[var] = T.IntImm("int32", var.value)
                    else:
                        self.sym_vars[var] = T.Var("new_sym", "int32")
                _replace_var(func.params[extra_args_offset + i], self.sym_vars[var])
            elif isinstance(var, Expr):
                if var in self.relax_var_to_entry:
                    var_entry = self.relax_var_to_entry[var]
                    _replace_var(func.params[extra_args_offset + i], var_entry.tir_var)
                    _replace_buffer(
                        func.buffer_map[func.params[extra_args_offset + i]], var_entry.tir_buffer
                    )
                else:
                    binding = self.builder_.lookup_binding(var)
                    assert isinstance(binding, relax.TupleGetItem), (
                        f"unknown source of call_tir_device extra_args: {var} with binding {binding}"  # noqa: E501
                    )
                    var_entry = VarEntry(
                        var,
                        func.params[extra_args_offset + i],
                        func.buffer_map[func.params[extra_args_offset + i]],
                    )
                    self.var_entry_mapping.add_element(var_entry)
                    self.relax_var_to_entry[var] = var_entry
                self.in_var_entries.append(var_entry)
            else:
                raise ValueError(f"Unsupported extra arg type: {type(var)}")
        # handle the output buffer
        out_buf_list = [func.buffer_map[func.params[input_dim + i]] for i in range(output_dim)]
        out_func = self._create_func(
            job_id,
            buffer_replace_map,
            var_replace_map,
            func,
            out_buf_list,
            input_dim - len(extra_args) - 1,
        )
        return out_func

    def rewrite(
        self, stmt: Stmt, buffer_replace_map: dict[Buffer, Buffer], cur_buf_list: list[Buffer]
    ) -> tuple[Stmt, list[PrimExpr]]:
        # Some output buffers may not require remapping after replacement.
        # Fall back to the original buffer in that case.
        self.cur_buf_set = {
            buffer_replace_map.get(buf, buf): i for i, buf in enumerate(cur_buf_list)
        }
        self.cur_out_prim_expr = [None] * len(cur_buf_list)
        new_stmt = self.visit_stmt(stmt)
        for prim_expr in self.cur_out_prim_expr:
            assert prim_expr is not None, "dependency handling failed to set all output prim exprs"
        # warp the new_stmt into a block
        # it will be processed to fix the scope of the variables declared in the input dependencies in the post process step  # noqa: E501
        new_stmt = AttrStmt(
            T.StringImm("__mark_wrapper"), "marks", T.StringImm("beg_mark"), new_stmt
        )
        return new_stmt, self.cur_out_prim_expr

    def visit_buffer_store_(self, op):
        if op.buffer in self.cur_buf_set:
            assert len(op.indices) == 1 and op.indices[0] == 0
            entry_mark = AttrStmt(
                T.StringImm("__mark_wrapper"), "marks", T.StringImm("entry_mark"), Evaluate(0)
            )
            if isinstance(op.value, IntImm):
                self.cur_out_prim_expr[self.cur_buf_set[op.buffer]] = op.value
                return entry_mark
            else:
                self.cur_out_prim_expr[self.cur_buf_set[op.buffer]] = T.Var(
                    "var", dtype=op.value.dtype, span=op.value.span
                )
                bind_stmt = Bind(
                    var=self.cur_out_prim_expr[self.cur_buf_set[op.buffer]],
                    value=op.value,
                    span=op.span,
                )
                return SeqStmt([bind_stmt, entry_mark], op.span)
        return super().visit_buffer_store_(op)

    def _is_entry_mark(self, stmt: Stmt) -> bool:
        return (
            isinstance(stmt, AttrStmt)
            and stmt.attr_key == "marks"
            and stmt.value.value == "entry_mark"
        )

    def _extract_entry_binding(
        self, stmt: Stmt
    ) -> tuple[Var, PrimExpr, object] | tuple[None, None, None] | None:
        if self._is_entry_mark(stmt):
            return (None, None, None)
        if isinstance(stmt, SeqStmt) and len(stmt.seq) == 2:
            bind_stmt, mark_stmt = stmt.seq
            if isinstance(bind_stmt, Bind) and self._is_entry_mark(mark_stmt):
                return (bind_stmt.var, bind_stmt.value, bind_stmt.span)
        return None

    def _build_entry_mark_with_bindings(self, bindings: list[tuple[Var, PrimExpr, object]]) -> Stmt:
        entry_mark = AttrStmt(
            T.StringImm("__mark_wrapper"), "marks", T.StringImm("entry_mark"), Evaluate(0)
        )
        if len(bindings) == 0:
            return entry_mark
        bind_stmts = [
            Bind(var=var, value=value, span=span) for var, value, span in reversed(bindings)
        ]
        return SeqStmt([*bind_stmts, entry_mark])

    def visit_seqstmt_(self, op):
        op = super().visit_seqstmt_(op)
        inv_seq = []
        flag = True
        out_bindings: list[tuple[Var, PrimExpr, object]] = []
        for stmt in reversed(op.seq):
            entry_binding = self._extract_entry_binding(stmt)
            if entry_binding is not None and flag:
                if entry_binding[0] is not None:
                    out_bindings.append(entry_binding)
                continue
            if flag:
                flag = False
                inv_seq.append(self._build_entry_mark_with_bindings(out_bindings))
            inv_seq.append(stmt)
        if flag:
            return self._build_entry_mark_with_bindings(out_bindings)
        else:
            assert len(inv_seq) > 1
            return SeqStmt(list(reversed(inv_seq)), op.span)


class InitEtensorFuncHandler(StmtExprMutator):
    def __init__(self):
        super().__init__()

    def rewrite(self, stmt: Stmt, output_buf: Buffer, etensor_buffer: Buffer, idx) -> Stmt:
        self.output_buf = output_buf
        self.etensor_buffer = etensor_buffer
        self.idx = idx
        return self.visit_stmt(stmt)

    def visit_buffer_store_(self, op):
        if op.buffer == self.output_buf:
            return BufferStore(
                buffer=self.etensor_buffer,
                value=op.value * (SemaphoreBase.base + 1),
                indices=[self.idx[0]],
                span=op.span,
            )
        return super().visit_buffer_store_(op)


@mutator
class _Rewriter(PyExprMutator):
    def __init__(
        self,
        mod: IRModule,
        strategy: Literal["static", "dynamic"],
        rewrite_func_name: str | list[str],
        fusion_prefix: str,
        tile_scheduler_class: type[TileSchedulerBase],
        semaphore_class: type[SemaphoreBase],
        profiler_on: bool,
    ):
        super().__init__(mod)
        self.mod = mod
        self.strategy = strategy
        self.call_tir_device_op = tvm.ir.Op.get("relax.call_tir_device")
        self.call_alloc_evt_op = tvm.ir.Op.get("relax.alloc_event_tensor")
        self.rewrite_func_name = set(rewrite_func_name)
        self.fusion_prefix = fusion_prefix
        self.tile_scheduler_class = tile_scheduler_class
        self.semaphore_class = semaphore_class
        self.profiler_on = profiler_on
        self.device_func_exec_scope = None
        self.device_func_infos: dict[relax.Call, DeviceFuncInfo] = {}
        self.persistent_var_infos: dict[str, PersistentVarInfo] = {}
        self.host_templates: list[Stmt] = []
        self.dyn_shared_mem_buffers = []
        self.event_buffers = {}
        self.ret_value_entries: dict[relax.Call, list[int]] = defaultdict(list)
        self.new_ret_index = {}
        self.var_entry_mapping = DisjointSet()
        self.relax_var_to_entry: dict[Var, VarEntry] = {}
        self.workspace_cur_offset: dict[str, int] = {}
        self.in_var_entries: list[VarEntry] = []
        self.gvar_to_remove = list()
        self.evt_handle_helper = EventHandleHelper(mod, self.strategy)
        self.profiler_buf_size = int(1e7)

    def clear_state(self):
        # user might want to rewrite 2 megakernels, so clear state before each rewrite
        self.device_func_exec_scope = None
        self.device_func_infos = {}
        self.persistent_var_infos = {}
        self.host_templates = []
        self.dyn_shared_mem_buffers = []
        self.event_buffers = {}
        self.ret_value_entries = defaultdict(list)
        self.new_ret_index = {}
        self.var_entry_mapping = DisjointSet()
        self.relax_var_to_entry = {}
        self.cur_rewrite_func_name = None
        self.workspace_cur_offset = {}
        self.in_var_entries = []

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function) and g_var.name_hint in self.rewrite_func_name:
                self.clear_state()
                self.cur_rewrite_func_name = g_var.name_hint
                self.builder_.update_func(g_var, self.visit_expr(func))
                # self.build_new_function(func)
        return self.builder_.finalize()

    def get_output_num(self, call: relax.Call) -> int:
        if isinstance(call.struct_info, TensorStructInfo):
            return 1
        elif isinstance(call.struct_info, TupleStructInfo):
            return len(call.struct_info.fields)
        else:
            raise ValueError(f"Unsupported struct info: {call.struct_info}")

    def flatten_sinfo(self, sinfo: StructInfo) -> list[TensorStructInfo]:
        if isinstance(sinfo, TensorStructInfo):
            return [sinfo]
        elif isinstance(sinfo, TupleStructInfo):
            return sinfo.fields
        else:
            raise ValueError(f"Unsupported struct info: {sinfo}")

    def _get_event_buffer(self, event: Expr) -> Buffer:
        if event not in self.event_buffers:
            assert isinstance(event.struct_info, TensorStructInfo), f"event {event} is not a tensor"
            shape = [T.int32(s) for s in event.struct_info.shape.values]
            alloc_evt_info = self.evt_handle_helper.alloc_evt_infos[event]
            assert alloc_evt_info.workspace in self.relax_var_to_entry
            if alloc_evt_info.workspace not in self.workspace_cur_offset:
                self.workspace_cur_offset[alloc_evt_info.workspace] = 0
            self.event_buffers[event] = T.buffer(
                shape,
                event.struct_info.dtype,
                data=self.relax_var_to_entry[alloc_evt_info.workspace].tir_buffer.data,
                scope="global",
                buffer_name="event",
                elem_offset=self.workspace_cur_offset[alloc_evt_info.workspace],
            )
            size = functools.reduce(lambda x, y: x * y, shape, 1)
            self.workspace_cur_offset[alloc_evt_info.workspace] += size
        return self.event_buffers[event]

    def _update_persistent_var_info(
        self,
        persistent_var_info_tile_class: PersistentVarInfo,
        persistent_var_info_megakernel: PersistentVarInfo,
        class_name: str | None,
    ):
        if class_name is None:
            return
        # megakernel persistent var is mapped with ""
        for name, persistent_info in zip(
            ["", class_name], [persistent_var_info_megakernel, persistent_var_info_tile_class]
        ):
            if name not in self.persistent_var_infos:
                self.persistent_var_infos[name] = persistent_info
            else:
                existing_info = self.persistent_var_infos[name]
                assert len(persistent_info.vars) == 0 or len(persistent_info.vars) == len(
                    existing_info.vars
                ), (
                    f"non empty persistent var num mismatch: {len(persistent_info.vars)} != {len(existing_info.vars)}"  # noqa: E501
                )
                if len(persistent_info.vars) > 0:
                    for var_name, var_list in existing_info.vars.items():
                        assert var_name in persistent_info.vars, (
                            f"var {var_name} not found in persistent_var_info"
                        )
                        var_list.extend(persistent_info.vars[var_name])
                    tvm.ir.assert_structural_equal(
                        persistent_info.init_body, existing_info.init_body
                    )
                    tvm.ir.assert_structural_equal(
                        persistent_info.finalize_body, existing_info.finalize_body
                    )

    def visit_function_(self, func: relax.Function) -> Expr:
        self.evt_handle_helper.infer(func)
        for i, para in enumerate(func.params):
            assert isinstance(para, relax.Var), (
                f"expected Var for parameter {para}, but got {type(para)}"
            )
            if isinstance(para.struct_info, TensorStructInfo):
                tir_var = Var(para.name_hint, "handle")
                shape = [T.int32(s) for s in para.struct_info.shape.values]
                tir_buf = T.buffer(
                    shape,
                    para.struct_info.dtype,
                    scope="global",
                    buffer_name=f"func_arg{i}",
                    offset_factor=1,
                )
                var_entry = VarEntry(para, tir_var, tir_buf)
                self.var_entry_mapping.add_element(var_entry)
                self.relax_var_to_entry[var_entry.relax_var] = var_entry
        return super().visit_function_(func)

    def _deep_copy(self, func: PrimFunc) -> PrimFunc:
        return load_json(save_json(func))

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed
        if call.op == self.call_tir_device_op:
            tile_num = call.args[2]
            assert len(tile_num) == 3, f"tile_num dimension mismatch: {len(tile_num)} != 3"
            tir_gvar = call.args[0]
            in_deps = call.args[3]
            out_deps = call.args[4]
            in_events = call.attrs.in_events
            out_events = call.attrs.out_events
            in_extra_args = call.attrs.in_extra_args
            out_extra_args = call.attrs.out_extra_args
            inplace_indices = list(int(i) for i in call.attrs.inplace_indices)
            prim_func = self._deep_copy(self.mod[tir_gvar])
            self.gvar_to_remove.append(tir_gvar)
            # check if this device function is well-formed and has consistent exec scope
            verify_tirx_well_formed(prim_func, device_func=True)
            # collect event tensors and insert event commit/wait
            in_event_tensors = []
            out_event_tensors = []
            for e in in_events:
                in_event_tensors.append(self._get_event_buffer(e))
            for e in out_events:
                out_event_tensors.append(self._get_event_buffer(e))
            tile_idx = prim_func.params[-len(tile_num) :]
            body, host_template = HostStmtCollector.rewrite(prim_func.body)
            self.host_templates.append(host_template)
            exec_scope = _find_outermost_exec_scope(prim_func.body)
            if self.device_func_exec_scope is None:
                self.device_func_exec_scope = exec_scope.name
            else:
                assert self.device_func_exec_scope == exec_scope.name, (
                    f"device function exec scope mismatch: {self.device_func_exec_scope} != {exec_scope.name}"  # noqa: E501
                )
            body, persistent_var_info_tile_class, persistent_var_info_megakernel = (
                PersistentVarCollector.rewrite(body)
            )
            func_type = prim_func.attrs.get("megakernel.device_func")
            self._update_persistent_var_info(
                persistent_var_info_tile_class, persistent_var_info_megakernel, func_type
            )

            dyn_shared_mem_buffer, body = DynSharedMemCollector.rewrite(body)
            if dyn_shared_mem_buffer is not None:
                self.dyn_shared_mem_buffers.append(dyn_shared_mem_buffer)
            # gather information for merging
            in_var_entries = []
            out_var_entries = []
            for i, arg in enumerate(call.args[1]):
                assert isinstance(arg, relax.Var), (
                    f"call_tir_device args[1] should be a list of relax.Var, but got {arg}"
                )
                var_entry = VarEntry(
                    arg, prim_func.params[i], prim_func.buffer_map[prim_func.params[i]]
                )
                in_var_entries.append(var_entry)
                self.var_entry_mapping.add_element(var_entry)
                self.relax_var_to_entry[var_entry.relax_var] = var_entry
                binding = self.builder_.lookup_binding(arg)
                if isinstance(binding, relax.Call):
                    self.var_entry_mapping.join(
                        var_entry, self.device_func_infos[binding].out_var_entries[0]
                    )
                elif isinstance(binding, relax.TupleGetItem):
                    tuple_binding = self.builder_.lookup_binding(binding.tuple_value)
                    if tuple_binding in self.device_func_infos:
                        self.var_entry_mapping.join(
                            var_entry,
                            self.device_func_infos[tuple_binding].out_var_entries[binding.index],
                        )
                elif binding is None:
                    pass
                else:
                    raise ValueError(f"Unsupported binding: {binding}, var: {arg}")
            output_num = self.get_output_num(call)
            non_inplace_output_num = 0
            for i in range(output_num):
                if inplace_indices[i] != -1:
                    var_entry = VarEntry(
                        None,
                        in_var_entries[inplace_indices[i]].tir_var,
                        in_var_entries[inplace_indices[i]].tir_buffer,
                    )
                    self.var_entry_mapping.add_element(var_entry)
                    self.relax_var_to_entry[var_entry.relax_var] = var_entry
                    self.var_entry_mapping.join(var_entry, in_var_entries[inplace_indices[i]])
                else:
                    var_entry = VarEntry(
                        None,
                        prim_func.params[len(in_var_entries) + non_inplace_output_num],
                        prim_func.buffer_map[
                            prim_func.params[len(in_var_entries) + non_inplace_output_num]
                        ],
                    )
                    non_inplace_output_num += 1
                    self.var_entry_mapping.add_element(var_entry)
                    self.relax_var_to_entry[var_entry.relax_var] = var_entry
                out_var_entries.append(var_entry)

            # handle the dep
            sym_vars = dict()
            in_info_list = []
            out_info_list = []
            push_info_list = []
            dep_handler = DependencyHandler(
                tile_idx,
                in_var_entries,
                sym_vars,
                self.relax_var_to_entry,
                self.builder_,
                self.var_entry_mapping,
            )
            for idx in range(len(in_deps)):
                in_dep_func = dep_handler.handle(
                    self._deep_copy(self.mod[in_deps[idx]]),
                    in_extra_args[idx],
                    len(tile_idx) + 1 + len(in_extra_args[idx]),
                    len(in_events[idx].struct_info.shape) + 2,
                )
                in_info_list.append(in_dep_func)
            if not (self.strategy == "static" and call in self.evt_handle_helper.end_points):
                # ignore notify for entry point in static scheduler
                for idx in range(len(out_deps)):
                    out_dep_func = dep_handler.handle(
                        self._deep_copy(self.mod[out_deps[idx]]),
                        out_extra_args[idx],
                        len(tile_idx) + 1 + len(out_extra_args[idx]),
                        len(out_events[idx].struct_info.shape) + 2,
                    )
                    out_info_list.append(out_dep_func)
            if self.strategy == "dynamic":
                for i, evt in enumerate(out_events):
                    notify_dep_func = dep_handler.handle(
                        self._deep_copy(self.mod[out_deps[i]]),
                        out_extra_args[i],
                        len(tile_idx) + 1 + len(out_extra_args[i]),
                        len(out_events[i].struct_info.shape) + 2,
                    )
                    push_info = (notify_dep_func, [])
                    if call in self.evt_handle_helper.end_points:
                        # handle the special case where the task is end point, which means we need to insert END task pushing  # noqa: E501
                        assert len(self.evt_handle_helper.in_evt[evt]) == 0

                        def push_map_func(idx: T.Var, emit=True):
                            return lambda idx: [JobType.END.value, KernelConfig.SM_NUMBER, 0, 0, 0]

                        push_info[1].append(push_map_func)
                    else:
                        for j, push_call in self.evt_handle_helper.in_evt[evt]:
                            assert len(push_call.args[3]) == 1
                            push_dep_func = dep_handler.handle(
                                self._deep_copy(self.mod[push_call.args[5][j]]),
                                push_call.attrs.inv_in_extra_args[j],
                                len(push_call.attrs.inv_in_events[j].struct_info.shape)
                                + 2
                                + len(push_call.attrs.inv_in_extra_args[j]),
                                len(tile_idx) + 1,
                                job_id=push_call.attrs.job_id,
                            )

                            def push_map_func(
                                idx: T.Var,
                                notify_dep_func=notify_dep_func,
                                push_dep_func=push_dep_func,
                                emit=True,
                            ):
                                return lambda push_idx: push_dep_func(
                                    push_idx, *(notify_dep_func(idx, emit=emit)[1:]), emit=emit
                                )

                            push_info[1].append(push_map_func)

                    push_info_list.append(push_info)
            self.device_func_infos[call] = DeviceFuncInfo(
                body,
                in_var_entries,
                out_var_entries,
                tile_idx,
                in_event_tensors,
                out_event_tensors,
                sym_vars,
                in_info_list,
                out_info_list,
                push_info_list,
                call.attrs.job_id,
                self.evt_handle_helper.event_handle_configs[call],
            )
            return super().visit_call_(call)
        elif call.op == self.call_alloc_evt_op:
            return call
        else:
            raise ValueError(
                "StaticHorizontalFusion can only be applied to relax function with only call_tir_device"  # noqa: E501
            )

    def _build_persistent_kernel(
        self,
        buffer_replace_map: dict[relax.Call, dict[Buffer, Buffer]],
        var_replace_map: dict[Var, Var],
    ):
        @T.meta_class
        class PersistentVars:
            def __init__(
                self, persistent_var_infos: dict[str, PersistentVarInfo], smem_manager: SmemManager
            ):
                nonlocal buffer_replace_map
                self.buffers = []
                self.inits = []
                self.finalizes = []
                for persistent_var_info in persistent_var_infos.values():
                    for buffer_name, old_buffers in reversed(persistent_var_info.vars.items()):
                        scope_resolve_table = {
                            "shared.persistent": "shared.dyn",
                            "local.persistent": "local",
                            "tmem.persistent": "tmem",
                        }
                        # TODO: Need to handle these more robustly
                        if old_buffers[0].scope() == "shared.persistent":
                            new_buffer = smem_manager.alloc(
                                old_buffers[0].shape,
                                old_buffers[0].dtype,
                                scope=scope_resolve_table[old_buffers[0].scope()],
                                align=old_buffers[0].data_alignment,
                                layout=old_buffers[0].layout,
                                name=buffer_name,
                                method="persistent",
                            )
                        elif old_buffers[0].scope() == "tmem.persistent":
                            new_buffer = T.decl_buffer(
                                old_buffers[0].shape,
                                old_buffers[0].dtype,
                                scope=scope_resolve_table[old_buffers[0].scope()],
                                align=old_buffers[0].data_alignment,
                                layout=old_buffers[0].layout,
                                allocated_addr=int(old_buffers[0].allocated_addr[0]),
                            )
                        else:
                            new_buffer = T.alloc_buffer(
                                old_buffers[0].shape,
                                old_buffers[0].dtype,
                                scope=scope_resolve_table[old_buffers[0].scope()],
                                align=old_buffers[0].data_alignment,
                                layout=old_buffers[0].layout,
                            )
                        self.buffers.append(new_buffer)
                        for old_buffer in old_buffers:
                            buffer_replace_map[old_buffer] = new_buffer
                for persistent_var_info in persistent_var_infos.values():
                    replacer = BufferReplacer(
                        buffer_map=buffer_replace_map, var_map=var_replace_map
                    )
                    self.inits.append(replacer.visit_stmt(persistent_var_info.init_body))
                    self.finalizes.append(replacer.visit_stmt(persistent_var_info.finalize_body))

            def init(self):
                for init in self.inits:
                    T.add_to_parent(init)

            def finalize(self):
                for finalize in self.finalizes:
                    T.add_to_parent(finalize)

        def switch_task_type(
            tile_scheduler: TileSchedulerBase, lane_id, profiler: CudaProfiler | None
        ):
            idxs, type = tile_scheduler.get_idx_and_task_type()
            total_type_count = len(self.device_func_infos)
            if_frames = [
                T.If(type == info.job_type_id) for info in self.device_func_infos.values()
            ] + [
                T.If(type == JobType.INIT_ETENSOR.value),
                T.If(type == JobType.WAIT_ETENSOR_INIT.value),
            ]
            then_frames = [T.Then() for i in range(total_type_count + 2)]
            else_frames = [T.Else() for i in range(total_type_count + 2)]
            for i, (call, func) in enumerate(self.device_func_infos.items()):
                if_frames[i].__enter__()
                new_body = EventOpInserter.rewrite(
                    self.strategy, func, tile_scheduler, self.semaphore_class
                )
                needed_idx_num = len(func.tile_idx)
                replacer = BufferReplacer(
                    buffer_map=buffer_replace_map,
                    var_map=var_replace_map
                    | {func.tile_idx[j]: idxs[j] for j in range(needed_idx_num)},
                )
                new_body = replacer.visit_stmt(new_body)
                with then_frames[i]:
                    if profiler is not None:
                        profiler.start(
                            map_job_type_to_profile_event_type[func.job_type_id], lane_id == 0
                        )
                    T.add_to_parent(new_body)
                    if profiler is not None:
                        profiler.end(
                            map_job_type_to_profile_event_type[func.job_type_id], lane_id == 0
                        )
                else_frames[i].__enter__()

            # manually allocate evt_wait_etensor_init_complete for static scheduler
            workspace, offset = next(iter(self.workspace_cur_offset.items()))
            self.workspace_cur_offset[workspace] += 1
            evt_wait_etensor_init_complete = T.buffer(
                [1],
                "int32",
                self.relax_var_to_entry[workspace].tir_buffer.data,
                buffer_name="event_init_complete",
                elem_offset=offset,
            )

            if_frames[-2].__enter__()
            with then_frames[-2]:
                init_etensor_impl(idxs, tile_scheduler, evt_wait_etensor_init_complete)
            else_frames[-2].__enter__()

            if_frames[-1].__enter__()
            with then_frames[-1]:
                wait_etensor_init_impl(evt_wait_etensor_init_complete)
            else_frames[-1].__enter__()

            T.add_to_parent(Evaluate(T.cuda.trap_when_assert_failed(False)))

            for i in range(total_type_count + 1, -1, -1):
                else_frames[i].__exit__(None, None, None)
                if_frames[i].__exit__(None, None, None)

        sym_map = {}

        def init_etensor_impl(
            idxs, tile_scheduler: TileSchedulerBase, evt_etensor_init_complete_buf: Buffer
        ):
            def convert_1d_index_to_nd(idx, shape):
                nd_idx = []
                for i in reversed(range(len(shape))):
                    nd_idx.append(idx % shape[i])
                    idx = idx // shape[i]
                return list(reversed(nd_idx))

            def handle_f_init(alloc_evt_info: AllocEventInfo, idx):
                event = alloc_evt_info.event_tensor
                etensor_buf_1d = self.event_buffers[event].view(-1)
                shape = [T.int32(s) for s in event.struct_info.shape.values]
                st_idx = convert_1d_index_to_nd(idx[0], shape)
                var_replace_map = {}
                buffer_replace_map = {}

                def _replace_var(src: Var, dst: Var):
                    var_replace_map[src] = dst

                def _replace_buffer(src: Buffer, dst: Buffer):
                    buffer_replace_map[src] = dst
                    for i in range(len(src.shape)):
                        if isinstance(src.shape[i], Var):
                            var_replace_map[src.shape[i]] = dst.shape[i]

                f_init = alloc_evt_info.f_init
                for i, var in enumerate(st_idx):
                    _replace_var(f_init.params[i], var)
                # handle extra tensors and tir sym vars
                extra_args_offset = len(st_idx)
                for i, var in enumerate(alloc_evt_info.extra_args):
                    if isinstance(var, Var | PrimExpr):
                        if var not in sym_map:
                            if isinstance(var, IntImm):
                                sym_map[var] = T.IntImm("int32", var.value)
                            else:
                                sym_map[var] = T.Var("new_sym", "int32")
                        _replace_var(f_init.params[extra_args_offset + i], sym_map[var])
                    elif isinstance(var, Expr):
                        if var in self.relax_var_to_entry:
                            var_entry = self.relax_var_to_entry[var]
                            _replace_var(f_init.params[extra_args_offset + i], var_entry.tir_var)
                            _replace_buffer(
                                f_init.buffer_map[f_init.params[extra_args_offset + i]],
                                var_entry.tir_buffer,
                            )
                        else:
                            binding = self.builder_.lookup_binding(var)
                            assert isinstance(binding, relax.TupleGetItem), (
                                f"unknown source of call_tir_device extra_args: {var} with binding {binding}"  # noqa: E501
                            )
                            var_entry = VarEntry(
                                var,
                                f_init.params[extra_args_offset + i],
                                f_init.buffer_map[f_init.params[extra_args_offset + i]],
                            )
                            self.var_entry_mapping.add_element(var_entry)
                            self.relax_var_to_entry[var] = var_entry
                        self.in_var_entries.append(var_entry)
                    else:
                        raise ValueError(f"Unsupported extra arg type: {type(var)}")
                # handle the output buffer
                out_buffer = f_init.buffer_map[f_init.params[-1]]

                replacer = BufferReplacer(buffer_replace_map, var_replace_map)
                func_body = replacer.visit_stmt(f_init.body)
                # Some init paths do not remap the output buffer attributes; in that case
                # keep using the original output buffer for the rewrite target.
                rewritten_out_buffer = buffer_replace_map.get(out_buffer, out_buffer)
                stmt = InitEtensorFuncHandler().rewrite(
                    func_body, rewritten_out_buffer, etensor_buf_1d, idx
                )
                T.add_to_parent(stmt)

            with T.cta():
                tid = T.thread_id([KernelConfig.NUM_THREADS])
                num_evt_infos = len(self.evt_handle_helper.alloc_evt_infos)
                if_frames = [T.If(idxs[0] == i) for i in range(num_evt_infos)]
                then_frames = [T.Then() for i in range(num_evt_infos)]
                else_frames = [T.Else() for i in range(num_evt_infos)]
                idx = T.alloc_local([1], "int32")
                T.buffer_store(idx, tid, [0])
                for i, alloc_evt_info in enumerate(self.evt_handle_helper.alloc_evt_infos.values()):
                    if_frames[i].__enter__()
                    with then_frames[i]:
                        shape = [
                            T.int32(s) for s in alloc_evt_info.event_tensor.struct_info.shape.values
                        ]
                        nelem = functools.reduce(lambda x, y: x * y, shape, 1)
                        with T.While(idx[0] < nelem):
                            handle_f_init(alloc_evt_info, idx)
                            T.buffer_store(idx, idx[0] + KernelConfig.NUM_THREADS, [0])
                    else_frames[i].__enter__()

                if self.strategy == "static":
                    # init the evt_etensor_init_complete
                    with T.If(idxs[0] == num_evt_infos):
                        with T.Then():
                            with T.If(tid == 0):
                                with T.Then():
                                    T.buffer_store(
                                        evt_etensor_init_complete_buf,
                                        (num_evt_infos + 1 + KernelConfig.SM_NUMBER)
                                        * (SemaphoreBase.base + 1),
                                        [0],
                                    )

                for i in range(num_evt_infos - 1, -1, -1):
                    else_frames[i].__exit__(None, None, None)
                    if_frames[i].__exit__(None, None, None)

                if self.strategy == "static":
                    with T.If(idxs[0] < num_evt_infos + 1):
                        with T.Then():
                            evt_etensor_init_complete = self.semaphore_class(
                                evt_etensor_init_complete_buf
                            )
                            tile_scheduler.notify(
                                evt_etensor_init_complete,
                                lambda notify_idx: (1, -1, 0),
                                scope="cta",
                                release=True,
                            )

        @T.inline
        def wait_etensor_init_impl(evt_etensor_init_complete_buf: Buffer):
            with T.thread():
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
                lane_id = T.lane_id([32])
                # if self.profiler_on:
                #     self.profiler.start(ProfileEventType.WAIT_ETENSOR_INIT, lane_id == 0)
                state = T.alloc_local([1], "int32")
                state[0] = -1
                while 1:
                    if lane_id == 0:
                        T.ptx.ld_global_acquire(state[0], evt_etensor_init_complete_buf.ptr_to([0]))
                    if any_sync(
                        0xFFFFFFFF,
                        state[0] <= KernelConfig.SM_NUMBER * (SemaphoreBase.base + 1)
                        and state[0] > 0,
                    ):
                        if lane_id == 0 and warp_id == 0:
                            T.cuda.atomic_add(
                                evt_etensor_init_complete_buf.ptr_to([0]), -(SemaphoreBase.base + 1)
                            )
                        break
                    T.cuda.nano_sleep(40)

        def replace_dyn_shared_mem_buffer(new_buffer):
            for buffer in self.dyn_shared_mem_buffers:
                buffer_replace_map[buffer] = new_buffer

        def set_tile_scheduler(queue, tasks, head, tail, smem_manager):
            if self.strategy == "static":
                tile_scheduler = self.tile_scheduler_class("mega_", queue, smem_manager)
            else:
                tile_scheduler = self.tile_scheduler_class(tasks, head, tail, smem_manager)
            return tile_scheduler

        def set_profiler(profiler_buf):
            if self.profiler_on:
                num_groups = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
                profiler_write_stride = KernelConfig.SM_NUMBER * num_groups
                profiler = CudaProfiler(profiler_buf, profiler_write_stride, num_groups)
                return profiler
            else:
                return None

        def init_profiler(warp_id, profiler):
            if self.profiler_on:
                profiler.init(warp_id)

        @T.inline
        def persistent_body(queue, tasks, head, tail, profiler_buf):
            with T.kernel():
                T.scope_id(
                    [KernelConfig.SM_NUMBER], parent="kernel", cur=self.device_func_exec_scope
                )
                T.thread_id([KernelConfig.NUM_THREADS])
                warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
                lane_id = T.lane_id([32])
                smem_buffer = T.alloc_buffer(
                    KernelConfig.MAX_SMEM_SIZE, "uint8", scope="shared.dyn", align=16
                )
                replace_dyn_shared_mem_buffer(smem_buffer)
                smem_manager = SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, smem_buffer.data)
                tile_scheduler = set_tile_scheduler(queue, tasks, head, tail, smem_manager)
                persistent_vars = PersistentVars(self.persistent_var_infos, smem_manager)
                profiler = T.meta_var(set_profiler(profiler_buf))
                init_profiler(warp_id, profiler)
                persistent_vars.init()
                tile_scheduler.init()
                if self.strategy == "dynamic":
                    tile_scheduler.next_tile()
                while tile_scheduler.valid():
                    switch_task_type(tile_scheduler, lane_id, profiler)
                    tile_scheduler.next_tile()
                if profiler is not None:
                    profiler.finalize(lane_id == 0)
                persistent_vars.finalize()

        if self.strategy == "static":
            if self.profiler_on:

                @T.prim_func(tirx=True, private=True)
                def persistent_kernel(
                    queue: T.buffer(
                        (KernelConfig.SM_NUMBER, self.tile_scheduler_class.MAX_TASKS), "int32"
                    ),
                    profiler_buf: T.buffer((self.profiler_buf_size,), "uint64"),
                ):
                    persistent_body(queue, None, None, None, profiler_buf)
            else:

                @T.prim_func(tirx=True, private=True)
                def persistent_kernel(
                    queue: T.buffer(
                        (KernelConfig.SM_NUMBER, self.tile_scheduler_class.MAX_TASKS), "int32"
                    ),
                ):
                    persistent_body(queue, None, None, None, None)
        else:
            if self.profiler_on:

                @T.prim_func(tirx=True, private=True)
                def persistent_kernel(
                    tasks: T.buffer((self.tile_scheduler_class.MAX_TASKS), "int32"),
                    head: T.buffer((1,), "int32"),
                    tail: T.buffer((1,), "int32"),
                    profiler_buf: T.buffer((self.profiler_buf_size,), "uint64"),
                ):
                    persistent_body(None, tasks, head, tail, profiler_buf)
            else:

                @T.prim_func(tirx=True, private=True)
                def persistent_kernel(
                    tasks: T.buffer((self.tile_scheduler_class.MAX_TASKS), "int32"),
                    head: T.buffer((1,), "int32"),
                    tail: T.buffer((1,), "int32"),
                ):
                    persistent_body(None, tasks, head, tail, None)

        new_body = persistent_kernel.body
        for template in self.host_templates:
            template = BufferReplacer(
                buffer_map=buffer_replace_map, var_map=var_replace_map
            ).visit_stmt(template)
            new_body = seek_kernel_replace_point(template, new_body)
        # collect the minimal set of sym vars and build the var_map
        cast_var_map = defaultdict(list)
        var_collector = VarCollector()
        for info in self.device_func_infos.values():
            for var, new_var in info.sym_vars.items():
                var_collector.visit_expr(var)
                if isinstance(var, IntImm):
                    continue
                cast_var_map[var].append(new_var)
        for var, new_var in sym_map.items():
            var_collector.visit_expr(var)
            if isinstance(var, IntImm):
                continue
            cast_var_map[var].append(new_var)
        para_var_map = {}
        for var in var_collector.vars:
            para_var_map[var] = Var("sym", var.dtype)
        old_sym_vars = list(para_var_map.keys())
        new_sym_vars = [para_var_map[var] for var in para_var_map.keys()]
        # insert cast stmts at the top
        replacer = BufferReplacer(var_map=para_var_map)
        for var, cast_var_list in cast_var_map.items():
            new_var_expr = replacer.visit_expr(var)
            for cast_var in cast_var_list:
                new_body = _prepend_bind(new_body, cast_var, T.cast(new_var_expr, "int32"))
        return persistent_kernel.with_body(new_body), new_sym_vars, old_sym_vars

    def _build_gen_exec_queue_kernel(self):
        THREAD_NUM = 256
        if self.strategy == "static":
            assert len(self.device_func_infos) <= KernelConfig.SM_NUMBER
        else:
            assert len(self.evt_handle_helper.entry_points) + 1 <= KernelConfig.SM_NUMBER
        var_collector = VarCollector()

        def emit_task_round_robin(queue, idx, bx, tx):
            # init etensor
            with T.If(bx == 0):
                with T.Then():
                    with T.If(tx == 0):
                        with T.Then():
                            num_evt_infos = len(self.evt_handle_helper.alloc_evt_infos)
                            for_grid = T.grid(
                                *[num_evt_infos + 1, 1, 1]
                            )  # +1 for wait_etensor_complete evt
                            vars = for_grid.__enter__()
                            packed_val = pack_into_32bit(
                                vars[0], vars[1], vars[2], JobType.INIT_ETENSOR.value, host=False
                            )
                            T.buffer_store(
                                queue,
                                packed_val,
                                [idx[0] % KernelConfig.SM_NUMBER, idx[0] // KernelConfig.SM_NUMBER],
                            )
                            T.buffer_store(idx, idx[0] + 1, 0)
                            for_grid.__exit__(None, None, None)
                with T.Else():
                    T.buffer_store(idx, idx[0] + num_evt_infos + 1, 0)

            # set entry tasks
            for i, call in enumerate(self.evt_handle_helper.entry_points):
                device_func_info = self.device_func_infos[call]
                tile_num = call.args[2]
                with T.If(bx == i + 1):
                    with T.Then():
                        with T.If(tx == 0):
                            with T.Then():
                                for e in tile_num.values:
                                    var_collector.visit_expr(e)
                                for_grid = T.grid(*tile_num.values)
                                vars = for_grid.__enter__()
                                packed_val = pack_into_32bit(
                                    vars[0],
                                    vars[1],
                                    vars[2],
                                    device_func_info.job_type_id,
                                    host=False,
                                )
                                T.buffer_store(
                                    queue,
                                    packed_val,
                                    [
                                        idx[0] % KernelConfig.SM_NUMBER,
                                        idx[0] // KernelConfig.SM_NUMBER,
                                    ],
                                )
                                T.buffer_store(idx, idx[0] + 1, 0)
                                for_grid.__exit__(None, None, None)
                    with T.Else():
                        T.buffer_store(idx, idx[0] + tile_num[0] * tile_num[1] * tile_num[2], 0)

            # wait_etensor_init_complete
            with T.If(bx == 1 + len(self.evt_handle_helper.entry_points)):
                with T.Then():
                    with T.If(tx == 0):
                        with T.Then():
                            num_evt_infos = len(self.evt_handle_helper.alloc_evt_infos)
                            for_grid = T.grid(
                                *[KernelConfig.SM_NUMBER, 1, 1]
                            )  # +1 for wait_etensor_complete evt
                            vars = for_grid.__enter__()
                            packed_val = pack_into_32bit(
                                vars[0],
                                vars[1],
                                vars[2],
                                JobType.WAIT_ETENSOR_INIT.value,
                                host=False,
                            )
                            T.buffer_store(
                                queue,
                                packed_val,
                                [idx[0] % KernelConfig.SM_NUMBER, idx[0] // KernelConfig.SM_NUMBER],
                            )
                            T.buffer_store(idx, idx[0] + 1, 0)
                            for_grid.__exit__(None, None, None)
                with T.Else():
                    T.buffer_store(idx, idx[0] + KernelConfig.SM_NUMBER, 0)

            # generate rest tasks in topological order
            for i, call in enumerate(self.evt_handle_helper.topological_order):
                if call in self.evt_handle_helper.entry_points:
                    continue
                device_func_info = self.device_func_infos[call]
                tile_num = call.args[2]
                with T.If(bx == i + 2):
                    with T.Then():
                        with T.If(tx == 0):
                            with T.Then():
                                for e in tile_num.values:
                                    var_collector.visit_expr(e)
                                for_grid = T.grid(*tile_num.values)
                                vars = for_grid.__enter__()
                                packed_val = pack_into_32bit(
                                    vars[0],
                                    vars[1],
                                    vars[2],
                                    device_func_info.job_type_id,
                                    host=False,
                                )
                                T.buffer_store(
                                    queue,
                                    packed_val,
                                    [
                                        idx[0] % KernelConfig.SM_NUMBER,
                                        idx[0] // KernelConfig.SM_NUMBER,
                                    ],
                                )
                                T.buffer_store(idx, idx[0] + 1, 0)
                                for_grid.__exit__(None, None, None)
                    with T.Else():
                        T.buffer_store(idx, idx[0] + tile_num[0] * tile_num[1] * tile_num[2], 0)

            with T.If(bx == len(self.device_func_infos) + 2):
                with T.Then():
                    with T.If(tx == 0):
                        with T.Then():
                            with T.serial(0, KernelConfig.SM_NUMBER):
                                packed_val = pack_into_32bit(0, 0, 0, JobType.END.value, host=False)
                                T.buffer_store(
                                    queue,
                                    packed_val,
                                    [
                                        idx[0] % KernelConfig.SM_NUMBER,
                                        idx[0] // KernelConfig.SM_NUMBER,
                                    ],
                                )
                                T.buffer_store(idx, idx[0] + 1, 0)

        def emit_entry_task(tasks, head, tail, idx, bx, tx):
            # init etensor
            with T.If(bx == 0):
                with T.Then():
                    with T.If(tx == 0):
                        with T.Then():
                            num_evt_infos = len(self.evt_handle_helper.alloc_evt_infos)
                            for_grid = T.grid(
                                *[num_evt_infos + 1, 1, 1]
                            )  # +1 for wait_etensor_complete evt
                            vars = for_grid.__enter__()
                            packed_val = pack_into_32bit(
                                vars[0], vars[1], vars[2], JobType.INIT_ETENSOR.value, host=False
                            )
                            T.buffer_store(tasks, packed_val, [idx[0]])
                            T.buffer_store(idx, idx[0] + 1, 0)
                            for_grid.__exit__(None, None, None)
                with T.Else():
                    T.buffer_store(idx, idx[0] + num_evt_infos + 1, 0)

            # set entry tasks
            for i, call in enumerate(self.evt_handle_helper.entry_points):
                device_func_info = self.device_func_infos[call]
                tile_num = call.args[2]
                with T.If(bx == i + 1):
                    with T.Then():
                        with T.If(tx == 0):
                            with T.Then():
                                for e in tile_num.values:
                                    var_collector.visit_expr(e)
                                for_grid = T.grid(*tile_num.values)
                                vars = for_grid.__enter__()
                                packed_val = pack_into_32bit(
                                    vars[0],
                                    vars[1],
                                    vars[2],
                                    device_func_info.job_type_id,
                                    host=False,
                                )
                                T.buffer_store(tasks, packed_val, [idx[0]])
                                T.buffer_store(idx, idx[0] + 1, 0)
                                for_grid.__exit__(None, None, None)
                    with T.Else():
                        T.buffer_store(idx, idx[0] + tile_num[0] * tile_num[1] * tile_num[2], 0)

            with T.If(bx >= len(self.evt_handle_helper.entry_points) + 1):
                with T.Then():
                    with T.If(
                        tvm.tirx.all(tx == 0, bx == len(self.evt_handle_helper.entry_points) + 1)
                    ):
                        with T.Then():
                            T.buffer_store(head, 0, [0])
                            T.buffer_store(tail, idx[0], [0])
                    rest_sm_num = KernelConfig.SM_NUMBER - (
                        len(self.evt_handle_helper.entry_points) + 1
                    )
                    T.buffer_store(
                        idx,
                        idx[0]
                        + (rest_sm_num - (KernelConfig.SM_NUMBER - bx)) * KernelConfig.NUM_THREADS
                        + tx,
                        0,
                    )
                    with T.While(idx[0] < self.tile_scheduler_class.MAX_TASKS):
                        T.buffer_store(tasks, -1, [idx[0]])
                        T.buffer_store(idx, idx[0] + rest_sm_num * KernelConfig.NUM_THREADS, 0)

        if self.strategy == "static":

            @T.prim_func(tirx=True, private=True)
            def gen_exec_queue(
                queue: T.buffer(
                    (KernelConfig.SM_NUMBER, self.tile_scheduler_class.MAX_TASKS), "int32"
                ),
            ):
                with T.kernel():
                    bx = T.cta_id([KernelConfig.SM_NUMBER])
                    tx = T.thread_id([THREAD_NUM])
                    idx = T.alloc_buffer([1], "int32", scope="local")
                    idx[0] = 0
                    emit_task_round_robin(queue, idx, bx, tx)
        else:

            @T.prim_func(tirx=True, private=True)
            def gen_exec_queue(
                tasks: T.buffer((self.tile_scheduler_class.MAX_TASKS,), "int32"),
                head: T.buffer((1,), "int32"),
                tail: T.buffer((1,), "int32"),
            ):
                with T.kernel():
                    # fill tasks queue with -1 first
                    bx = T.cta_id([KernelConfig.SM_NUMBER])
                    tx = T.thread_id([THREAD_NUM])
                    idx = T.alloc_buffer([1], "int32", scope="local")
                    idx[0] = 0
                    emit_entry_task(tasks, head, tail, idx, bx, tx)

        new_vars = [Var("sym", v.dtype) for v in var_collector.vars]
        replace = BufferReplacer(var_map={v: new_vars[i] for i, v in enumerate(var_collector.vars)})
        new_body = replace.visit_stmt(gen_exec_queue.body)
        func_with_sym_vars = PrimFunc(
            params=list(gen_exec_queue.params) + new_vars,
            body=new_body,
            ret_type=gen_exec_queue.ret_type,
            buffer_map=gen_exec_queue.buffer_map,
            attrs=gen_exec_queue.attrs,
            span=gen_exec_queue.span,
        )
        return func_with_sym_vars, list(var_collector.vars)

    def merge_function(self):
        new_relax_params = []
        new_tir_params = []
        new_buffer_map = {}
        out_sinfo = []
        inplace_indices = []
        buffer_replace_map = {}
        var_replace_map = {}

        def replace_buffer(src: Buffer, dst: Buffer):
            buffer_replace_map[src] = dst
            for i in range(len(src.shape)):
                if isinstance(src.shape[i], Var):
                    var_replace_map[src.shape[i]] = dst.shape[i]

        input_var_entries_after_merge = []
        output_var_entries_after_merge = []

        def connected_with_any_entries(e, entries):
            for i, input_entry in enumerate(entries):
                if self.var_entry_mapping.is_connected(e, input_entry):
                    return i
            return None

        visited_entry_group = set()
        for call, func in self.device_func_infos.items():
            for entry in func.in_var_entries:
                leader_entry = self.var_entry_mapping.find(entry)
                if leader_entry not in visited_entry_group:
                    visited_entry_group.add(leader_entry)
                    # this is an input tensor after merging
                    new_relax_params.append(entry.relax_var)
                    new_tir_params.append(leader_entry.tir_var)
                    new_buffer_map[leader_entry.tir_var] = leader_entry.tir_buffer
                    input_var_entries_after_merge.append(entry)
                replace_buffer(entry.tir_buffer, leader_entry.tir_buffer)
            for entry in func.out_var_entries:
                leader_entry = self.var_entry_mapping.find(entry)
                visited_entry_group.add(leader_entry)
        for entry in self.in_var_entries:
            leader_entry = self.var_entry_mapping.find(entry)
            if leader_entry not in visited_entry_group:
                visited_entry_group.add(leader_entry)
                # this is an input tensor after merging
                new_relax_params.append(entry.relax_var)
                new_tir_params.append(leader_entry.tir_var)
                new_buffer_map[leader_entry.tir_var] = leader_entry.tir_buffer
                input_var_entries_after_merge.append(entry)
            replace_buffer(entry.tir_buffer, leader_entry.tir_buffer)

        workspace = set()
        for info in self.evt_handle_helper.alloc_evt_infos.values():
            if info.workspace not in workspace:
                assert info.workspace in self.relax_var_to_entry
                entry = self.relax_var_to_entry[info.workspace]
                workspace.add(info.workspace)
                new_buffer_map[entry.tir_var] = entry.tir_buffer
                new_tir_params.append(entry.tir_var)
                new_relax_params.append(info.workspace)

        total_input_num = len(new_relax_params)

        for call, func in self.device_func_infos.items():
            self.new_ret_index[call] = {}
            if len(self.ret_value_entries[call]) == 1 and self.ret_value_entries[call][0] == -1:
                call_ret_indices = list(range(len(func.out_var_entries)))
            else:
                call_ret_indices = self.ret_value_entries[call]
            call_out_sinfo = self.flatten_sinfo(call.struct_info)
            for i, entry in enumerate(func.out_var_entries):
                leader_entry = self.var_entry_mapping.find(entry)
                replace_buffer(entry.tir_buffer, leader_entry.tir_buffer)
                input_connected = connected_with_any_entries(entry, input_var_entries_after_merge)
                output_connected = connected_with_any_entries(entry, output_var_entries_after_merge)
                if i in call_ret_indices:
                    if output_connected is None:
                        self.new_ret_index[call][i] = len(output_var_entries_after_merge)
                    else:
                        self.new_ret_index[call][i] = output_connected
                if output_connected is None:
                    out_sinfo.append(call_out_sinfo[i])
                    output_var_entries_after_merge.append(entry)
                    new_buffer_map[leader_entry.tir_var] = leader_entry.tir_buffer
                    if input_connected is None:
                        inplace_indices.append(-1)
                        new_tir_params.append(leader_entry.tir_var)
                    else:
                        inplace_indices.append(input_connected)

        persistent_kernel, persistent_kernel_new_sym_vars, persistent_kernel_sym_vars = (
            self._build_persistent_kernel(buffer_replace_map, var_replace_map)
        )
        new_body = persistent_kernel.body
        for i, param in enumerate(persistent_kernel.params):
            if not self.profiler_on or i < len(persistent_kernel.params) - 1:
                new_tir_params.insert(total_input_num + i, param)
            else:
                new_tir_params.append(param)  # append profiler buffer at last
        new_tir_params.extend(persistent_kernel_new_sym_vars)
        new_buffer_map.update(persistent_kernel.buffer_map)
        new_prim_func = PrimFunc(
            new_tir_params,
            new_body,
            None,
            new_buffer_map,
            tvm.ir.make_node(
                "ir.DictAttrs",
                is_tirx=True,
                global_symbol=f"persistent_kernel_{self.cur_rewrite_func_name}",
            ),
        )
        new_gvar = self.builder_.add_func(
            new_prim_func, f"persistent_kernel_{self.cur_rewrite_func_name}"
        )
        if self.strategy == "static":
            queue_buffer = next(iter(persistent_kernel.buffer_map.values()))
            gen_exec_queue_tir, sym_vars = self._build_gen_exec_queue_kernel()
            gen_exec_queue_tir_gvar = self.builder_.add_func(
                gen_exec_queue_tir, f"gen_exec_queue_tir_{self.cur_rewrite_func_name}"
            )
            rx_sym_vars = relax.Var("rx_sym_vars", R.Shape([T.int64() for _ in sym_vars]))
            with self.builder_.function(
                f"gen_exec_queue_{self.cur_rewrite_func_name}", rx_sym_vars
            ):
                tir_var = []
                for i in range(len(sym_vars)):
                    tir_var.append(T.int64())
                _ = self.builder_.match_cast(rx_sym_vars, R.Shape(tir_var))
                queue = relax.call_tir(
                    gen_exec_queue_tir_gvar,
                    [],
                    out_sinfo=TensorStructInfo(shape=queue_buffer.shape, dtype=queue_buffer.dtype),
                    tir_vars=tir_var,
                )
                self.builder_.emit_func_output(queue)
            gen_exec_queue_gvar = self.builder_.add_func(
                self.builder_.get()[f"gen_exec_queue_{self.cur_rewrite_func_name}"],
                f"gen_exec_queue_{self.cur_rewrite_func_name}",
            )
            exec_queue = self.builder_.emit(
                relax.op.call_builtin_with_ctx(
                    "megakernel.horizontal_fusion.get_exec_queue_static",
                    args=[gen_exec_queue_gvar, sym_vars],
                    sinfo_args=TensorStructInfo(shape=queue_buffer.shape, dtype=queue_buffer.dtype),
                )
            )
            new_relax_params.append(exec_queue)
        else:
            task_buffer = next(iter(persistent_kernel.buffer_map.values()))
            head_buffer = list(persistent_kernel.buffer_map.values())[1]
            tail_buffer = list(persistent_kernel.buffer_map.values())[2]
            gen_exec_queue_tir, sym_vars = self._build_gen_exec_queue_kernel()
            gen_exec_queue_gvar = self.builder_.add_func(
                gen_exec_queue_tir, f"gen_exec_queue_{self.cur_rewrite_func_name}"
            )
            output_sinfo = [
                TensorStructInfo(shape=task_buffer.shape, dtype=task_buffer.dtype),
                TensorStructInfo(shape=head_buffer.shape, dtype=head_buffer.dtype),
                TensorStructInfo(shape=tail_buffer.shape, dtype=tail_buffer.dtype),
            ]
            packed_exec_queue = self.builder_.emit(
                relax.call_tir(gen_exec_queue_gvar, [], out_sinfo=output_sinfo, tir_vars=sym_vars)
            )
            new_relax_params.extend(
                [packed_exec_queue[0], packed_exec_queue[1], packed_exec_queue[2]]
            )
        if self.profiler_on:
            out_sinfo.append(TensorStructInfo(shape=(self.profiler_buf_size,), dtype="uint64"))
            inplace_indices.append(-1)
        if all(i == -1 for i in inplace_indices):
            return relax.call_tir(
                new_gvar, new_relax_params, out_sinfo, tir_vars=persistent_kernel_sym_vars
            ), len(out_sinfo)
        else:
            return relax.call_tir_inplace(
                new_gvar,
                new_relax_params,
                inplace_indices,
                out_sinfo,
                tir_vars=persistent_kernel_sym_vars,
            ), len(out_sinfo)

    def visit_seq_expr_(self, op: relax.SeqExpr) -> Expr:
        op = super().visit_seq_expr_(op)
        ret_order = []

        def _handle_var(var: relax.Var):
            binding = self.builder_.lookup_binding(var)
            if isinstance(binding, relax.Call):
                self.ret_value_entries[binding].append(-1)
                ret_order.append((binding, -1))
            elif isinstance(binding, relax.TupleGetItem):
                call_binding = self.builder_.lookup_binding(binding.tuple_value)
                if isinstance(call_binding, relax.Call):
                    self.ret_value_entries[call_binding].append(binding.index)
                    ret_order.append((call_binding, binding.index))
                else:
                    raise ValueError(f"Unsupported binding: {call_binding}")
            elif isinstance(binding, relax.expr.DataflowVar):
                binding = self.builder_.lookup_binding(binding)
                if isinstance(binding, relax.Call):
                    self.ret_value_entries[binding].append(0)
                    ret_order.append((binding, -1))
                else:
                    raise ValueError(f"Unsupported binding: {binding}")
            else:
                raise ValueError(f"Unsupported binding: {binding}")

        if isinstance(op.body, relax.Var):
            _handle_var(op.body)
        elif isinstance(op.body, relax.Tuple):
            for field in op.body.fields:
                _handle_var(field)
        else:
            raise ValueError(f"Unsupported body: {op.body}")
        self.builder_._begin_binding_block()
        merge_call, ret_len = self.merge_function()
        call_megakernel = self.builder_.emit(merge_call)
        if isinstance(call_megakernel.struct_info, TensorStructInfo):
            ret = call_megakernel
        else:
            ret_vars = []
            for call, idx in ret_order:
                if idx == -1:
                    # all the tir_call results will be returned
                    new_ret_index_call = self.new_ret_index[call]
                    assert len(new_ret_index_call) == len(
                        self.device_func_infos[call].out_var_entries
                    )
                    if len(new_ret_index_call) == 1:
                        ret_vars.append(
                            self.builder_.emit(
                                relax.TupleGetItem(call_megakernel, new_ret_index_call[0])
                            )
                        )
                    else:
                        ret_tuple = relax.Tuple(
                            [
                                self.builder_.emit(
                                    relax.TupleGetItem(call_megakernel, new_ret_index_call[idx])
                                )
                                for idx in sorted(new_ret_index_call.keys())
                            ]
                        )
                        ret_vars.append(ret_tuple)
                else:
                    if idx in self.new_ret_index[call]:
                        ret_vars.append(
                            self.builder_.emit(
                                relax.TupleGetItem(call_megakernel, self.new_ret_index[call][idx])
                            )
                        )
                    else:
                        raise ValueError(f"Call {call} with index {idx} not found in new_ret_index")
            if self.profiler_on:
                ret_vars.append(
                    self.builder_.emit(relax.TupleGetItem(call_megakernel, ret_len - 1))
                )
            if len(ret_vars) == 1:
                ret = ret_vars[0]
            else:
                ret = relax.Tuple(ret_vars)
        new_block = self.builder_._end_block()
        return relax.SeqExpr([*op.blocks, new_block], ret)
