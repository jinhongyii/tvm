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

"""Implementation of FMA operator dispatch for CUDA targets.

Registered op: fma (fused multiply-add: output = input * scale + bias).
Dispatch variant: "local" (priority=10).

Matching conditions:
- output and input are local-scope buffers
- exec_scope in {thread, warp, warpgroup}
- scale and bias can each be BufferRegion or PrimExpr scalar

Codegen paths:
- packed_f32x2: dtype=float32, even element count, SM100+ → Tx.ptx.fma_packed_f32x2
- scalar fallback: per-element loop with output[i] = input[i] * scale + bias
"""

import functools
import operator
import re

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimExpr, PrimFunc, TilePrimitiveCall
from tvm.tirx.expr import FloatImm
from tvm.tirx.operator.tile_primitive_dispatch import DispatchContext, fail
from tvm.tirx.operator.tile_primitive_dispatch.dispatcher import predicate

from ..common import get_indices, get_st_extent
from ..layout_utils import get_local_region


def _validate_fma(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if len(op_call.args) != 4:
        return False, f"fma expects 4 args, got {len(op_call.args)}"
    output, inp, scale, bias = op_call.args
    if not isinstance(output, BufferRegion):
        return False, "fma output must be BufferRegion"
    if not isinstance(inp, BufferRegion):
        return False, "fma input must be BufferRegion"
    if not isinstance(scale, BufferRegion | FloatImm | PrimExpr):
        return False, f"fma scale must be BufferRegion or PrimExpr, got {type(scale)}"
    if not isinstance(bias, BufferRegion | FloatImm | PrimExpr):
        return False, f"fma bias must be BufferRegion or PrimExpr, got {type(bias)}"
    return True, None


def _match_local_scope(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    output = op_call.args[0]
    inp = op_call.args[1]
    if output.buffer.scope() != "local":
        return False, f"fma output scope {output.buffer.scope()} != local"
    if inp.buffer.scope() != "local":
        return False, f"fma input scope {inp.buffer.scope()} != local"
    scale = op_call.args[2]
    if isinstance(scale, BufferRegion) and scale.buffer.scope() != "local":
        return False, f"fma scale scope {scale.buffer.scope()} != local"
    bias = op_call.args[3]
    if isinstance(bias, BufferRegion) and bias.buffer.scope() != "local":
        return False, f"fma bias scope {bias.buffer.scope()} != local"
    return True, None


def _validate_fma_local(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    scope = sctx.scope_kind
    if scope not in ["cta", "warpgroup", "warp", "thread"]:
        return False, f"unsupported exec_scope {scope} for fma"
    if scope == "thread":
        return True, None

    output, inp, scale, bias = op_call.args
    if output.buffer.layout is None or inp.buffer.layout is None:
        return False, "layouts are required for sub-CTA local fma"

    out_st, out_extent = get_st_extent(output)
    inp_st, inp_extent = get_st_extent(inp)
    out_local_info = get_local_region(
        output.buffer.layout, list(output.buffer.shape), out_st, out_extent
    )
    inp_local_info = get_local_region(inp.buffer.layout, list(inp.buffer.shape), inp_st, inp_extent)
    if out_local_info is None or inp_local_info is None:
        return False, "output/input layout is not supported for sub-CTA local fma"

    analyzer = Analyzer()
    out_total = functools.reduce(operator.mul, out_local_info[2], 1)
    inp_total = functools.reduce(operator.mul, inp_local_info[2], 1)
    if not analyzer.can_prove_equal(out_total, inp_total):
        return False, "output/input local element count mismatch for sub-CTA local fma"

    if isinstance(scale, BufferRegion):
        if scale.buffer.layout is None:
            return False, "buffer scale requires layout for sub-CTA local fma"
        scale_st, scale_extent = get_st_extent(scale)
        scale_info = get_local_region(
            scale.buffer.layout, list(scale.buffer.shape), scale_st, scale_extent
        )
        if scale_info is None:
            return False, "scale layout is not supported for sub-CTA local fma"
        scale_total = functools.reduce(operator.mul, scale_info[2], 1)
        if not analyzer.can_prove_equal(scale_total, out_total):
            return False, "scale local element count mismatch for sub-CTA local fma"

    if isinstance(bias, BufferRegion):
        if bias.buffer.layout is None:
            return False, "buffer bias requires layout for sub-CTA local fma"
        bias_st, bias_extent = get_st_extent(bias)
        bias_info = get_local_region(
            bias.buffer.layout, list(bias.buffer.shape), bias_st, bias_extent
        )
        if bias_info is None:
            return False, "bias layout is not supported for sub-CTA local fma"
        bias_total = functools.reduce(operator.mul, bias_info[2], 1)
        if not analyzer.can_prove_equal(bias_total, out_total):
            return False, "bias local element count mismatch for sub-CTA local fma"

    return True, None


def _can_use_packed_f32x2(op_call, sctx):
    """Check whether we can use fma_packed_f32x2."""
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    if sm_version < 100:
        return False

    output = op_call.args[0]
    inp = op_call.args[1]
    if output.buffer.dtype != "float32" or inp.buffer.dtype != "float32":
        return False

    dst_extent = [r.extent for r in output.region]
    n_elements = functools.reduce(operator.mul, dst_extent, 1)
    if n_elements % 2 != 0:
        return False

    return True


def _fma_local_thread_impl(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    output_br, inp_br = op.args[0], op.args[1]
    scale_arg, bias_arg = op.args[2], op.args[3]
    output_buf = output_br.buffer
    inp_buf = inp_br.buffer

    dst_st, dst_extent = get_st_extent(output_br)
    inp_st, inp_extent = get_st_extent(inp_br)
    n_elements = functools.reduce(operator.mul, dst_extent, 1)

    rounding_mode = op.config.get("rounding_mode", "rz")

    # Try packed f32x2 path
    if _can_use_packed_f32x2(op, sctx):
        scale_is_buf = isinstance(scale_arg, BufferRegion)
        bias_is_buf = isinstance(bias_arg, BufferRegion)

        if scale_is_buf:
            scale_buf = scale_arg.buffer
            scale_st, scale_extent = get_st_extent(scale_arg)
        if bias_is_buf:
            bias_buf = bias_arg.buffer
            bias_st, bias_extent = get_st_extent(bias_arg)

        if not scale_is_buf and not bias_is_buf:
            # scalar scale + scalar bias
            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                for s in Tx.serial(0, n_elements // 2):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_extent))
                    inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_st, inp_extent))
                    inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_st, inp_extent))
                    Tx.ptx.fma_packed_f32x2(
                        inp_buf[tuple(inp_idx_1)],
                        inp_buf[tuple(inp_idx_2)],
                        scale_arg,
                        scale_arg,
                        bias_arg,
                        bias_arg,
                        Tx.address_of(output_buf[tuple(dst_idx)]),
                        rounding_mode=rounding_mode,
                    )

            return impl

        elif scale_is_buf and not bias_is_buf:
            # buffer scale + scalar bias
            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                for s in Tx.serial(0, n_elements // 2):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_extent))
                    inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_st, inp_extent))
                    inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_st, inp_extent))
                    sc_idx_1 = Tx.meta_var(get_indices(2 * s, scale_st, scale_extent))
                    sc_idx_2 = Tx.meta_var(get_indices(2 * s + 1, scale_st, scale_extent))
                    Tx.ptx.fma_packed_f32x2(
                        inp_buf[tuple(inp_idx_1)],
                        inp_buf[tuple(inp_idx_2)],
                        scale_buf[tuple(sc_idx_1)],
                        scale_buf[tuple(sc_idx_2)],
                        bias_arg,
                        bias_arg,
                        Tx.address_of(output_buf[tuple(dst_idx)]),
                        rounding_mode=rounding_mode,
                    )

            return impl

        elif not scale_is_buf and bias_is_buf:
            # scalar scale + buffer bias
            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                for s in Tx.serial(0, n_elements // 2):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_extent))
                    inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_st, inp_extent))
                    inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_st, inp_extent))
                    bi_idx_1 = Tx.meta_var(get_indices(2 * s, bias_st, bias_extent))
                    bi_idx_2 = Tx.meta_var(get_indices(2 * s + 1, bias_st, bias_extent))
                    Tx.ptx.fma_packed_f32x2(
                        inp_buf[tuple(inp_idx_1)],
                        inp_buf[tuple(inp_idx_2)],
                        scale_arg,
                        scale_arg,
                        bias_buf[tuple(bi_idx_1)],
                        bias_buf[tuple(bi_idx_2)],
                        Tx.address_of(output_buf[tuple(dst_idx)]),
                        rounding_mode=rounding_mode,
                    )

            return impl

        else:
            # buffer scale + buffer bias
            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl():
                for s in Tx.serial(0, n_elements // 2):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_extent))
                    inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_st, inp_extent))
                    inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_st, inp_extent))
                    sc_idx_1 = Tx.meta_var(get_indices(2 * s, scale_st, scale_extent))
                    sc_idx_2 = Tx.meta_var(get_indices(2 * s + 1, scale_st, scale_extent))
                    bi_idx_1 = Tx.meta_var(get_indices(2 * s, bias_st, bias_extent))
                    bi_idx_2 = Tx.meta_var(get_indices(2 * s + 1, bias_st, bias_extent))
                    Tx.ptx.fma_packed_f32x2(
                        inp_buf[tuple(inp_idx_1)],
                        inp_buf[tuple(inp_idx_2)],
                        scale_buf[tuple(sc_idx_1)],
                        scale_buf[tuple(sc_idx_2)],
                        bias_buf[tuple(bi_idx_1)],
                        bias_buf[tuple(bi_idx_2)],
                        Tx.address_of(output_buf[tuple(dst_idx)]),
                        rounding_mode=rounding_mode,
                    )

            return impl

    # Scalar fallback: per-element loop
    vec_len = op.config.get("vec_len", 1)

    scale_is_buf = isinstance(scale_arg, BufferRegion)
    bias_is_buf = isinstance(bias_arg, BufferRegion)

    if not scale_is_buf and not bias_is_buf:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for s in Tx.serial(0, n_elements // vec_len):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len + vec)
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    inp_idx = Tx.meta_var(get_indices(fused, inp_st, inp_extent))
                    output_buf[tuple(dst_idx)] = inp_buf[tuple(inp_idx)] * scale_arg + bias_arg

    elif scale_is_buf and not bias_is_buf:
        scale_buf = scale_arg.buffer
        scale_st, scale_extent = get_st_extent(scale_arg)

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for s in Tx.serial(0, n_elements // vec_len):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len + vec)
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    inp_idx = Tx.meta_var(get_indices(fused, inp_st, inp_extent))
                    sc_idx = Tx.meta_var(get_indices(fused, scale_st, scale_extent))
                    output_buf[tuple(dst_idx)] = (
                        inp_buf[tuple(inp_idx)] * scale_buf[tuple(sc_idx)] + bias_arg
                    )

    elif not scale_is_buf and bias_is_buf:
        bias_buf = bias_arg.buffer
        bias_st, bias_extent = get_st_extent(bias_arg)

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for s in Tx.serial(0, n_elements // vec_len):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len + vec)
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    inp_idx = Tx.meta_var(get_indices(fused, inp_st, inp_extent))
                    bi_idx = Tx.meta_var(get_indices(fused, bias_st, bias_extent))
                    output_buf[tuple(dst_idx)] = (
                        inp_buf[tuple(inp_idx)] * scale_arg + bias_buf[tuple(bi_idx)]
                    )

    else:
        scale_buf = scale_arg.buffer
        scale_st, scale_extent = get_st_extent(scale_arg)
        bias_buf = bias_arg.buffer
        bias_st, bias_extent = get_st_extent(bias_arg)

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            for s in Tx.serial(0, n_elements // vec_len):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len + vec)
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    inp_idx = Tx.meta_var(get_indices(fused, inp_st, inp_extent))
                    sc_idx = Tx.meta_var(get_indices(fused, scale_st, scale_extent))
                    bi_idx = Tx.meta_var(get_indices(fused, bias_st, bias_extent))
                    output_buf[tuple(dst_idx)] = (
                        inp_buf[tuple(inp_idx)] * scale_buf[tuple(sc_idx)] + bias_buf[tuple(bi_idx)]
                    )

    return impl


def _local_view_can_use_packed_f32x2(sctx, output_buf, inp_buf, scale_arg, bias_arg, local_total):
    """Per-thread local view eligibility check for packed f32x2 fma."""
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    if sm_version < 100:
        return False
    if output_buf.dtype != "float32" or inp_buf.dtype != "float32":
        return False
    for arg in (scale_arg, bias_arg):
        if isinstance(arg, BufferRegion):
            if arg.buffer.dtype != "float32":
                return False
        else:
            if arg.dtype != "float32":
                return False
    return Analyzer().can_prove_equal(local_total % 2, 0)


def _fma_local_view_impl(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    output_br, inp_br = op.args[0], op.args[1]
    scale_arg, bias_arg = op.args[2], op.args[3]
    output_buf = output_br.buffer
    inp_buf = inp_br.buffer

    out_st, out_extent = get_st_extent(output_br)
    inp_st, inp_extent = get_st_extent(inp_br)
    out_local_info = get_local_region(output_buf.layout, list(output_buf.shape), out_st, out_extent)
    inp_local_info = get_local_region(inp_buf.layout, list(inp_buf.shape), inp_st, inp_extent)
    if out_local_info is None or inp_local_info is None:
        fail("output/input layout is not supported for sub-CTA local fma")

    out_local_shape, out_local_st, out_local_ext = out_local_info
    inp_local_shape, inp_local_st, inp_local_ext = inp_local_info
    local_total = functools.reduce(operator.mul, out_local_ext, 1)
    vec_len = op.config.get("vec_len", 1)

    scale_is_buf = isinstance(scale_arg, BufferRegion)
    bias_is_buf = isinstance(bias_arg, BufferRegion)

    if scale_is_buf:
        scale_buf = scale_arg.buffer
        scale_st, scale_extent = get_st_extent(scale_arg)
        scale_local_info = get_local_region(
            scale_buf.layout, list(scale_buf.shape), scale_st, scale_extent
        )
        if scale_local_info is None:
            fail("scale layout is not supported for sub-CTA local fma")
        scale_local_shape, scale_local_st, scale_local_ext = scale_local_info

    if bias_is_buf:
        bias_buf = bias_arg.buffer
        bias_st, bias_extent = get_st_extent(bias_arg)
        bias_local_info = get_local_region(
            bias_buf.layout, list(bias_buf.shape), bias_st, bias_extent
        )
        if bias_local_info is None:
            fail("bias layout is not supported for sub-CTA local fma")
        bias_local_shape, bias_local_st, bias_local_ext = bias_local_info

    use_packed = _local_view_can_use_packed_f32x2(
        sctx, output_buf, inp_buf, scale_arg, bias_arg, local_total
    )
    if use_packed:
        rounding_mode = op.config.get("rounding_mode", "rz")
        pair_count = local_total // 2

        if not scale_is_buf and not bias_is_buf:

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl_packed_ss():
                with Tx.thread():
                    output_local = output_buf.local(*out_local_shape)
                    inp_local = inp_buf.local(*inp_local_shape)
                    for s in Tx.unroll(pair_count):
                        out_idx = Tx.meta_var(get_indices(2 * s, out_local_st, out_local_ext))
                        inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_local_st, inp_local_ext))
                        inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_local_st, inp_local_ext))
                        Tx.ptx.fma_packed_f32x2(
                            inp_local[tuple(inp_idx_1)],
                            inp_local[tuple(inp_idx_2)],
                            scale_arg,
                            scale_arg,
                            bias_arg,
                            bias_arg,
                            Tx.address_of(output_local[tuple(out_idx)]),
                            rounding_mode=rounding_mode,
                        )

            return impl_packed_ss

        if scale_is_buf and not bias_is_buf:

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl_packed_bs():
                with Tx.thread():
                    output_local = output_buf.local(*out_local_shape)
                    inp_local = inp_buf.local(*inp_local_shape)
                    scale_local = scale_buf.local(*scale_local_shape)
                    for s in Tx.unroll(pair_count):
                        out_idx = Tx.meta_var(get_indices(2 * s, out_local_st, out_local_ext))
                        inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_local_st, inp_local_ext))
                        inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_local_st, inp_local_ext))
                        sc_idx_1 = Tx.meta_var(get_indices(2 * s, scale_local_st, scale_local_ext))
                        sc_idx_2 = Tx.meta_var(
                            get_indices(2 * s + 1, scale_local_st, scale_local_ext)
                        )
                        Tx.ptx.fma_packed_f32x2(
                            inp_local[tuple(inp_idx_1)],
                            inp_local[tuple(inp_idx_2)],
                            scale_local[tuple(sc_idx_1)],
                            scale_local[tuple(sc_idx_2)],
                            bias_arg,
                            bias_arg,
                            Tx.address_of(output_local[tuple(out_idx)]),
                            rounding_mode=rounding_mode,
                        )

            return impl_packed_bs

        if not scale_is_buf and bias_is_buf:

            @Tx.prim_func(tirx=True, check_well_formed=False)
            def impl_packed_sb():
                with Tx.thread():
                    output_local = output_buf.local(*out_local_shape)
                    inp_local = inp_buf.local(*inp_local_shape)
                    bias_local = bias_buf.local(*bias_local_shape)
                    for s in Tx.unroll(pair_count):
                        out_idx = Tx.meta_var(get_indices(2 * s, out_local_st, out_local_ext))
                        inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_local_st, inp_local_ext))
                        inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_local_st, inp_local_ext))
                        bi_idx_1 = Tx.meta_var(get_indices(2 * s, bias_local_st, bias_local_ext))
                        bi_idx_2 = Tx.meta_var(
                            get_indices(2 * s + 1, bias_local_st, bias_local_ext)
                        )
                        Tx.ptx.fma_packed_f32x2(
                            inp_local[tuple(inp_idx_1)],
                            inp_local[tuple(inp_idx_2)],
                            scale_arg,
                            scale_arg,
                            bias_local[tuple(bi_idx_1)],
                            bias_local[tuple(bi_idx_2)],
                            Tx.address_of(output_local[tuple(out_idx)]),
                            rounding_mode=rounding_mode,
                        )

            return impl_packed_sb

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl_packed_bb():
            with Tx.thread():
                output_local = output_buf.local(*out_local_shape)
                inp_local = inp_buf.local(*inp_local_shape)
                scale_local = scale_buf.local(*scale_local_shape)
                bias_local = bias_buf.local(*bias_local_shape)
                for s in Tx.unroll(pair_count):
                    out_idx = Tx.meta_var(get_indices(2 * s, out_local_st, out_local_ext))
                    inp_idx_1 = Tx.meta_var(get_indices(2 * s, inp_local_st, inp_local_ext))
                    inp_idx_2 = Tx.meta_var(get_indices(2 * s + 1, inp_local_st, inp_local_ext))
                    sc_idx_1 = Tx.meta_var(get_indices(2 * s, scale_local_st, scale_local_ext))
                    sc_idx_2 = Tx.meta_var(get_indices(2 * s + 1, scale_local_st, scale_local_ext))
                    bi_idx_1 = Tx.meta_var(get_indices(2 * s, bias_local_st, bias_local_ext))
                    bi_idx_2 = Tx.meta_var(get_indices(2 * s + 1, bias_local_st, bias_local_ext))
                    Tx.ptx.fma_packed_f32x2(
                        inp_local[tuple(inp_idx_1)],
                        inp_local[tuple(inp_idx_2)],
                        scale_local[tuple(sc_idx_1)],
                        scale_local[tuple(sc_idx_2)],
                        bias_local[tuple(bi_idx_1)],
                        bias_local[tuple(bi_idx_2)],
                        Tx.address_of(output_local[tuple(out_idx)]),
                        rounding_mode=rounding_mode,
                    )

        return impl_packed_bb

    if not scale_is_buf and not bias_is_buf:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                output_local = output_buf.local(*out_local_shape)
                inp_local = inp_buf.local(*inp_local_shape)
                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            out_idx = Tx.meta_var(get_indices(fused, out_local_st, out_local_ext))
                            inp_idx = Tx.meta_var(get_indices(fused, inp_local_st, inp_local_ext))
                            output_local[tuple(out_idx)] = (
                                inp_local[tuple(inp_idx)] * scale_arg + bias_arg
                            )

    elif scale_is_buf and not bias_is_buf:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                output_local = output_buf.local(*out_local_shape)
                inp_local = inp_buf.local(*inp_local_shape)
                scale_local = scale_buf.local(*scale_local_shape)
                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            out_idx = Tx.meta_var(get_indices(fused, out_local_st, out_local_ext))
                            inp_idx = Tx.meta_var(get_indices(fused, inp_local_st, inp_local_ext))
                            sc_idx = Tx.meta_var(
                                get_indices(fused, scale_local_st, scale_local_ext)
                            )
                            output_local[tuple(out_idx)] = (
                                inp_local[tuple(inp_idx)] * scale_local[tuple(sc_idx)] + bias_arg
                            )

    elif not scale_is_buf and bias_is_buf:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                output_local = output_buf.local(*out_local_shape)
                inp_local = inp_buf.local(*inp_local_shape)
                bias_local = bias_buf.local(*bias_local_shape)
                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            out_idx = Tx.meta_var(get_indices(fused, out_local_st, out_local_ext))
                            inp_idx = Tx.meta_var(get_indices(fused, inp_local_st, inp_local_ext))
                            bi_idx = Tx.meta_var(get_indices(fused, bias_local_st, bias_local_ext))
                            output_local[tuple(out_idx)] = (
                                inp_local[tuple(inp_idx)] * scale_arg + bias_local[tuple(bi_idx)]
                            )

    else:

        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                output_local = output_buf.local(*out_local_shape)
                inp_local = inp_buf.local(*inp_local_shape)
                scale_local = scale_buf.local(*scale_local_shape)
                bias_local = bias_buf.local(*bias_local_shape)
                for s in Tx.serial(0, Tx.ceildiv(local_total, vec_len)):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        if fused < local_total:
                            out_idx = Tx.meta_var(get_indices(fused, out_local_st, out_local_ext))
                            inp_idx = Tx.meta_var(get_indices(fused, inp_local_st, inp_local_ext))
                            sc_idx = Tx.meta_var(
                                get_indices(fused, scale_local_st, scale_local_ext)
                            )
                            bi_idx = Tx.meta_var(get_indices(fused, bias_local_st, bias_local_ext))
                            output_local[tuple(out_idx)] = (
                                inp_local[tuple(inp_idx)] * scale_local[tuple(sc_idx)]
                                + bias_local[tuple(bi_idx)]
                            )

    return impl


def _fma_local_impl(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    if sctx.is_thread:
        return _fma_local_thread_impl(op, sctx)
    return _fma_local_view_impl(op, sctx)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
from tvm.tirx.operator.tile_primitive_dispatch import register_dispatch  # noqa: E402


@register_dispatch(
    "fma",
    "cuda",
    variant="local",
    priority=10,
    when=[
        predicate("validate_fma", _validate_fma),
        predicate("local_scope", _match_local_scope),
        predicate("local_valid", _validate_fma_local),
    ],
)
def _fma_local_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _fma_local_impl(op, sctx)
