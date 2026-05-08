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

"""Implementation of cast/local_view dispatch variant for CUDA targets."""

import functools
import operator

from tvm.arith import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, BufferRegion, IntImm, PrimFunc
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.tile_primitive_dispatch import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import get_indices, get_st_extent
from ..layout_utils import (
    get_local_region,
    get_sublayout_from_region,
    layout_signature,
    resolve_thread_var,
    sig_equal,
)
from .utils import dtypex2_dict, vec2_cast_cuda_intrinsic_dict

_LOCAL_CASE_VIEW_FULL = "view_full"
_LOCAL_CASE_VIEW_SLICED = "view_sliced"
_LOCAL_CASE_THREAD_WISE = "thread_wise"


def _is_contiguous_region(analyzer, st, ext, shape):
    """Check if region [st:st+ext] maps to a contiguous block in row-major layout.

    Scans from innermost to outermost dimension. Inner dimensions must be fully
    covered (st=0, ext=shape) up to one "break" dimension (any st/ext). All
    dimensions outside the break must have unit extent.
    """
    found_break = False
    for i in reversed(range(len(st))):
        is_full = analyzer.can_prove_equal(st[i], 0) and analyzer.can_prove_equal(ext[i], shape[i])
        if found_break:
            if not analyzer.can_prove_equal(ext[i], 1):
                return False
        else:
            if not is_full:
                found_break = True
    return True


def _linear_offset(st, shape):
    """Compute row-major linear offset of position st in buffer of given shape."""
    offset = 0
    stride = 1
    for i in reversed(range(len(st))):
        offset = offset + st[i] * stride
        stride = stride * shape[i]
    return offset


def _is_offset_aligned_for_vec2(analyzer, offset):
    """Check if offset is provably even (required for vec2 alignment)."""
    if isinstance(offset, int):
        return offset % 2 == 0
    if isinstance(offset, IntImm):
        return int(offset.value) % 2 == 0
    return analyzer.can_prove_equal(offset % 2, 0)


def _has_nontrivial_layout(buf: Buffer) -> bool:
    layout = buf.layout
    return layout is not None and not layout.is_trivial()


def _classify_cast_local_case(
    dst_region: BufferRegion, src_region: BufferRegion, sctx: DispatchContext
) -> str | None:
    """Classify local cast implementation path."""

    def _full_region(buf_region: BufferRegion) -> bool:
        st, ext = get_st_extent(buf_region)
        analyzer = Analyzer()
        return all(
            analyzer.can_prove_equal(e, s) for e, s in zip(ext, buf_region.buffer.shape)
        ) and all(analyzer.can_prove_equal(s, 0) for s in st)

    if sctx.is_thread:
        return _LOCAL_CASE_THREAD_WISE
    if sctx.scope_kind in ["warp", "warpgroup", "cta", "cluster"]:
        if _full_region(dst_region) and _full_region(src_region):
            return _LOCAL_CASE_VIEW_FULL
        return _LOCAL_CASE_VIEW_SLICED
    return None


def validate_cast_local_view(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, sctx: DispatchContext
) -> bool:
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer

    if not (
        src.scope() == "local"
        and dst.scope() == "local"
        and sctx.is_cuda()
        and sctx.scope_kind in ["warp", "warpgroup", "cta", "cluster"]
    ):
        return False

    src_has_layout = _has_nontrivial_layout(src)
    dst_has_layout = _has_nontrivial_layout(dst)
    if not (src_has_layout or dst_has_layout):
        return False

    analyzer = Analyzer()
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    def _validate_layout_partition(layout, buf, st, ext) -> tuple[bool, tuple | None]:
        if layout.is_swizzle():
            return False, None
        if not isinstance(layout, TileLayout):
            return False, None
        if not getattr(layout, "shard", None):
            return False, None
        if not any(it.axis.is_thread() for it in layout.shard):
            return False, None
        for it in layout.shard:
            if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                return False, None
        replica = getattr(layout, "replica", None) or []
        if any(it.axis.is_thread() for it in replica):
            return False, None
        local_info = get_local_region(layout, list(buf.shape), st, ext)
        if local_info is None:
            return False, None
        return True, local_info

    if src_has_layout and dst_has_layout:
        src_region_extents = [r.extent for r in src_buffer_region.region]
        dst_region_extents = [r.extent for r in dst_buffer_region.region]
        if len(src_region_extents) != len(dst_region_extents):
            return False
        if not all(
            analyzer.can_prove_equal(s, d) for s, d in zip(src_region_extents, dst_region_extents)
        ):
            return False
        if src.layout.size() != dst.layout.size():
            return False

        src_ok, _ = _validate_layout_partition(src.layout, src, src_st, src_extent)
        if not src_ok:
            return False
        dst_ok, _ = _validate_layout_partition(dst.layout, dst, dst_st, dst_extent)
        if not dst_ok:
            return False

        # Slice -> Canonicalize
        src_sliced = get_sublayout_from_region(src.layout, src.shape, src_st, src_extent)
        dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)

        src_can = src_sliced.canonicalize() if hasattr(src_sliced, "canonicalize") else src_sliced
        dst_can = dst_sliced.canonicalize() if hasattr(dst_sliced, "canonicalize") else dst_sliced

        src_sig = layout_signature(src_can)
        dst_sig = layout_signature(dst_can)

        # Here check the canonicalized layouts are semantically equal.
        if not sig_equal(analyzer, src_sig, dst_sig):
            return False

        # Resolve thread vars once; reuse for launch count check
        thread_vars_list = []
        thr_extents = []
        for it in src_sliced.shard:
            if it.axis.is_thread():
                var = resolve_thread_var(it.axis, sctx)
                if var is None:
                    return False
                thread_vars_list.append(var)
                thr_extents.append(it.extent)

        if thread_vars_list and "threadIdx.x" in sctx.launch_params:
            expected = functools.reduce(operator.mul, thr_extents, 1)
            actual = sctx.launch_params["threadIdx.x"].dom.extent
            if len(set(id(v) for v in thread_vars_list)) == 1:
                if thread_vars_list[0] is sctx.launch_params["threadIdx.x"].var:
                    if not analyzer.can_prove_equal(actual, expected):
                        return False
        return True

    if src_has_layout:
        src_ok, src_local_info = _validate_layout_partition(src.layout, src, src_st, src_extent)
        if not src_ok:
            return False
        assert src_local_info is not None
        _, _, src_local_ext = src_local_info
        src_local_total = functools.reduce(operator.mul, src_local_ext, 1)
        dst_total = functools.reduce(operator.mul, dst_extent, 1)
        return analyzer.can_prove_equal(src_local_total, dst_total)

    dst_ok, dst_local_info = _validate_layout_partition(dst.layout, dst, dst_st, dst_extent)
    if not dst_ok:
        return False
    assert dst_local_info is not None
    _, _, dst_local_ext = dst_local_info
    dst_local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    src_total = functools.reduce(operator.mul, src_extent, 1)
    return analyzer.can_prove_equal(dst_local_total, src_total)


def _emit_cast_local_view_full(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, dst_local_info
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    _, _, dst_local_ext = dst_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    # Decide vec2 availability (offset is 0 for full region, so alignment is guaranteed)
    use_vec2 = (src.dtype, dst.dtype) in vec2_cast_cuda_intrinsic_dict
    local_total_imm = int(local_total.value) if isinstance(local_total, IntImm) else None

    # Prepare vec2 intrinsic codegen if needed
    if use_vec2:
        intrinsic_name = vec2_cast_cuda_intrinsic_dict[(src.dtype, dst.dtype)]
        src_dtypex2 = dtypex2_dict[src.dtype]
        dst_dtypex2 = dtypex2_dict[dst.dtype]
        func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""
    else:
        func_name = None
        source_code = None

    # fmt: off
    @Tx.prim_func(tirx=True, check_well_formed=False)
    def impl():
        with Tx.thread():
            base_src = Tx.decl_buffer((local_total,), src.dtype, src.data, scope=src.scope())
            base_dst = Tx.decl_buffer((local_total,), dst.dtype, dst.data, scope=dst.scope())

            if use_vec2 and (local_total_imm is not None):
                # vec2 loop + tail scalar (odd case)
                n2 = Tx.meta_var(local_total_imm // 2)
                tail = Tx.meta_var(local_total_imm % 2)

                for s in Tx.serial(0, n2):
                    local_idx = Tx.meta_var(s * 2)
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(base_dst[local_idx]),
                        Tx.address_of(base_src[local_idx]),
                        source_code=source_code,
                    )

                if tail == 1:
                    local_idx = Tx.meta_var(n2 * 2)
                    base_dst[local_idx] = Tx.cast(base_src[local_idx], dst.dtype)
            else:
                for s in Tx.serial(0, local_total):
                    local_idx = Tx.meta_var(s)
                    base_dst[local_idx] = Tx.cast(base_src[local_idx], dst.dtype)
    # fmt: on

    return impl


def _emit_cast_local_view_sliced(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, dst_local_info, src_local_info
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    src_local_shape, src_local_st, src_local_ext = src_local_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    # Decide vec2 availability
    use_vec2 = (src.dtype, dst.dtype) in vec2_cast_cuda_intrinsic_dict
    local_total_imm = int(local_total.value) if isinstance(local_total, IntImm) else None

    if use_vec2 and local_total_imm is not None:
        analyzer = Analyzer()
        src_cont = _is_contiguous_region(analyzer, src_local_st, src_local_ext, src_local_shape)
        dst_cont = _is_contiguous_region(analyzer, dst_local_st, dst_local_ext, dst_local_shape)
        if src_cont and dst_cont:
            src_off = _linear_offset(src_local_st, src_local_shape)
            dst_off = _linear_offset(dst_local_st, dst_local_shape)
            if not (
                _is_offset_aligned_for_vec2(analyzer, src_off)
                and _is_offset_aligned_for_vec2(analyzer, dst_off)
            ):
                use_vec2 = False
        else:
            use_vec2 = False
    else:
        use_vec2 = False

    if use_vec2:
        intrinsic_name = vec2_cast_cuda_intrinsic_dict[(src.dtype, dst.dtype)]
        src_dtypex2 = dtypex2_dict[src.dtype]
        dst_dtypex2 = dtypex2_dict[dst.dtype]
        func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""
        src_full_size = functools.reduce(operator.mul, src_local_shape, 1)
        dst_full_size = functools.reduce(operator.mul, dst_local_shape, 1)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                base_src = Tx.decl_buffer((src_full_size,), src.dtype, src.data, scope=src.scope())
                base_dst = Tx.decl_buffer((dst_full_size,), dst.dtype, dst.data, scope=dst.scope())

                n2 = Tx.meta_var(local_total_imm // 2)
                tail = Tx.meta_var(local_total_imm % 2)

                for s in Tx.serial(0, n2):
                    src_idx = Tx.meta_var(src_off + s * 2)
                    dst_idx = Tx.meta_var(dst_off + s * 2)
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(base_dst[dst_idx]),
                        Tx.address_of(base_src[src_idx]),
                        source_code=source_code,
                    )

                if tail == 1:
                    src_idx = Tx.meta_var(src_off + n2 * 2)
                    dst_idx = Tx.meta_var(dst_off + n2 * 2)
                    base_dst[dst_idx] = Tx.cast(base_src[src_idx], dst.dtype)
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                src_local = src.local(*src_local_shape)
                dst_local = dst.local(*dst_local_shape)
                for s in Tx.serial(0, local_total):
                    fused = Tx.meta_var(s)
                    dst_indices = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                    src_indices = Tx.meta_var(get_indices(fused, src_local_st, src_local_ext))
                    casted = Tx.cast(src_local[tuple(src_indices)], dst.dtype)
                    dst_local[tuple(dst_indices)] = casted
        # fmt: on

    return impl


def _emit_cast_local_view_src_layout_to_flat(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, src_local_info
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    src_local_shape, src_local_st, src_local_ext = src_local_info
    dst_st, dst_extent = get_st_extent(dst_buffer_region)
    local_total = functools.reduce(operator.mul, src_local_ext, 1)

    # Decide vec2 availability
    use_vec2 = (src.dtype, dst.dtype) in vec2_cast_cuda_intrinsic_dict
    local_total_imm = int(local_total.value) if isinstance(local_total, IntImm) else None

    if use_vec2 and local_total_imm is not None:
        analyzer = Analyzer()
        src_cont = _is_contiguous_region(analyzer, src_local_st, src_local_ext, src_local_shape)
        dst_cont = _is_contiguous_region(analyzer, dst_st, dst_extent, list(dst.shape))
        if src_cont and dst_cont:
            src_off = _linear_offset(src_local_st, src_local_shape)
            dst_off = _linear_offset(dst_st, list(dst.shape))
            if not (
                _is_offset_aligned_for_vec2(analyzer, src_off)
                and _is_offset_aligned_for_vec2(analyzer, dst_off)
            ):
                use_vec2 = False
        else:
            use_vec2 = False
    else:
        use_vec2 = False

    if use_vec2:
        intrinsic_name = vec2_cast_cuda_intrinsic_dict[(src.dtype, dst.dtype)]
        src_dtypex2 = dtypex2_dict[src.dtype]
        dst_dtypex2 = dtypex2_dict[dst.dtype]
        func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""
        src_full_size = functools.reduce(operator.mul, src_local_shape, 1)
        dst_full_size = functools.reduce(operator.mul, list(dst.shape), 1)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                base_src = Tx.decl_buffer((src_full_size,), src.dtype, src.data, scope=src.scope())
                base_dst = Tx.decl_buffer((dst_full_size,), dst.dtype, dst.data, scope=dst.scope())

                n2 = Tx.meta_var(local_total_imm // 2)
                tail = Tx.meta_var(local_total_imm % 2)

                for s in Tx.serial(0, n2):
                    src_idx = Tx.meta_var(src_off + s * 2)
                    dst_idx = Tx.meta_var(dst_off + s * 2)
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(base_dst[dst_idx]),
                        Tx.address_of(base_src[src_idx]),
                        source_code=source_code,
                    )

                if tail == 1:
                    src_idx = Tx.meta_var(src_off + n2 * 2)
                    dst_idx = Tx.meta_var(dst_off + n2 * 2)
                    base_dst[dst_idx] = Tx.cast(base_src[src_idx], dst.dtype)
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                src_local = src.local(*src_local_shape)
                for s in Tx.serial(0, local_total):
                    fused = Tx.meta_var(s)
                    src_indices = Tx.meta_var(get_indices(fused, src_local_st, src_local_ext))
                    dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    dst[tuple(dst_indices)] = Tx.cast(src_local[tuple(src_indices)], dst.dtype)
        # fmt: on

    return impl


def _emit_cast_local_view_flat_to_dst_layout(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, dst_local_info
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
    src_st, src_extent = get_st_extent(src_buffer_region)
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    # Decide vec2 availability
    use_vec2 = (src.dtype, dst.dtype) in vec2_cast_cuda_intrinsic_dict
    local_total_imm = int(local_total.value) if isinstance(local_total, IntImm) else None

    if use_vec2 and local_total_imm is not None:
        analyzer = Analyzer()
        src_cont = _is_contiguous_region(analyzer, src_st, src_extent, list(src.shape))
        dst_cont = _is_contiguous_region(analyzer, dst_local_st, dst_local_ext, dst_local_shape)
        if src_cont and dst_cont:
            src_off = _linear_offset(src_st, list(src.shape))
            dst_off = _linear_offset(dst_local_st, dst_local_shape)
            if not (
                _is_offset_aligned_for_vec2(analyzer, src_off)
                and _is_offset_aligned_for_vec2(analyzer, dst_off)
            ):
                use_vec2 = False
        else:
            use_vec2 = False
    else:
        use_vec2 = False

    if use_vec2:
        intrinsic_name = vec2_cast_cuda_intrinsic_dict[(src.dtype, dst.dtype)]
        src_dtypex2 = dtypex2_dict[src.dtype]
        dst_dtypex2 = dtypex2_dict[dst.dtype]
        func_name = f"tvm_builtin_cast_{src.dtype}x2_{dst.dtype}x2"
        source_code = f"""
__forceinline__ __device__ void {func_name}(void* dst, void* src) {{
    (({dst_dtypex2}*)dst)[0] = {intrinsic_name}((({src_dtypex2}*)src)[0]);
}}
"""
        src_full_size = functools.reduce(operator.mul, list(src.shape), 1)
        dst_full_size = functools.reduce(operator.mul, dst_local_shape, 1)

        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                base_src = Tx.decl_buffer((src_full_size,), src.dtype, src.data, scope=src.scope())
                base_dst = Tx.decl_buffer((dst_full_size,), dst.dtype, dst.data, scope=dst.scope())

                n2 = Tx.meta_var(local_total_imm // 2)
                tail = Tx.meta_var(local_total_imm % 2)

                for s in Tx.serial(0, n2):
                    src_idx = Tx.meta_var(src_off + s * 2)
                    dst_idx = Tx.meta_var(dst_off + s * 2)
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(base_dst[dst_idx]),
                        Tx.address_of(base_src[src_idx]),
                        source_code=source_code,
                    )

                if tail == 1:
                    src_idx = Tx.meta_var(src_off + n2 * 2)
                    dst_idx = Tx.meta_var(dst_off + n2 * 2)
                    base_dst[dst_idx] = Tx.cast(base_src[src_idx], dst.dtype)
        # fmt: on
    else:
        # fmt: off
        @Tx.prim_func(tirx=True, check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                for s in Tx.serial(0, local_total):
                    fused = Tx.meta_var(s)
                    src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                    dst_indices = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                    dst_local[tuple(dst_indices)] = Tx.cast(src[tuple(src_indices)], dst.dtype)
        # fmt: on

    return impl


def cast_local_view_impl(
    dst_buffer_region: BufferRegion, src_buffer_region: BufferRegion, sctx: DispatchContext
) -> PrimFunc | None:
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if sctx.scope_kind not in ["warp", "warpgroup", "cta", "cluster"]:
        fail(f"unsupported exec_scope {sctx.scope_kind} for local-view cast")

    src_has_layout = _has_nontrivial_layout(src)
    dst_has_layout = _has_nontrivial_layout(dst)

    if src_has_layout and dst_has_layout:
        local_case = _classify_cast_local_case(dst_buffer_region, src_buffer_region, sctx)

        dst_st, dst_extent = get_st_extent(dst_buffer_region)
        dst_local_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
        if not dst_local_info:
            fail("dst layout is not supported for local-view cast")

        if local_case == _LOCAL_CASE_VIEW_FULL:
            return _emit_cast_local_view_full(dst_buffer_region, src_buffer_region, dst_local_info)

        # view_sliced
        src_st, src_extent = get_st_extent(src_buffer_region)
        src_local_info = get_local_region(src.layout, list(src.shape), src_st, src_extent)
        if not src_local_info:
            fail("src layout is not supported for local-view cast")

        return _emit_cast_local_view_sliced(
            dst_buffer_region, src_buffer_region, dst_local_info, src_local_info
        )

    if src_has_layout:
        src_st, src_extent = get_st_extent(src_buffer_region)
        src_local_info = get_local_region(src.layout, list(src.shape), src_st, src_extent)
        if not src_local_info:
            fail("src layout is not supported for local-view cast")
        return _emit_cast_local_view_src_layout_to_flat(
            dst_buffer_region, src_buffer_region, src_local_info
        )

    if dst_has_layout:
        dst_st, dst_extent = get_st_extent(dst_buffer_region)
        dst_local_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
        if not dst_local_info:
            fail("dst layout is not supported for local-view cast")
        return _emit_cast_local_view_flat_to_dst_layout(
            dst_buffer_region, src_buffer_region, dst_local_info
        )

    fail("at least one local buffer must carry layout for local-view cast")


# === Variant: cast/local_view (priority=15) ===
#
# When: both dst and src are local-scope TileLayout buffers with matching
# canonical layout signatures, at warp/warpgroup/cta/cluster scope.
# Higher priority than thread_wise — preferred when layout partition is valid.
#
# Before (TilePrimitiveCall):
#     with Tx.warp():
#         Tx.cast(dst_view[0:16, 0:128], src_view[0:16, 0:128])
#         # src: local float16, dst: local float32, both WGMMA layout
#
# After — view_full path (local_total=64, fp16→fp32 uses vec2 intrinsic):
#     with Tx.thread():
#         base_dst = Tx.decl_buffer((64,), "float32", dst.data, scope="local")
#         base_src = Tx.decl_buffer((64,), "float16", src.data, scope="local")
#         for s in Tx.serial(64 // 2):
#             # __half22float2: converts 2 fp16 values to 2 fp32 at once
#             Tx.cuda.func_call("__half22float2", &base_dst[s*2], &base_src[s*2])
#         # If dtype pair has no vec2 intrinsic, falls back to:
#         # base_dst[idx] = Tx.cast(base_src[idx], "float32")
#
# After — view_sliced path (partial region, e.g. [4:12, 0:128]):
#     with Tx.thread():
#         src_local = src.local(*src_local_shape)
#         dst_local = dst.local(*dst_local_shape)
#         for s in Tx.serial(local_total):
#             dst_indices = get_indices(s, dst_local_st, dst_local_ext)
#             src_indices = get_indices(s, src_local_st, src_local_ext)
#             dst_local[dst_indices] = Tx.cast(src_local[src_indices], dst.dtype)
#
@register_dispatch(
    "cast",
    "cuda",
    variant="local_view",
    priority=15,
    when=[
        predicate(
            "validate_cast_local_view",
            lambda op, sctx: (
                validate_cast_local_view(
                    TilePrimitiveCall.downcast(op).output,
                    TilePrimitiveCall.downcast(op).input,
                    sctx,
                ),
                "validate_cast_local_view failed",
            ),
        ),
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.scope_kind in ["warp", "warpgroup", "cta", "cluster"],
                f"unsupported exec_scope {sctx.scope_kind} for local cast",
            ),
        ),
    ],
)
def cast_dispatch_local_view(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    return cast_local_view_impl(op_call.output, op_call.input, sctx)
