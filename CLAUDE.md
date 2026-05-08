<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TIR Project

A compiler for high-performance AI kernels, built on top of Apache TVM. Adds **TIRX** — an extended tensor IR with hardware-aware layout abstractions, hierarchical execution scopes, async primitives, and an operator dispatch framework. Targets NVIDIA GPUs (Ampere/Hopper/Blackwell) and AWS Trainium.

## Project Structure

The codebase is an Apache TVM fork. Custom additions are concentrated in these areas:

### Core IR Extensions (`python/tvm/tir/`, `src/tir/ir/`, `include/tvm/tir/`)
- **Layout system** — `layout.py`, `src/tir/ir/layout/`, `layout.h` — Axe Layout abstraction (TileLayout, SwizzleLayout, ComposeLayout, Iter, Axis)
- **Execution scopes** — `exec_scope.py`, `exec_scope.h`, `exec_scope.cc` — Hierarchical scope model (world > kernel > cluster > cta > warpgroup > warp > thread)
- **TIRX statements** — `tirx_stmt.h/.cc`, `tirx_op.h` — OpCall node for operator invocation
- **Device op codegen** — `python/tvm/tir/device_op_codegen/cuda/` — PTX/CUDA code generation for barriers, MMA, wgmma, TMA, nvshmem

### TIRX DSL & Compiler (`python/tvm/tirx/`)
- **Operator framework** — `operator/tile_primitive.py` — Base operator classes (Unary, Binary, Reduction, etc.)
- **Scope op dispatch** — `operator/tile_primitive_dispatch/cuda/`, `operator/tile_primitive_dispatch/trn/` — Target-specific kernel scheduling (GEMM, copy_async, reduction, etc.)
- **Device native codegen** — `operator/intrinsics/cuda/` — PTX/CUDA code generation for barriers, MMA, wgmma, TMA, nvshmem
- **Transforms** — `transform/` — Buffer allocation, event tensor legalization
- **Lang** — `lang/pipeline.py`, `lang/tile_scheduler.py`, `lang/warp_role.py` — Async pipeline management, tile scheduling, warp role dispatch
- **Bench** — `bench.py` — Benchmarking utilities (profiling, CUDA timing, Proton integration)

### Script Frontend (`python/tvm/script/`, `src/script/`)
- **TIRX DSL API** — `python/tvm/script/tirx.py`, `ir_builder/tir/tirx.py` — 40+ builtin ops (Tx.gemm_async, Tx.copy_async, etc.)
- **Printer/Parser** — `src/script/printer/tir/`, `src/script/ir_builder/tir/` — TIRX-aware printing and parsing

### Lowering Passes (`src/tir/transform/`)
- `lower_tirx.cc` — Entry point (composes TIRX passes and strips ExecScopeStmt at the end)
- `lower_tirx_scope_ids.cc` — Scope ID resolution
- `lower_tirx_scope_slices.cc` — Execution scope slicing
- `lower_tirx_dispatch_ops.cc` — Op dispatch lowering
- `lower_tirx_dedup_tensormap.cc` — TensorMap deduplication
- `lower_tirx_cleanup.cc` — Post-lowering optimization
- `lower_tirx_opaque.cc` — Opaque block handling

### Tests (`tests/python/tirx/`)
- Core: `test_layout.py`, `test_parser_printer.py`, `test_verifier.py`, `test_exec_scope.py`
- Codegen: `codegen/test_codegen_cuda.py`, `test_codegen_hopper.py`, `test_codegen_blackwell.py`, `test_codegen_nki.py`
- Transforms: `transform/test_transform_lower_tirx.py`, `test_stmt_functor.py`, `test_expr_functor.py`
- Kernels: `kernels/cuda/sm80/`, `sm90a/`, `sm100a/`; `kernels/trn/`

### Other
- `kernels/sm100/` — Standalone SM100 GEMM kernels (fp16, fp8, nvfp4)
- `tirx_doc/` — Tutorials (e.g., `gemm_tutorial.ipynb`)
- `python/tvm/tir/pipeline.py` — Compilation pipelines: `default_tir_pipeline()`, `tirx_pipeline()`, `trn_pipeline()`

## Axe Layout

Axe Layout is the hardware-aware layout abstraction in TIR/TIRX that maps logical tensor coordinates to a multi-axis physical space via named axes. When working with `TLayout`, `TileLayout`, `SwizzleLayout`, `ComposeLayout`, `Iter`, `Axis`, or layout operations (tile, slice, canonicalize, group, direct_sum, apply, is_tile_inner/outer), use the **axe-layout** skill for detailed reference.

Key files:
- Python API: `python/tvm/tir/layout.py`
- C++ implementation: `src/tir/ir/layout/`
- Tests: `tests/python/tirx/test_layout.py`

## Building, Testing, and Benchmarking

Use the following skills (slash commands) for standard operations:

- `/tir-build` — Build TVM (initial or incremental)
- `/tir-test` — Run the full TIRX test suite. First pick an idle GPU, start a background monitor that flags foreign processes landing on it, then run pinned to that GPU:
  ```bash
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
  GPU_LOG="/tmp/tir_test_gpu_${CUDA_VISIBLE_DEVICES}.log"
  bash .claude/scripts/monitor_gpu.sh --gpu "$CUDA_VISIBLE_DEVICES" --interval 5 --log "$GPU_LOG" &
  MON_PID=$!
  trap 'kill $MON_PID 2>/dev/null' EXIT
  pytest tests/python/tirx/ -n 16
  kill $MON_PID 2>/dev/null
  grep -E 'FOREIGN USER|\[FOREIGN\]' "$GPU_LOG" || echo "no foreign GPU usage observed"
  ```
- `/tir-bench` — Run GEMM performance benchmarks (fp16, fp8, nvfp4)

**SM100a tests** require the [tirx-kernels](https://github.com/mlc-ai/tirx-kernels) repo (kernel definitions are maintained separately). Install it as an editable pip package:
```bash
git clone git@github.com:mlc-ai/tirx-kernels.git ~/tirx-kernels
pip install -e ~/tirx-kernels
```
SM100a kernel tests use the unified `tirx_kernels` package registry — the single `test_kernels.py` file discovers and runs all registered kernels via `tirx_kernels.registry.discover_kernels()`.

## Code Style

All formatting and linting is managed via `pre-commit` (see `.pre-commit-config.yaml`):

- **Python**: `ruff check --fix` (lint) + `ruff format` (format)
- **C++**: `clang-format`
- **Cython**: `cython-lint`

Run on staged files:
```bash
pre-commit run
```

Run on all files:
```bash
pre-commit run --all-files
```

## Commits and Branches

Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
<type>(<scope>): <short imperative description>

Types: feat, fix, refactor, perf, test, docs, chore, ci, build
Scopes: layout, kernel, op, op-dispatch, lower-tirx, tvmscript, infra
```

Default branch: **`tirx`** (not `main`). Always use `--base tirx` when creating PRs.

**NEVER force push to `tirx`** — no exceptions, no `--force`, no `--force-with-lease`. Always verify the current branch before any push.

Branch naming: `<type>/<kebab-case-description>`, e.g. `feat/direct-sum`, `fix/slice-swizzle`.

## Common Gotchas

- **C++ ↔ Python sync**: Changing IR nodes (layout, stmt, op) requires updating both the C++ side (`include/` + `src/`) and Python bindings (`python/tvm/tir/`). Don't forget FFI registration.
- **Printer ↔ Parser sync**: Modifications to IR printing (`src/script/printer/tir/`) must have corresponding parser changes (`python/tvm/script/parser/tir/`) and vice versa. Run `test_parser_printer.py` to verify round-trip.
- **New op checklist**: Adding a TIRX operator requires registration in `operator/tile_primitive.py`, a schedule in `operator/tile_primitive_dispatch/cuda/` (and/or `trn/`), a DSL entry in `ir_builder/tir/tirx.py`, and tests.
- Keep documentation in sync with code changes: when modifying code that is referenced in this document or in `.claude/skills/`, update the corresponding documentation immediately.
- **`Tx.meta_var` is not an accumulator**: `Tx.meta_var` creates a TIR expression, not a mutable variable. Writing `acc = Tx.meta_var(0); for i: acc = acc + x` does NOT accumulate — each reassignment creates a new expression and the compiler may optimize the loop away. For accumulation, write directly into a buffer: `buf[d] = buf[d] + x` (using `Tx.alloc_buffer` or SMEM buffers). Read-only scalars (`val = Tx.meta_var(expr)`) are fine.
- **Flaky tests**: If tests fail intermittently, first check `nvidia-smi` to see if the GPU is occupied by other workloads. Switch to an idle GPU before re-running.
