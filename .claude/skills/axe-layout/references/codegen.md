# Axe Layout Code Generation Patterns

## Table of Contents

1. [TMA Asynchronous Copy (NVIDIA Hopper/Blackwell)](#1-tma-asynchronous-copy)
2. [AI Accelerator: Trainium TensorEngine GEMM](#2-trainium-tensorengine-gemm)
3. [Thread-Local Loop Transformation (CuTe style)](#3-thread-local-loop-transformation)
4. [CTA Collective Semantics (Triton style)](#4-cta-collective-semantics)
5. [Execution Scopes](#5-execution-scopes)
6. [Distributed Tensor](#6-distributed-tensor)

---

## 1. TMA Asynchronous Copy

TMA (Tensor Memory Access) allows specifying a multi-dimensional box region in global memory
to copy to shared memory. The Axe compiler dispatches async copy operators to TMA.

### Algorithm

Given global tensor G (layout L_G, shape E_G) and shared tensor S (layout L_S, shape E_S),
copying region R_G in G to R_S in S:

**Step 1: Slice view.** Derive `L_G[R_G:E_G]` and `L_S[R_S:E_S]`.

**Step 2: Determine shared-memory copy atom (with swizzle).**
A TMA atom given S with dtype d and swizzle mode `a in {32, 64, 128} bytes`:
- Atom logical shape E_{d,a} has innermost two dims `(8, a/sizeof(d))`, others `1`.
- Atom intra-box layout L_{d,a} is a hardware swizzle (SwizzleLayout).
- Require a tiler T such that: `(L_S)_{||E_S} = T_{||E_o} x (L_{d,a})_{||E_{d,a}}`
- Loop over iters of T to enumerate shared-memory atoms.

**Step 3: Craft CuTensorMap for global memory.**
Translate shared-memory atom shape E_{d,a} to global counterpart E_{d,a}^G.
After grouping `(L_G)_{||E_G}`, verify for each group i that `E_{d,a}^G(i)` is a suffix
product of iter extents in group i (or L_G admits a direct-sum decomposition).

### Key functions (copy_async.py)

- `tma_atom_layout()` — generate swizzle layout for TMA atom
- `tma_shared_layout()` — generate full shared memory layout with swizzle
- `compute_box_dim()` — extract TMA box dimensions from layout shards

### Implementation

`python/tvm/tirx/op_dispatch/cuda/copy_async.py` (~875 lines).

---

## 2. Trainium TensorEngine GEMM

The Trainium TensorEngine computes `C = A.T @ B` with strict layout constraints.

### Hardware constraints

- **Memory axes**: SBUF and PSUM are 2D memories with 128 partitions (P) and contiguous free (F).
- **Layout**: A[K,M] and B[K,N] must have K on P-axis, M/N on F-axis.
  C[M,N] has P from M and F from N.
- **Max tile**: 128x128x512 (MxNxK).

### Algorithm

**Step 1: Group.** Find S_M, S_N, S_K such that:
- `L_A' = (L_A)_{||(S_M, S_K)}` has iter extents exactly (S_M, S_K)
- `L_B' = (L_B)_{||(S_N, S_K)}` has iter extents exactly (S_N, S_K)
- `L_C' = (L_C)_{||(S_M, S_N)}` has iter extents exactly (S_M, S_N)

**Step 2: K Intersection.** For iter pairs `I_1 = (e_1, s_1@a)` and `I_2 = (e_2, s_2@a)`:
- Compute `I_1 ∩ I_2 = (e, s@a)` producing the exact intersection of values.
- Enumerate L_A' and L_B' iters; if both have axis P, append intersection.

**Step 3: MN Intersection.**
- M intersection: From L_A' and L_C', keep iters where L_A'[i] has axis F and L_C'[i] has axis P.
  Find index set I such that L_C'[I] canonicalizes to a single iter (as R set) and contains
  the smallest-stride iter of L_C'. Keep L_M^A = L_A'[I], L_M^C = L_C'[I].
- N intersection: Similar for L_B' and L_C', picking L_C'[i] with axis F.

**Step 4: Finalize.** The extents of L_M, L_N, L_K give the largest possible M, N, K instruction
shapes. Remaining iters generate loops over the instruction.

---

## 3. Thread-Local Loop Transformation

CuTe-style programming: define algorithm atoms and split work per thread.

### Example

Copy 16x64 region from global memory using 128 threads with 4-element vectorized loads:

```python
# Work partition layout over source tensor
work = TileLayout(shard=([16, 8, 2, 4], [128, 8, 4, 1]))  # all @m
# + offset 2112@m

# Bind first 2 loops to threadIdx.x: tx//8, tx%8
# Per-thread remaining loops:
per_thread = TileLayout(shard=([2, 4], [4, 1]))  # all @m
# + offset = (tx//8)*128 + (tx%8)*8 + 2112 @m
```

The inner iter `(4, 1@m)` corresponds to a single vectorized-load atom.
The outer iter `(2, 4@m)` iterates over atoms.

### Axe DSL representation

```python
# Define CTA-scope source tensor slice
C_slice = C[16:32, 64:128]  # shape [16, 64], scope=cta

# Thread-level binding: each thread gets its portion
# Layout tells the compiler how to derive per-thread addresses
```

---

## 4. CTA Collective Semantics

Triton-style programming: organize local registers as a CTA-collective tensor.

### Example

Same copy, but expressed collectively:

```python
# C_local is a CTA-collective register tensor
# Layout shows how 128*8 = 1024 elements are distributed across 128 threads
C_local = TileLayout(
    shard=([16, 8, 8],
           [8 @ Axis.tx, 1 @ Axis.tx, 1])  # 1 = 1@m (register)
)
# Semantics: C_local = C[16:32, 64:128]
```

The layout explicitly shows:
- First dim (16): 8 values sharded across tx, 2 per thread
- Second dim (64): 8 values on tx, 8 values in registers per thread

### Axe unifies both

Axe can represent both the CuTe thread-local view and the Triton collective view
in the same layout formalism. The compiler can generate code from either representation.

---

## 5. Execution Scopes

Axe DSL uses execution scopes to denote thread groups executing operators together:

- `kernel` — all threads in a kernel launch
- `cta` — thread block (cooperative thread array)
- `warpgroup` — group of warps
- `warp` — 32 threads
- `thread` — single thread

Scope slicing: within a CTA of 256 threads (8 warps), assign roles:
```python
# warps [0:3] = producers (async copy)
# warps [4:7] = consumers (compute)
producer_scope = cta.slice(warp=[0, 3])
consumer_scope = cta.slice(warp=[4, 7])
```

Operators are invoked relative to a scope — they execute per entity in that scope.

---

## 6. Distributed Tensor

Axe layout naturally expresses distributed tensor sharding:

```python
# DTensor with shape (4, 64, 64) sharded on first dim across 4 GPUs
# Each GPU holds (1, 64, 64)
input_layout = TileLayout(
    shard=([4, 64, 64], [1 @ gpuid, 64, 1])  # 64, 1 are on @m
)

# Output after reduce-scatter: (64, 64) sharded on first dim
output_layout = TileLayout(
    shard=([4, 16, 64], [1 @ gpuid, 64, 1])
)
```

The compiler generates runtime checks to ensure DTensor sharding matches declared layouts
and dispatches to appropriate communication primitives (all-gather, reduce-scatter, etc.)
implemented via NVSHMEM or similar.

For multi-GPU GEMM + Reduce-Scatter, Axe composes a distributed tensor, invokes the sum
operator, and the compiler dispatches to `multimem.ld_reduce` on B200 for fine-grained
communication-computation overlap within a single kernel.
