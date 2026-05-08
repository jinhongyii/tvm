# Axe Layout Algorithms — Detailed Reference

## Table of Contents

1. [Canonicalize](#1-canonicalize)
2. [Group-By-Shape](#2-group-by-shape)
3. [Tile (Kronecker Product)](#3-tile-kronecker-product)
4. [Tile-Of Check & C Recovery](#4-tile-of-check--c-recovery)
5. [Slice](#5-slice)
6. [Direct Sum](#6-direct-sum)

---

## 1. Canonicalize

Simplifies a layout to a unique canonical form without changing its semantics.

### D-part rewrite rules (applied repeatedly until fixpoint):

**D0 (remove unit extent):** If `e_i = 1`, delete the iter.

**D1 (merge adjacent same-axis):** If two consecutive iters target the same axis and
`s_i = e_{i+1} * s_{i+1}` (i.e., the outer stride equals inner extent * inner stride),
merge into `(e_i * e_{i+1}, s_{i+1}, a_i)`.

Example:
```
(4, 8@m), (8, 1@m) -> s_0=8 == e_1*s_1=8*1 -> merge to (32, 1@m)
```

### (O,R)-part rewrite rules (per axis):

**C0 (remove unit extent):** Remove any replica iter with `e = 1`.

**C1 (normalize sign):** If `s < 0`, replace with `s' = -s` and update `O += (e-1)*s @ axis`.

**C2 (absorb multiples):** If two replica iters on the same axis have `s_j = q * s_i`
where `1 <= q < e_i`, merge into `(e_i + q*(e_j - 1), s_i, axis)`.

### Uniqueness

Under the **gap condition (GC)** — that per-axis replica strides are well-separated:
`sigma_{k+1} > E_k * sigma_k` for consecutive stride levels — the canonical form is **unique**.
This means two layouts with the same induced function will have identical canonical representations.

### Implementation

C++ in `src/tir/ir/layout/tile_canonicalize.cc`.
Python: `layout.canonicalize()`.

---

## 2. Group-By-Shape

Refines a D-list so it can be partitioned into `r` consecutive blocks whose extent products
match a target shape `S = (S_0, ..., S_{r-1})`.

### Algorithm (GCD-driven)

```
Input: D = [(e_0, s_0, a_0), ...], S = [S_0, ..., S_{r-1}] with prod(e_k) = prod(S_i)
Output: refined D' with block boundaries

For each block i:
    T = S_i (target product), cur = 1
    While cur < T:
        (e, s, a) = current iter at position j
        rem = T / cur
        g = gcd(e, rem)
        If g == 1: FAIL
        Split iter: head = (g, e/g * s, a), tail = (e/g, s, a)
        Append head to D', cur *= g
        If tail extent > 1: replace current iter with tail
        Else: advance j
    Record block boundary
```

### Key properties

- The GCD choice is the **largest** admissible head factor at each step.
- Produces the **minimum** number of splits among all successful refinements.
- **Split rule**: `(e, s, a)` with `e = e1 * e2` can be split into `(e1, e2*s, a), (e2, s, a)`
  without changing the induced map.
- **Fuse rule** (inverse): consecutive `(e1, e2*s, a), (e2, s, a)` can be fused to `(e1*e2, s, a)`.

### Implementation

C++ in `src/tir/ir/layout/tile_tile_ops.cc` (the grouping is called as a subroutine).
Python: `tile_layout.group(shape)`.

---

## 3. Tile (Kronecker Product)

Composes two grouped layouts A (outer) and B (inner) into a tiled layout T:

```
f_T(x || y) = f_A(x) * span(f_B) + f_B(y)
```

where `*` is the axis-wise (Hadamard) product and `span` is taken per axis.

### Algorithm

1. **Group both inputs**: `A_{||S_A}` and `B_{||S_B}` into r blocks each.

2. **Compute scaling vector** (per-axis span of B):
   ```
   Sigma[a] = 1 + sum(|s_I|*(e_I-1) for I in D^B where axis=a)
              + sum(|s_J|*(e_J-1) for J in R^B where axis=a)
   ```

3. **Emit D^T**: For each rank position i=0..r-1:
   - Append all iters of B^A_i, **scaled**: `(e, Sigma[a]*s, a)` for each `(e, s, a)`.
   - Append all iters of B^B_i **as-is**.

4. **Emit R^T**: Scaled R^A union R^B:
   ```
   R^T = {(e, Sigma[a]*s, a) : (e,s,a) in R^A} union R^B
   ```

5. **Emit O^T**: `O^T = O^A * Sigma + O^B` (axis-wise).

The resulting D^T is grouped by the interleaved shape `(S_A[0], S_B[0], ..., S_A[r-1], S_B[r-1])`.

### Example

```
A = (2, 3):(3, 1)    B = (8, 8):(8, 1)
span(B) = 64

A x B = (2, 8, 3, 8):(192, 8, 64, 1)
```

This is a 16x24 matrix stored as a 2x3 grid of 8x8 tiles.

### Implementation

C++ in `src/tir/ir/layout/tile_tile_ops.cc`.
Python: `inner.tile(outer, outer_shape, inner_shape)` or `inner.tile_to(to_shape, current_shape)`.

---

## 4. Tile-Of Check & C Recovery

Given layouts A (with shape S_A) and B (with shape S_B), decide if A is a tile of B
(i.e., A = C x B for some C) and recover C.

### Algorithm

1. **Shape check**: S_B must divide S_A coordinate-wise. Set S_C[j] = S_A[j] / S_B[j].

2. **Group both**: A_{||S_A} and B_{||S_B}.

3. **Compute span** W = span(f_{B_{||S_B}}).

4. **For each rank position j**: Scan block A^(j) left-to-right:
   - If current iter matches next B-iter exactly: consume both (B-subsequence match).
   - Else: check if iter is W-scaled (W[a] divides s), descale to get C-iter.
   - If neither: FAIL.
   - After scanning: all B-iters must be consumed.

5. **Offset check**: `O_A = O_C * W + O_B` must hold axis-wise.

### Implementation

C++ in `src/tir/ir/layout/tile_tile_ops.cc`.
Python:
- `inner.is_tile_inner(tiled, tiled_shape, inner_shape)` → returns outer or None
- `outer.is_tile_outer(tiled, tiled_shape, outer_shape)` → returns inner or None

---

## 5. Slice

Extract a sub-layout for an axis-aligned rectangular region `R = [b_0:b_0+T_0) x ... x [b_{r-1}:b_{r-1}+T_{r-1})`
within shape S admitted by L.

### Algorithm (per canonicalized block)

1. **Compute offset**: O* = f_{L<S>}(b) (the address of the region origin).

2. **Compute start digits**: d_k^0 for each iter position.

3. **Greedy peeling** (from fastest digit rightward):
   - An iter j is **peelable** iff `d_j^0 = 0` and `E_j | T`.
   - Peel: append `(E_j, S_j@a_j)` to output, `T /= E_j`.
   - Continue until no more peelable iters.

4. **If T = 0**: Return peeled iters + O*. Done.

5. **Pivot** (rightmost unpeeled digit k). Two sufficient forms:

   **No-wrap**: If `d_k^0 + T <= E_k`, emit `(T, S_k@a_k)` before peeled iters.

   **Symmetric one-wrap**: If T is even, `d_k^0 + T/2 = E_k`, and (if k>0) `d_{k-1}^0 + 1 <= E_{k-1}`:
   ```
   c = T/2
   Delta = S_{k-1}@a_{k-1} - (E_k - c)*S_k@a_k   (drop S_{k-1} term if k=0)
   Emit: (2, Delta), (c, S_k@a_k), then peeled iters
   ```

6. **Otherwise**: FAIL (region not representable as single Axe layout).

### Example

```
L = (2, 8, 3, 8):(192, 8, 64, 1)@m
S = (16, 24), R = [0:8) x [8:24)

Sliced: (1, 8, 2, 8):(192, 8, 64, 1)@m + 64@m
```

### Implementation

C++ in `src/tir/ir/layout/tile_slice.cc`.
Python: `layout.slice(shape=[16, 24], region=[(0, 8), (8, 24)])`.

---

## 6. Direct Sum

Composes two layouts over the same interleaved domain **without** span scaling (unlike tile).

```
f_{A+B}(x || y) = f_A(x) + f_B(y)
```

### Construction

For each rank position j, concatenate blocks: `A^(j) || B^(j)`.

```
D^{A+B} = [A^(0) || B^(0) || ... || A^(r-1) || B^(r-1)]
R^{A+B} = R^A || R^B
O^{A+B} = O^A + O^B
```

### Relationship to Tile

```
(A * W) + B  =  A x B     (where W = span(f_B))
```

Direct sum is the unscaled counterpart of tiling. Tiling inserts span scaling into the A part.

### When to use direct sum vs tile

- **Tile (x)**: When the inner layout is a compact atom and tiles must not overlap.
  E.g., TMA shared-memory box with contiguous atom `(2,2):(2,1)`.
- **Direct sum (+)**: When the instruction accepts a strided atom and overlap is managed by
  the hardware. E.g., TMA global-memory box with pitch `(2,2):(4,1)`.

### Example

```
B = (2, 2):(4, 1)@m    A = (2, 2):(8, 2)@m

A + B = (2, 2, 2, 2):(8, 2, 4, 1)@m
After reordering same-axis and canonicalizing: (16):(1)@m = contiguous!
```

But A x B cannot produce (16):(1) because span(B) = 6 and residues mod 6 don't cover {2, 3}.

### Implementation

C++ in `src/tir/ir/layout/tile_direct_sum_ops.cc`.
Python: `right.direct_sum(left, left_shape, right_shape)`.
Recognition: `right.is_direct_sum_right(...)`, `left.is_direct_sum_left(...)`.
