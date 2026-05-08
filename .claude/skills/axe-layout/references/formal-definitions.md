# Axe Layout Formal Definitions & Axis Registry

## Table of Contents

1. [Formal Definitions](#1-formal-definitions)
2. [Axis Registry](#2-axis-registry)
3. [Scope Hierarchy](#3-scope-hierarchy)
4. [SwizzleLayout Details](#4-swizzlelayout-details)
5. [Trainium-Specific Layouts](#5-trainium-specific-layouts)
6. [Non-Bit-Linear Layouts](#6-non-bit-linear-layouts)

---

## 1. Formal Definitions

### Axis Space

Let `A = {a_0, ..., a_{n_A-1}}` be named axes (e.g., m, laneid, warpid, gpuid).
The axis space is:
```
ZA = { sum_i z_i @ a_i | z_i in Z }
```
with componentwise addition and scalar multiplication.

- **Minkowski sum**: `X + Y = {x + y | x in X, y in Y}`
- **Hadamard product**: `(sum z_i@a_i) * (sum z'_i@a_i) = sum (z_i * z'_i)@a_i`

### Iter

An iter `I = (e_I, s_I, a_I)` with:
- extent `e_I > 0`
- stride `s_I != 0`
- axis `a_I in A`

Induces `f_I: [0, e_I) -> ZA`, `f_I(x) = (x * s_I) @ a_I`.

### Layout Triple

An Axe layout `L = (D, R, O)`:
- `D = (I_0, ..., I_{n_D-1})` — ordered tuple of sharded iters (n_D >= 1)
- `R = (J_0, ..., J_{n_R-1})` — multiset of replicated iters (n_R >= 0)
- `O in ZA` — offset

Total domain size: `E_D = prod(e_{I_k})`.
With lexicographic unflattening `iota: [0, E_D) -> prod [0, e_{I_k})`:

```
f_D(x) = sum_{k=0}^{n_D-1} (iota(x)_k * s_{I_k}) @ a_{I_k}
```

### Induced Set-Valued Map

```
f_L(x) = { f_D(x) + f_R(r) + O | r in prod [0, e_{J_t}) }
```

If R is empty, `f_L(x) = {f_D(x) + O}` (singleton).
Otherwise `|f_L(x)| = E_R = prod(e_{J_t})`.

### Shape-Admitted View

For shape `S = (S_0, ..., S_{r-1})` with `prod S_i = E_D`:
```
f_{L<S>}(u) = f_L(flat_S(u))
```
where `flat_S` is the row-major flattener.

### Axis-wise Span (Closed Form)

```
span_a(f_L) = 1 + sum_{I in D, a_I=a} |s_I|(e_I - 1)
                + sum_{J in R, a_J=a} |s_J|(e_J - 1)
```

Total span: `span(f_L) = sum_{a in A} span_a(f_L) @ a`.

---

## 2. Axis Registry

Axes are registered at C++ level via `TVM_REGISTER_AXIS`. Each axis has:
- **name**: string identifier
- **scope**: the execution scope it belongs to (optional, for thread axes)
- **subscope**: the child scope (optional)
- **fuser**: function to fuse loop iters onto this axis
- **splitter**: function to split an axis iter into (outer, inner)

### Thread Axes and Their Scopes

| Axis | Scope | Subscope | Description |
|------|-------|----------|-------------|
| `pid` | kernel | kernel | Program/kernel ID |
| `bx` | kernel | cta | Block index X |
| `by` | kernel | cta | Block index Y |
| `bz` | kernel | cta | Block index Z |
| `cbx` | kernel | cta | Cluster block X |
| `cby` | kernel | cta | Cluster block Y |
| `cbz` | kernel | cta | Cluster block Z |
| `tx` | cta | thread | Thread index X |
| `warpid` | cta | warp | Warp ID within CTA |
| `laneid` | warp | thread | Lane ID within warp |
| `wgid` | cta | warpgroup | Warpgroup ID |
| `tid_in_wg` | warpgroup | thread | Thread within warpgroup |
| `wid_in_wg` | warpgroup | warp | Warp within warpgroup |

### Memory Axes

| Axis | Description |
|------|-------------|
| `m` | Default memory (linear address space) |
| `P` | Partition dimension (Trainium SBUF/PSUM) |
| `F` | Free dimension (Trainium SBUF/PSUM) |
| `Bank` | Memory bank (Trainium PSUM) |
| `TCol` | Tensor memory column (Blackwell) |
| `TLane` | Tensor memory lane (Blackwell) |

### Scope Well-Formedness

A layout's verify_well_formed() checks that thread axes have **well-connected scopes**.
If a layout uses `laneid` (scope=warp) and `warpid` (scope=cta), the subscope chain
must be connected: warp is a subscope of cta.

---

## 3. Scope Hierarchy

```
kernel
├── cta (thread block)
│   ├── warpgroup
│   │   ├── warp
│   │   │   └── thread
│   │   └── thread (via tid_in_wg)
│   ├── warp (via warpid)
│   │   └── thread (via laneid)
│   └── thread (via tx)
└── cta (via bx, by, bz, cbx, cby, cbz)
```

The `get_scope()` method on TileLayout returns a `(scope, subscope)` pair representing
the execution context: `scope` is the coarsest scope of all thread axes, `subscope`
is the finest scope.

---

## 4. SwizzleLayout Details

SwizzleLayout models XOR-based swizzling for shared memory to avoid bank conflicts.

Parameters:
- `per_element`: log2 of element size in bytes
- `swizzle_len`: number of swizzle bits (typically 1-5)
- `atom_len`: log2 of atom size (the repeating swizzle pattern)
- `swizzle_inner`: if true, inner bits are swizzled; if false, outer bits

The swizzle operation XORs certain bits of the memory address to permute data placement
across shared memory banks. For a given linear index:

```
inner_bits = index & inner_mask
outer_bits = index & outer_mask
swizzled_addr = index ^ ((outer_bits >> shift) & inner_mask)
```

Where masks and shift are derived from the parameters.

### Common configurations

- **128B swizzle**: `SwizzleLayout(per_element=1, swizzle_len=3, atom_len=4)` for FP16
- **64B swizzle**: `SwizzleLayout(per_element=1, swizzle_len=2, atom_len=4)` for FP16
- Adjusted per_element for different dtypes (e.g., 0 for FP32, 2 for FP8)

---

## 5. Trainium-Specific Layouts

Trainium has 2D scratchpad memories (SBUF and PSUM) with:
- 128 partitions (P axis)
- Contiguous free dimension (F axis)

### Construction via `TileLayout.trainium(annotation, shape)`

```python
# annotation is a string of 'P' and 'F' characters
# shape gives the extent of each dimension

# 128 partitions x 512 free
layout = TileLayout.trainium("PF", (128, 512))
# -> TileLayout(shard=[(128, 1@P), (512, 1@F)])

# Free x Partition
layout = TileLayout.trainium("FP", (256, 64))
# -> TileLayout(shard=[(256, 1@F), (64, 1@P)])

# Multiple P dimensions (product must be <= 128)
layout = TileLayout.trainium("PPF", (4, 32, 512))

# Large P dimension (> 128 splits into P + F)
layout = TileLayout.trainium("PF", (256, 512))
# -> splits 256 into 2@F * 128@P
```

### PSUM conversion

```python
psum_layout = layout.to_psum()
```

Converts F-axis iters into Bank-axis iters where stride crosses the 512-element bank boundary.
PSUM has 8 banks of 512 elements each.

---

## 6. Non-Bit-Linear Layouts

Axe layout is strictly more expressive than bit-linear (F_2-linear) layouts used in systems
like Triton.

A layout function f is **bit-linear** if `f(x XOR y) = f(x) XOR f(y)` for all x, y.

**Counterexample**: A 24x24 tensor with column-major layout:
```
f(i) = floor(i/24) + (i % 24) * 24
```
has `f(1) = 24`, `f(2) = 48`, `f(1 XOR 2) = f(3) = 72`, but `f(1) XOR f(2) = 40 != 72`.

This means non-power-of-two sizes common in cross-device partitions (e.g., 3-way sharding,
DeepGEMM's BLK_N=112) cannot be represented by bit-linear layouts but are naturally
expressible in Axe.

Axe handles arbitrary integer extents and strides, making it suitable for:
- Non-power-of-two tile sizes
- Distributed sharding across arbitrary device counts
- Hardware with non-standard memory dimensions
