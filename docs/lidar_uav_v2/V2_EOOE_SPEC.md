# LiDAR UAV V2 — EOOE-v1

EOOE-v1 (Explicit Octant Occupancy Encoding) preserves local topology while
L0 voxels merge into L1 and L1 voxels merge into L2. It changes only the two
bottom-up `SparseMerge` stages. SBE-Lite, spatial attention, SparseUp,
CandidateHead, Query-Causal temporal processing, supervision, EQS, and UQP are
unchanged.

## Audited predecessor

Before EOOE, each child was transformed from

```text
LayerNorm(child_feature[128]) + relative_octant_xyz[3]
    -> Linear(131, 128) -> GELU
```

and each parent projection received:

```text
max transformed child feature       128
mean transformed child feature      128
log1p(parent raw-point count)          1
child voxel count / 8                  1
                                      ---
                                      258
```

The first scalar is raw-point density. `SparseLevel.point_count` starts as the
number of raw points in each L0 voxel, and `HierarchyBuilder` sums it through
L1 and L2. This statistic cannot be recovered from occupied child slots and is
retained unchanged.

The second scalar is occupancy ratio. It is exactly
`occupancy.sum(-1) / 8`, so EOOE removes it rather than retaining redundant
information.

## Relative position and explicit topology

For child coordinate `g_c`, its parent coordinate `g_p`, and `parent_map`:

```text
local = g_c - 2 * g_p[parent_map]
rel   = local - 0.5
```

Every component of `local` must be 0 or 1, making each component of `rel`
either -0.5 or +0.5. Relative position answers where an existing individual
child lies. EOOE answers which of all eight possible positions exist for the
parent. These signals are complementary.

The fixed slot convention is shared with SBE-Lite:

```text
slot = 4 * local_x + 2 * local_y + local_z
```

Thus slots 0..7 correspond to `(000), (001), (010), (011), (100), (101),
(110), (111)`. A vectorized flattened index,
`parent_map * 8 + slot`, constructs binary occupancy `[N_parent, 8]`. Runtime
invariants require local coordinates in `{0,1}³` and require occupancy sum to
equal each parent's unique child voxel count.

## Final parent representation

Both L0→L1 and L1→L2 use independent `SparseMerge` parameters and the same
input definition:

```text
max transformed child feature       128
mean transformed child feature      128
log1p(parent raw-point count)          1
explicit ordered occupancy             8
                                      ---
                                      265

Linear(265, 128) -> LayerNorm(128)
```

There is no occupancy MLP, pattern ID embedding, learned scale, residual
occupancy branch, or occupancy input to PositionEncoding/SparseUp. The final
linear projection directly owns distinct weights for every fixed octant.

## Hierarchical representation

At L0, SBE-Lite first forms ordered `2×2×2` raw-point structure with eleven
physical statistics per sub-voxel, then expands `88→128` once per occupied
voxel. During L0→L1 and L1→L2 expansion, EOOE combines transformed child
semantics, relative position, ordered occupancy topology, and propagated raw
point density. This is a hierarchical geometry-preserving sparse
representation; the wording does not claim an empirical accuracy result.

## Verified structure audit

The read-only audit is stored under
`outputs/own_multimodal_research/lidar_uav_v2/eooe_structure_audit/`. On the
real eight-query `seq0001` smoke clip it processed 70,545 input points and
produced 29,865 / 14,409 / 6,131 L0/L1/L2 tokens. Parent projection inputs were
`[14409,265]` and `[6131,265]`; occupancy tensors were `[14409,8]` and
`[6131,8]`. All full-model outputs were finite and no future event was used.

EOOE, UQP exactness, EQS, sequence isolation, Query-Causal leakage checks,
SBE-Lite, and original spatial regressions all passed. Only forward/no-grad and
test backward operations were performed: optimizer steps, scheduler steps,
training epochs, and checkpoint optimization were all zero.
