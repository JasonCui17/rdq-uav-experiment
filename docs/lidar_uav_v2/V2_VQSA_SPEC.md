# LiDAR UAV V2 — VQSA-v1

VQSA-v1 (Voxel Query Slot Aggregation) adds content-adaptive aggregation to
the eight ordered SBE-Lite sub-voxel slots without restoring a point-wise
learned expansion. It modifies only the `8-slot → L0 token` stage.

## Three geometry responsibilities

- **SBE-Lite** computes explicit physical structure inside each 0.5 m L0
  voxel: eight ordered sub-voxels with eleven fixed statistics per slot.
- **VQSA** uses one learned voxel summary query to aggregate those L0 slots
  according to their content and fixed location.
- **EOOE** preserves explicit eight-octant topology when L0 merges into L1 and
  L1 merges into L2.

The `voxel_query` is a voxel-summary query. It is not a UAV token, object class
token, subtype label, or additional supervised target.

## Preserved SBE statistics

VQSA receives `[V0,8,11]` in the unchanged slot order:

```text
0:3   mean residual XYZ
3:6   max absolute residual XYZ
6     log1p(point count)
7     occupancy
8     Avia ratio
9     latest age
10    temporal standard deviation
```

Slots use `slot = 4*x + 2*y + z`. Their fixed normalized centers are registered
as a non-learned buffer; every axis is -0.25 for bit 0 and +0.25 for bit 1.

## Dynamic branch

Each slot concatenates its statistics and fixed center:

```text
[V0,8,11] + [8,3] -> [V0,8,14]
shared Linear(14,16) -> GELU -> [V0,8,16]
```

A single learned `voxel_query [1,1,16]` expands to `[V0,1,16]`. It is the query
of `MultiheadAttention(embed_dim=16, heads=2, dropout=0, batch_first=True)`;
the eight slot tokens are keys and values. This is `1×8` cross-attention, not
nine-token self-attention and not `8×8` slot self-attention.

The key padding mask comes from integer sub-voxel point counts. Empty slots are
masked even though their fixed centers are nonzero. Every real L0 voxel must
have at least one visible slot; empty LiDAR samples create no fake voxel and
continue through the existing missing-query mechanism. Attention weights are
returned only when explicitly requested for debugging.

## Final L0 token

The fixed structural path remains the ordered flattened SBE descriptor:

```text
fixed   = flatten(slot_stats)       [V0,88]
dynamic = VQSA(slot_stats, counts)  [V0,16]
concat                              [V0,104]
Linear(104,128) -> LayerNorm        [V0,128]
```

There is no old `88→128` residual, learned mixing scale, point MLP, slot-slot
self-attention, VQSA auxiliary loss, or L1/L2 semantic query.

## Compute contract

Per occupied L0 voxel, excluding bias/GELU/softmax bookkeeping:

| Operation | MACs |
|---|---:|
| Shared slot projection `8×14×16` | 1,792 |
| MHA Q/K/V/output projections | 4,608 |
| QK dot products, two heads | 128 |
| AV aggregation, two heads | 128 |
| Final projection increment `16×128` | 2,048 |
| Total increment over fixed SBE | 8,704 |

The query length is one and key/value length is fixed at eight. Logical visible
attention pairs equal the number of occupied slots. Raw points still undergo
only vectorized physical statistics; there is no `O(P × high_dim)` learned
point expansion.

## Training boundary

CandidateHead, spatial/temporal losses, Query-Causal, EQS, UQP, recent-mask
contract, and evaluation remain unchanged. Structure verification uses forward,
no-grad smoke, and regression backward only. It performs no optimizer step,
scheduler step, training epoch, or checkpoint optimization.
