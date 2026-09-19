# SBE-Lite v1 — Structure Before Expansion

## Scope and hypothesis
Only the point→L0 token embedding is replaced. The hypothesis is that point-wise
5→32→64 learned expansion followed by symmetric max/mean pooling may create
redundant compute and lose intra-voxel spatial arrangement. SBE first constructs
an ordered 2×2×2 physical description and expands once per occupied L0 voxel.
This is a research hypothesis, not a trained accuracy or speedup claim.

V1 is untouched. Pre-SBE Query-Causal is recoverable at commit `7c5ae42`
(`feat: complete lidar v2 query-causal backbone`). Unrelated staged changes were
excluded from that commit. The SBE experiment retains arbitrary query times,
causal temporal attention, empty-observation tokens, CandidateHead/Selector,
spatial and temporal supervision, and all downstream spatial modules.

SBE does not recover absent UAV observations. NO_CURRENT_SUPPORT remains the
responsibility of the existing Query-Causal temporal route, whose effectiveness
still requires a trained experiment. Neither stage is trained here.

## Geometry and fixed descriptor
Reuse HierarchyBuilder.point_to_l0. L0 size is 0.5m with fixed zero origin and
mathematical floor, including negative coordinates. No second voxelization.

For a point p and its voxel center c:

```
q = (p - c) / 0.5
bits = clamp(floor(2 * (q + 0.5)), 0, 1)
slot = 4*bits.x + 2*bits.y + bits.z
slot_center = 0.5*bits - 0.25
r = q - slot_center
sub_id = point_to_l0 * 8 + slot
```

Slots are permanently ordered 0..7. Residual coordinates are dimensionless,
relative to the sub-voxel center, not the L0 center. Each slot permanently stores:

| Index | Statistic |
|---|---|
| 0–2 | mean residual x/y/z |
| 3–5 | max absolute residual x/y/z |
| 6 | log(1 + point count) |
| 7 | occupancy, 1 if nonempty else 0 |
| 8 | count(sensor_id == 0) / point count, Avia ratio |
| 9 | latest age = -max(delta_t), seconds |
| 10 | temporal std = sqrt(max(mean(delta_t²) - mean(delta_t)², 0)), seconds |

Avia=0, Mid360=1. Delta time is event_timestamp-query_time<=0 and does not depend
on GT. Latest age is clamped nonnegative. Singleton variance is explicitly zero
to avoid cancellation artifacts. Empty slots are all-zero, including max_dt
sentinels before age computation. True empty queries still have no voxels.

All reductions use FP32 and vectorized index_add/scatter_reduce/bincount over
sub_id, with no voxel/slot Python loop and no [P,64] learned activation. Statistics
are [V0,8,11], flattened in slot-major order to [V0,88]. The only learned layers
are Linear(88,128) and LayerNorm(128,eps=1e-5). Output is [V0,128].

## Implementation and configuration
`src/rdq_uav/lidar_v2/sbe.py` defines `SBELiteVoxelEmbed`,
`subvoxel_coordinates`, and `SLOT_DESCRIPTOR`. In model.py the old implementation
is named `LegacyVoxelEmbed`; it remains available only through explicit
`model.voxel.embedding: legacy`. Default YAML selects `sbe_lite` with subdivisions
2, slots8, slot_dim11, flattened_dim88, output_dim128. In SBE mode no legacy
module, Point MLP, or trainable sensor embedding is instantiated.

No change to event selection, reader cleanup, geometry hierarchy, data, losses,
position encoding, spatial attention, merge/up, heads, temporal modules or NMS.
No GPSVE, residual geometry branch, lambda, Slot MLP, extra time window, GT
interpolation, query jitter, velocity, RGB or KV cache is introduced.

## Weight inheritance
`LiDARUAVDetector.load_pre_sbe_weights(state_dict)` drops source voxel_embed.*,
requires all downstream keys with matching shapes, and permits missing keys only:

- voxel_embed.proj.weight
- voxel_embed.proj.bias
- voxel_embed.norm.weight
- voxel_embed.norm.bias

All other missing/unexpected/shape-mismatched keys raise before loading. This
includes temporal weights: a spatial-only V1 checkpoint is not silently accepted
as a complete pre-SBE Query-Causal checkpoint. The test uses the committed
pre-SBE class's initialized state_dict; no trained temporal checkpoint is claimed.
Old/new spatial logits equality is intentionally not a correctness criterion.

## Structure verification and efficiency protocol
Run `tools/check_lidar_uav_v2_sbe.py` for unit/causal/spatial regression,
seq0001's same eight queries, GT-free midpoint inference, inheritance checks,
unchanged-label checks and benchmarks. It never constructs an optimizer.

Output: `outputs/own_multimodal_research/lidar_uav_v2/sbe_lite_v1_structure_smoke/`.
Original Query-Causal smoke and checkpoint files are not overwritten. A rerun
requires a fresh output directory if structure_report.json already exists.

Each embedding and full-model benchmark uses the same real eight-query clip,
eval/no_grad FP32, three warmups and at least 20 measured repeats. Embedding timing
excludes hierarchy construction; full-model timing includes it and both branches.
CUDA timing synchronizes each repetition and records peak allocated memory when
available. CPU fallback reports no invented CUDA measurements. CPU thread count,
mean/median/P95 and raw input/call shapes are recorded in structure_report.json.

Synthetic backward only checks the existing spatial/temporal gradient paths;
there is no optimizer step, scheduler step, epoch or checkpoint optimization.

## Measured result (CPU only)
All 14 SBE tests, 13 Query-Causal tests and nine spatial regressions pass;
py_compile passes. Tests run through unittest plus the existing function-based
regressions (pytest is not installed in this environment). The frozen V1 hash
manifest and downstream class/method AST comparisons pass. Data/geometry/loss/
selector/temporal/trainer helper source files are byte-identical to pre-SBE.

| Parameters | Count |
|---|---:|
| Legacy VoxelEmbed | 19,202 |
| SBE-Lite | 11,648 |
| Spatial including SBE | 1,040,168 |
| Query/temporal modules | 282,723 |
| Full V2 | 1,322,891 |

Reduction from 1,330,445: 7,554 (0.568% full-model; 39.34% embedding).
Default SBE contains no Point MLP or sensor embedding parameters.

Same seq0001_g000019..g000026 clip: P=338,058; V0/V1/V2=69,443/27,996/10,846;
P/V0=4.868. SBE descriptor=[69443,8,11], flattened=[69443,88], L0 token=[69443,128].
Query token/temporal hidden=[1,8,128]; temporal XYZ=[1,8,3]. Outputs finite, future
events=0. GT-free midpoint inference passes. Labels and hierarchy are unchanged.

Linear hooks on the clip record legacy point-level inputs [338058,5] and
[338058,32] (1,690,290 and 10,817,856 input elements), then [69443,129] at voxel
level. SBE records only [69443,88] (6,110,984 elements). Point-level learned calls=0.

CPU, FP32, four threads, PyTorch2.1.1+cu118, 3 warmups and 20 repeats:

| Median forward time | Legacy | SBE |
|---|---:|---:|
| Embedding | 164.42ms | 68.33ms |
| Complete eight-query model | 6022.24ms | 6069.28ms |

Embedding speedup=2.406×. Full-model ratio=0.992×: **this measurement does not
show end-to-end acceleration**. Attention and hierarchy work are unchanged;
embedding gains alone must not be reported as full training speedup. CUDA is
unavailable to this process: CUDA_NOT_TESTED, GPU peak memory unmeasured.

Future-LiDAR change max difference=0; future-GT change=0; truncated-future max
difference=1.1175870895385742e-08. Empty input and NO_CURRENT_SUPPORT temporal
gradient tests pass. Downstream inheritance is exact with no unexplained keys.

Detailed timings, shapes, call counts and source hashes are preserved in
`V2_SBE_LITE_VERIFICATION.json` and the structure-smoke output directory.
No optimizer/scheduler step, training epoch or formal checkpoint optimization.
