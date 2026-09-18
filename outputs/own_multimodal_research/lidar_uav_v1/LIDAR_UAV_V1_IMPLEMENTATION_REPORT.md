# LiDAR UAV Transformer V1 Implementation Report

## 1. Files

- `configs/lidar_uav_v1.yaml`: sole default V1 configuration.
- `src/rdq_uav/lidar_v1/data.py`: unified last-20 dataset and packed collate.
- `src/rdq_uav/lidar_v1/geometry.py`: one-pass L0 voxelization and parent maps.
- `src/rdq_uav/lidar_v1/model.py`: VoxelEmbed, SparseMerge, SpatialTransformer, SparseUp, CandidateHead, detector.
- `src/rdq_uav/lidar_v1/loss.py`: labels, focal loss, regression loss, NO_CURRENT_SUPPORT.
- `src/rdq_uav/lidar_v1/selector.py`: raw Top20 and stable radius-NMS Top20.
- `src/rdq_uav/lidar_v1/runtime.py`: optimizer grouping, update scheduler and metrics.
- `tools/train_lidar_uav_v1.py`, `tools/evaluate_lidar_uav_v1.py`, `tools/export_lidar_uav_candidates.py`, `tools/smoke_lidar_uav_v1.py`: public entries.
- `tests/test_lidar_uav_v1.py`: V1 model-chain regression tests.

## 2. Engineering structure

The public chain is `LiDARUAVDataset -> collate_lidar_samples -> LiDARUAVDetector -> CandidateLoss / CandidateSelector`.
Geometry owns packed coordinates/maps; neural modules consume them without rescanning source points. Training and evaluation depend only on detector outputs.

## 3. Model

`VoxelEmbed -> L0 local blocks x2 -> Merge01 -> L1 local blocks x2 -> Merge12 -> L2 global blocks x2 -> Up21 -> Up10 -> FinalLN -> CandidateHead`.

| Module | Input | Output | Parameters |
|---|---|---|---:|
| VoxelEmbed | P points: XYZ, sensor id, delta-t | N0 x 128 | 19,202 |
| L0 SpatialTransformer | N0 x 128, 0.5 m coords | N0 x 128 | 270,092 |
| SparseMerge 0.5->1 m | N0 x 128 + parent map | N1 x 128 | 50,560 |
| L1 SpatialTransformer | N1 x 128, 1 m coords | N1 x 128 | 270,092 |
| SparseMerge 1->2 m | N1 x 128 + parent map | N2 x 128 | 50,560 |
| L2 SpatialTransformer | N2 x 128 per-sample global | N2 x 128 | 270,092 |
| SparseUp 2->1 m | F1, F2, parent map | N1 x 128 | 50,048 |
| SparseUp 1->0.5 m | F0, D1, parent map | N0 x 128 | 50,048 |
| Final LayerNorm | N0 x 128 | N0 x 128 | 256 |
| CandidateHead | N0 x 128 | logits N0, residual N0x3 | 16,772 |

Total: **1,047,722 parameters**.

## 4. Forward data flow

Avia/Mid360 metadata are stably merged, restricted to `timestamp <= t0`, and truncated to the most recent 20 total events. Cleaning removes only non-finite XYZ and confirmed all-zero padding. Points carry sensor id and event delta-t seconds. Fixed-origin mathematical floor creates L0 once; L1/L2 and point counts come only from parent maps. Packed tokens remain sample-separated at every attention level.

The detector returns logits, residual/predicted XYZ, 128D fine features, centers, stable source ids, batch ids, layouts and token statistics.

## 5. Design conformance

- D=128, four heads, 256D FFN, two blocks per level.
- Shared per-level three-axis relative tables, zero initialized, clipped to [-32,32].
- Morton ordering; 64-token local groups with non-circular 32-token shift at L0/L1.
- **L2 uses true per-sample global attention.** No local fallback exists.
- SparseMerge uses child position, max/mean, log point count and child-count/8. No AttentionPool or occupancy input.
- SparseUp has adjacent links only. No 2m-to-0.5m link.
- Head exposes FinalLN feature directly; no query head.
- Denoising is false. No valid point is filtered, capped, deduplicated or randomly removed.
- Labels follow recent<=1m positive, otherwise all<=2m ignore, otherwise negative.
- Focal/SmoothL1 are normalized per sample by positive count and averaged over supervised samples.

## 6. Necessary adaptations

The repository had no sparse CUDA kernels, so V1 uses packed native PyTorch operations and explicit grouped attention. The actual backend is `explicit_pytorch_scaled_dot_product_with_additive_axis_bias`. It creates no dense spatial grid. GT interpolation is unnecessary because prediction times are GT timestamps.

## 7. Not run or unavailable

- The requested 100-epoch formal training was intentionally not started.
- Independent-flight metrics are not emitted because a confirmed sequence-to-independent-flight mapping is unavailable.
- Multi-GPU execution was not exercised: this host exposes one RTX 3070 Laptop GPU. The loss path globally synchronizes whether any rank is supervised, but the current entry was validated in single-GPU mode.
- Activation checkpointing was unnecessary.

## 8. Known risks

- Explicit L2 attention is quadratic in each sample's L2 token count. No architectural fallback is present.
- Training/evaluation repeatedly loads 20 source events; production throughput may benefit from a provenance-safe read cache.
- The frozen `EXISTING_MMUAV_COORDINATE_ASSUMPTION` remains an engineering assumption. G2 observed strong Mid360/Avia point-count imbalance; V1 does not alter coordinates.

## 9. Tests

`py_compile` passes. Eighteen regression functions pass using the installed rdq environment (pytest is absent, so functions were executed directly). Coverage includes event causality, unified last-20, cleaning, negative floor, all maps, octant positions, single/large voxels without cap, batch isolation, Morton identity, non-circular shifts, global batch isolation, finite loss/backward, ignore and NO_CURRENT_SUPPORT, residual decoding, synchronized selection, and empty prediction.

Candidate export smoke produced 723 synchronized rows and a `(723,128)` feature array. The checkpoint embeds model/data/loss/selector configs, optimizer, scheduler, epoch/step, seed, commit, best metric and `denoise=false`.

## 10. Real single-batch smoke

- Sample: `seq0001_g000015`; 17 historical events; 33,523 valid points.
- Tokens L0/L1/L2: 8,078 / 3,312 / 1,290.
- Shapes: logits `[8078]`, residual/pred XYZ `[8078,3]`, fine feature `[8078,128]`.
- Loss/cls/reg: 2.11923 / 1.13014 / 0.49454; pos/neg/ignore: 1 / 8,075 / 2.
- Forward/backward: 1.461 / 1.058 seconds.
- Peak GPU allocation: 540.01 MB.
- VoxelEmbed, Merge01, L2 QKV, Up10 and regression-head gradients are present and finite.
- Future leakage: false.

## 11. Tiny-set overfit

- Deterministic first 32 train samples with recent support; all happen to be from `seq0001`.
- Full V1, batch 2, accumulation 1, BF16 autocast, 500 optimizer updates, AdamW lr 2e-4, weight decay 0.
- Loss: 2.23059 -> 0.0002367, a 99.989% reduction.
- NMS Recall@10@1m: **100%**. Raw and NMS Top1 Success@0.5/1/2m: **100%**.
- NMS Top1 error median/P90/P95: 0.4767 / 0.4928 / 0.4960 m.
- Runtime: 1,892.1 seconds. Peak GPU allocation: 979.63 MB.
- No OOM, NaN, architecture fallback, point filtering, future leakage, confirmed axis error or GT-unit mismatch occurred.

## 12. Provenance and next step

Current git commit recorded in checkpoints: `a214c303fd01ed885e9c3333c48aa9fcb2decd93`. The worktree contains the new implementation. Review the public interfaces, real-batch shapes and tiny diagnostics before manually starting formal training.
