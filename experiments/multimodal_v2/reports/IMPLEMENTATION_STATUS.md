# Multimodal V2 implementation status

## Code fact and old-to-new mapping

| V1 component | V2 disposition |
|---|---|
| `P5MultimodalBackbone` / three pre-stage HCI blocks | Removed from V2 execution |
| `LiDARV2PyramidAdapter` + `RadarCandidateBuilder` | Independent `LiDARCandidateModel` |
| shared `SwinPyramidAdapter` + `DINOAdapter` + RGB builder | Independent `VisionCandidateModel` |
| global candidate association + ReliabilityGate | Deterministic 16px geometry match + independent evidence-gated 3D/2D corrections |
| `TypedSharedQuery`, Fusion Decoder | Removed |
| `FusionPredictionHeads.xyz` | Removed; LiDAR XYZ is copied exactly |
| V-only absolute XYZ | Forbidden by `has_xyz=False` contract |

The V2 call chain is:

```text
data.prepare_model_batch
  -> LiDARCandidateModel + VisionCandidateModel
  -> CandidateCrossAttention
  -> CandidateScoring
  -> MultimodalOutput
  -> CandidateRankingLoss / FP32 evaluation
```

`CandidateScoring` does not compare raw cross-encoder feature cosine
similarity. It sorts feasible pairs by `(d_box + 0.01*d_center_normalized,
radar_source_index, vision_source_index)` and greedily accepts a one-to-one
match. Query diagnostics record candidate counts, feasible pairs, R/RV/V
counts, invalid projections, conflicts, and accepted RV geometry. Geometric
feasibility is not called association correctness without independent 2D GT.

The output owns two disjoint score streams. `score_3d_*` starts from and can
only correct `radar_score`; `score_2d_*` starts from and can only correct
`vision_score`. V-only rows cannot enter 3D ranking, R-only rows cannot enter
2D output, and B2 never changes LiDAR XYZ.

## Established E5 result (not a V2 result)

The archived 4,800-query aggregate validation reports:

| checkpoint | Success@1m | median error | mean error |
|---|---:|---:|---:|
| E5 Best, epoch 1 | 0.839583 | 0.371650 m | 4.1723 m |
| E5 Last, epoch 12 | 0.205833 | 21.722412 m | 22.5350 m |

The archived 32-query smoke diagnosis found 13 Best-success/Last-failure
pairs; all 13 selected V-only at Last while a LiDAR candidate within 1 m was
still present. This is exploratory evidence and does not replace the supplied
full 4,800-query diagnostic command.

## Current validation boundary

- CPU contract tests: 17 passed after the task-separated and mixed-dtype
  contract updates. The dedicated CUDA FP16 autocast test is present but was
  skipped because CUDA is unavailable in the current environment.
- Dataset audit: 8,000 train / 4,800 validation queries, 20 train sequences.
- Strict CPU construction of LiDAR, DINO, and E5 visual weights was established
  for the initial V2 implementation. It was not rerun after this revision
  because the user requested code changes only.
- The single-owner DINO state contract and strict in-memory roundtrip are CPU
  tested. Real-weight strict construction remains part of the manual gate.
- Real GPU B0/B1 identity: implemented, not run in the current CUDA-blocked environment.
- B2 two-update and formal B2 training/evaluation: intentionally not run; the
  user executes them manually.
- B3: deferred because only five validation queries have valid independent 2D GT.
