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

- CPU contract tests: 18 passed after the task-separated, mixed-dtype, and
  autocast-safe focal-loss updates. The dedicated CUDA FP16 autocast test is
  present but was skipped because CUDA is unavailable in the current
  environment.
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

## 2026-10-02: query-time data refactor

The earlier validation boundary above is historical. Current progress and the
full audit are recorded in [DATA_REFACTOR_PROGRESS_20261002.md](DATA_REFACTOR_PROGRESS_20261002.md).

- V2 builds each complete Sample directly, without LiDARUAVDataset wrapping.
- Radar selects all events in the inclusive configurable historical window;
  left RGB selects the latest frame in [query_time-max_image_gap_s, query_time]; future RGB is forbidden.
- Sample/collate/preprocess remain three explicit layers in data.py.
- V2 owns the migrated ProjectionContext and unchanged calibration math;
  its active model path no longer uses the V1 InteractionContext wrapper.
- Final CPU regression: 48 passed (34 V2 + 14 legacy data/geometry), one
  existing attention mask dtype warning. Compileall and diff check passed.
- Real sequence smoke: blocked by absent MMAUD data in this environment.
- Earlier GPU updates and 4,800-query B0/B1 are user-reported under the old
  data definition; new baseline and GPU gates remain pending.
- This checkout starts at 6cbb240, before the user's later AMP/BCE fixes.
  Preserve those fixes when integrating this refactor into the latest branch.

### Follow-up: causal historical RGB selection

The latest user instruction supersedes symmetric RGB matching. Select the
latest image timestamp <= query_time and reject it if older than
query_time-max_image_gap_s (default 1 second, inclusive endpoints).
Missing historical RGB is masked even when a closer future frame exists.
Updated labels and validity checks use this historical frame only.
Regression result: 55 passed (41 V2 + 14 legacy), one existing mask-type warning;
compileall and diff check passed. Real-data/GPU gates remain pending.
