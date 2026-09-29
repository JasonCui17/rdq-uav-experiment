# Multimodal V2 implementation status

## Code fact and old-to-new mapping

| V1 component | V2 disposition |
|---|---|
| `P5MultimodalBackbone` / three pre-stage HCI blocks | Removed from V2 execution |
| `LiDARV2PyramidAdapter` + `RadarCandidateBuilder` | Independent `LiDARCandidateModel` |
| shared `SwinPyramidAdapter` + `DINOAdapter` + RGB builder | Independent `VisionCandidateModel` |
| global candidate association + ReliabilityGate | Geometry association + evidence-gated bounded score correction |
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

- CPU contract tests: implemented and run.
- Dataset audit: 8,000 train / 4,800 validation queries, 20 train sequences.
- Strict CPU construction of LiDAR, DINO, and E5 visual weights: run.
- Constructed parameters: 49,383,561 total; B2 exposes 137,218 trainable parameters.
- Real GPU B0/B1 identity: implemented, not run in the current CUDA-blocked environment.
- Formal B2 training/evaluation: intentionally not run; the user executes it manually.
- B3: deferred because only five validation queries have valid independent 2D GT.
