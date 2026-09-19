# LiDAR UAV V2 Pre-Training Correctness Gate v1

This gate freezes the current SBE-Lite + VQSA + EOOE + Query-Causal model
mathematics and hardens the boundaries used by formal training and evaluation.
It does not add a model module or change a loss weight.

## F1: candidate ordering

Candidate Top-K and radius-NMS priority use the raw objectness logits converted
to FP32. The reported/exported confidence is `sigmoid(logits.float())`. This
preserves real-valued monotonic ordering while avoiding BF16 sigmoid saturation.
Equal logits retain the deterministic ascending `source_token_id` tie break.

## Evaluation precision

`evaluation.precision` is validated and defaults to FP32. Train-time validation,
standalone evaluation, and checkpoint re-evaluation all call
`rdq_uav.lidar_v2.training.validate` with the same precision policy. Training
AMP remains independent and may use BF16.

## F2: occurrence-level endpoint supervision

`spatial_supervise_mask_occurrence [N_occurrence]` is the only boundary that
decides which query occurrences enter spatial loss aggregation. All valid
training occurrences remain supervised. Rolling validation enables only its
scored endpoint; history and padding are context only.

UQP still computes each unique spatial query once. CandidateLoss maps the
selected occurrence mask through `occurrence_to_unique`, so duplicate selected
occurrences preserve their original weight and NO_CURRENT_SUPPORT remains
excluded from spatial loss. Temporal validation supervision uses the same
endpoint mask and remains in occurrence space.

The relevant index spaces are:

- `num_occurrences`: temporal clip slots before UQP.
- `num_spatial_queries`: unique queries evaluated by the spatial branch.
- `occurrence_to_unique`: occurrence-to-spatial-query mapping.

## F3: effective configuration

Every train/evaluate startup validates the frozen V2 architecture. Fixed fields
such as SBE, VQSA, EOOE, global L2 attention, causal temporal processing, and
evaluation precision fail immediately when set to unsupported values. The stale
`train.checkpoint_metric` field was removed and is explicitly rejected if it
reappears.

Each entry point writes `effective_config.yaml` and `effective_config.json`.
These record the architecture, voxel scales, temporal settings, EQS, UQP,
training/evaluation precision, losses, residual scale, and checkpoint policy
that actually govern the run.

## F4: residual coordinate contract

This V2 version only supports `residual_scale_m = 1.0`. Both model construction
and the shared codec reject any other scale. Spatial targets use
`encode_residual(gt_xyz, voxel_center, 1.0)` and predictions use
`decode_residual(residual, voxel_center, 1.0)`.

## F5: evaluation boundary

Formal evaluation currently requires occurrence-aligned spatial queries and
therefore UQP must be disabled in validation/evaluation collate. Evaluation
entry points check this invariant and raise a clear `RuntimeError` for packed
UQP layouts. UQP remains enabled for training, where its weighted occurrence
loss contract is tested independently.

## Checkpoint policy

Formal training writes three independent files:

- `last.pt`: updated each epoch for resume.
- `best_spatial.pt`: ALL-sample NMS Recall@10@1m, then spatial Top1
  Success@1m, then lower spatial median error, then earlier epoch.
- `best_temporal.pt`: ALL-sample Temporal Success@1m, then lower temporal
  median error, then lower temporal P90 error, then earlier epoch.

CURRENT_SUPPORT and NO_CURRENT_SUPPORT metrics remain required reports, but do
not select either best checkpoint. Checkpoints include optimizer/scheduler
state, global optimizer step, effective configuration, both precision modes,
selection metrics, Git/run identity, and EQS cycle metadata.

## Query-readout diagnostics

The candidate-aware pool exposes a no-gradient optional diagnostic helper for
pool entropy, maximum objectness probability, highest-logit reference XYZ,
pooled XYZ, candidate count, and optional GT-near accumulated pool weight. The
helper changes neither the forward output contract nor any loss.

No optimizer step, scheduler step, training epoch, or checkpoint optimization
is part of this gate audit.
