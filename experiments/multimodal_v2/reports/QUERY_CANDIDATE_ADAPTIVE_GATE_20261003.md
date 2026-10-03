# Query / candidate / adaptive gate implementation

## SBE audit
Local `sbe.py` already uses `time_weighted_v2`, 0.2s configurable half-life,
weighted residual means and population standard deviations, true counts,
occupancy, raw Avia ratio, mean absolute time weight and unweighted time std.
VQSA and [V0,8,11] -> 88+16 -> 128 are already present. This update does not
modify SBE, its configuration, tests or TIME_WEIGHTED_SBE report. Existing
centered two-pass variance is retained for numerical stability. Configuration
remains `model.voxel.sbe.time_half_life_s`, not a new duplicate temporal key.

## Query and target
`QueryRecord(sequence_id, query_time, sample_id)` contains no target path.
`MultimodalV2Dataset(root, query_records, *, target_3d_index=None, ...)` requires
explicit records. `query_records_from_3d_gt` is the current offline query
source and reads filenames only. `build_3d_target_index` independently indexes
exact-time GT. No implicit nearest target is used. Missing or nonfinite target
has zero XYZ and `target_valid=False`; malformed target shape is an error.
Provider paths are checked for sequence and filename time agreement.
Training input filtering depends solely on radar/image availability.
`build_datasets`, real-data diagnostic and synthetic fixtures use the new API.
Sample/packed Batch/model shapes are unchanged. `target_source_path` is audit
metadata only and is not included in the model batch.

## Candidate limits
Shared `configs/multimodal_v1/p6_candidates.yaml`: radar raw=10, pre-NMS=50,
final=10, radius=1m; RGB pre=50, final=10, IoU=.7. Limits apply per sample.
The larger NMS pool is independent of raw diagnostic Top10.
This shared config also affects V1 runs which consume it; LiDAR-only selector
configuration is unchanged. B0/B1 compare against this new candidate budget,
not historical 50-candidate outputs.

## Common adaptive gate
`geometry.py` provides validated common inverse-range configuration and margins.
M(r)=clip(8+160/r,8,48), r=norm(radar XYZ) in meters about the radar-frame origin.
Output is FP32 in source-camera pixels, independent of image resize.
Both V<-R evidence admission and RV association consume the same config.
Finite [N,3] XYZ is required; zero range is clamped to 1e-3m then saturated.
Calibration math, association cost, tie policy, attention bias and XYZ/box
outputs are unchanged. V2 initialization no longer reads V1 geometry_gate_px
or the old V2 box_margin_px. Constructors default to the same adaptive settings.
In B2, association changes immediately; R<-V 3x3 image evidence reading does
not use a box gate. V<-R remains disabled by the B2 configuration.
The inverse-range law is an empirical heuristic; it does not establish
calibration covariance and may underrepresent range-independent rotational
error. Parameters need real-data assessment before claiming improvement.

## Checkpoints and validation
Do not treat shape-compatible pre-time-weighted SBE weights as the new trained
baseline. `LiDARUAVDetector.load_pre_sbe_weights` already supports transferring
downstream weights while leaving SBE initialized; train LiDAR to obtain a new
checkpoint before formal B2. This patch does not retrain or enforce checkpoint
provenance. Existing loading can still accept old weights: verification pending.
New tests cover no-GT queries, filename-only query source, input-only filtering,
GT identity and invalid GT, gate values/config validation, near/far rejection,
shared evidence/association admission and larger pre-NMS pool behavior.
Per the user's deferred validation instruction, pytest, model forward, GPU,
checkpoint identity and real-data smoke have NOT been executed this round.
Only syntax/YAML, diff whitespace and installer checks are performed.
