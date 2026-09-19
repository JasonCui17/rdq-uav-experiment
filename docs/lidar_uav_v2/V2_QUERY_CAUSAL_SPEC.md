# LiDAR UAV V2 QueryCausal v1

## Task and time semantics
Predict absolute UAV XYZ at arbitrary `query_time` from Avia/Mid360 events with timestamp <= query_time. `target_timestamp`, `target_xyz`, and `target_valid` are separate supervision metadata. Model forward never reads targets. First-stage train queries use train GT timestamps; validation uses the CSV Timestamp/Position with Classification retained only as UAV-type metadata. No timestamp interpolation is performed.

`LiDARQueryBuilder` only reads sensor streams. It selects the latest 20 merged events (not 20 per sensor); fewer events remain fewer. Recent points belong to the last four selected actual events. Delta time is seconds relative to query time. Shared released-XYZ cleaning remains unchanged; no denoising, point cap, background removal, or coordinate transform is added. Coordinate convention remains EXISTING_MMUAV_COORDINATE_ASSUMPTION.

## Spatial branch
VoxelEmbed, 0.5/1/2m hierarchy, 128D spatial Transformers (including true per-query L2 global attention), SparseMerge, adjacent SparseUp, final LN, CandidateHead and selector retain V2-base mathematics and spatial state keys. `load_spatial_v2_base_weights()` requires every spatial key and rejects unexpected keys; only the five new temporal-related module prefixes may be missing.

## Query token and temporal branch
Within each flattened query, softmax(objectness logits) weights all fine features and all spatial XYZ predictions. Their weighted XYZ is embedded by 3→32→128 GELU at a fixed /100m scale and added to the weighted feature before LayerNorm. No hard TopK selection enters pooling.

Empty queries use a learned missing-LiDAR token without fabricated candidates. A two-entry presence embedding distinguishes observation absence from temporal padding. Observation absence does not mask attention keys.

TimeEncoding embeds [query_time - clip_start, query_time - previous_query] in seconds through 2→32→128 GELU (first delta is zero). Timestamp subtraction uses float64 before casting; no Unix timestamp enters the MLP directly.

Two independent Pre-LN causal temporal blocks use dim128, four heads, FFN256 and zero dropout. Mask is zero for j<=i and -inf for j>i. Padding has no influence on valid queries; padded rows are handled safely to avoid all-masked softmax. Missing observations remain valid temporal positions. TemporalXYZHead is 128→64→3 GELU and outputs absolute XYZ, independently of spatial candidate predictions.

## Data and supervision
TemporalQueryClipDataset creates same-sequence rolling histories of at most eight consecutive queries, stride one. Sequence-start prefixes are right-padded; no event/query duplication. Collation flattens B×T spatial queries and preserves query-valid, target-valid and scoring masks. Training supervises all valid queries in each clip; validation scores only the endpoint, so each CSV reference contributes exactly once, including empty observations.

Spatial focal/SmoothL1 labels and normalization are unchanged: positive recent distance <=1m; ignore nonpositive all-history distance <=2m; negative >2m. No-positive queries skip spatial supervision. TemporalPositionLoss averages SmoothL1(beta1) over XYZ and every valid target query, including NO_CURRENT_SUPPORT. Total = spatial loss + temporal loss (weight1). Thus missing current evidence does not discard temporal supervision.

The latest-four marker is stored as `supervision_recent_mask` and is consumed
only by spatial label construction and evaluation grouping. It is not a learned
model feature. Learned point timing uses continuous `delta_t`; temporal query
timing uses the continuous query-time encoding. See
`V2_RECENT_INPUT_CONTRACT.md` for the enforced data contract.

## Public interfaces
- `LiDARQueryBuilder.build(sequence_id, query_time, ...)`: optional targets.
- `build_query_history(builder, sequence_id, query_times)`: GT-free last-eight query history.
- `LiDARUAVDetector(batch)`: spatial outputs plus query_token, temporal_hidden, temporal_pred_xyz, presence/validity/time metadata.
- `QueryCausalLoss(outputs, batch)`: spatial and temporal losses separately.
- `tools/infer_lidar_uav_v2.py`: GT-free query-history inference.
- V2 train/evaluate/export tools use temporal clips and retain spatial metrics, adding temporal Success@0.5/1/2m and mean/median/P90/P95 error in ALL/current-support/no-current-support groups.

Default future run directory: outputs/own_multimodal_research/lidar_uav_v2/query_causal_v1/. Structure artifacts are in query_causal_v1_structure_smoke/. V1 and prior V2-base equivalence artifacts remain untouched.

## Verification (structure only)
Spatial parameters: 1,047,722. Added parameters: 282,723. Total: 1,330,445.

13 query/temporal tests and 9 spatial regressions pass. The environment lacks pytest; unittest runs the query tests and the structure checker invokes existing spatial regression functions directly. py_compile also passes.

Frozen V2-base commit: 38f939a7c4be8215a5f53ae5d7d0d84f29679c56. Real-sample spatial logits/residual/XYZ/features/centers max differences are all zero. Only added modules are missing during spatial weight loading. Future-LiDAR change: 0; future-GT change: 0; future-query removal: 7.450580596923828e-09 (FP32).

Real seq0001 eight-query CPU no_grad check: 338,058 valid input points; L0/L1/L2 totals 69,443/27,996/10,846. Query token and temporal hidden are [1,8,128]; temporal XYZ is [1,8,3]. Every query has 20 past events; future event count=0. A real midpoint query without targets also passes. Per-query timestamps/counts are recorded in structure_report.json. Random temporal predictions are not reported as localization results.

V1 frozen code/shared-reader/split SHA256 checks all pass. No optimizer/scheduler update, training epoch or checkpoint optimization was executed. Synthetic backward is used only to verify gradient flow (including NO_CURRENT_SUPPORT); real-data smoke is eval/no_grad.

## Limits and intentionally absent features
No GT interpolation/jitter, fixed one-second window, velocity/acceleration, RGB/radar, trajectory/consistency loss, bidirectional attention, KV cache or altered spatial architecture. No training/convergence claim. CUDA training memory and performance have not been measured; eight-query spatial activation cost may be substantial. DDP is explicitly unsupported by this V2 entry rather than silently running independently. Future training requires separate authorization. Causal histories alone do not establish improved prediction under missing observations; that requires a later trained experiment.
