# V2 sequence isolation audit and hardening

## Finding
The previous TemporalQueryClipDataset was already sequence-isolated: it grouped
records by sequence_id, sorted timestamps inside each group, and built rolling
prefixes there. Spatial voxel keys at L0/L1/L2 already include flattened sample
identity. Temporal attention already operates on [B,H,T,T], not [1,H,B*T,B*T].
Arbitrary-query inference already takes one explicit sequence and has no global
query-state list or KV cache. No actual cross-sequence contamination was found.

## Hardening
- A shared assert_temporal_clip_integrity validates one sequence, nonempty identity,
  strictly increasing finite query times, unique sample identity, explicit selected
  event sequence membership and event_timestamp<=query_time. Errors include clip
  index, sequences, sample IDs and times. Metadata-only requests can omit events;
  loaded queries, collate and precheck require event metadata.
- Dataset validates records at construction and loaded clips at access, catching
  both bad metadata and a reader returning a different sequence.
- LiDARQueryBuilder.select_events checks each actual LidarFrameEvent.sequence_id,
  not just the stream dictionary key. Returned queries retain event_sequence_ids.
- Collate keeps stable sequence_id/sample_id/clip_batch_index/clip_position. The
  model validates these identities and restores [B,T] explicitly with clip indices,
  never by timestamp. Each temporal row must represent exactly one sequence.
- Arbitrary history validates the full requested ordered history before truncating
  to its last eight queries, and rejects mixed-sequence builder results.
- The causality mask remains unchanged. Missing observation is still a valid query;
  only padding is invalid. No SBE, attention mathematics, loss or supervision change.

## Full metadata audit
Every clip was traversed, checking every selected event reference; no point arrays
were needed for this metadata audit. Repeated references in rolling windows count
repeatedly. There is no event/frame encoding cache or stride optimization.

| Split | Sequences | Clips | Selected event references |
|---|---:|---:|---:|
| Frozen train_sub | 72 | 28,600 | 4,416,168 |
| Official validation CSV | 16 | 1,600 | 220,632 |

Both splits: cross_sequence_clips=0, unsorted_clips=0,
duplicate_timestamp_clips=0, empty_sequence_id=0, duplicate_sample_identity_clips=0,
event_sequence_mismatch=0, event_path_sequence_mismatch=0, future_event_count=0.
No duplicate timestamp needed an exception to strict ordering.

## Tests
11 isolation tests + 13 Query-Causal tests + 14 SBE tests pass, as do nine spatial
regressions and py_compile. Test details and machine-readable full audit are in
outputs/own_multimodal_research/lidar_uav_v2/sequence_isolation_audit/.

- B=2,T=4 temporal token perturbation: seqA max difference=0.
- Identical timestamps in seqA/seqB, changing seqB points: seqA max difference=0.
- Same XYZ in separate flattened samples: separate L0/L1/L2 tokens, PASS.
- Rolling validation resets at sequence boundary, PASS.
- Mixed arbitrary history rejected, PASS.
- Explicit clip-index restoration under permuted flat query identity, PASS.
- Future LiDAR/GT perturbations: max difference=0.
- Future query removal: max difference=1.1175870895385742e-08.
- Empty observations and NO_CURRENT_SUPPORT temporal gradient: PASS.

Within the audited public builder/dataset/collate/model pipeline, there is no
sequence-A to sequence-B temporal state propagation path. A raw tensor-only
attention block cannot infer semantic sequence labels by itself; the model boundary
therefore enforces that each attention batch row is a single sequence before calling
it. The two independent conditions are sequence identity and causal time ordering.

No optimizer/scheduler steps, training epochs or parameter optimization occurred.
Synthetic backward only executes existing gradient regression tests. V1 remains
unchanged. No stride experiment or temporal cache was introduced.
