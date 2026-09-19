# LiDAR UAV V2 recent4 input contract

LiDAR UAV V2 uses continuous time in its learned path. Point-level model inputs
are XYZ, `sensor_id`, continuous `delta_t = event_timestamp - query_time`, and
packed `point_batch_index`. SBE-Lite derives latest age and temporal standard
deviation from `delta_t`; Query-Causal uses continuous query-time encoding.

The latest four selected events remain a supervision and evaluation concept.
The dataset marks their points with `supervision_recent_mask`. CandidateLoss
uses that metadata to compute `d_recent` and preserves the frozen labels:

```text
d_recent <= 1m                         Positive
d_recent > 1m and d_all <= 2m          Ignore
d_all > 2m                             Negative
no Positive                            NO_CURRENT_SUPPORT
```

Validation uses the same metadata for CURRENT_SUPPORT/NO_CURRENT_SUPPORT and
recent-neighbor grouping. Temporal loss continues to supervise every valid
target, including NO_CURRENT_SUPPORT queries.

`supervision_recent_mask` is not read by `LiDARUAVDetector`, SBE-Lite, EOOE,
PositionEncoding, spatial attention, CandidateHead, CandidateAwareQueryPool,
Continuous Time Encoding, or the causal Temporal Transformer. It is not a
learned feature, binary time embedding, attention bias, or recent/old token.

The boolean tensor is currently transferred to the model device because
CandidateLoss computes point-to-GT `d_recent` there and validation grouping
shares those tensors. This preserves the existing label and metric mathematics;
it does not make the mask a neural-network input.

The contract is regression-tested by deleting the supervision mask before
model forward and requiring exact equality of spatial logits/XYZ/features,
query tokens, temporal hidden states, and temporal XYZ. A frozen pre-cleanup
label implementation verifies exact label, loss, and representative gradient
equivalence. No optimizer or scheduler step is involved.
