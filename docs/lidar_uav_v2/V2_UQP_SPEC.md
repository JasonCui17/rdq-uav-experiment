# UQP-v1 — Unique Query Packing

## Purpose and exact semantics

EQS reduces how many overlapping clips enter an epoch. UQP removes a different
redundancy: identical queries repeated inside the same DataLoader batch execute
SBE, Spatial Transformer, CandidateHead, and CandidateAwareQueryPool once.

A typical stride-4, clip-batch-2 case is:

```
clip A: q1 q2 q3 q4 q5 q6 q7 q8
clip B:             q5 q6 q7 q8 q9 q10 q11 q12

16 query occurrences -> 12 unique spatial queries
```

The shared spatial token for q5 is computed once. Its temporal states remain
separate: in clip A it sees q1…q5, while in clip B it sees only q5. UQP never
deduplicates temporal hidden states, predictions, or losses.

## Stable identity and packing

Train and validation records have a sequence-local `query_uid`. Formal dedup key
is `(sequence_id, query_uid)`; timestamp alone is never an identity. Train UIDs
are ground-truth record ordinals inside each sequence. Validation UIDs are
assigned after sequence/time sorting. Compatibility queries fall back to stable
sample IDs, still paired with sequence identity.

Collation retains B×T occurrence metadata and constructs:

- `occurrence_to_unique [B*T]`, with -1 only for padding;
- U unique spatial samples and point_batch_index in 0…U-1;
- per-unique target metadata and occurrence count;
- original clip_batch_index, clip_position, sequence/sample IDs and times.

Repeated identity is accepted only when sample ID, query time, selected event
timestamps/sequence IDs, points, sensor IDs, delta_t and recent mask are exactly
equal. A conflicting UID fails closed. Unique query tokens are gathered by the
inverse map, then scattered using explicit clip batch/position—not timestamp—into
`[B,T,128]` before the unchanged causal temporal network.

Empty real queries remain unique temporal observations using the existing missing
LiDAR token. Padding is never made a unique query. Sequence isolation and causal
time checks run before model forward.

## Exact spatial-loss weighting

The reference path treats every repeated query as a separate supervised spatial
sample. UQP computes per-unique-query focal and regression losses, then gathers
them back in occurrence order through `occurrence_to_unique` and averages over
supervised occurrences. Thus a shared query appearing twice retains weight two.
NO_CURRENT_SUPPORT occurrences remain excluded from spatial loss with their old
multiplicity, while every valid temporal occurrence retains temporal supervision.

No detach, feature cache, CPU roundtrip, or cross-forward state is used. Gradients
from repeated spatial-loss occurrences and all temporal contexts accumulate into
the single shared spatial graph. No feature survives an optimizer update.

## Overlap-aware batching

`OverlapAwareBatchSampler` wraps the existing epoch-cyclic sampler. It preserves
the selected index set exactly, groups indices per sequence by anchor ordinal,
forms adjacent same-sequence batches, pools only incomplete per-sequence leftovers,
and deterministically shuffles complete batch groups with `seed + epoch`.
Cross-sequence leftovers are ordinary independent samples and cannot deduplicate.
Resume uses the absolute one-based epoch. No selected clip is copied, dropped, or
borrowed across trajectories.

Default training config enables UQP and overlap-aware batching with
`dedup_key: sequence_query_uid`. Setting `enabled: false` keeps the reference
duplicated compute path. Validation continues on the reference collation path and
all 1,600 endpoints.

## Point materialization boundary

The current Dataset loads each query before collate, so repeated occurrences can
read/materialize raw point arrays twice. UQP then packs only unique points for the
model, reducing SBE/Spatial compute and its input tensor. It does **not** claim
disk-I/O deduplication. It also does not remove overlap between different queries'
latest-20 histories, such as q8=e1…e20 versus q9=e2…e21.

There is no event encoder cache, incremental voxelization, epoch cache, global
autograd cache, recurrent voxel map, or temporal KV cache.

## Audited results

For real epoch-1 stride-4 metadata:

- 7,150 selected clips in 3,575 batches;
- 56,480 query occurrences;
- 42,396 unique spatial queries;
- 14,084 removed duplicate spatial occurrences;
- 24.9363% batch-local spatial-query reduction;
- all 3,575 batches are same-sequence overlap batches; mixed leftovers=0.

The compute funnel is 226,784 dense stride-1 spatial queries → 56,480 EQS
occurrences → 42,396 UQP queries. Dense-to-UQP reduction is 81.3056%; this is a
query-count estimate, not an equal claim about wall time or event I/O.

One real typical batch has 241,478 points materialized by Dataset/reference and
179,195 points in the UQP model input. It maps 16 occurrences to 12 unique queries.

Reference versus UQP FP32 differences are exactly zero for logits, XYZ, fine
features, query tokens, temporal hidden/output and all five losses. Representative
gradient maximum absolute differences range from zero to 7.45058e-9; shared
multi-context gradient aggregation differs by zero.

CPU/FP32, four threads, three warmups and 20 repeats on the same real batch:

| Path | Median full forward |
|---|---:|
| Reference duplicated | 6137.12 ms |
| UQP | 4088.72 ms |

Observed speedup is 1.501×. CUDA was unavailable, so GPU peak memory is not
reported. This benchmark measures model forward only, after point loading/collate.

All UQP, EQS, sequence isolation, Query-Causal, SBE and spatial regressions pass
(65 tests total). Future LiDAR/GT invariance differences remain zero; future-query
removal difference is 1.11759e-8. Planned updates remain 178,800 with 8,940 warmup.
No optimizer/scheduler step, training epoch, or checkpoint optimization occurred.
