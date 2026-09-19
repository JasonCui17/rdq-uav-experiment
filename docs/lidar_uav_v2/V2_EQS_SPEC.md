# EQS-v1 — Epoch-Cyclic Query Subsampling

## Scope

EQS reduces repeated temporal clips per training epoch without changing any
model, loss, query, event history, validation sample, or clip contents. The four
data levels are:

| Level | Definition |
|---|---|
| Event | One Avia or Mid360 LiDAR frame |
| Query | One query_time and its latest 20 merged causal events |
| Clip | Up to eight consecutive same-sequence queries, temporal stride one |
| Batch | B complete clips; batch_size counts clips, not queries |

EQS stride applies only to selecting complete clips. It does not stride events,
does not change latest-20 selection, and does not turn an eight-query clip into
`q1,q5,q9...`. All valid query slots in every selected clip retain spatial and
temporal supervision. NO_CURRENT_SUPPORT behavior remains unchanged.

## Dataset and sampler

`TemporalQueryClipDataset` remains the full dense dataset and exposes lightweight
`clip_metadata`: dataset index, sequence, sequence-local anchor ordinal, anchor
time/ID, valid slot count, and underlying query indices. No point array is loaded
to choose a clip.

`EpochCyclicQuerySampler` uses one-based absolute epoch numbers:

```
offset = (epoch - 1) % stride
selected iff sequence_local_anchor_ordinal % stride == offset
```

Ordinals restart at zero for each sequence. `set_epoch(17)` gives the same set
and deterministic order whether epochs 1–16 ran in this process or training was
resumed directly. After selection, optional shuffling uses a fresh CPU generator
seeded with `base_seed + epoch`. Short sequences may contribute zero clips for an
offset; nothing is copied or borrowed across sequences.

Training constructs the dataset with dense stride one, supplies the sampler and
sets DataLoader `shuffle=False`. Validation has no EQS sampler and retains all
1,600 official CSV endpoints. Sequence-isolation assertions remain active.

## Scheduler planning and logging

`planned_epoch_stats` evaluates every absolute epoch before scheduler creation:

```
batches_e = ceil(selected_clips_e / clip_batch_size)
updates_e = ceil(batches_e / accumulation)
total_updates = sum_e updates_e
```

This matches the existing end-of-epoch partial accumulation update. It estimates
updates from DataLoader batches; it does not load point clouds or predict
NO_CURRENT_SUPPORT. Warmup/cosine use the subsampled total. Resume restores the
saved scheduler state and uses the absolute resumed epoch for sampler offset.

Terminal progress includes stride, offset and selected/full clip count. Epoch JSON
and CSV records also include selection ratio and valid query slots. Effective
batch semantics remain two clips/GPU and accumulation two by default: at most
32 query slots per optimizer update.

## Full-data audit

The frozen train split contains 72 sequences, 28,600 dense clips, 28,600 source
queries, and 226,784 dense valid query slots.

| Stride | Offset selected clips | Ratio | Valid slots by offset |
|---:|---:|---:|---|
| 1 | 28,600 | 100% | 226,784 |
| 2 | 14,300 | 50% | 113,248; 113,536 |
| 4 | 7,150 | 25% | 56,480; 56,624; 56,768; 56,912 |
| 8 | 3,575 | 12.5% | 28,096; 28,168; 28,240; 28,312; 28,384; 28,456; 28,528; 28,600 |

Stride-1 appearances per underlying query are mean 7.9295, median/P90/max 8/8/8.
Stride-4 per-epoch means are 1.9748, 1.9799, 1.9849, and 1.9899;
median/P90/max are 2/2/2 for every offset. Across four epochs, every anchor is
selected exactly once and every underlying target query is supervised at least
once: both coverage ratios are 100%, with zero missing or duplicate anchors.

For 100 epochs, clip batch two and accumulation two, planned optimizer updates are
178,800; warmup at 5% is 8,940. This is a dry-run calculation: no scheduler step,
optimizer step, backward training update, or epoch was executed.

## Verification and boundaries

Ten EQS tests, 11 sequence-isolation tests, 13 Query-Causal tests, 14 SBE tests,
and nine spatial regressions pass (57 total), along with py_compile. Tested cases
include stride 1/2/4/8, full stride-one equivalence, epoch cycle, resume order,
dense clip context, clip-batch semantics, short sequences, four-epoch coverage,
future invariance, same-timestamp sequence isolation, arbitrary queries, empty
LiDAR, and NO_CURRENT_SUPPORT temporal gradients.

No model module or loss was changed. This version adds no event/feature cache,
incremental voxelization, temporal KV cache, alternate stride schedule, epoch
offset augmentation, or non-overlapping clip construction.

Machine-readable results and per-sequence/query tables are under
`outputs/own_multimodal_research/lidar_uav_v2/query_subsampling_audit/`.
