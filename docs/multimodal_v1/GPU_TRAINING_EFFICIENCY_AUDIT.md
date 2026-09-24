# Multimodal V1 GPU training efficiency audit

Date: 2026-09-24
Baseline commit: `c2e2ea5c6d6c0bff25e040b11ffd8b11b703b26f`

This audit follows the `optimize-for-gpu` workflow: preserve the scientific
contract, measure representative inputs, remove host/device synchronization
from hot paths, and require numerical/selection equivalence tests.

## Finding

The refactor did not duplicate the LiDAR or DINO forward. It already improved
the path by decoding/resizing images in the Dataset, transferring image data as
`uint8`, caching calibration tensors, reusing P5/DINO outputs for all three
losses, and suppressing diagnostic tensor allocation during training.

The annotated Lightning configuration still selected zero DataLoader workers,
so the new preprocessing path ran serially on the training process. On 48 real
queries, input-only throughput was:

| Workers | Queries/s | Elapsed (48 queries) |
|---:|---:|---:|
| 0 | 10.44 | 4.599 s |
| 2 | 18.39 | 2.611 s |
| 4 | 30.51 | 1.573 s |

The formal default is now two workers. This is the conservative WSL2 setting;
four workers were faster in isolation but increase prefetch and host-memory
pressure.

Two candidate-selection loops were also GPU synchronization hot spots:

- RGB NMS executed up to 100 Python iterations over CUDA tensors. It now uses
  fused `torchvision.ops.nms`, with synthetic priority scores preserving the
  old stable score/source order, including tied scores.
- LiDAR radius suppression synchronized once per candidate. It now computes
  the same pairwise FP distance predicate in one device region, transfers one
  at-most 100x100 boolean matrix, and runs the unchanged greedy ordering on it.

CPU microbenchmarks over representative 100-candidate inputs were:

| Hot path | Before | After | CPU speedup |
|---|---:|---:|---:|
| RGB NMS | 9.267 ms | 0.082 ms | 113.4x |
| LiDAR radius suppression | 0.835 ms | 0.508 ms | 1.64x |

These local timings do not predict end-to-end GPU speedup. The expected CUDA
benefit comes primarily from removing sequential launches and host
synchronizations. The current agent environment reports `cuda_available=false`
and `GPU access blocked by the operating system`, so a synchronized end-to-end
GPU benchmark remains required on the user's WSL terminal.

## Correctness

- Stable RGB NMS output identity, including score ties: PASS.
- LiDAR raw/NMS source-token identity against the legacy loop: PASS.
- Worker image preparation identity and cached calibration identity: PASS.
- Multimodal, Lightning, and spatial-query regression suite: 74 passed, 3
  subtests passed.
- Model architecture, losses, optimizer grouping, stage schedule, batch size,
  accumulation, and precision policy were not changed.

## RTX 3070 gate

The 20-update gate completed 80 micro-batches in about 86 seconds (0.98
iterations/s in Lightning's progress display). It peaked at 4.70 GiB allocated
and 6.20 GiB reserved, with no OOM or non-finite loss.

Checkpoint inspection exposed a separate FP16 correctness issue: the default
scale of 65536 overflowed six initial optimizer attempts. Lightning reported
global step 20 while the main active parameters had optimizer step 14, and the
scheduler had already advanced to step 20. The entry point now creates an
explicit scaler with initial scale 1024, the stable scale reached by the gate.
A repeated gate must confirm that optimizer and scheduler steps remain aligned.

The repeated gate passed: all 149 continuously active parameters reached
optimizer step 20, the scaler stayed at 1024 for all 20 updates, and the
scheduler ended at step 20. Four conditional 2D box-head parameters received
gradients only on the first update; this is branch activity rather than an AMP
skip. The run completed 80 micro-batches in 83 seconds of displayed training
time (1.02 iterations/s), with unchanged 4.70/6.20 GiB allocated/reserved
peaks.

The CPU microbenchmarks above remain local hot-path measurements; they are not
being reported as end-to-end GPU speedups.

## Method reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
