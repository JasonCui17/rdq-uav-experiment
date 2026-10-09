# B1 DINO training core restoration

Base: `refactor-v2-standalone`, `b11bcf6`.

## Code change

- Preserve the existing COCO checkpoint loading, single-class head adaptation,
  shared Swin, four-scale transformer and project batch contract.
- In B1 training, build compact visual targets before the transformer. Reuse
  detrex `prepare_for_cdn`, its attention mask, `dn_post_process`, and the
  criterion with `dn_meta`, including regular, auxiliary and encoder losses.
- Remove denoising tokens from logits, boxes and decoder features before
  candidate selection. No GT-generated token can become a V2 candidate.
- Exclude unannotated images from both regular and denoising losses.
- Validation/inference and B2/B3 receive no GT-derived queries.
- No third-party source, radar code, ranking loss, or evaluation threshold was changed.

This restores the missing CDN training path in the detrex-based implementation.
It is not a byte-identical port of IDEA-Research/DINO, nor an identical COCO
training protocol. MMAUD preprocessing, optimizer/schedule, and V2 Top-K/NMS
remain as before. Single-class label corruption cannot introduce another object
class; native positive/negative box denoising still operates.

## IoU decision

The B1 detection loss is native Hungarian assignment plus classification,
L1 and GIoU losses. `box_positive_iou=0.5` belongs to V2 candidate ranking,
not B1 detection assignment; ranking diagnostics do not replace B1's optimized
loss. Validation IoU50 is a measurement, not a candidate filter.

An epoch-based IoU curriculum was not implemented. IoU>0 still rejects nearby
non-overlapping tiny boxes and accepts oversized boxes containing the GT.
Adding a center-distance AND condition reduces some false positives but does
not rescue zero-overlap boxes or constrain predicted size. Keep Hungarian
matching for the restored baseline. If tiny-target assignment remains a
problem, compare a separate soft center-and-size cost (e.g. NWD), rather than
changing assignment, evaluation, and inference thresholds together. Ground
truth-based conditions are never valid inference filters.

References: DINO https://arxiv.org/abs/2203.03605 ; ATSS
https://arxiv.org/abs/1912.02424 ; NWD https://arxiv.org/abs/2110.13389 .
ATSS/NWD's published anchor-detector results do not establish gains for this
DINO/MMAUD implementation; transfer requires an independent experiment.

## Validation

CPU, Python 3.12, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu,
Lightning 2.6.6: V2 tests 104 passed, 1 CUDA test skipped. Compile and
diff checks passed. CDN unit tests use a detector double; real detrex CUDA
forward/backward, numerical parity, memory use and accuracy are not validated
here. Recheck on the server's PyTorch 2.1.2/CUDA 11.8 environment before formal
training. Start a fresh output directory; do not resume an old no-CDN run.
