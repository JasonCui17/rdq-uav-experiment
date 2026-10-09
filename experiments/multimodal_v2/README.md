# Multimodal V2

V2's own data, radar SBE/backbone, candidate builders, Swin/DINO adapters, geometry, interaction, scoring, losses and training code live in `src/rdq_uav/multimodal_v2`. Detrex/DINO/Swin remain third-party code under `third_party/detrex`.

## Stages

| Stage | Training supervision | Initialization | Evaluation |
| --- | --- | --- | --- |
| B0 | Radar's all-input-point 3D candidate loss | Random | 3D success and recall |
| B1 | DINO criterion on the selected image's YOLO box | Full COCO-pretrained DINO; newly initialized single-class UAV heads | 2D top-1 IoU ≥ 0.5 |
| B2 | Radar ranking with vision evidence | This run's B0 and B1 best checkpoints | 3D and 2D |
| B3 | Bidirectional candidate interaction and ranking | This run's B0 and B1 best checkpoints | 3D and 2D |

B0 starts from random initialization. By default, B1 loads the full COCO-pretrained DINO checkpoint specified by `initialization.dino_checkpoint` into the original 80-class model, then replaces its classification heads with newly initialized single-class UAV heads and fine-tunes on MMAUD. Backbone, transformer and box-regression weights are retained during head adaptation. If `dino_checkpoint` is omitted or null, B1 starts without pretrained weights. A configured checkpoint that is missing or incompatible raises an error; there is no silent fallback to random initialization.

Pretrained initialization loads model weights; `--resume` continues a training run by restoring its training state. B2/B3 require the `b0_checkpoint` and `b1_checkpoint` paths from this run. Review the trained B0/B1 metrics before starting a formal B2/B3 run. Historical random-initialization B1 results and new COCO-initialized runs must be recorded separately; current defaults do not establish how an earlier run was initialized.

B1 training now restores detrex DINO's native contrastive denoising (CDN):
view-normalized targets are prepared before the transformer, passed to
`prepare_for_cdn`, and supervised through `dn_post_process` and the native
criterion's `dn_meta`. The regular decoder, auxiliary decoder and encoder
losses remain enabled. GT-derived denoising tokens are removed from both
prediction tensors and query features before candidate selection. Missing
annotations contribute neither regular detection nor denoising losses.
Validation/inference never receive GT queries; B2/B3 use the existing ordinary
candidate forward without CDN. Third-party source files are unchanged.

`loss.box_positive_iou` and `loss.box_ignore_iou` control V2 candidate ranking,
not B1's native Hungarian detection assignment. B1's `val/2d_iou50` is a fixed
evaluation metric, not a training or inference filter. No IoU curriculum is
enabled. This restores the DINO training core, not an identical COCO training
protocol: the MMAUD data preparation, optimizer/schedule and V2 Top-K/NMS
postprocessing remain project-specific. Start a new output directory without
`--resume` to compare this CDN-restored run against the previous B1 run.

```bash
PYTHONPATH=src python experiments/multimodal_v2/train.py --config experiments/multimodal_v2/configs/b0_standalone.yaml
PYTHONPATH=src python experiments/multimodal_v2/train.py --config experiments/multimodal_v2/configs/b1_standalone.yaml
PYTHONPATH=src python experiments/multimodal_v2/evaluate.py --config experiments/multimodal_v2/configs/b0_standalone.yaml --mode B0 --checkpoint outputs/own_multimodal_research/multimodal_v2/b0_seed42/checkpoints/best.ckpt --output outputs/own_multimodal_research/multimodal_v2/b0_eval
PYTHONPATH=src python experiments/multimodal_v2/evaluate.py --config experiments/multimodal_v2/configs/b1_standalone.yaml --mode B1 --checkpoint outputs/own_multimodal_research/multimodal_v2/b1_seed42/checkpoints/best.ckpt --output outputs/own_multimodal_research/multimodal_v2/b1_eval
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/audit_dataset.py --config experiments/multimodal_v2/configs/b0_standalone.yaml --output /tmp/v2_dataset_audit.json
```

The 20 annotated sequences use `splits/mmaud_v2_annotated20_seed42.json`: 14 train, 3 validation, and 3 held-out test sequences. Training and checkpoint selection use train and validation only. Evaluation defaults to `--split validation_sub`; pass `--split heldout_test_sub` for final test evaluation after model selection. The audit reads the three groups without loading a model or checkpoint and counts B0 supervision only when a valid 3D target has an input radar point within 1 m.

After reviewing those evaluations, train B2 or B3 with `b2_radar_reads_vision.yaml` or `b3_bidirectional.yaml`. Each training run saves `best.ckpt` and `last.ckpt` under its output directory; use `--resume auto` or `--resume PATH` only to continue that run. `--max-updates 2 --train-limit 16 --val-limit 2 --accelerator cpu --devices 1 --precision 32-true --num-workers 0` provides a small CPU smoke when the model dependencies permit it.

For a focused real-data smoke, training accepts `--train-indices` and `--val-indices`; evaluation accepts `--indices`. These select dataset indices after the configured split and cannot be combined with the corresponding `--train-limit`, `--val-limit`, or `--limit` option.

The dataset keeps QueryRecord separate from optional 3D GT. Same-sequence radar history and the nearest historical image use `[t-1s,t]`; relative times are observation minus query. The YOLO label uses the selected image stem. Batch fields pack only observed modalities and retain original sample indices. Missing modalities skip their network path. Radar SBE uses a 0.2s time half-life and its loss supervises every input point. Candidate budgets are pre-NMS 50 and final at most 10 per modality. Interaction and association share the distance-adaptive geometry gate.

Historical E0/E5 diagnostics and earlier reports remain in this directory for reproduction. The former B0/B1 zero-head identity check is now `diagnostics/check_interaction_identity.py`; it is a diagnostic, not either trainable baseline.

## Paper evaluation: XYZ RMSE and image-level COCO AP

Evaluation now retains the existing `summary.json` and adds `paper_metrics.json`,
`paper_metrics.csv`, and enriched `per_query.jsonl` (GT/predicted XYZ, selected image,
GT box, all final 2D boxes and scores). Training adds validation XYZ/3D RMSE logs;
losses and architecture are unchanged; see the live-monitoring section below for
B1 checkpoint selection. Existing checkpoints
can be re-evaluated without training again. Old per-query files lack XYZ/boxes and
must first be regenerated with the updated evaluation entry point.

For N valid-GT queries and M finite Top1 outputs, axis RMSE is
`sqrt(sum((prediction_axis - GT_axis)^2) / M)`; 3D RMSE is
`sqrt(sum(||prediction - GT||^2) / M)`. Thus squared 3D RMSE equals the sum of
squared axis RMSEs. RMSE is conditional on an output and must be reported alongside
`coverage=M/N`, `missing_queries` and Success@0.5/1/2m. Success uses all N queries,
including missing outputs as failures. No output means null RMSE, never zero.
B1 has no metric 3D output and leaves all meter RMSE columns empty.
XYZ axes use the existing GT frame; physical axis directions remain to be verified.

Install the official COCO evaluator without reinstalling PyTorch:

```bash
python -m pip install 'pycocotools>=2.0.7,<3'

PYTHONPATH=src python experiments/multimodal_v2/evaluate.py \
  --config experiments/multimodal_v2/configs/b0_standalone.yaml --mode B0 \
  --checkpoint outputs/b0_formal_seed42/checkpoints/best.ckpt \
  --split heldout_test_sub --output outputs/b0_formal_seed42/eval_best_heldout_paper

PYTHONPATH=src python experiments/multimodal_v2/evaluate.py \
  --config experiments/multimodal_v2/configs/b1_standalone.yaml --mode B1 \
  --checkpoint outputs/b1_formal_seed42/checkpoints/best.ckpt \
  --split heldout_test_sub --output outputs/b1_formal_seed42/eval_best_heldout_paper
```

Use the config corresponding to each experiment/checkpoint, including A1/A2.
Validation can use `--split validation_sub`; select the checkpoint on validation,
then use heldout only for final reporting. Runtime data/calibration/third-party
paths must already be configured as for previous evaluations.

COCO reports AP (IoU 0.50:0.05:0.95), AP50, AP75, AP_small and AR100 as fractions
(multiply by 100 for percentages). It evaluates **final V/RV candidates**, currently
usually at most 10 per image, with official maxDets=[1,10,100]. AR100 does not mean
100 predictions were supplied. This is final-pipeline AP, not AP of all raw DINO
queries. Only images with a valid box are evaluated: missing/empty label files
are not assumed to be verified negative images. AP_small uses area in source
left-view pixels, not resized input pixels.

Repeated query records selecting the same sequence/image are counted once for AP.
Keep the query closest to image time; break ties by query time and sample ID,
without looking at prediction quality. `coco/` contains GT, predictions and chosen
query records for audit. Existing query-level Top1 IoU metrics remain unchanged;
they are different from image-level AP. `--skip-coco` explicitly disables AP.

Day/night is disabled unless reliable labels are supplied with `--conditions`.
The JSON schema is `{"sequences": {"sequence_id": "day or night"},
"samples": {"sample_id": "day or night"}}`: replace the values with exactly
`day` or `night`; per-sample overrides per-sequence. Every evaluated query must
have a label. Never infer conditions from absolute timestamps or assign defaults.
Overall RMSE pools all query errors; `macro_mean_day_night_rmse_m` separately
averages the two group RMSEs and is not the pooled RMSE. Empty groups stay null.
`--method` labels the CSV method; `--bandwidth` accepts an independently measured
value with units. Bandwidth and day/night columns remain blank when unavailable.

After generating enriched records, tables can be recomputed without GPU/model:

```bash
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/export_paper_metrics.py \
  --input outputs/b0_formal_seed42/eval_best_heldout_paper/per_query.jsonl \
  --mode B0 --split heldout_test_sub --output outputs/b0_paper_table
```

State conditional RMSE, coverage, split, image deduplication and candidate scope
in the paper. Results on different datasets/splits are not automatically comparable.

## Live training and validation monitoring (2026-10-09)

The progress bar shows stage-specific metrics instead of displaying B1's undefined
3D errors. Each training batch reports its **actual optimized loss**, learning rate,
CUDA peak reserved memory (GiB), and native loss components. B0 shows weighted
classification/regression; B1 shows weighted final classification, L1 box and GIoU,
plus disjoint CDN (`dn`), auxiliary decoder (`aux`), encoder (`enc`) and other terms.
These components sum to the existing loss; the loss definition is not changed.
`loss_avg` averages processed optimization batches weighted by batch size; skipped
loss batches are flagged and excluded. Peak reserved memory is a lifetime peak,
not current allocation.

Progress labels:

- `T/rx,ry,rz,r3,Cov,S1,N`: current-epoch **training** cumulative XYZ/3D RMSE,
  output coverage, Success@1m and GT query count, updated every batch. They use
  changing model weights/training mode and are diagnostics, not validation results.
  Missing outputs count in Coverage/Success, not RMSE. The accumulator stores sums
  of squared errors and counts rather than averaging batch RMSEs. CSV/TensorBoard
  also receive Success@0.5m/2m and output counts.
- `V~/...`: provisional results for the already processed part of the **current
  validation pass**, not full-validation results. Radar updates each batch; COCO
  AP refreshes every 50 validation batches by default. AP is re-evaluated on the
  cumulative, deduplicated image set; batch APs are never averaged.
- `V/...`: completed validation metrics, recomputed at the end of the pass. The
  complete summary is printed on a separate line and written to existing loggers.
- `lastV/AP,AP50,AP75,AP_small,AR100`: last completed visual validation results shown
  during training. They are not recomputed using each training batch. Before the
  first complete validation no last-validation metric is fabricated.

AP is a fraction (0.3 means 30%). Undefined metrics display `--`; B1 has no XYZ RMSE.
The COCO scope and image selection are identical to the paper-evaluation section:
only valid labeled images, final candidate boxes (typically <=10/image).

B1 now selects best.ckpt using **val/AP (IoU 0.50:0.95)** rather than Top1 IoU50;
val/2d_iou50 remains logged for older-result comparisons. Radar B0 and B2/B3 retain
val/success_1m selection. Start the new B1 run in a fresh output directory; changing
an old run's checkpoint monitor during resume does not preserve its model-selection
history. This patch neither changes model parameters nor trains a visual 3D head.

Default validation is still once per epoch. To obtain more frequent fixed-weight
validation results, add the following options to your existing train.py command:

```bash
--val-check-interval 0.25 --ap-every-val-batches 50
```

`0.25` runs the full configured validation split at approximately each quarter of
an epoch, increasing validation cost. AP can update every validation batch with
`--ap-every-val-batches 1`, but repeatedly sorting/evaluating the cumulative image
set costs CPU time. No extra train-mode forward pass is added for radar diagnostics.
The current implementation is verified for the project's single-device protocol;
DDP/global metric aggregation is not implemented or claimed here.
