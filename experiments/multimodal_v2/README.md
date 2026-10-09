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
