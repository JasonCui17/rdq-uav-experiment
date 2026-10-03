# Multimodal V2

V2's own data, radar SBE/backbone, candidate builders, Swin/DINO adapters, geometry, interaction, scoring, losses and training code live in `src/rdq_uav/multimodal_v2`. Detrex/DINO/Swin remain third-party code under `third_party/detrex`.

## Stages

| Stage | Training supervision | Initialization | Evaluation |
| --- | --- | --- | --- |
| B0 | Radar's all-input-point 3D candidate loss | Random | 3D success and recall |
| B1 | DINO criterion on the selected image's YOLO box | Random | 2D top-1 IoU ≥ 0.5 |
| B2 | Radar ranking with vision evidence | This run's B0 and B1 best checkpoints | 3D and 2D |
| B3 | Bidirectional candidate interaction and ranking | This run's B0 and B1 best checkpoints | 3D and 2D |

B0 and B1 do not load any project or third-party pretrained checkpoint. Only an explicit `--resume` restores a training run. B2/B3 require the `b0_checkpoint` and `b1_checkpoint` paths in their configs. Review the trained B0/B1 metrics before starting a formal B2/B3 run.

```bash
PYTHONPATH=src python experiments/multimodal_v2/train.py --config experiments/multimodal_v2/configs/b0_standalone.yaml
PYTHONPATH=src python experiments/multimodal_v2/train.py --config experiments/multimodal_v2/configs/b1_standalone.yaml
PYTHONPATH=src python experiments/multimodal_v2/evaluate.py --config experiments/multimodal_v2/configs/b0_standalone.yaml --mode B0 --checkpoint outputs/own_multimodal_research/multimodal_v2/b0_seed42/checkpoints/best.ckpt --output outputs/own_multimodal_research/multimodal_v2/b0_eval
PYTHONPATH=src python experiments/multimodal_v2/evaluate.py --config experiments/multimodal_v2/configs/b1_standalone.yaml --mode B1 --checkpoint outputs/own_multimodal_research/multimodal_v2/b1_seed42/checkpoints/best.ckpt --output outputs/own_multimodal_research/multimodal_v2/b1_eval
```

After reviewing those evaluations, train B2 or B3 with `b2_radar_reads_vision.yaml` or `b3_bidirectional.yaml`. Each training run saves `best.ckpt` and `last.ckpt` under its output directory; use `--resume auto` or `--resume PATH` only to continue that run. `--max-updates 2 --train-limit 16 --val-limit 2 --accelerator cpu --devices 1 --precision 32-true --num-workers 0` provides a small CPU smoke when the model dependencies permit it.

For a focused real-data smoke, training accepts `--train-indices` and `--val-indices`; evaluation accepts `--indices`. These select dataset indices after the configured split and cannot be combined with the corresponding `--train-limit`, `--val-limit`, or `--limit` option.

The dataset keeps QueryRecord separate from optional 3D GT. Same-sequence radar history and the nearest historical image use `[t-1s,t]`; relative times are observation minus query. The YOLO label uses the selected image stem. Batch fields pack only observed modalities and retain original sample indices. Missing modalities skip their network path. Radar SBE uses a 0.2s time half-life and its loss supervises every input point. Candidate budgets are pre-NMS 50 and final at most 10 per modality. Interaction and association share the distance-adaptive geometry gate.

Historical E0/E5 diagnostics and earlier reports remain in this directory for reproduction. The former B0/B1 zero-head identity check is now `diagnostics/check_interaction_identity.py`; it is a diagnostic, not either trainable baseline.
