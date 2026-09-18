# LiDAR UAV V1 Formal Training Readiness

## Status

The formal training entry is ready for a manual single-GPU launch. No 100-epoch run was started by Codex.

## Validation reference

- File: `/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv`
- Exact columns: `Sequence`, `Timestamp`, `Position`, `Classification`
- Rows: 1,600 valid / 1,600 total
- Sequences: 16 matched / 16 referenced
- Duplicate `(Sequence, Timestamp)` keys: 0
- Missing or invalid reference rows: 0
- Timestamp: parsed as float64 seconds and matched to LiDAR filename timestamps
- XYZ: parsed from `Position` as three float values; unit is not declared by the CSV and remains `CODE_ASSUMED_METER_FROM_EXISTING_MMUAV_CONVENTION`
- GT rows with no historical LiDAR event: 21
- GT rows whose merged input has no valid LiDAR points: 23
- GT rows with fewer than 20 historical events: 310

All 1,600 valid GT rows remain in validation. Empty inputs count as failed predictions.

## Precheck

- Train source: `sequence_ground_truth`
- Validation source: `validation_ref_csv`
- Train samples: 28,600 from the frozen `train_sub` split
- Validation samples: 1,600
- Deterministic timing checks: 32 train + 32 validation samples
- `t0 == gt_timestamp`: PASS
- Future events: 0
- Model parameters: 1,047,722
- Denoise: false
- L2 attention: global

## Tests and smoke

- Python compilation: PASS
- Existing V1 regression functions: 8/8 PASS
- Formal-training adapter/scheduler tests: 4/4 PASS
- GPU smoke: 4 optimizer updates, including checkpoint resume
- Device: NVIDIA GeForce RTX 3070 Laptop GPU
- Precision: BF16
- Batch/GPU: 2
- Accumulation: 2
- Peak allocated GPU memory: 0.97 GiB
- NaN/Inf/OOM/future leakage: none
- Full 100-epoch training started: no

The four-sample smoke validation is an execution check only and is not a model-quality result.

## Manual command

```bash
cd /home/jasoncui/projects/rdq-uav-experiment
/home/jasoncui/miniconda3/envs/rdq/bin/python tools/train_lidar_uav_v1.py \
  --config configs/lidar_uav_v1.yaml \
  --train-root /home/jasoncui/datasets/MMAUD/official/train \
  --val-root /home/jasoncui/datasets/MMAUD/official/val \
  --val-reference "/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv" \
  --epochs 100 \
  --device 0
```

Reliable DDP is not implemented in this V1 CLI, so there is no supported dual-GPU command. The entry rejects a multi-process launch instead of silently running incorrect duplicated training.
