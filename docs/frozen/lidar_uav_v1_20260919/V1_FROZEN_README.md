# LiDAR UAV Transformer V1 — Frozen 2026-09-19

This directory records the immutable V1 baseline contract before any V2 work.

## Data contract

- Training: `/home/jasoncui/datasets/MMAUD/official/train` using sequence `ground_truth`.
- Validation LiDAR: `/home/jasoncui/datasets/MMAUD/official/val`.
- Validation GT: `/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv`.
- `t0` is the exact GT timestamp. The input is the latest 20 events after merging Avia and Mid360 streams, restricted to `event_timestamp <= t0`; events are never duplicated and future events are forbidden.
- Point inputs are XYZ, sensor ID, and event delta time. NaN, Inf, and confirmed `[0,0,0]` padding are removed. Denoising is disabled.
- The most recent four selected events define recent support.

## Model contract

V1 uses 0.5/1/2 m sparse levels, D=128, two spatial blocks per level, local L0/L1 attention, true per-sample global L2 attention, adjacent SparseUp, and a fine-token candidate head. It has **1,047,722 parameters**. Outputs are objectness logit, XYZ residual/prediction, source token ID, and a 128D candidate feature.

## Supervision and loss

Positive means `d_recent <= 1m`; ignore means non-positive with `d_all <= 2m`; negative means `d_all > 2m`. `loss_cls` is objectness focal loss and `loss_reg` is XYZ SmoothL1. Total loss is `loss_cls + 2 * loss_reg`. The validation CSV `Classification` field means UAV model class 0/1/2/3. It is retained only as metadata and **does not enter CandidateLoss**.

## Frozen result status

`FORMAL_100_EPOCH_TRAINING_NOT_FOUND`. The available baseline is the completed 3-epoch seed-42 pilot. Its selected checkpoint metric was NMS Recall@10@1m = 0.931875; reloading the frozen best checkpoint produced 0.933750. The best checkpoint is `pilot_3epoch_seed42/best.pt` inside the frozen result snapshot.

The original result directory remains unchanged. Its reflink/copy snapshot is `outputs/own_multimodal_research/lidar_uav_v1_frozen_20260919/`; `RESULTS_SHA256.txt` verifies all 89 source files and the critical checkpoint/metric hashes.
