# LiDAR UAV Transformer V2-base

V2-base contains **no algorithmic innovation**. It is an independent source copy of the frozen V1 baseline so future architecture work cannot contaminate V1.

## Frozen behavior

The data, supervision, loss, architecture, selector, optimizer, and evaluation definitions are unchanged. Inputs remain the latest 20 causal events from the merged Avia/Mid360 stream at exact GT time `t0`; recent means the last four events. Voxel scales remain 0.5/1/2 m, D remains 128, L2 attention remains global, and denoising remains disabled. `loss_cls` is objectness focal loss derived from GT/point distances; validation CSV `Classification` remains UAV-model metadata and is excluded from loss.

Only `model.name` differs between `configs/lidar_uav_v1.yaml` and `configs/lidar_uav_v2.yaml`. The V2 CLI uses an independent default output root: `outputs/own_multimodal_research/lidar_uav_v2/`.

## Paths

- Model package: `src/rdq_uav/lidar_v2/`
- Config: `configs/lidar_uav_v2.yaml`
- Train: `tools/train_lidar_uav_v2.py`
- Evaluate: `tools/evaluate_lidar_uav_v2.py`
- Export: `tools/export_lidar_uav_v2_candidates.py`
- Regression tests: `tests/test_lidar_uav_v2.py`

No V2 training was started. All future model ideas must be introduced in later commits after this V2-base checkpoint.
