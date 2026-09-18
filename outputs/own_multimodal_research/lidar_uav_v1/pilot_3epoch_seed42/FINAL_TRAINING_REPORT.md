# LiDAR UAV V1 Formal Training Report

- Device: NVIDIA GeForce RTX 3070 Laptop GPU
- PyTorch: 2.1.1+cu118
- CUDA: 11.8
- Attention backend: explicit PyTorch QK matmul + FP32 softmax + additive axis bias
- Train samples: 28600
- Validation GT rows: 1600
- Validation schema: Sequence, Timestamp, Position, Classification
- t0: exact GT timestamp from sequence ground_truth (train) or validation reference CSV (val)
- Future event count: 0
- Parameters: 1,047,722
- Best epoch: 3
- Best NMS Recall@10@1m: 0.931875
- Peak GPU memory: see metrics.csv per epoch
- Best checkpoint: /home/jasoncui/projects/rdq-uav-experiment/outputs/own_multimodal_research/lidar_uav_v1/pilot_3epoch_seed42/best.pt
- Candidate features: /home/jasoncui/projects/rdq-uav-experiment/outputs/own_multimodal_research/lidar_uav_v1/pilot_3epoch_seed42/candidate_exports/candidate_features.npz
- Denoise: false
- L2 attention: global
- Architecture fallback: none

See `metrics.csv`, `best_validation_metrics.json`, and `per_sequence_metrics.csv` for full results.
