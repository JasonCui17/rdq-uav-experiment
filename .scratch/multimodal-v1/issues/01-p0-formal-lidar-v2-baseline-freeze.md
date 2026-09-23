# P0 — Formal LiDAR V2 Baseline Freeze (E0)

Status: ready-for-agent
Blocked by: (none — 可与 01 并行)
Type: task

## Goal

正式训练并冻结 E0 LiDAR V2 baseline，形成多模态实验不可悄悄改变的 3D 单模态对照基准。记录完整的可复现性信息。

## Non-goals

- 不修改 LiDAR V2 数学、监督定义、数据协议（顺手优化 = 隐性收益，禁止）。
- 不做多模态任何内容。
- 不调参冲分——用既定 config 正式训练一次并冻结结果。

## Files / modules expected to touch

- `tools/train_lidar_uav_v2.py`（调用，原则上不修改；若需 batch/fp16 工程适配见下）
- `configs/lidar_uav_v2.yaml`（生成 effective config 副本，不改原文件语义）
- `runs/`、`weights/`、`results/`（输出产物）

## Dependencies

无。与 01（P1 adapter）可并行：本 ticket 在服务器/GPU 侧跑训练，01 在开发侧写代码。

## Frozen constraints

- V1.1 §2：SBE-Lite/VQSA/EOOE/SpatialTransformer/SparseUp/CandidateHead/CandidateLoss/CandidateSelector 全部只读复用。
- V1.1 §21-24：E0 是冻结矩阵第一项，其结果定义 LiDAR baseline，后续不得悄悄改变。

## Implementation notes

- 显存约束（3070 8GB WSL）：小 batch + AMP 是 V1.1 §20 预案内工程手段，允许；但 optimizer/scheduler/监督定义不得因显存改动语义。
- 若训练脚本本身有阻塞 bug：**停下来开 prerequisite ticket**，不顺手改 `lidar_v2/` 数学。

## Acceptance criteria

存在一个冻结记录（建议 `results/e0_baseline_freeze.md` + checkpoint），包含全部：git commit、effective config、data split、seed、precision、optimizer/scheduler、checkpoint 路径 + hash、validation metrics（Recall@K within 0.5/1/2m、Top1 Success@1m、XYZ mean/median/P90/P95）、训练命令、环境信息（Python/torch/CUDA/GPU）。

## Tests

- 无新代码则无新测试；若写了任何辅助脚本，走 contract 测试风格。
- 断言 checkpoint 可被 `tools/evaluate_lidar_uav_v2.py` 加载并复现冻结 metrics（±数值精度）。

## Artifacts / reports expected

- 冻结 checkpoint（如 `weights/e0_lidar_v2_best.pt`）
- `results/e0_baseline_freeze.md`（上述全部记录字段）
- effective config 快照

## Stop condition

- 训练需要改 `lidar_v2/` 数学或监督定义才能跑通 → 停止，开 prerequisite ticket，记录冲突。
- 显存不足以任何 batch size 完成 → 停止，开 engineering-risk ticket（gradient accumulation 等工程方案在 ticket 内决策，不改监督）。

## Comments
