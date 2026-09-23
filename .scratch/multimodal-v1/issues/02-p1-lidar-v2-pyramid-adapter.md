# P1 — LiDARV2PyramidAdapter（Stage Adapter + Numerical Identity）

Status: ready-for-agent
Blocked by: (none — 可与 00 并行)
Type: task

## Goal

把 `LiDARUAVDetector.spatial_forward()` 的一口气执行拆成可插 HCI 的阶段接口，形成 LiDARV2PyramidAdapter，并通过严格数值等价验收。这是 multimodal_v1 的第一道实现门。

## Non-goals

- 不重写/不改动任何 LiDAR V2 数学（SBE/VQSA/EOOE/Transformer/SparseUp/CandidateHead）。
- 不实现 HCI（哪怕 identity 之外的空壳）。
- 不碰视觉侧。

## Files / modules expected to touch

- 新建 `src/rdq_uav/multimodal_v1/registry.py`、`contracts.py`
- 新建 `src/rdq_uav/multimodal_v1/radar/lidar_v2_adapter.py`
- 新建 `tests/test_multimodal_v1_p1_identity.py`
- 只读复用：`src/rdq_uav/lidar_v2/`（全部）

## Dependencies

无（与 00 并行）。**P1 FAIL 时禁止开始 05（P5 HCI）**。

## Frozen constraints

- V1.1 §2 / §21-01：LiDAR V2 数学不大改，只做 Stage Adapter。
- V1.1 §16：旧 `lidar_v2/` 只读复用，多模态逻辑不得反向污染。

## Implementation notes

- Adapter 接口（spec Solution 第 1 条）。**关键约束：merge01 / merge12 / SparseUp / CandidateHead 都依赖原始 SparseHierarchy 的拓扑与缓存——同一次 `prepare()` 构建出的 hierarchy 必须贯穿传递到链路末端，每个方法都接收 ctx；禁止任何 stage 重新构造 hierarchy**（重新构造 = 布局漂移 = 等价性破坏）：

  ```
  ctx = prepare(batch)          # 内含 hierarchy + R0_pre
  R0_post = run_stage0(ctx.R0_pre, ctx)
  R1_pre  = merge01(R0_post, ctx)
  R1_post = run_stage1(R1_pre, ctx)
  R2_pre  = merge12(R1_post, ctx)
  R2_post = run_stage2(R2_pre, ctx)
  fine    = decode_to_fine(R0_post, R1_post, R2_post, ctx)
  out     = candidate_head(fine, ctx)   # logits / residual / pred_xyz
  ```

- ctx 是 hierarchy 及各级索引/中间结构的载体（不是重新计算的入口）；HCI 插入点（P5）也在同一 ctx 上操作，保证交互后的特征与原 hierarchy 结构一致。
- 内部模块直接复用原 `LiDARUAVDetector` 实例的方法/子模块，不复制代码。
- Registry 骨架在本 ticket 建立（name→class），后续 ticket 只往里注册。

## Acceptance criteria

同一 batch、同一 state_dict、FP32、eval 模式下：

LiDARV2PyramidAdapter ≡ LiDARUAVDetector，逐项 max_abs_diff ≈ 0：
- logits
- residual_xyz
- pred_xyz
- fine_features
- voxel_centers（以及 token ids / sparse layouts 完全一致）

## Tests

`tests/test_multimodal_v1_p1_identity.py`：
- 用小规模真实或合成 batch，双模型同权重跑前向，五项输出逐项 allclose（fp32 容差）。
- 随机化若干 batch（不同稀疏度/voxel 分布）重复断言。
- **单次 prepare 的 hierarchy 贯穿断言**：ctx 中的 hierarchy 对象 identity（或其布局缓存）在 stage0→candidate_head 全链为同一份；逐 stage 比较 voxel/token 布局一致。
- Prior art：`tests/test_lidar_uav_v2_correctness_gate.py` 的范式。

## Artifacts / reports expected

- 等价性验证报告（简短：每项 max_abs_diff 表格）可并入 PR 描述。

## Stop condition

- 任何一项数值不一致且根因指向"必须改 lidar_v2 数学才能 stage 化" → 停止，开 prerequisite ticket，记录冲突，不改 spec。

## Comments
