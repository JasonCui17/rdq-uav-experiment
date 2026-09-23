# P3 — Multimodal Data Contract + UQP 扩展

Status: ready-for-agent
Blocked by: 03（P2 视觉链路，弱依赖——数据结构可先行设计，联调需 03）
Type: task

## Goal

定义并实现多模态数据契约：每个 unique query 绑定 LiDAR events + left RGB + 时间戳 + 标定 handle + 模态有效 mask，UQP 语义同时覆盖两模态，使每个 unique query 的双 backbone 各只运行一次。

## Non-goals

- 不做 geometry audit（05/P4）。
- 不改 LiDAR V2 侧已有 data 语义（EQS/UQP/sequence isolation 原样复用）。
- 不做右鱼眼。

## Files / modules expected to touch

- 新建 `src/rdq_uav/multimodal_v1/data.py`
- 只读复用：`src/rdq_uav/lidar_v2/data.py`（UQP/EQS/sequence isolation）
- 参考：`src/rdq_uav/calibration/spatiotemporal.py`（time_offset_s ≈ -0.124153668）
- 新建 `tests/test_multimodal_v1_p3_data_contract.py`

## Dependencies

- 03（视觉链路存在才能联调 UQP-on-RGB）。
- 阻塞下游：05（P4 geometry audit 需要时间绑定数据）。

## Frozen constraints

- V1.1 §21-21：第一版只使用 left RGB。
- V1.1 §21-11 共享 backbone 原则的工程面：UQP 扩展到 RGB，同一 unique query 不重复跑视觉 backbone。
- Query-Causal 约束：所有点时间戳 ≤ query_time；RGB 图像按 nearest/matched policy，显式记录 gap。

## Implementation notes

- **时间偏移公式（写死，防符号用反）**：仓库约定 `query_time = image_time + time_offset_s`（time_offset_s ≈ -0.124153668）。因此匹配 left RGB 时按 **`gap = query_time - (image_time + time_offset_s)`** 选 |gap| 最小的图像。**禁止**直接拿 image_time 与 query_time 做 nearest（漏掉 offset 等于把符号/量级用错）。
- **UQP 唯一键（写死）**：沿用 LiDAR V2 实际使用的复合键 **`(sequence_id, query_uid)`**。不得只用 query_uid——不同 sequence 的 query_uid 可能碰撞。occurrence→unique 映射、RGB 复用、缓存全部以该复合键索引。
- 契约字段：unique key (sequence_id, query_uid)、LiDAR events（packed）、left RGB 图像或其引用、image_time、query_time、gap、sequence_id、calibration handle、m_R/m_V 有效 mask、occurrence→unique 映射（同时映射 LiDAR 与 RGB）。
- 不把"同文件序号"当天然同步；gap 阈值策略显式可配并在报告中说明。
- 缺失 RGB（左相机无匹配图像）→ m_V=0，不伪造图像。

## Acceptance criteria

- 任一 unique query：LiDAR backbone 与视觉 backbone 各恰好前向一次（可由计数断言）。
- 无跨 sequence 样本；所有 LiDAR 点时间戳 ≤ query_time；时间 gap 分布有报告。
- 缺失模态样本走显式 mask 路径。

## Tests

- UQP 唯一性：重复 spatial query 的 occurrence 映射正确；**构造两个不同 sequence 含相同 query_uid 的样本，断言它们是两个不同 unique query**（复合键防碰撞）。
- 时间匹配：合成时间轴上断言 gap 按修正公式计算（offset 符号正确——用已知 offset 构造"正确图像 gap=0、错误图像 gap=|offset|"的判别样本）；选中的 RGB 恒为 |gap| 最小者。
- 时间泄漏：构造边界样本断言无未来信息。
- Sequence isolation：同 batch 无跨 sequence。
- 契约字段完整性（shape/dtype）。
- Prior art：`tests/test_lidar_uav_v2_query_causal.py`、`test_lidar_uav_v2_spatial_query_pipeline.py`。

## Artifacts / reports expected

- 时间对齐统计报告（gap 分布、匹配率、缺失率）。

## Stop condition

- 发现 MMAUD 官方数据的时间戳/图像语义与 spatiotemporal 标定假设矛盾 → 停止，开 calibration/data-semantics prerequisite ticket（这正是 P4 要审的问题，提前暴露更好）。

## Comments
