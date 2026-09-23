# P5 — Geometry-guided Hierarchical Bi-HCI（冻结主方法）

Status: ready-for-agent
Blocked by: none（P1/P2 已 PASS；P4 已 engineering-closed with known limitation，原 formal FAIL 保留）
Type: task

## Goal

实现冻结主交互方法 Geometry Bi-HCI：每级 stage 前（Pre-Stage）执行 Projection → Geometry Sampling → Local Cross-Attention → Interaction Residual Gate → Residual。

## Non-goals

- 不实现 LatentBridge 的主路径化（LatentBridge 只允许在独立的 baseline/debug ticket 中存在）。
- 不加 deformable attention、不加 FFN、不加 self-attention 到 HCI 内（§21-08/§18）。
- 不做 alignment 消融变体（shifted/full-hierarchy 属后续消融 backlog）。

## Files / modules expected to touch

- 新建 `src/rdq_uav/multimodal_v1/interaction/geometry_local.py`（主方法）
- 新建 `src/rdq_uav/multimodal_v1/interaction/identity.py`（正式组件，E2 对照用）
- 新建 `src/rdq_uav/multimodal_v1/interaction/bi_hci.py`（组装：Projection→Sampling→CA→Gate→Residual）
- 新建 `tests/test_multimodal_v1_p5_hci.py`

## Dependencies

- 02（P1：LiDAR stage 接口）
- 03（P2：视觉 stage 接口）
- **05（P4）**——原 preregistered formal gate FAIL 保留；经人工 bbox 审计确认当前 geometry 足以提供 coarse local prior，已 engineering-close 并放行 P5。精细映射作为后续独立优化问题，不再阻塞本 ticket。
- 阻塞下游：07（P6 候选装配）。

## Frozen constraints（全部实现期不可改）

- Pre-Stage 位置（§21-03）。
- interaction_dim=128（§21-05）；视觉 native channel 仅在 HCI 内投影/反投影。
- local visual window=3×3，K=9（§21-07）。
- 双向从同一组原始 R/V **并行**计算（§21-04）：V→R 每个 Radar token 读其投影位置附近 3×3 visual tokens；R→V 只更新 Radar-supported sparse visual locations，不做全图密集 R→V attention（物理不对称，Q11/Q12）。
- token-wise scalar gate，near-zero init（§21-08）；残差回原模态，不覆盖。
- 默认 alignment：R0↔V0、R1↔V1、R2↔V2（same-level，§21-10）。
- HCI 只含五个组成（Q14）。

## Implementation notes

- 投影 Π 使用 P4 audit 通过的 transform/标定 handle（从 InteractionContext 传入，不硬编码）。
- gate bias 初始化为负值 → 初始模型≈单模态 backbone，保护预训练特征。
- modality-valid mask（m_R/m_V）贯穿：**任一模态缺失时，所有依赖该模态的 cross-modal residual 置 0；现存模态只继续自己的 backbone，不接受 HCI 更新**。即 m_V=0 时 Radar 与 Vision 的 HCI residual 均为 0（V→R 无从发生，Radar 不被更新）；m_R=0 同理。缺失侧特征走数据契约的 missing 路径，不伪造。

## Acceptance criteria

- synthetic backward：随机输入梯度存在且 finite。
- 真实 batch 前向 finite。
- gate 关闭（near-zero 初值下）时输出≈各自单模态特征（近等价，非严格——这是行为检查不是 identity seam）。
- tiny subset 可 overfit（T0 条件）。

## Tests

- Contract 测试（非数值 identity seam）：shape 保持、finite、梯度、**mask 行为：任一模态缺失时，所有依赖该模态的 cross-modal residual 置 0；现存模态只继续自己的 backbone，不接受 HCI 更新**——m_V=0 时 Radar 与 Vision 的 HCI residual 均为 0（没有 Vision，V→R 也无法发生，Radar 不被更新）；m_R=0 同理。
- near-zero gate 初值下输出与输入差范数 < 阈值。
- 投影索引正确性：合成标定数据下 Radar token 只 attend 其投影邻域内 visual tokens。

## Artifacts / reports expected

- gate 初值行为检查记录（overfit tiny subset 的 loss 曲线）。

## Stop condition

- 若 P5 实际训练/真实 batch 证明 3×3/K=9 无法有效利用 UAV visual region，则停止并回到独立 alignment/neighborhood 优化；不得在本 ticket 内静默改大窗口。
- 任何"加个 FFN 就能收敛"的冲动 → §18 边界，backlog。

## Comments
