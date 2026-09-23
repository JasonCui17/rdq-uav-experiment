# P2 — Shared DINO-Swin-T Pyramid Adapter（第二数值等价 seam）

Status: ready-for-agent
Blocked by: (none — 原则上依赖 registry 骨架，可与 02 同批)
Type: task

## Goal

建立唯一在线视觉链路：RGB → Swin-T → V0/V1/V2/V3 → HCI 接口 + DINO detection head → RGB candidates 基础设施。Swin-T backbone 同时服务 HCI 与 DINO 检测头，一套权重、一次前向。

## Non-goals

- 不实现 HCI 本身（只暴露 stage 前后接口）。
- 不做 RGB candidate 的最终装配（属 06/P6）。
- 不引入 YOLO 到任何主路径。
- 不另起第二套在线视觉 backbone（冻结禁止项）。

## Files / modules expected to touch

- 新建 `src/rdq_uav/multimodal_v1/vision/swin_adapter.py`
- 新建 `src/rdq_uav/multimodal_v1/vision/dino_adapter.py`
- 只读参考：`src/rdq_uav/models/backbones.py`、`src/rdq_uav/calibration/omni.py`
- 新建 `tests/test_multimodal_v1_p2_vision_identity.py`

## Dependencies

- 弱依赖 01（P1）的 registry/contracts 骨架（可同批并行，先写模块后注册）。
- 阻塞下游：04（P3 数据）、05（P5 HCI）、06（P6 候选）。

## Frozen constraints

- V1.1 §21-02：RGB 主干 Swin-T，主 detector 为共享 backbone 的 DINO-Swin-T。
- V1.1 §4.1/§21-11：RGB candidate 必须由 HCI-enhanced Swin/DINO 路径产生；YOLO 仅 RGB-only baseline。
- V1.1 §21-05：interaction_dim=128；视觉保留 native channels，仅 HCI 内投影。
- 层级：V0(/4)、V1(/8)、V2(/16) 对齐 R0/R1/R2；V3(/32) 保留给 DINO 高层语义。

## Implementation notes

- **Reference 固定（等价性前提）**：当前仓库没有现成 DINO-Swin-T 链路，因此必须先固定一个成熟 DINO-Swin-T reference implementation + checkpoint（如官方 DINO/DINO-style detector 的 Swin-T 变体，具体选型记录在本 ticket 评论，加载方式与权重版本冻结）。identity 验收的对象就是这个 reference。
- **真 stage boundary，不是只读中间特征**：SwinPyramidAdapter 必须暴露真实 stage 边界——HCI 修改后的 V_l_pre 能继续进入原 Swin stage 和后续 DINO head（即逐 stage 可回写地执行，不是 `features_only=True` 一次性跑完再读中间特征）。
- Adapter 显式暴露各 stage 前后特征（V0_pre/V0_post…），HCI 插入点为 stage 前。
- DINO head 挂在共享 Swin 输出之上（V3 或按 DINO 结构需要的多层），其 queries/boxes 输出预留为 RGB candidate 源。
- Native channel ↔ 128D 的投影属 HCI 内部（05），本 ticket 只保证接口能承接。
- DINO 依赖/显存问题 → engineering-risk ticket，不换 detector 路线。

## Acceptance criteria

- 先固定 reference implementation + checkpoint 并记录选型；Identity 路径（无 HCI 注入）下，adapter 与该 reference 的 **Swin stage outputs 与 DINO detection outputs** 数值/shape 一致（第二 numerical identity seam）。
- **Stage 可回写验证**：把某一 V_l_pre 原样传回 adapter 的 stage 执行路径，输出与不拦截的完整前向一致；HCI 修改 V_l_pre 后能继续进入原 Swin stage 与 DINO head（本 ticket 验证回写机制本身，交互语义属 05）。
- 一次前向同时产出 HCI 可用的 V0–V3 stage 特征与 DINO 检测输出——证明无第二 backbone。
- Config 中 candidate source 显式（`dino_query`），无隐藏 YOLO 路径。

## Tests

- Identity 数值等价：同输入同权重（reference checkpoint），adapter 逐 stage 执行的 Swin outputs 与 DINO detection outputs vs reference 直接前向，allclose。
- Stage 回写：拦截 V_l_pre 原样传回 → 输出与完整前向一致（证明真 stage boundary）；替换 V_l_pre 为扰动张量 → 后续 stage 正常消费（证明可回写）。
- Shape/stride 契约：V0–V3 分辨率与 channel 数符合 Swin-T 结构。
- 单 backbone 断言：前向计数（或参数集合）证明视觉分支只有一套 Swin。

## Artifacts / reports expected

- 无独立报告；identity 验证结果并入 PR。

## Stop condition

- 发现 DINO 检测头无法挂在共享 Swin 层级上而"必须"改 Swin 结构或加第二 backbone → 停止，开 prerequisite ticket 记录冲突（这触碰 §21-02/11，不得自行决定）。
- 显存不足以载入 Swin-T+DINO → engineering-risk ticket（AMP/UQP 属预案内）。

## Comments
