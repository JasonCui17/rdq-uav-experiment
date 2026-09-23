# Multimodal V1 — Ticket Index & Dependency Graph

Source of truth: `.scratch/multimodal-v1/spec.md` + V1.1 §21 冻结清单（不可覆盖）。
纪律：工程问题 → prerequisite / risk / fallback ticket；不改 frozen architecture。

## Tickets

| # | Ticket | Phase | Type |
|---|---|---|---|
| 01 | [P0 Formal LiDAR V2 Baseline Freeze](01-p0-formal-lidar-v2-baseline-freeze.md) | P0 | task |
| 02 | [P1 LiDARV2PyramidAdapter](02-p1-lidar-v2-pyramid-adapter.md) | P1 | task |
| 03 | [P2 Shared DINO-Swin-T Adapter](03-p2-shared-dino-swin-adapter.md) | P2 | task |
| 04 | [P3 Multimodal Data Contract + UQP](04-p3-multimodal-data-contract-uqp.md) | P3 | task |
| 05 | [P4 LiDAR→Image Geometry Correctness Gate](05-p4-lidar-image-geometry-correctness-gate.md) | P4 | audit |
| 06 | [P5 Geometry Bi-HCI（主方法）](06-p5-geometry-bi-hci.md) | P5 | task |
| 07 | [P6 Dual Candidates + Frozen Association](07-p6-dual-candidates-association.md) | P6 | task |
| 08 | [P7a c2D/c3D Target Contract](08-p7a-c2d-c3d-target-contract.md) | P7a | 契约设计 |
| 09 | [P7b Reliability Gate + SharedQuery + Decoder](09-p7b-reliability-gate-shared-query-decoder.md) | P7b | task |
| 10 | [P8 Training Pipeline + E0–E5](10-p8-training-pipeline-e0-e5.md) | P8 | task |
| 11 | [F1 LatentBridge Baseline/Fallback](11-fallback-latent-bridge-baseline.md) | fallback | baseline |
| 13 | [A1 Post-Strongest Ablation Backlog](13-post-strongest-ablation-backlog.md) | 消融 | backlog |

（编号 12 已移除——YOLO11 RGB-only baseline ticket 不再单列；YOLO 仅作为 §21-11 冻结约束下的对照概念存在于各 ticket 的 Frozen constraints 中，如需正式跑对照再开票。）

## Dependency Graph

```
01 (P0 E0 baseline) ─────────────────────────────┐
   ∥ (并行)                                        │
02 (P1 LiDAR adapter) ──┬──────────────┐          │
                        │              │          │
03 (P2 DINO-Swin) ──┬───┼──┐           │          │
                    │   │  │           │          │
                    ▼   │  │           ▼          │
        04 (P3 data+UQP)│  └──► 06 (P5 HCI) ◄─ 05 (P4 gate PASS)
                    │   │        │    ▲              ▲
                    │   │        │    └── P4 FAIL ──► prerequisite
                    │   │        ▼                   (修标定，06 保持阻塞)
                    │   │   07 (P6 dual+assoc) ◄── 06, 03
                    │   │        │
                    │   │        ▼
                    │   │   08 (P7a target contract) ◄── 07
                    │   │        │
                    │   │        ▼
                    │   └──► 09 (P7b gate+query+decoder) ◄── 07, 08
                    │            │
                    │            ▼
                    └──► 10 (P8 training + E0–E5) ◄── 01, 09
                                 │
                                 ▼
                            13 (A1 ablations, backlog)

11 (F1 LatentBridge baseline) ◄── 02, 03   （独立，不在主线；仅 debug/baseline）
```

关键边：
- 01 ∥ 02（P0 与 P1 并行：服务器训练 / 开发侧实现）
- 02 → 06；03 → 04/06/07；04 → 05；05 PASS → 06；06+03 → 07；07+08 → 09；01–09 → 10
- 08（P7a）是 09（P7b）的 mandatory prerequisite：L_valid 的 target contract 批准前不得实现 generic BCE。
- 05（P4）有两阶段阈值冻结流程门：先提交固定 PASS/FAIL 阈值，再首次正式运行。

## 首批可并行执行

**Batch 1（立即可开，三条并行线）：**
1. **01（P0）**：E0 LiDAR V2 正式训练——GPU 侧长任务，先启动。
2. **02（P1）**：LiDARV2PyramidAdapter + identity 等价——开发侧，无依赖。
3. **03（P2）**：DINO-Swin 共享 adapter——开发侧，弱依赖 02 的 registry 骨架（可先写模块；reference 选型需用户确认）。

**Batch 2（Batch 1 部分完成后）：**
- 04（P3）← 03；11（F1）可随时穿插（debug 用）。

## 变更纪律备忘

发现标定/数据/显存/依赖/checkpoint/接口问题时的唯一出口：
1. prerequisite ticket（修前置）；
2. engineering-risk ticket（记录 + 预案内工程手段）；
3. fallback/debug ticket（baseline 身份，不升主方法）。

禁止：GeometryLocal→LatentBridge 主方法化、DINO→YOLO 主方法化、geometry gate→feature-only 主方法化、第二套在线 RGB backbone、任何 §21 冻结项修改。
