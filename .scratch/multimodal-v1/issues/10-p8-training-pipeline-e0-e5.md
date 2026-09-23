# P8 — Training Pipeline (T0–T3) + E0–E5 Experiment Matrix

Status: ready-for-agent
Blocked by: 00（P0 E0 baseline）、09（P7b 全链可用）
Type: task

## Goal

搭建多模态训练管线：

T0 correctness
→ T1 fusion warm-up
→ T2 fusion training
→ T3 final fine-tune

并跑完冻结实验矩阵 E0–E5。

**第一优先级：取得 strongest model。**

## Non-goals

* 严谨消融（alignment / direction / router / gate / candidate source / hypothesis type / modality dropout）不阻塞本 ticket，strongest model 完成后再做。
* 不改任何 P1–P7b 已验收模块的语义。
* 发现模块问题时开 prerequisite / engineering-risk ticket，不顺手改 frozen architecture。
* E3/E4 之外不做额外方向消融。
* 不在 E0–E5 中插入额外新模块。

## Files / modules expected to touch

* 新建 `src/rdq_uav/multimodal_v1/training.py`
  或复用既有训练框架，视仓库现有 engine 结构决定。
* 新建 `tools/train_multimodal_v1.py`
* 新建 `tools/evaluate_multimodal_v1.py`
* 新建 `configs/multimodal_v1/`

  * `e0_lidar.yaml`
  * `e1_rgb.yaml`
  * `e2_late_fusion.yaml`
  * `e3_single_hci.yaml`
  * `e4_hierarchical_hci.yaml`
  * `e5_full_v1.yaml`
  * T0–T3 schedule 配置
* 新建 `tests/test_multimodal_v1_p8_pipeline.py`

## Dependencies

* 00（P0）：提供冻结 E0 LiDAR V2 baseline。
* 09（P7b）：提供完整可训练多模态模型。
* E1 只依赖视觉链路完成，可与 E2–E5 的后续准备并行。

## Frozen constraints

### 1. 初始化权重必须统一

为了保证 E1–E5 可比：

* E1–E5 的视觉分支必须统一使用同一份固定的 DINO-Swin-T reference / pretrained checkpoint。
* E2–E5 的 Radar 分支必须统一从 P0 冻结的 E0 LiDAR V2 checkpoint 初始化。
* 禁止不同实验使用不同 backbone 初始化权重。
* 所有 checkpoint 来源必须写入对应实验 config / run metadata。

因此：

```text
E0:
P0 frozen LiDAR checkpoint

E1:
fixed DINO-Swin-T pretrained/reference checkpoint

E2–E5:
Radar ← same P0 E0 checkpoint
Vision ← same DINO-Swin-T checkpoint
```

除实验定义本身要求的模块开关外，初始化条件保持一致。

### 2. T0–T3 主要用于 E2–E5 多模态训练

V1.1 §12.3 / Q38：

#### T0 — Correctness

验证：

* identity / contract；
* finite；
* synthetic backward；
* tiny subset overfit。

多数 correctness 已在 P1–P7b 分别验证，本 ticket 做完整模型集成复验。

#### T1 — Fusion warm-up

冻结：

* Swin/DINO 主体；
* 尽量保持 LiDAR baseline。

主要训练：

* HCI；
* interaction projections / adapters；
* Reliability Gate；
* SharedQuery；
* Decoder；
* fusion heads。

目标：

让新增融合模块先学会使用两模态特征，不立即破坏预训练 backbone。

#### T2 — Fusion training

继续训练：

* HCI；
* Gate；
* SharedQuery；
* Decoder；
* fusion heads。

按冻结策略允许逐步解冻：

* Radar 后级；
* DINO detector head / 后级。

目标：

获得稳定多模态性能。

#### T3 — Final fine-tune

解冻最后 2 个 Swin stages。

要求：

```text
LR_RGB < LR_new_modules
```

RGB backbone 使用明显更低学习率。

目标：

冲最终 strongest result。

### 3. E0 / E1 不机械套用融合训练阶段

* **E0**：直接引用 P0 已冻结结果，不在本 ticket 中重新训练，也不进入 T1–T3。
* **E1**：采用 RGB-only DINO-Swin-T 的正常训练 / fine-tune 流程，不套用不存在的 HCI / Gate / Decoder fusion warm-up。
* **E2–E5**：执行完整 T0–T3 多模态训练流程。

## Modality Dropout

V1.1 §12.2 / §21-23：

Modality Dropout 只用于多模态实验。

初始建议：

```text
p ≈ 0.1
```

但属于可调超参数，不是冻结常数。

必须满足：

* 同一样本不能同时 drop Radar 与 RGB；
* dropout 后正确更新 `m_R / m_V`；
* Reliability Gate / SharedQuery 使用真实 mask；
* 不通过伪造 feature 代替缺失模态。

E0 / E1 单模态 baseline 不使用 modality dropout。

## E0–E5 Experiment Matrix

### E0 — LiDAR Only

模型：

```text
LiDAR V2
```

直接使用 P0 frozen result。

不重新训练。

作用：

3D 单模态 baseline。

---

### E1 — RGB Only

模型：

```text
RGB
→ DINO-Swin-T
→ 2D detection
```

不使用 Radar。

初始化：

```text
fixed DINO-Swin-T reference/pretrained checkpoint
```

作用：

视觉单模态 baseline。

---

### E2 — Radar + RGB Late Fusion

结构：

```text
Radar branch
+
RGB DINO-Swin branch
+
Dual Candidates
+
Association
+
Shared Hypothesis
+
Reliability Gate
+
Typed Shared Query
+
Decoder
```

HCI：

```text
Identity
```

即不进行 backbone-level cross-modal interaction。

作用：

回答：

```text
仅加入双模态候选 + 后融合是否已经有收益？
```

---

### E3 — Single-Level HCI

在 E2 基础上：

只在最深默认对齐层启用单向 Geometry HCI。

same-level 默认：

```text
R2 ↔ V2
```

但 E3 使用固定单向方向。

该方向必须在实现 E3 前确定并固定，E4 继续使用完全相同方向。

不得在 E3/E4 中间更换方向。

作用：

回答：

```text
加入一次 feature-level cross-modal interaction 是否有效？
```

---

### E4 — 3-Stage Hierarchical HCI

在 E3 基础上扩展为：

```text
R0 ↔ V0
R1 ↔ V1
R2 ↔ V2
```

但仍保持与 E3 完全相同的单向方向。

作用：

回答：

```text
hierarchical interaction 是否优于 single-level interaction？
```

---

### E5 — Full V1

完整冻结主模型：

```text
3-stage
Bidirectional
Geometry-guided HCI
+
token-wise residual gate
+
Dual Candidates
+
Geometry Gate
+
Feature Cost
+
Hungarian
+
Reliability Gate
+
Typed Shared Query
+
2-layer Transformer Decoder
```

作用：

取得完整模型 strongest result。

## Training Consistency

E1–E5 应尽量保持以下实验条件一致：

* data split；
* preprocessing；
* seed；
* optimizer family；
* scheduler family；
* evaluation script；
* checkpoint selection policy；
* 输入分辨率；
* 数据时间绑定协议。

如果某实验因结构不同必须使用不同训练参数，必须显式写进 config，不允许隐藏差异。

显存工程手段允许：

* AMP；
* gradient accumulation；
* UQP；
* 调整 per-device batch size。

但必须保持等效训练语义，并记录：

```text
effective batch size
precision
accumulation steps
```

## Loss

E2–E5 使用已冻结：

```text
L_total =
    λ_R * L_R
  + λ_V * L_V
  + λ_F * L_F
```

其中：

```text
L_F =
    λ_cls   * L_cls
  + λ_2D    * L_box
  + λ_3D    * L_xyz
  + λ_valid * L_valid
```

`L_valid` 必须使用 P7a 已批准 contract。

各实验通过模块开关实现，而不是临时重写 loss 语义。

## Evaluation

统一评估表保留 V1.1 §15 的七层指标。

### Radar Candidate

* Recall@K within 0.5 / 1 / 2m
* Top1 Success@1m
* XYZ mean / median / P90 / P95 error

### RGB Candidate

* proposal / detection recall
* AP
* IoU
* center error
* tiny-object subset

### Association

* matched-pair recall / precision
* correct-match rate
* H^RV / H^R / H^V 比例

### Final 3D

* Success@0.5 / 1 / 2m
* mean / median / P90 / P95 localization error

### Final 2D

* AP
* IoU
* center error

只在有 2D GT 的 valid subset 上统计。

### Joint

* 2D + 3D joint success
* CURRENT_SUPPORT / NO_CURRENT_SUPPORT
* modality-missing groups

### Efficiency

* Params
* FLOPs
* GPU memory
* latency / FPS
* Radar token count
* Vision token count
* candidate count

## 单模态实验的指标处理

不是所有实验都具备全部七层输出。

因此统一结果表保留全部指标列，但：

```text
结构上不存在的指标 = N/A
```

禁止为了填表额外增加预测头或伪造指标。

### E0

主要报告：

* Radar candidate；
* 3D localization；
* efficiency。

2D / association / joint fusion 等不存在的指标标记 `N/A`。

### E1

主要报告：

* RGB candidate；
* 2D detection；
* efficiency。

Radar / association / fused 3D 等不存在的指标标记 `N/A`。

### E2–E5

报告所有结构上适用的完整指标。

## Strongest Model

第一阶段目标不是完成所有消融，而是先获得 strongest model。

原则上 E5 是 frozen full model，也是 strongest-model 主候选。

结果报告必须同时列出 E0–E5 的核心指标，不因结果不符合预期而隐藏。

重点观察：

* Final 3D Success@1m；
* Final 2D AP；
* Joint success；
* tiny-UAV subset；
* Radar / RGB candidate recall；
* efficiency。

若 E5 某指标并非最高，不现场改定义或修改实验矩阵，而是记录结果并进入后续诊断 / ablation。

## Acceptance criteria

* T0–T3 可以由 config 独立控制。
* T1 freeze policy 正确。
* T3 只按冻结策略解冻最后 2 个 Swin stages，并使用低 RGB LR。
* E0–E5 全部完成对应实验。
* E0 直接引用 P0 frozen checkpoint，不重新训练。
* E1–E5 使用统一视觉初始 checkpoint。
* E2–E5 使用统一 E0 Radar checkpoint 初始化。
* 每个实验均保存：

  * checkpoint；
  * effective config；
  * metrics；
  * training log；
  * seed；
  * commit；
  * checkpoint initialization source。
* E5 完成训练并生成正式评估结果。
* Modality dropout on/off 均可运行，off 用于诊断。

## Tests

### Training stage contract

T1：

断言被冻结模块：

```text
Swin/DINO backbone
LiDAR frozen modules（按 T1 policy）
```

训练模块：

```text
HCI
Gate
SharedQuery
Decoder
Fusion heads
```

### T3

断言：

```text
only last 2 Swin stages are unfrozen
```

并检查 RGB LR 低于新模块 LR。

### Initialization consistency

断言：

```text
E1–E5 vision checkpoint source identical
E2–E5 radar checkpoint source identical
```

### Modality Dropout

断言：

* 同一样本禁止双 drop；
* `m_R/m_V` 正确更新；
* mask 一直传播到 Reliability Gate。

### E2 Identity

E2 必须：

```text
HCI = Identity
```

完整前向可以运行。

不要求与 LiDAR-only 数值等价。

### Config

每个实验保存 effective config snapshot。

## Artifacts / reports expected

* `results/multimodal_v1_e0_e5.md`
* E0–E5 主结果表
* 各实验 checkpoint
* effective config
* training logs
* evaluation outputs
* strongest model checkpoint
* 后续 ablation backlog

主结果表至少包含：

```text
Experiment
Radar candidate metrics
RGB candidate metrics
Association metrics
Final 2D
Final 3D
Joint
Efficiency
Checkpoint
```

不适用指标统一标记：

```text
N/A
```

## Stop condition

### Negative transfer

如果 E5 明显低于 E0：

不立即修改冻结架构。

先检查：

* near-zero HCI gate；
* T1 freeze policy；
* E2 Identity baseline；
* modality dropout；
* candidate recall；
* association quality；
* loss 是否正常。

若最终确认必须修改 §21 frozen architecture 才能解决：

停止并进入重新决策流程。

### OOM

如果 UQP / AMP / gradient accumulation 后仍无法完成训练：

创建 engineering-risk ticket。

不得通过修改冻结模型结构解决显存问题。

### Architecture drift

任何在 E0–E5 中增加新模块的想法：

进入 backlog。

不得直接进入本 ticket。
