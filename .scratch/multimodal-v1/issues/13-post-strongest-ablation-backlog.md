# A1 — Post-Strongest-Model Ablation Backlog

Status: backlog（E0–E5 完成后才开工）
Blocked by: 10（P8 E0–E5 完成、主结果确定）
Type: task（消融矩阵）

## Goal

在 E0–E5 全部完成后，按 V1.1 §14 补齐正式消融实验，用于解释 Full V1 各关键组件的实际贡献。

A1 不阻塞 strongest-model 主线，也不改变 E0–E5 的主结果定义。

## Ablation Anchor

A1 的统一消融基准固定为：

```text
E5 — Full V1
```

“Post-Strongest-Model”只表示执行时机：

```text
先完成 E0–E5
→ 得到主结果
→ 再开始 A1
```

它不表示自动选择 E0–E4 中数值最高的实验作为消融基准。

所有消融都以 E5 为 reference，只修改当前研究变量，其余条件保持与 E5 一致。

若 E0–E4 中某实验最终数值高于 E5：

* 如实报告；
* 不自动改变 A1 anchor；
* 不重新定义 Full V1；
* 是否修改主方法需另走重新决策流程。

## Backlog Matrix

| 消融               | 对照                                    | 目的                                       |
| ---------------- | ------------------------------------- | ---------------------------------------- |
| Alignment        | same-level / shifted / full-hierarchy | 验证层级对齐策略；full-hierarchy 同时报 Params/FLOPs |
| Direction        | R→V / V→R / Bi                        | 验证双向交互价值                                 |
| HCI stage count  | 1 / 3                                 | 验证 hierarchical interaction 贡献           |
| Interaction gate | off / token scalar                    | 验证 near-zero gated residual 的价值          |
| Router           | GeometryLocal / LatentBridge          | 显式几何局部交互 vs geometry-free baseline       |
| Candidate source | R-only / V-only / dual                | 验证双候选对 recall / robustness 的贡献           |
| Hypothesis type  | 仅 H^RV / H^RV+H^R+H^V                 | 验证保留单模态 hypothesis 的价值                   |
| Modality dropout | off / on                              | 验证模态缺失与 gate collapse 鲁棒性                |

## Alignment Ablation

对照：

```text
same-level
shifted
full-hierarchy
```

### same-level

使用 E5 默认：

```text
R0 ↔ V0
R1 ↔ V1
R2 ↔ V2
```

### shifted

按 V1.1 已定义 shifted alignment 实现。

### full-hierarchy

仅在该消融配置中实现额外 Radar stage：

```text
R3 = SparseMerge23 + Stage3
```

要求：

* 仅做最小必要延伸；
* 不改变原 LiDAR V2 主干定义；
* 不进入 E5 默认路径；
* 同时报告 Params / FLOPs / memory / latency。

## Direction Ablation

对照：

```text
R→V
V→R
Bi
```

其中：

```text
Bi = E5 默认双向
```

E3/E4 主实验中使用的固定单向方向，在此扩展为完整方向比较。

除 interaction direction 外，其余全部保持 E5 一致。

## HCI Stage Count Ablation

对照：

```text
1-stage
3-stage
```

### 1-stage

仅保留最深默认 aligned stage 的 HCI。

### 3-stage

使用 E5 默认三级：

```text
R0/V0
R1/V1
R2/V2
```

Direction、gate、candidate、decoder 等保持 E5 设置不变。

## Interaction Gate Ablation

对照：

```text
gate off
token-wise scalar gate
```

### off

保留相同 HCI interaction，仅移除 gated scaling。

### token scalar

使用 E5 默认 near-zero initialized token-wise scalar gate。

不得同时改变 HCI 结构。

## Router Ablation

对照：

```text
GeometryLocal
LatentBridge
```

### GeometryLocal

使用 E5 默认 Geometry-guided Local HCI。

### LatentBridge

复用 F1 已实现 baseline。

LatentBridge 在此只作为 geometry-free router baseline，不改变其身份，不得成为默认主方法。

除 Router 外，其余后续 candidate / hypothesis / decoder / training protocol 尽可能保持一致。

## Candidate Source Ablation

对照：

```text
R-only
V-only
dual
```

### R-only

仅 Radar candidates 发起 hypothesis。

### V-only

仅 RGB candidates 发起 hypothesis。

### dual

使用 E5 默认：

```text
Radar candidates
+
RGB candidates
```

目的：

验证双候选是否提升 candidate recall、最终 recall 与 modality robustness。

不得通过修改 candidate detector 本身补偿单源配置。

## Hypothesis Type Ablation

对照：

```text
H^RV only
vs
H^RV + H^R + H^V
```

### H^RV only

只保留成功 association 的 paired hypotheses。

### Full

使用 E5 默认：

```text
H^RV
H^R
H^V
```

三类全部保留。

目的：

验证未匹配单模态 hypotheses 是否真正提升 recall / robustness。

## Modality Dropout Ablation

对照：

```text
off
on
```

### off

训练过程中不进行 modality dropout。

### on

使用 E5 默认配置及概率。

禁止同一样本双 drop 的规则保持不变。

## Non-goals

* 不修改 E5 Full V1 定义。
* 消融结果即使优于 E5，也不自动替换主方法。
* 不在 A1 中加入 §18 禁止的新模块。
* 不在完成 §14 既定消融前扩张新的创新点。
* 不通过改变数据 split、loss、schedule 等方式人为放大消融差异。

## Files / modules expected to touch

新建：

```text
configs/multimodal_v1/ablations/
```

每个消融至少一个独立 config。

示例：

```text
alignment_same.yaml
alignment_shifted.yaml
alignment_full_hierarchy.yaml

direction_r2v.yaml
direction_v2r.yaml
direction_bi.yaml

stage_1.yaml
stage_3.yaml

gate_off.yaml
gate_token_scalar.yaml

router_geometry_local.yaml
router_latent_bridge.yaml

candidate_r_only.yaml
candidate_v_only.yaml
candidate_dual.yaml

hypothesis_paired_only.yaml
hypothesis_all.yaml

modality_dropout_off.yaml
modality_dropout_on.yaml
```

训练与评估复用 P8 已完成 pipeline。

## Dependencies

* P8 E0–E5 已全部完成。
* E5 checkpoint / config / schedule 已固定。
* F1 LatentBridge 已实现后才能运行 Router ablation。
* full-hierarchy 的 R3 只在对应 alignment ablation 开工时实现。

## Frozen constraints

每组消融必须满足：

```text
only one studied variable changes
```

其他条件与 E5 一致，包括：

* data split；
* preprocessing；
* initialization；
* training stages；
* optimizer family；
* scheduler；
* loss；
* modality settings；
* seed；
* evaluation script；
* checkpoint selection policy。

若某消融因为结构本身必须改变额外工程字段，只允许修改实现该结构所必需的字段，并在 config 中显式记录。

## Experimental Principle

推荐每个消融组直接从 E5 config 派生：

```text
E5 config
   ↓
change one field / one structural option
   ↓
ablation config
```

避免重新写完整独立 config 导致隐藏差异。

所有结果与 E5 reference 并列报告。

## Acceptance criteria

V1.1 §14 所有消融维度均有结果。

每个实验至少具备：

* config；
* checkpoint；
* metrics；
* training log；
* seed；
* initialization source；
* evaluation output。

每组消融必须能够明确回答：

```text
相对 E5，
只改变该变量后，
性能 / recall / localization / efficiency 如何变化？
```

## Tests

### Config Difference Test

自动检查同一消融组：

```text
ablation config
vs
E5 reference config
```

只允许预期字段不同。

例如 Direction ablation：

```text
allowed difference:
interaction.direction
```

其他关键字段不得静默变化。

### Runtime Contract

每个配置至少通过：

* forward；
* finite；
* evaluation；
* checkpoint load。

full-hierarchy 等新增结构需额外检查 shape / memory contract。

## Metrics

复用 P8 评估体系，重点观察：

* Radar candidate recall；
* RGB candidate recall；
* association quality；
* Final 2D；
* Final 3D；
* Joint success；
* modality failure groups；
* Params / FLOPs / latency / memory。

不同消融根据研究问题选择核心指标，但保留统一结果表。

## Artifacts / reports expected

生成：

```text
results/multimodal_v1_ablations.md
```

作为论文 Ablation Study 的直接数据来源。

建议总表结构：

```text
Ablation Group
Variant
Final 2D
Final 3D
Joint
Candidate Recall
Association
Params
FLOPs
Latency
Delta vs E5
```

同时保留每组独立详细结果。

## Result Interpretation

A1 只回答：

```text
组件是否有效？
组件带来了什么收益/代价？
```

不在该 ticket 内重新设计模型。

若某个简单 ablation 数值高于 E5：

* 记录结果；
* 分析可能原因；
* 进入论文讨论；
* 如确实需要调整主方法，单独触发重新决策流程。

不得直接修改 frozen E5。

## Stop condition

若消融出现以下情况：

1. 必须改变 §21 frozen architecture 才能继续；
2. 需要引入新的核心模块才能解释结果；
3. 需要改变主方法定义；
4. 需要改变 E5 anchor；

则停止当前 A1 修改，并记录为：

```text
engineering finding
research finding
limitation
或重新决策候选
```

不得在 A1 内直接改主架构。
