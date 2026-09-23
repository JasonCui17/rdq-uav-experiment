# P7b — Reliability Gate + Typed Shared Query + Standard Transformer Decoder

Status: ready-for-agent
Blocked by: 07（P6 hypothesis 输入）、08（P7a target contract **必须先批准**）
Type: task

## Goal

实现融合解码链：

Hypothesis Reliability Gate
→ Typed Shared Query（128D）
→ 标准 2-layer Transformer Decoder
→ 输出 2D box / 3D XYZ / fused score / c_2D / c_3D。

## Non-goals

* 不发明 decoder（标准 Transformer Decoder ×2，§21-17）。
* 不做 alignment 消融的 memory 变体（shifted/full-hierarchy 属后续消融）。
* 不做后处理创新：最终去重只用简单 2D NMS + 3D radius suppression（Q45）。
* 不新增 learned association / contrastive / consistency 等额外模块或损失。

## Files / modules expected to touch

* 新建 `src/rdq_uav/multimodal_v1/candidate/reliability_gate.py`
* 新建 `src/rdq_uav/multimodal_v1/candidate/shared_query.py`
* 新建 `src/rdq_uav/multimodal_v1/decoder/transformer_decoder.py`
* 新建 `src/rdq_uav/multimodal_v1/loss.py`
* 新建 `src/rdq_uav/multimodal_v1/model.py`
* 新建 `tests/test_multimodal_v1_p7b_fusion.py`

## Dependencies

* 07（P6：三类 hypothesis）
* **08（P7a：target contract 批准前不得实现 L_valid）**
* 阻塞下游：10（P8 训练管线）

Gate / SharedQuery / Decoder 可先实现，但 `L_valid` 必须在 P7a target contract 批准后才能实现。

## Frozen constraints

### 1. Hypothesis Reliability Gate

V1.1 §10.1 / §21-15：

Reliability Gate 学习三路证据权重：

```text
w_R
w_V
w_J
```

由：

```text
[w_R, w_V, w_J] = softmax(masked_logits)
```

得到。

输入至少包含：

```text
s_R
s_V
f_R
f_V
association_info
m_R
m_V
hypothesis type / presence information
```

高高 / 高低 / 低高 / 低低仅作为学习语义，不写人工 if/else。

该 Gate 与 HCI 内 `Interaction Residual Gate` 是两个完全不同组件。

### 2. 缺失模态行为

presence mask：

```text
m_R, m_V ∈ {0,1}
```

必须显式进入 Gate。

缺失模态：

* feature 使用显式 missing embedding 或 zero + explicit mask；
* 禁止伪造 score；
* 禁止复制另一模态 feature；
* 禁止普通 0 feature 在没有 mask 的情况下冒充真实特征。

权重 mask 必须在 softmax 前执行，或 mask 后重新归一化。

推荐实现：

```text
m_R = 0 → radar logit = -inf
m_V = 0 → vision logit = -inf
```

然后：

```text
[w_R, w_V, w_J] = softmax(masked_logits)
```

必须满足：

```text
m_R = 0 ⇒ w_R = 0
m_V = 0 ⇒ w_V = 0
w_R + w_V + w_J = 1
```

`w_J` 始终允许存在。

### 3. Joint score

最终：

```text
s_fusion = w_R * s_R
         + w_V * s_V
         + w_J * s_joint
```

其中 `s_joint` 必须由当前 hypothesis 的融合表示学习得到，不允许手工平均 `s_R / s_V`。

定义：

```text
joint_input =
    [f_R,
     f_V,
     association_info,
     presence/type information]
```

缺失模态使用对应 missing embedding + mask。

然后：

```text
s_joint = MLP_joint(joint_input)
```

输出 scalar/logit。

H^R / H^V 同样使用统一 `MLP_joint`，通过 missing embedding 与 mask 表示缺失证据。

### 4. Typed Shared Query

V1.1 §10.2 / §21-16：

统一输出：

```text
q ∈ R^128
```

三种 hypothesis：

```text
H^RV:
q = MLP([f_R; f_V; association_info]) + e_RV

H^R:
q = MLP([f_R; e_missing_V]) + e_R

H^V:
q = MLP([e_missing_R; f_V]) + e_V
```

type embedding 与 missing embedding 必须显式存在。

三类 hypothesis 使用同一个 decoder。

### 5. Decoder

V1.1 §10.3 / §21-17/18：

使用标准 Transformer Decoder：

```text
dim = 128
layers = 2
```

不新增 decoder 范式。

Decoder memory 只读取当前 alignment 的最终交互层。

same-level 默认：

```text
Radar memory:
Proj(R2_post) → 128D

Vision memory:
Proj(V2_post) → 128D
```

每个 sample 的 memory：

```text
memory_b =
    concat(
        Proj(R2_post_b),
        Proj(flatten(V2_post_b))
    )
```

Radar token 数量是变长的，因此 batch 化时：

* 对 memory 做 padding；
* 显式生成 `memory_key_padding_mask`；
* Transformer Decoder 必须使用该 mask；
* 禁止不同 sample 的 Radar/Vision token 混入同一 memory。

Vision feature flatten 后必须保持正确 spatial/sample ownership。

### 6. 输出与 residual refinement

最终输出：

```text
B_2D
c_2D
P_3D
c_3D
s_fusion
```

#### H^RV

已有 RGB box + Radar XYZ：

```text
B_hat = B_V + ΔB
P_hat = P_R + ΔP
```

#### H^R

已有 Radar XYZ：

```text
P_hat = P_R + ΔP
```

2D box 以：

```text
Π(P_R)
```

作为 2D center reference。

Decoder 预测例如：

```text
Δcx
Δcy
log_w
log_h
```

或等价明确参数化。

完整 box：

```text
cx = Π(P_R)_x + Δcx
cy = Π(P_R)_y + Δcy
w  = exp(log_w)
h  = exp(log_h)
```

再转换为：

```text
[x1, y1, x2, y2]
```

禁止直接把投影点当作完整 box。

#### H^V

已有 RGB box：

```text
B_hat = B_V + ΔB
```

XYZ 由：

```text
query + decoder memory
```

直接补全：

```text
P_hat = PredXYZ(decoder_output)
```

缺失维度允许直接从 memory 预测。

## Loss

总体：

```text
L_total = λ_R * L_R
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

固定：

```text
L_cls = Focal-style classification
L_box = L1 + GIoU
L_xyz = SmoothL1
```

`L_valid` 必须严格按照 P7a 已批准 target contract 实现。

在 P7a 未批准前：

```text
禁止实现 generic BCE validity target
```

不得把：

```text
GT exists = 1
```

直接解释为：

```text
prediction reliable = 1
```

## Implementation notes

* `m_R / m_V` 从 P3 数据契约 / P6 hypothesis provenance 一路传入，不在 P7b 中重新猜测。
* HCI=Identity 路径必须能够完整运行，用于 E2 late-fusion 对照。
* P7b 不修改 P1/P2/P5/P6 已定义的 candidate provenance。
* Decoder / Gate / SharedQuery 都必须支持 H^RV / H^R / H^V 混合 batch。

## Acceptance criteria

* synthetic batch 全链 forward/backward 正常。
* 真实 batch 输出 finite。
* 三类 hypothesis 共用同一个 decoder，并得到统一输出 shape。
* 缺失模态行为正确：

  * `m_R=0 ⇒ w_R=0`
  * `m_V=0 ⇒ w_V=0`
  * `w_R+w_V+w_J=1`
* missing feature 不使用伪造 evidence。
* H^RV / H^R / H^V residual 规则正确。
* Decoder memory batch/padding/mask 正确，无跨 sample token 混合。
* `L_valid` 严格实现 P7a target contract。
* tiny subset 可以 overfit（T0 条件）。

## Tests

### Full model seam

不做 LiDAR-only numerical identity。

只测试：

* input/output shape；
* semantics；
* finite；
* gradient；
* candidate provenance；
* hypothesis preservation；
* missing-modality behavior。

### Candidate provenance

Identity HCI 路径下：

```text
RGB candidates
必须来自
DINO-Swin-T
```

config 中 source 必须显式为：

```text
dino_query
```

不得静默切换 YOLO。

### Reliability Gate

测试：

```text
m_R=1, m_V=1
m_R=1, m_V=0
m_R=0, m_V=1
```

断言：

```text
m_R=0 → w_R=0
m_V=0 → w_V=0
sum(weights)=1
```

并检查 missing embedding / mask 生效。

### Shared Query

分别构造：

```text
H^RV
H^R
H^V
```

检查统一输出：

```text
[N_hypothesis, 128]
```

### Decoder memory

构造不同 Radar token 数量的 batch：

```text
sample A: N_Ra
sample B: N_Rb
```

检查：

* padding 正确；
* `memory_key_padding_mask` 正确；
* 不同 sample token 不混合。

### Residual heads

检查：

```text
H^RV:
box/xyz 都使用已有 prior refinement

H^R:
xyz residual + Π(P_R) center-based box prediction

H^V:
box residual + direct XYZ completion
```

### L_valid

按 P7a contract 逐条测试，例如：

* 无 2D GT → 不产生 c_2D loss；
* H^V + 有 3D GT → c_3D 有监督；
* H^R + 有 2D GT → c_2D 有监督；
* modality support 不被直接作为 reliability target。

### T0

tiny subset overfit。

## Artifacts / reports expected

* tiny subset overfit loss 曲线；
* missing-modality behavior 简单结果表：

```text
m_R | m_V | w_R | w_V | weight_sum | PASS
```

## Stop condition

* 实现 `L_valid` 时发现 P7a contract 有缺口 → 停止，回 P7a 补契约，不得现场发明标签。
* 发现 decoder 需要 >2 层或额外 multiscale memory 才能收敛 → 记录 engineering risk / 后续消融候选，不得修改冻结结构。
* 发现必须修改 P6 hypothesis contract 才能完成 → 停止并上报接口冲突，不在 P7b 内偷偷重定义。
