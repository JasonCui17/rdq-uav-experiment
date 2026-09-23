# F1 — LatentBridge Baseline / Fallback / Debug（非主方法）

Status: ready-for-agent
Blocked by: 02（P1 adapter）、03（P2 视觉链路）
Type: task

**身份：baseline / fallback / debug。永不进入主方法路径。**

## Goal

实现 LatentBridge 作为 geometry-free cross-modal interaction baseline，用途限定为：

1. **Debug**：当 P4 Geometry Correctness Gate 尚未通过时，用于验证不依赖几何投影的多模态链路，包括：

   * Radar / Vision 双 backbone；
   * cross-modal interaction；
   * HCI 后特征传播；
   * RGB / Radar candidate generation。

2. **Baseline / Ablation**：在 strongest model 完成后，与 GeometryLocal 进行 Router 对照实验，验证显式 geometry-guided local interaction 的价值。

若 P4 FAIL，LatentBridge **只能绕过 P5 HCI 对 geometry 的依赖**。

它不能自动绕过 P6 正式 Association 中的：

```text
Geometry Hard Gate
→ cosine feature cost
→ Hungarian
```

若需要在 P4 FAIL 期间调试完整 candidate → hypothesis → decoder 链路，必须显式使用独立的：

```text
feature-only association debug / fallback
```

该路径只允许用于 debug / baseline，永不进入正式主方法。

## Non-goals

* **绝不**作为主方法或默认 config。
* 不在 E0–E5 主线实验矩阵中替代 Geometry HCI。
* 不允许因为 P4 未通过就使用 LatentBridge 跑“伪 E5”。
* 不修改正式 P6 Association：

  ```text
  Geometry Gate → cosine feature cost → Hungarian
  ```
* 不把 feature-only association 升级为主方法。

## Files / modules expected to touch

* 新建：
  `src/rdq_uav/multimodal_v1/interaction/latent_bridge.py`
* 注册名：
  `latent_bridge`
* 新建：
  `tests/test_multimodal_v1_f1_latent_bridge.py`

如需要完整链路 debug，可复用/增加独立的 feature-only association debug 配置，但不得修改正式 P6 association 默认行为。

## Dependencies

* 02（P1 Radar stage interface）
* 03（P2 Vision stage interface）

不阻塞任何主线 ticket。

主线不得等待 F1 完成。

## Frozen constraints

* V1.1 §5.4 / §21-09：
  LatentBridge 仅属于：

  * baseline；
  * fallback；
  * debug。

* 不得替换 GeometryLocal 主方法。

* 不得进入默认 YAML。

* 不得自动替代 P4 Geometry Correctness Gate。

* 不得修改正式 association 路径。

真正被冻结的是 **LatentBridge 的身份和用途**，而不是其内部 latent 数量等 baseline 超参数。

## Implementation notes

LatentBridge 可采用如下轻量 geometry-free 实现：

```text
Radar tokens
    ↓
Radar latent summary
    ↓
cross-modal bridge
    ↓
broadcast to Vision / Radar
    ↓
token-wise gated residual
```

Vision 侧同理。

默认建议：

```text
interaction_dim = 128
num_latents K = 16
```

其中：

```text
K=16
```

只是 F1 baseline 的默认工程参数，可配置，不属于 V1.1 frozen architecture。

基本形式：

```text
R tokens → Z_R
V tokens → Z_V

Z_R ↔ Z_V

ΔR = broadcast(Z_V → R)
ΔV = broadcast(Z_R → V)

R' = R + g_R * ΔR
V' = V + g_V * ΔV
```

Gate 使用 near-zero residual initialization，避免初始化时强烈扰动原 backbone。

缺失模态时，对应 cross-modal residual 置 0，不伪造另一模态证据。

## Debug 使用规则

### P4 PASS

正式主线使用：

```text
GeometryLocal HCI
→ Geometry Association
```

LatentBridge 不参与主线。

### P4 FAIL

允许使用：

```text
Radar
   ↘
 LatentBridge
   ↗
Vision
```

验证：

* 双 backbone；
* multimodal interaction；
* candidate generation；
* gradient / finite；
* downstream interface。

但正式 P6 Association 仍然被 geometry dependency 阻塞。

如果确实需要调试完整链路，可显式运行：

```text
LatentBridge
→ Dual Candidates
→ Feature-only Association [DEBUG ONLY]
→ Shared Hypothesis
→ Decoder
```

该配置必须明确标记：

```text
debug_only: true
```

不得作为正式 E0–E5 结果。

## Acceptance criteria

* LatentBridge 可以通过 Registry 显式配置运行。
* Radar / Vision 输入输出 shape 保持正确。
* forward / backward finite。
* 缺失模态 mask 行为正确。
* near-zero gate 初始化时不会显著破坏原模态特征。
* 默认 V1 config 不实例化 `latent_bridge`。
* 主方法 config 仍然固定为 GeometryLocal。
* 使用 feature-only association 时必须显式进入 debug/fallback 配置，不能静默切换。

## Tests

### Contract

检查：

* shape；
* finite；
* gradient；
* modality mask；
* gated residual。

### Default-path protection

断言默认主模型 YAML：

```text
router != latent_bridge
```

并且不会实例化该模块。

### Debug-path protection

若配置：

```text
router: latent_bridge
```

则必须显式识别为 baseline/debug 模式。

如果同时启用 feature-only association：

```text
association: feature_only
```

必须要求：

```text
debug_only = true
```

防止该组合被误当作正式模型。

## Artifacts / reports expected

无独立正式实验报告。

strongest model 完成后，如进行 Router ablation，再记录：

```text
GeometryLocal
vs
LatentBridge
```

的对照结果。

## Stop condition

以下任一情况立即停止并记录：

* 尝试将 LatentBridge 设置为默认 router；
* 尝试用 LatentBridge 替代 P4 calibration 修复；
* 尝试用 LatentBridge + feature-only association 作为正式 E5；
* 尝试将 feature-only association 升级为正式主方法。

以上均违反 V1.1 frozen architecture。
