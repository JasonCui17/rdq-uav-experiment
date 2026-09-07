# Stage 4.7：RDQ Attention Mechanism Audit

## 实验问题

本轮只检查两个机制问题：Radar-conditioned query 是否把真实cross-attention放在UAV附近；如果attention已经找到UAV，单个fused token是否又丢失其空间位置。实验使用stride8-only RDQ、固定20个train样本、seed42、batch2、两组LR均为`1e-4`、600 steps，并且只用normalized center L1训练。未读取test，未使用FPN，也没有修改Radar encoder、RDQ或CrossAttention。

## 权重与网格契约

`CrossAttentionBlock`原本已经调用PyTorch `MultiheadAttention(..., average_attn_weights=False)`，因此本轮没有修改模型来制造attention。实际返回shape为：

```text
[batch, heads, query, tokens] = [2, 8, 1, 3456]
```

每个视图的真实feature map为`36×48`，双视图按`DualViewTokenizer`的顺序拼成3456 tokens。输入是两个`384×288`视图，故每个token cell对应`8×8 px`；左视图token先排列，右视图token的x坐标再加384，形成`768×288`stitched canvas。所有attention中心均由这套实际坐标计算。

## 中心误差结果

### Fused-token最佳checkpoint（step 250）的严格同点比较

| Center estimator | Mean / median error | P(<2px) | P(<4px) | P(<8px) |
|---|---:|---:|---:|---:|
| Fused-token regression | **7.997 / 1.297 px** | 0.60 | 0.70 | 0.75 |
| Mean-head attention soft-argmax | 209.730 / 208.444 px | 0 | 0 | 0 |
| Mean-head attention peak | 362.359 / 359.654 px | 0 | 0 | 0 |
| Best fixed head soft-argmax | 89.469 / 88.987 px | 0 | 0 | 0 |
| Per-sample oracle-best head | 89.469 / 88.987 px | 0 | 0 | 0 |
| Stride8 grid oracle | **2.865 / 2.872 px** | 0.20 | 0.85 | 1.00 |

`best fixed head`是在完整固定20样本上选出的同一个head；`oracle-best head`允许每个样本选择误差最小的head，只是诊断下界，不能作为可部署预测。即使使用更乐观的oracle-best，attention仍没有任何样本进入8 px。

### Tail稳定性（steps 400/450/500/550/600）

| Center estimator | Tail mean error ± sample std | Tail min | Tail P(<8px) |
|---|---:|---:|---:|
| Fused-token regression | **9.669 ± 1.019 px** | 8.901 | 0.38 |
| Mean-head attention soft-argmax | 204.437 ± 23.567 px | 176.594 | 0 |
| Mean-head attention peak | 309.301 ± 106.219 px | 120.132 | 0 |
| Best fixed head soft-argmax | 134.299 ± 44.886 px | 77.545 | 0 |
| Per-sample oracle-best head | 134.022 ± 44.563 px | 77.545 | 0 |
| Stride8 grid oracle | **2.865 ± 0.000 px** | 2.865 | 1.00 |

Fused-token的tail误差稳定在约10 px，而任何attention读出均远差于它。由此不能把fused-token的较好结果解释成“attention先找到UAV、随后单token压缩丢失位置”。

## Attention熵与GT邻域质量

Tail归一化熵为`0.99461 ± 0.00121`（1代表完全均匀），说明attention非常接近全图均匀分布。Tail平均GT邻域质量为：

| Region | Observed attention mass | Uniform reference |
|---|---:|---:|
| GT中心±1 cell（x/y各±8 px） | 0.001036 | 0.001157 |
| GT中心±2 cells（x/y各±16 px） | 0.004167 | 0.004630 |

GT邻域质量不仅没有明显高于均匀基线，反而略低。最终step各head soft-argmax mean error为`[140.44, 247.56, 185.88, 233.11, 224.26, 250.86, 193.45, 248.91] px`；不存在少数已经可靠定位目标的head。

## 结论

- **Hypothesis 1**：Radar query把attention放在UAV附近。
- **Result / Status**：**rejected**。mean、peak、best fixed head和oracle-best head全部远离GT，熵接近均匀，GT邻域mass不高于均匀参考。
- **Hypothesis 2**：attention已经找到UAV，但single fused token丢失空间信息。
- **Result / Status**：**inconclusive（没有支持证据）**。其必要前提“attention已找到UAV”不成立；attention readout比fused regression差一个数量级。
- **Grid quantization**：不是当前主瓶颈。stride8 oracle mean仅2.865 px，85%样本小于4 px，全部样本小于8 px。
- **Interpretation**：当前fused center能够拟合，并不来自可观测的目标对齐attention。由于attention block含query residual且分类/定位特征还含radar skip，head可以绕过空间选择，利用Radar/query与全局视觉统计回归中心。
- **Next action**：只建议进入**Radar correspondence audit**：固定模型做temporal-shift eval并同时观察center与attention是否随时间错配退化。在该证据出现前，不建议开发FPN或single-token spatial decoder。

完整原始结果位于`outputs/stage4_attention_audit_20260908_002135/`。本轮在CPU执行，训练与固定全集评估共约198.37秒。
