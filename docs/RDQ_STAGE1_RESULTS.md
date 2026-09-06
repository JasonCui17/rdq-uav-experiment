# RDQ 第一阶段结果（seed 0）

## 1. 做了什么实验

在冻结的 `manifests_oracle_left_fixed256_bbox`、官方 bbox `2×` context crop、256×256
输入和完全相同训练超参数下，比较 RGB-only、Radar-only、naive concat、Learned Query 与
Radar Dynamic Query。所有结果均来自 validation，checkpoint 按 Macro-F1 选择；未查看 test。
RGB 复用此前完全相同配置的 seed0 有效结果，其余四组重新训练10个 epoch。

## 2. 为什么做

核心问题是：在共享 RGB backbone、visual tokens、CrossAttentionBlock、classifier、radar
skip、数据和训练设置时，由当前 Radar 生成 Query 是否优于固定 Learned Query。

## 3. 得到了什么数字

| Model | Val Accuracy | Val Macro-F1 | Mavic2 Recall | Mavic3 Recall | Avata Recall | M300 Recall | Pham4 Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB | 78.55% | 73.20% | 73.47% | 95.16% | 93.81% | 14.75% | 98.77% |
| Radar | 35.18% | 35.38% | 39.80% | 33.87% | 40.71% | 0.00% | 49.38% |
| Concat | 80.72% | 75.86% | 91.84% | 77.42% | 97.35% | 31.15% | 83.95% |
| Learned Query | 84.10% | 81.12% | 84.69% | 56.45% | 100.00% | 65.57% | 96.30% |
| **RDQ** | **89.16%** | **86.73%** | **98.98%** | 56.45% | **100.00%** | **72.13%** | **100.00%** |

最佳 epoch 依次为 RGB 10、Radar 2、Concat 7、Learned Query 2、RDQ 5。

关键差值：

```text
Δ_RDQ-LQ     = +5.61 Macro-F1 百分点
Δ_RDQ-Concat = +10.88 Macro-F1 百分点
```

## 4. 数字说明什么

seed0 支持 RDQ 主假设：RDQ 同时超过 Learned Query 与 naive concat，而且 M300 recall 相比
RGB 从 14.75% 提升到 72.13%。但这是单 seed、同一 V1 class-session coupling 条件下的
validation 结果，尚不能证明提升稳定，也不能证明 Query 利用了当前时刻正确配对的 Radar。

## 5. 下一步只做什么

进入预先规定的 Radar 因果干预。固定 temporal shift 为 `+1.0 s`，使用同一个 RDQ seed0
best checkpoint，在 validation 比较 normal、zero、shuffle_same_class 与 shift；不改模型、
数据协议或超参数，也不查看 test。

## Radar 因果干预结果

固定使用 RDQ seed0 的 epoch 5 best checkpoint，temporal shift 在评估前固定为 `+1.0 s`。

| RDQ Radar Input | Val Accuracy | Val Macro-F1 | 相对 Normal |
| --- | ---: | ---: | ---: |
| Normal | 89.16% | 86.73% | 0.00 pp |
| Zero | 34.46% | 22.53% | -64.20 pp |
| Same-class Shuffle | 89.64% | 87.41% | +0.67 pp |
| Temporal Shift +1.0 s | 87.71% | 84.72% | -2.01 pp |

Zero 的巨大下降证明该 checkpoint 强烈依赖 Radar 输入，但不能单独证明时刻对应关系，因为
全零输入属于训练分布之外。更严格的同类别 Shuffle 没有下降，反而轻微上升；`+1.0 s`
Shift 也只下降 2.01 个百分点。因此当前证据支持“Radar 模态为分类提供了信息”，但不支持
“RDQ 已利用当前时刻正确配对的 radar-visual correspondence”。它可能主要利用同类/session
的 Radar 总体统计。这个结论不会通过调整干预参数或修改网络来回避，下一步只验证
Concat、Learned Query、RDQ 在 seed 1/2 下的性能差异是否稳定。
