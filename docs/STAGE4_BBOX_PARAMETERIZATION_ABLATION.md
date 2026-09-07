# Stage 4：BBox Parameterization 配对消融

## 实验范围

本实验只改变 `MultiModalLocalizer` 的 bbox 解码：

- A：`sigmoid_cxcywh`
- B：`sigmoid_center_log_size`

两组共享相同的前 20 个 train 样本、seed 42、batch size 2、600 optimizer steps、样本顺序、RDQ、encoder、fusion、optimizer、学习率、loss 权重、augmentation 与 xyz head。每 50 steps 均在 `model.eval() + torch.no_grad()` 下重新评估完整 20 样本。没有读取 test，也没有正式训练。

基线直接复用 `stage4_bbox_sanity_20260907_123848`。两次运行的初始 center loss、center pixel error、xyz loss 和 3D error 完全相同，进一步验证除尺寸解码外的条件一致。

## 完整 Train Bbox 统计

统计只使用 `manifests_oracle_left_fixed256_bbox/train.csv` 的全部 1,828 帧；width/height 是 2560×960 panorama（等价于处理后 768×288 双视图画布）上的 normalized 值。

| Statistic | Width | Height |
|---|---:|---:|
| Mean | 0.01219590 | 0.02937352 |
| Median | **0.00859400** | **0.01875000** |
| Population std | 0.00993098 | 0.02545028 |

因此 log-size 使用：

```text
w_ref = 0.008594
h_ref = 0.018750
wh = ref_wh × exp(clamp(delta_wh, -4, 4))
```

最后一层 w/h bias 初始化为 0。初始预测均值为：

| Parameterization | Pred width | Pred height | GT width（20 samples） | GT height（20 samples） |
|---|---:|---:|---:|---:|
| sigmoid CXCYWH | 0.542893 | 0.556911 | 0.024492 | 0.060677 |
| log-size | 0.010030 | 0.023784 | 0.024492 | 0.060677 |

log-size 的初始尺寸量级明显更合理，但 full-train reference 小于这 20 个较大目标，因此仍需学习正的 log-scale offset。

## Initial / Best / Final 结果

| Metric | sigmoid CXCYWH | log-size |
|---|---:|---:|
| Initial Mean IoU | 0.000438 | 0.000000 |
| **Best Mean IoU** | **0.399954** | **0.224896** |
| Best step | **550** | **400** |
| Best Recall@IoU0.5 | **0.55** | **0.05** |
| Best center error | **8.184 px** | 9.834 px |
| Best width abs error | **0.007313** | 0.016374 |
| Best height abs error | **0.009571** | 0.015314 |
| Best-step 3D error | 1.272 m | **0.420 m** |
| Final Mean IoU | **0.200047** | 0.164403 |
| Final Recall@IoU0.5 | 0.00 | 0.00 |
| Final center error | **10.709 px** | 18.644 px |
| Final width abs error | **0.005360** | 0.033275 |
| Final height abs error | **0.025810** | 0.044406 |
| Final 3D error | **0.317 m** | 0.833 m |

## 曲线观察

`sigmoid_cxcywh` 在 step 550 达到 Mean IoU 0.400、Recall@0.5 0.55，随后回落。`sigmoid_center_log_size` 在 step 400 达到自身最佳 Mean IoU 0.225、Recall@0.5 0.05，随后也回落。

log-size 虽改善初始预测尺寸，但训练中 exp 尺寸出现明显震荡。例如 step 100 的 width/height error 上升到 0.128/0.259；step 300 尺寸误差已很小，但 center error 为 25.85 px，Mean IoU 仍仅 0.0018。它没有改善 tiny-box 的整体优化稳定性。

## 决策

`sigmoid_center_log_size` 明显差于原始 sigmoid baseline：

- Best Mean IoU：0.225 vs 0.400
- Best Recall@0.5：0.05 vs 0.55
- best center/width/height error 均更差
- final 3D error 也更差

因此不把 log-size 设为默认 head，保留 `sigmoid_cxcywh`。两种参数化的 Best Mean IoU 都没有超过 0.5，按预先规定的决策标准，本阶段暂停；下一步应先研究既有 loss optimization，而不是启动正式 localization seed0。
