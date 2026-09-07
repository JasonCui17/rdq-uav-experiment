# Stage 4.2：BBox Loss 配对消融

本实验仅改变 bbox regression loss 与 GIoU 权重。模型、数据、参数化、优化器、学习率、样本顺序和 XYZ 分支保持一致；未访问 test。

## 配对结果

| Loss | Initial IoU | Best IoU | Best step | Best Recall@0.5 | Center error (px) | Width error | Height error | XYZ error (m) | Final IoU | Final Recall@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smooth_l1_giou | 0.000481 | 0.390692 | 600 | 0.150 | 7.364 | 0.012372 | 0.021477 | 0.550 | 0.390692 | 0.150 |
| l1_giou | 0.000481 | 0.392629 | 450 | 0.500 | 10.248 | 0.004471 | 0.010701 | 0.481 | 0.294774 | 0.000 |
| l1_only | 0.000481 | 0.512551 | 500 | 0.700 | 5.392 | 0.003485 | 0.004659 | 0.199 | 0.003006 | 0.000 |

## Tiny-box synthetic sensitivity

使用 train bbox median size，processed canvas 为 768×288。10% 尺寸偏差指宽高均放大 10%；中心偏差仅沿 x 轴。

| Case | Offset (px) | L1 | GIoU loss | IoU |
|---|---:|---:|---:|---:|
| perfect | 0.0 | 0.00000000 | 0.000620 | 1.000000 |
| width_height_10pct_larger | 0.0 | 0.00068360 | 0.173979 | 0.826445 |
| center_x_plus_1px | 1.0 | 0.00032552 | 0.263544 | 0.736852 |
| center_x_plus_2px | 2.0 | 0.00065105 | 0.465363 | 0.534892 |
| center_x_plus_4px | 4.0 | 0.00130208 | 0.754794 | 0.245301 |
| center_x_plus_8px | 8.0 | 0.00260417 | 1.095848 | 0.000000 |

## Head gradient norm（step 50–600）

| Loss | Box-head mean | Box-head min | Box-head max | Position-head mean |
|---|---:|---:|---:|---:|
| smooth_l1_giou | 59.395218 | 23.829950 | 157.178253 | 0.824474 |
| l1_giou | 35.999392 | 4.919549 | 104.662178 | 0.890254 |
| l1_only | 3.908030 | 1.359488 | 4.981252 | 0.384394 |

## 可复现信息

完整历史与机器可读报告位于 `outputs/stage4_bbox_loss_20260907_141337`。Best checkpoint criterion 为固定 20 样本全集上的 Mean IoU。
