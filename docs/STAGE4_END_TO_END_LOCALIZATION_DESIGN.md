# Stage 4：Minimal End-to-End RDQ 联合 2D/3D 定位

## 研究问题与任务边界

Stage 3 的任务是 `RGB + Radar → UAV type`。Stage 4 不再预测型号，而回答：

> 在完全相同的视觉、Radar 与 cross-attention 编码路径下，Radar-conditioned Dynamic Query 是否能改善单架 UAV 的 2D bbox 与 3D xyz 联合定位？

当前数据没有可靠的 no-UAV negative samples，也不预测 objectness，因此任务准确名称是：

`single-UAV joint 2D/3D localization`

它不是 open-world detection。

## 无 GT 泄漏的输入协议

定位任务使用 `image_mode: dual_full`：读取完整 2560×960 panorama，只按固定中线拆为两个 1280×960 鱼眼，再分别 resize 为 384×288。输入变换不读取 official bbox、Oracle ROI、GT xyz 或目标中心。

```text
原始 panorama [B, 3, 960, 2560]
              │ 固定中线拆分（不依赖 GT）
              ▼
双视图输入 [B, 2, 3, 288, 384]
```

禁止使用旧分类实验的 `oracle_left`、`bbox_mode: crop/erase/foreground_only`。虽然 Stage 4 manifest 只包含具有 official 2D annotation 的帧，但 annotation 只生成监督标签，不控制图像 crop/mask。

Train 1,828 帧和 validation 415 帧的 official bbox 均通过边界检查，位于原始左鱼眼 1280×960 内。没有读取 test。

## Bbox 坐标定义

两个处理后视图按 `left | right` 横向组成一个仅用于坐标定义的虚拟画布：

```text
processed canvas: H=288, W=384×2=768
```

网络监督和输出均为该画布上的 normalized CXCYWH：

```text
cx = (x1 + x2) / (2 × 2560)
cy = (y1 + y2) / (2 × 960)
w  = (x2 - x1) / 2560
h  = (y2 - y1) / 960
```

这与先将左右视图 resize、再映射到 768×288 画布完全等价。输出经过 sigmoid，四个分量都在 `[0,1]`。

## 模型

旧 `MultiModalClassifier` 保留。新建 `MultiModalLocalizer`，复用其 `forward_features`，因此 Stage 3 的 encoder/fusion 权重路径没有另起实现。

```text
image [B,2,3,288,384]
  → ResNet18
  → existing 1×1 projection
  → existing camera embedding + 2D sine position
  → visual tokens [B, 864, 256]（ResNet18 layer3 的典型网格为 18×24/视图）

radar [B,768,3] + mask [B,768]
  → existing MaskedPointMLP
  → existing masked global max pooling
  → radar token [B,256]

Learned Query 或 RDQ query
  → existing CrossAttentionBlock
  → attended token [B,256]
  → optional existing radar_skip concat
  → fused token [B,512]（learned_query/rdq 且 radar_skip=true）
      ├── box_head → sigmoid → box [B,4]
      └── position_head → normalized xyz [B,3]
```

模型返回：

```python
{
    "box": box_pred,
    "position": position_pred,
    "attention": attention,
    "features": fused,
    "visual_grid": visual_grid,
}
```

## XYZ normalization

只从 `manifests_oracle_left_fixed256_bbox/train.csv` 的 1,828 帧计算 population mean/std（`ddof=0`）：

```text
mean = [1.0976047486, 3.6788042521, 11.0677567366]
std  = [1.6146068627, 5.3843482310, 5.7585334895]
```

训练目标：

```text
position_normalized = (position_m - train_mean) / train_std
```

loss 在 normalized space 计算；metrics 前恢复到米。每个训练/smoke output 目录保存 `position_stats.json`，其中记录 train manifest 路径、样本数与 std 约定。val/test 绝不参与统计。

## Loss

第一版固定：

```text
L = 5.0 × SmoothL1(box)
  + 2.0 × (1 - GIoU)
  + 1.0 × SmoothL1(position_normalized)
```

GIoU 使用 torchvision 的维护实现。日志分别记录：

- `total_loss`
- `bbox_l1_loss`
- `giou_loss`
- `position_loss`

## Validation metrics 与预测记录

2D：

- mean/median IoU
- Recall@IoU 0.5
- bbox center error 的 mean/median pixel error（768×288 画布）
- mean normalized center error

3D：

- mean/median Euclidean position error（米）
- X/Y/Z MAE（米）
- mean/median range error（米）

训练入口把每轮完整指标写入 `history.csv`，最新/最佳 validation metrics 写入 JSON。最佳 validation predictions 同时写 JSON/CSV，字段包括 sample id、预测/真值 bbox、IoU、预测/真值 xyz、3D error、distance、sequence、temporal block 与 GT time。

## 物理先验接口

配置已预留但默认关闭：

```yaml
loss:
  projection_consistency:
    enabled: false
    weight: 0.0

physical_priors:
  velocity:
    enabled: false
    v_max_soft_mps: 40.0
```

虽然仓库已有鱼眼投影代码，但 GT/Radar/camera 的坐标系、时间约定与最终标定尚未同时验证。因此开启 projection consistency 会显式报错，不会静默使用不可靠投影。velocity 只保留配置语义；单帧模型不计算 temporal loss，也不 hard clip。

本阶段没有 GT-radius Radar crop。此前 1 m/2 m 非空率仅约 24.56%/38.29%，hard crop 会制造空帧 shortcut。

## 实验配置

配置位于：

```text
configs/localization/base.yaml
configs/localization/rgb.yaml
configs/localization/radar.yaml
configs/localization/concat.yaml
configs/localization/learned_query.yaml
configs/localization/rdq.yaml
```

它们共享数据、loss、head、优化器和训练超参数，仅改变既有 fusion variant。本阶段没有启动任何正式训练。

## Smoke tests

代码单元测试共 14 项全部通过，包含五种 localizer variant 的 forward/backward、bbox 映射和 perfect-prediction metric sanity。

真实 train 数据上的 RDQ smoke：

```text
image       [2,2,3,288,384]
radar       [2,768,3]
radar_mask  [2,768]
bbox        [2,4]
position    [2,3]
pred_box    [2,4]，范围 [0.517071, 0.574849]
pred_xyz    [2,3]（normalized space）
gradient finite: true
metric sanity: true
```

20-sample、300-step CPU overfit：

| 项目 | Initial | Final |
|---|---:|---:|
| bbox SmoothL1 | 0.074793 | 0.000493 |
| GIoU loss | 1.034741 | 0.813527 |
| xyz SmoothL1 | 0.289103 | 0.002269 |
| mean IoU | 0.000438 | 0.240973 |
| bbox center error mean | 216.99 px | 6.39 px |
| mean 3D error | 6.02 m | 0.54 m |

bbox 与 xyz loss 均明显下降，forward/backward 数据链路成立。不过 20 样本最终 Recall@IoU0.5 仍为 0，GIoU 收敛明显慢于参数 L1；这是启动正式训练前需要关注的优化风险，而不是通过增加新模块规避的问题。

## 明确未执行

- 没有读取 test
- 没有正式 seed0 训练
- 没有 soft voting
- 没有 Oracle Target-RDQ
- 没有 temporal model
- 没有新增 PointROPE、token pruning、gating、LiDAR distillation 或 DETR decoder
