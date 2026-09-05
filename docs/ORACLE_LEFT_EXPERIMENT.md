# 左鱼眼 Oracle ROI 实验协议

## 1. 目的和边界

完整双鱼眼图在 `32×24` 下仍可达到 100% 五分类准确率，证明模型主要利用了与类别绑定的
采集 session 和全局背景。本协议使用已经验证的左鱼眼时空标定，把 GT 三维中心投影到
图像后裁剪局部区域，重新验证 RGB、Radar 和 RDQ。

这是 **Oracle localization + UAV classification**，不是端到端 UAV detection。所有结果
必须标记为 Oracle ROI。

## 2. ROI 不能泄露类别

官方训练框反推的有效包围半径 pooled P99 约为 `0.94 m`。工程配置统一使用：

```text
shared_radius_m = 1.0
context_scale  = 1.5
min_side_px    = 128
max_side_px    = 512
```

五类全部使用同一个半径，不能根据真实类别分别使用 Mavic2/M300 尺寸。ROI 根据三维距离和
严格 `omni+radtan` 投影自然改变像素大小；靠近边缘时整体平移回图像内部，不使用黑色
padding，避免产生新的边界捷径。

## 3. 数据规模

| Split | 原始同步样本 | 左相机可见并保留 | 删除 |
| --- | ---: | ---: | ---: |
| Train | 3308 | 2925 | 383 |
| Val | 790 | 671 | 119 |
| Test | 779 | 735 | 44 |

删除原因仅包括：补偿后的图像时刻超出 GT 轨迹，或投影中心不在左相机有效画面。原有时间块
划分不变。Radar 均值和方差已经只用保留后的 2,925 个训练样本重新计算。

ROI 尺寸分布：训练集 median `128 px`、P95 `313 px`、最大 `414 px`。官方框审计结果：

- 全部 4,120 个可评估框完整覆盖率：`99.93%`；
- val 时间块：`99.85%`；
- test 时间块：`100%`。

可视化见 `calibration/oracle_left_roi_montage.jpg`。

## 4. 自适应 ROI 捷径控制结果

提供两份严格配对配置：

```bash
python tools/train.py --config configs/oracle_left_lowres.yaml
python tools/train.py --config configs/oracle_left_lowres_masked.yaml
```

第二组以真实投影位置为中心遮住 ROI 的 80%，即使 ROI 在鱼眼边缘发生位置平移，也能覆盖
UAV。遮挡区域填入 ImageNet 均值颜色，归一化后接近零。

GPU 10 epoch、seed 0 的 best checkpoint 结果：

| 输入 | Best Val Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: |
| 24×24 自适应 Oracle ROI | 71.17% | 78.50% | 78.81% |
| 24×24，目标中心遮挡 80% | 74.67% | 86.26% | 85.61% |

遮挡目标后性能反而提高，不能解释为目标视觉信息有效。原因包括：自适应物理 ROI 在 resize
后通过背景缩放暴露距离、边缘 ROI 平移后通过目标/遮挡位置暴露方向，以及类别绑定 session
的天空和曝光差异。因此 `configs/oracle_left.yaml` 保留为自适应 ROI 消融，但不再作为 A～E
主协议。

## 5. 推荐主协议：固定 256 px、严格居中

新的 `manifests_oracle_left_fixed256` 使用以下规则：

```text
ROI side       = 256 px（所有帧固定）
target center  = ROI 几何中心
boundary       = 越界即删除，不平移、不 padding
resize         = 256×256（不改变空间尺度）
```

因此遮挡控制始终位于固定中心，ROI 尺度也不再编码 GT 距离。数据量为：

| Split | 样本数 |
| --- | ---: |
| Train | 1996 |
| Val | 417 |
| Test | 565 |

样本减少是切断泄漏通道的必要代价。各类仍均有训练、验证和测试样本；Radar normalization
仅使用这 1,996 个训练样本重新计算。

## 6. 固定 256 px 捷径控制结果

GPU 10 epoch、seed 0 的 best checkpoint 结果：

| 输入 | Best Val Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: |
| 24×24 固定 256 px ROI | 75.48% | 93.27% | 93.15% |
| 24×24，目标中心遮挡 80% | 87.26% | 85.66% | 83.97% |

这里的“遮挡 80%”表示宽和高分别遮挡 80%，即遮掉中心 64% 面积，仅保留约 36% 的外围
边框。测试集上，保留目标比遮挡目标高 9.18 个 Macro-F1 百分点，说明中心区域提供了额外
信息；但是仅凭外围边框仍达到 83.97%，证明模型主要性能仍可由背景、曝光、轨迹或采集
session 特征获得。验证集上遮挡组反而更高，同时未遮挡组从验证 75.48% 跳到测试 93.15%，
说明不同时间块难度和背景分布也不稳定。

因此固定 ROI 消除了自适应尺度和边界平移两条明显捷径，但无法消除“一个类别对应一个
采集 session”这个数据集层面的混杂因素。此时不能把高分类准确率直接解释成 UAV 外观识别，
也不应立即用 A～E 的数值排序论证 Radar Dynamic Query 的跨 session 泛化能力。

官方 2D 框与固定 256 px manifest 的精确匹配覆盖率为：Train `91.6%`、Val `99.5%`、
Test `98.8%`。下一步应在完全相同的匹配子集上做三组互补控制：

```text
Full ROI          完整固定 ROI
BBox erased       仅擦除官方 UAV bbox（含统一 margin）
BBox only         仅保留官方 UAV bbox（含统一 context），其余填均值
```

这三组用于进一步区分目标外观与目标外背景，不能解决单 session/class 的根本可辨识性问题。

## 7. 后续运行顺序

以下固定中心低分辨率控制已经完成：

```bash
python tools/train.py --config configs/oracle_left_fixed256_lowres.yaml
python tools/train.py --config configs/oracle_left_fixed256_lowres_masked.yaml
```

旧自适应 ROI 的 80 样本实验在第 4 epoch 已达到 100% train accuracy，已经证明训练链路可
过拟合。当前优先完成官方 bbox 的互补控制；在此之前暂停 A～E 正式比较。若后续仍运行
A～E，只能将其定义为 MMAUD V1 单 session/class 条件下的探索性实验：

```bash
python tools/train.py --config configs/oracle_left_fixed256.yaml \
  --set experiment.name=A_fixed256_rgb --set model.variant=rgb

python tools/train.py --config configs/oracle_left_fixed256.yaml \
  --set experiment.name=B_fixed256_radar --set model.variant=radar

python tools/train.py --config configs/oracle_left_fixed256.yaml \
  --set experiment.name=C_fixed256_concat --set model.variant=concat

python tools/train.py --config configs/oracle_left_fixed256.yaml \
  --set experiment.name=D_fixed256_learned_query --set model.variant=learned_query

python tools/train.py --config configs/oracle_left_fixed256.yaml \
  --set experiment.name=E_fixed256_rdq --set model.variant=rdq
```

所有正式组使用相同 `manifests_oracle_left_fixed256`、ROI 参数和训练超参，并分别运行
seed 0/1/2。

## 8. 官方 bbox 反事实实验（seed 0）

使用完全相同的官方框匹配子集，Train/Val/Test 为 `1828/415/558`；输入为 256×256，
所有目标框统一使用 `2×` context。第一轮只查看验证集，未用这组配置继续查看测试集。

| 输入 | Best epoch | Best Val Accuracy | Best Val Macro-F1 |
| --- | ---: | ---: | ---: |
| Full fixed ROI | 5 | 100.00% | 100.00% |
| BBox erased | 6 | 97.59% | 97.97% |
| Foreground-only | 10 | 76.14% | 71.20% |
| BBox crop + resize | 10 | 78.55% | 73.20% |

`Full` 与 `BBox erased` 逐样本配对时，前者独占正确 10 帧，后者独占正确 0 帧；目标区域有
增益，但擦除 UAV 后仍正确 405/415 帧，说明总体分类几乎可由 session/background 完成。

逐类别正确率：

| 类别 | Full | Erased | Foreground-only | Crop |
| --- | ---: | ---: | ---: | ---: |
| Mavic2 | 100.0% | 100.0% | 80.6% | 73.5% |
| Mavic3 | 100.0% | 100.0% | 90.3% | 95.2% |
| Avata | 100.0% | 92.9% | 95.6% | 93.8% |
| M300 | 100.0% | 96.7% | 19.7% | 14.8% |
| Pham4 | 100.0% | 100.0% | 75.3% | 98.8% |

结论不是“目标完全无用”：部分型号的局部外观明显可辨，且 Full 相比 Erased 有一致的配对
增益。但单帧 Crop 对 M300 几乎失效，说明有效帧分布、目标尺度和清晰度高度不均。下一步
迁移 UG2 获胜方案的关键帧选择与序列 soft vote，并先用 seed 1/2 检查上述差异是否稳定；
在此之前不运行 RDQ 正式对比。

## 9. 官方 bbox 反事实实验（三个 seed）

seed 0/1/2 的最佳验证集结果如下。表中为 Macro-F1 的均值与样本标准差；模型选择只使用
验证集，没有继续查看测试集。

| 输入 | Best Val Macro-F1（mean ± std） |
| --- | ---: |
| Full fixed ROI | 99.30 ± 0.77% |
| BBox erased | 96.98 ± 2.00% |
| BBox crop + resize | 76.26 ± 2.93% |

Full 与 Erased 相差约 2.32 个百分点，说明目标区域有贡献；但是擦除目标后仍接近 97%，背景
捷径在不同随机种子下稳定存在。Crop 明显高于五分类的 20% 随机水平，证明目标外观确实包含
类别信息，但 M300 三个 seed 的逐类正确率仅为 14.8%、18.0% 和 19.7%。因此下一步首先审计
M300 的框、同步、可见像素和类别混淆，不能把 Full 的接近满分结果作为 UAV 外观分类性能。

当前已实现按 `(sequence_id, temporal_block)` 聚合帧概率的 soft vote，以及按官方 bbox 面积
选择 5 个关键帧、关键帧间至少相隔 1.5 秒的诊断协议。该协议迁移自视频 UAV 分类中的
“检测后选清晰帧再投票”思路；由于官方框没有检测置信度，这里明确使用 bbox 面积作为 Oracle
可见度代理。时序聚合代码已经通过合成单元测试，下一步只在 validation 上评估，不能根据
test 结果继续选择聚合规则。
