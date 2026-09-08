# Stage 4.10 Radar Coordinate-Frame Resolution

## Gap 复核

### 已验证且本轮未重复

- 官方左鱼眼使用 Kalibr `omni+radtan`，不是 pinhole 或 OpenCV equidistant 模型。
- `GT -> left camera` 已经完成24种 proper axis/sign bootstrap、连续 SE(3)+时间偏移拟合和独立 calibration-validation 检查。
- 官方2D图像与原始左半幅的像素级时间戳匹配已经完成。
- Radar/GT 已做过恒等坐标下的最近点距离和同序列半程错配审计。
- Stage 4.9 已做 raw/shuffle/oracle 图像投影；不重复。

### 已排除

- GT 到左鱼眼的投影模型不是当前主要 gap。
- 直接假设 `Radar XYZ == GT XYZ` 无法提供可靠 tiny-UAV 图像对应。
- 简单时间补偿没有形成强 oracle upper bound。

### 唯一未解析关系

```text
Radar XYZ -> GT reference frame
```

现有 `bootstrap_axis_aligned.py` 的轴搜索属于 GT→camera，不属于 Radar→GT。此前没有对 Radar 做轴交换/符号、平移或刚体拟合。

## 本轮新增测试

只读取 `manifests_oracle_left_fixed256_bbox/{train,val}.csv`：train 1828，val 415，test 未读取。

1. Identity baseline。
2. Train 上搜索 Radar→GT 的24种 proper signed-axis rotation，并用最近点 ICP 估计每轴不超过±2 m的固定平移。
3. 从最佳轴解出发做最小连续 rigid ICP refinement。
4. 用同一个 fixed transform 在 val 验证，同时做 val same-sequence half-cycle shuffle。
5. 因全局解对 M300 不一致，触发按 sequence 的 train-fit/val-verify；没有用 val 拟合。

拟合目标是每帧“变换后 Radar 点到 radar-time GT 中心的最近距离”。它仍可能选择 clutter，因此 val 与 shuffle 是必需控制。

## 全局固定变换

最佳离散轴关系：

```text
x_gt = -y_radar + 0.556 m
y_gt =  x_radar + 0.465 m
z_gt =  z_radar + 0.030 m
```

即约 `+90° yaw`。连续刚体 refinement 为：

```text
Euler xyz = [1.93°, -1.46°, 88.48°]
translation = [-0.037, 0.422, -0.004] m
```

| Transform | Split | 3D nearest mean/median | Hit <=2m | image nearest mean/median | Cov@16 | Cov@32 | Cov@64 |
|---|---|---:|---:|---:|---:|---:|---:|
| Identity | Train | 6.24 / 5.03 m | 38.24% | 220.92 / 289.54 px | 6.56% | 15.75% | 30.31% |
| Axis+translation | Train | 4.23 / 2.64 m | 44.04% | 93.31 / 71.77 px | 13.95% | 26.86% | 46.61% |
| Rigid refinement | Train | 4.18 / 2.72 m | 45.84% | 86.59 / 64.77 px | 17.61% | 29.05% | 49.73% |
| Identity | Val | 6.79 / 4.80 m | 39.52% | 253.15 / 289.98 px | 0.48% | 3.13% | 16.87% |
| Axis+translation | Val | 4.35 / 1.88 m | 51.33% | 93.09 / 81.79 px | 0.96% | 15.18% | 40.48% |
| Rigid refinement | Val | **4.33 / 1.85 m** | **51.57%** | **87.15 / 72.75 px** | 1.69% | **19.04%** | **44.82%** |
| Rigid + shuffled GT | Val | 5.36 / 3.57 m | 33.01% | 116.61 / 89.22 px | 2.89% | 13.01% | 40.24% |

相对 identity，global rigid 的 val 3D median 改善61.42%，Coverage@32增加15.90个百分点。真实配对在3D median与Coverage@32上优于 shuffle，说明 `+90° yaw` 不是纯随机拟合。

但它还不能作为 tiny-UAV hard gate：Cov@16只有1.69%，且在Cov@16上甚至低于shuffle；Cov@64的real/shuffle差距也只有4.58个百分点。

## 跨 sequence 一致性

同一个 global rigid transform 在 val 上：

| Sequence | Identity 3D median | Global 3D median | Global image median | Global Cov@32 |
|---|---:|---:|---:|---:|
| Mavic2 | 1.76 m | 1.35 m | 34.9 px | 48.0% |
| Mavic3 | 8.80 m | 4.76 m | 66.9 px | 1.6% |
| Avata | 1.92 m | 0.92 m | 110.7 px | 0.9% |
| M300 | 20.09 m | 14.52 m | 41.1 px | 16.4% |
| Pham4 | 2.16 m | 1.32 m | 60.6 px | 24.7% |

方向改善广泛存在，但精确目标点对应高度不一致。

## Per-sequence 对照

独立拟合得到的 yaw 分别约为：Mavic2 -90°、Mavic3 86°、Avata 90°、M300 -179°、Pham4 167°，不构成稳定的 sequence-specific 安装差异。更重要的是：

- Mavic2 val median 3D error 为3.05 m，比global 1.35 m更差。
- Mavic3 val为5.17 m，且Cov@32为0。
- Avata val为2.37 m，比 global 0.92 m更差。
- M300 train median可到0.23 m，但val崩至20.98 m，是明显的clutter overfit。
- 只有Pham4保持部分val对应：0.65 m、Cov@32 49.4%。

因此证据不支持“只需每个sequence分别标定”。

## 结论

1. Radar→GT 存在一个可信的**粗方向关系**：约 `+90° yaw` 加约0.42 m平移。
2. 单一固定变换显著改善总体 val，但仍无法提供16 px级别的目标空间锚点。
3. Per-sequence 拟合多数不能泛化，说明剩余问题不是简单的 sequence-specific rigid calibration。
4. 当前唯一下一方向是 **Radar target association / 数据语义解析**：确认官方 `radar_enhance_pcl` 中哪些点是真实 UAV return、坐标字段/聚类语义和帧生成逻辑，再评估几何先验。
5. 在此之前不进入 Geometry-RDQ 或 hard geometry gating。

机器可读结果：`calibration/radar_frame_resolution.json`。运行归档：
`outputs/stage4_radar_frame_resolution_20260908_155128/report.json`。
