# Stage 4.9 Radar-to-Image Geometry Feasibility Audit

## 手动执行

Codex 只准备和检查脚本，不自动运行完整审计。手动执行后，新结果写入：

```text
outputs/stage4_geometry_audit_<timestamp>/
```

进度条格式为：

```text
split | processed/total | valid% | median nearest px | coverage@32 | samples/s | ETA
```

命令见文末；`--with-oracle` 只额外生成独立 oracle 结果，不改变 raw 主路径。

## 研究问题与边界

本阶段只判断：未经 GT 筛选的原始 Radar XYZ，经现有标定投影后，是否与 tiny-UAV 的图像位置存在可用的空间对应。

审计读取完整 train（1828 帧）与 val（415 帧），未读取 test。主路径为：

```text
全部有限 Radar XYZ
-> 当前假设的 Radar/GT 共用坐标系
-> fitted camera_from_gt transform
-> OmniRadtanCamera
-> 2560x960 stitched canvas 左半区
-> 与官方 GT bbox 比较（GT 只用于评分）
```

主路径没有使用 GT target gate、GT Radar 点选择或 GT motion compensation。同序列 null control 使用固定半周期置换，把每个 Radar 帧与同一 split、同一 sequence 的另一帧 GT bbox 配对。

重要限制：官方 2D 标注只覆盖左鱼眼；当前只有左相机拟合外参。鱼眼内参和 `GT -> left camera` 投影已验证，但官方材料没有提供已解析的 camera-radar 外参。因此“Radar XYZ 与 GT 共用坐标系”仍是待验证假设，不能把本审计解释为已知 camera-radar 标定的最终精度。

## 总体结果

覆盖率表示一帧至少有一个有效投影点落在 GT bbox 中心指定半径内。

| Split / Correspondence | nearest mean / median (px) | Cov@8 | Cov@16 | Cov@32 | Cov@64 | No valid projection |
|---|---:|---:|---:|---:|---:|---:|
| Train raw | 220.92 / 289.54 | 4.49% | 6.56% | 15.75% | 30.31% | 0.00% |
| Train shuffled | 244.99 / 312.96 | 6.24% | 9.57% | 13.24% | 19.31% | 0.00% |
| Val raw | 253.15 / 289.98 | 0.00% | 0.48% | 3.13% | 16.87% | 0.00% |
| Val shuffled | 254.88 / 319.10 | 0.00% | 0.00% | 3.13% | 11.08% | 0.00% |
| Combined raw | 226.88 / 289.98 | 3.66% | 5.44% | 13.42% | 27.82% | 0.00% |
| Combined shuffled | 246.82 / 314.54 | 5.08% | 7.80% | 11.37% | 17.79% | 0.00% |

raw 在 32/64 px 上只略高于 shuffle，在更关键的 8/16 px 上反而低于 shuffle。特别是 val 的 raw Coverage@16 只有 0.48%，Coverage@32 只有 3.13%。这不构成可用于 tiny-UAV 定位的可靠几何对应。

## Oracle 上界诊断

Oracle 使用 `GT radar-time position ±2 m` 选择点，并将所选点按 GT 轨迹补偿到有效 image time；它与主结果严格分开。

| Split | gated non-empty | nearest mean / median (px) | Cov@8 | Cov@16 | Cov@32 | Cov@64 |
|---|---:|---:|---:|---:|---:|---:|
| Train oracle（全部帧） | 38.24% | 48.34 / 36.09 | 4.27% | 6.84% | 16.08% | 29.49% |
| Val oracle（全部帧） | 39.52% | 123.03 / 73.30 | 0.00% | 0.48% | 3.13% | 16.14% |
| Combined oracle（全部帧） | 38.48% | 62.54 / 38.00 | 3.48% | 5.66% | 13.69% | 27.02% |

在 gated non-empty 帧上，combined 的条件覆盖率为 @8 9.04%、@16 14.72%、@32 35.57%、@64 70.22%；但 val 条件覆盖率仅为 @16 1.22%、@32 7.93%、@64 40.85%。Oracle 没有在全体帧或 val 上形成强上界，说明问题不能只归因于 clutter target association 或简单运动补偿。

## Sequence、距离与时间差

Val raw 按 sequence：

| Sequence | Frames | nearest median (px) | Cov@16 | Cov@32 | Cov@64 |
|---|---:|---:|---:|---:|---:|
| Avata | 113 | 263.69 | 0.00% | 0.00% | 0.00% |
| M300 | 61 | 382.29 | 0.00% | 0.00% | 0.00% |
| Mavic2 | 98 | 48.55 | 0.00% | 8.16% | 52.04% |
| Mavic3 | 62 | 366.51 | 0.00% | 0.00% | 0.00% |
| Pham4 | 81 | 149.78 | 2.47% | 6.17% | 23.46% |

结果高度依赖 sequence。Mavic2/Pham4 明显好于其余三类，但即使 Mavic2 在 tiny-box 所需的 16 px 内仍为 0%。这更像坐标约定、传感器对应或特定序列标定差异，而非一个跨序列稳定的投影关系。

Val raw 按 GT range：

| Range | Frames | nearest median (px) | Cov@16 | Cov@32 | Cov@64 |
|---|---:|---:|---:|---:|---:|
| 0–5 m | 140 | 250.41 | 0.00% | 0.00% | 6.43% |
| 5–10 m | 55 | 289.03 | 3.64% | 9.09% | 18.18% |
| 10–15 m | 56 | 381.87 | 0.00% | 0.00% | 0.00% |
| 15–20 m | 98 | 48.55 | 0.00% | 8.16% | 52.04% |
| >=20 m | 66 | 380.09 | 0.00% | 0.00% | 0.00% |

距离趋势非单调，且 range bin 与 sequence/class 强耦合，不能据此声称距离本身是原因。

Val raw 按 `abs(image_time-radar_time)`：

| Time gap | Frames | nearest median (px) | Cov@16 | Cov@32 | Cov@64 |
|---|---:|---:|---:|---:|---:|
| 0–5 ms | 82 | 212.01 | 1.22% | 1.22% | 15.85% |
| 5–10 ms | 46 | 323.87 | 0.00% | 0.00% | 2.17% |
| 10–20 ms | 72 | 35.12 | 0.00% | 9.72% | 56.94% |
| 20–40 ms | 203 | 342.20 | 0.49% | 2.46% | 6.40% |
| >=40 ms | 12 | 353.06 | 0.00% | 0.00% | 16.67% |

同步差与误差不呈单调关系，并被 sequence 构成混淆。几十毫秒的 image-radar gap 不是当前数百像素偏差的充分解释。

## 结论与停止条件

状态：**raw geometry feasibility = rejected（在当前变换假设下）**。

- Raw 没有在关键的 @8/@16 指标上稳定优于同序列 shuffle。
- Val raw 的中位最近距离约 290 px，远大于约 5–10 px 的 UAV bbox。
- Oracle 仍有约 60.5% val 帧没有 GT-gated 可投影点，且 val @16/@32 很弱。
- 因此现在不应实现 hard geometry gating 或 Geometry-RDQ。

最可能的问题优先级：

1. **Radar-to-GT/camera 坐标约定或传感器 correspondence 未解析**：左鱼眼内参及 GT-to-camera 投影本身已有低像素误差证据，但 Radar 到该参考系的刚体关系没有官方外参支撑。
2. **Target association / Radar 对 UAV 的真实回波稀疏**：2 m oracle gate 仅约 38–40% 帧非空，且其中可能包含 clutter。
3. **Time alignment**：可能有贡献，但当前分桶无单调关系，简单 GT motion compensation 也没有形成强 val upper bound，因此不是首要解释。

完整逐帧与分组数据位于：

- `outputs/stage4_radar_image_geometry_20260908_135511/frame_metrics.csv`
- `outputs/stage4_radar_image_geometry_20260908_135511/grouped_metrics.csv`
- `outputs/stage4_radar_image_geometry_20260908_135511/report.json`

以上是此前审计的只读归档。按当前工具复现时使用的新目录前缀为
`stage4_geometry_audit_`，不会覆盖归档。

```bash
python tools/radar_image_geometry_audit.py --with-oracle --oracle-gate-m 2.0
```
