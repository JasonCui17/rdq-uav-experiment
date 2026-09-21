# Stage 4 LiDAR → GT 独立坐标求解

## 目的与边界

本实验分别求解 `lidar_360 → GT` 和 `livox_avia → GT`。两者从 identity
独立开始拟合，不加载 Radar transform，也不共享旋转或平移。只读取固定的
train/val manifest，没有读取 test。

处理流程：

1. 对每个 GT 时间戳，在同一 sequence 内寻找时间最近的传感器 NPY；
2. 删除 NaN/Inf 和 `(0,0,0)` 填充点；
3. 先按传感器独立有效距离标记 `OUT_OF_RANGE`；
4. 范围内的空点云标记 `NO_VALID_CLOUD`；
5. 仅 `ELIGIBLE = IN_RANGE + VALID_CLOUD` 帧参与拟合和 Hit@X 统计；
6. 在 train 上依次拟合 identity、24 个 proper signed-axis rotations + 有界平移、
   continuous rigid refinement；
7. 在 val 和 val same-sequence half-cycle shuffle 上只评估，不选择参数。

采用的保守距离范围为：Mid-360 `0.1–40 m`，Avia `1–130 m`。当前受控
train/val 的 GT 距离约为 `3–27 m`，因此没有样本被距离范围排除。产品 FOV
虽然已知，但安装后的传感器轴相对 GT 正是待求量；为避免循环筛选，本轮没有启用
角度 observability mask。

## 数据状态

| Sensor | Split | Total | In range | Out of range | Valid cloud / eligible | No valid cloud | median dt |
|---|---|---:|---:|---:|---:|---:|---:|
| lidar_360 | Train | 1828 | 1828 | 0 | 1828 | 0 | 23.00 ms |
| lidar_360 | Val | 415 | 415 | 0 | 415 | 0 | 21.25 ms |
| livox_avia | Train | 1828 | 1828 | 0 | 1449 | 379 | 27.76 ms |
| livox_avia | Val | 415 | 415 | 0 | 307 | 108 | 31.83 ms |

所有 Hit@X 的分母都是表中的 eligible，而不是 total。

## lidar_360 → GT

连续 refinement 相对最佳离散解的 train robust objective 仅改善 `1.69%`，低于
预先固定的 `5%` 阈值，因此保留离散轴解：

```text
x_gt =  z_lidar + 0.6383
y_gt = -x_lidar + 0.2640
z_gt = -y_lidar + 0.2353
```

即：

```text
R = [[ 0,  0,  1],
     [-1,  0,  0],
     [ 0, -1,  0]]
t = [0.6383, 0.2640, 0.2353] m
```

| Evaluation | Mean | Median | P75 | P90 | Hit@0.5 | Hit@1 | Hit@2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train normal | 2.241 m | 1.158 m | 2.455 m | 5.307 m | 22.92% | 45.24% | 67.23% |
| Val normal | 3.161 m | 1.461 m | 3.419 m | 9.386 m | 7.23% | 24.10% | 56.63% |
| Val shuffle | 3.156 m | 1.466 m | 3.390 m | 9.479 m | 7.71% | 24.58% | 56.39% |

Identity 的 val median 为 `4.618 m`，离散解改善 `68.36%`。但是 normal 与
shuffle 的 median 分别为 `1.461/1.466 m`，改善只有 `0.36%`。因此该解只能作为
一个候选坐标约定，不能证明找到了当前 UAV 的 LiDAR 回波，也不能作为可信的目标
外参。密集环境点使最近邻目标可以在错误时间配对下同样取得较小距离。

Val 按 GT 距离：

| GT range | N | Mean | Median | P75 | P90 |
|---|---:|---:|---:|---:|---:|
| 0.1–10 m | 195 | 1.497 | 1.423 | 2.328 | 2.506 |
| 10–20 m | 154 | 2.235 | 1.257 | 3.410 | 5.782 |
| 20–30 m | 66 | 10.240 | 11.577 | 12.883 | 13.017 |

远距离区间明显失效。

## livox_avia → GT

最佳离散解约为 `180° yaw + translation`。连续 refinement 在第 36 次自然停止，
相对离散解的 train robust objective 改善 `83.48%`，超过 5% 阈值，因此选用独立
continuous rigid 解：

```text
Euler xyz = [1.0446°, 0.0603°, 164.8813°]
t = [0.0883, 0.1875, -0.0862] m
```

```text
R = [[-0.965387, -0.260796,  0.003739],
     [ 0.260820, -0.965222,  0.017874],
     [-0.001053,  0.018230,  0.999833]]
```

| Evaluation | Mean | Median | P75 | P90 | Hit@0.5 | Hit@1 | Hit@2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train normal | 0.796 m | 0.044 m | 0.100 m | 0.197 m | 98.76% | 99.10% | 99.10% |
| Val normal | 1.211 m | 0.047 m | 0.099 m | 0.267 m | 96.09% | 99.02% | 99.02% |
| Val shuffle | 4.450 m | 2.654 m | 4.098 m | 6.898 m | 4.89% | 6.84% | 28.66% |

Identity 的 val median 为 `1.919 m`；独立刚体解降至 `0.047 m`，改善 `97.55%`。
同序列 shuffle 为 `2.654 m`，normal 相对 shuffle 改善 `98.23%`。因此在
**非空有效云帧**上，Avia → GT 的固定变换具有很强的时刻对应证据。

Val 按 GT 距离：

| GT range | N | Mean | Median | P75 | P90 |
|---|---:|---:|---:|---:|---:|
| 1–10 m | 184 | 1.886 | 0.029 | 0.058 | 0.112 |
| 10–20 m | 99 | 0.177 | 0.060 | 0.109 | 0.405 |
| 20–30 m | 24 | 0.298 | 0.268 | 0.327 | 0.517 |

近距离区间的 mean 被少数极端离群帧拉高，因此必须同时阅读 median/P90。Avia
仍有大量 `NO_VALID_CLOUD`：train `379/1828`，val `108/415`。这些空云没有参与
外参拟合，也不能被解释为外参失败。

## 结论

- `lidar_360`：**不可信/未解析完成**。identity 改善明显，但 normal 不优于
  shuffle，最近邻结果很可能主要来自密集背景几何。
- `livox_avia`：**可信（仅针对有效非空云）**。val 显著优于 identity，且 normal
  显著优于同 sequence shuffle。
- 两个输出完全独立，不能互换，更不能复用 Radar transform。

机器可读结果：

- `calibration/lidar360_frame_resolution.json`
- `calibration/livox_avia_frame_resolution.json`

完整运行归档：`outputs/stage4_lidar_frame_resolution_20260910_203032/`。
