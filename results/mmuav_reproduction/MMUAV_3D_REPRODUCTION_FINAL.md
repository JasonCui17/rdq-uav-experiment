# MMUAV 3D Reproduction Final

## 1. Objective

重建一版可运行、可评价的 MMUAV LiDAR 3D 定位/轨迹分支，并规范比较论文数值。
固定 sequence-level split：72 train / 15 validation / 15 heldout，seed=42。
本次收尾只读既有预测和 GT，不训练、不拟合、不重跑。Heldout 未参与模型或配置选择。

## 2. Reconstructed Pipeline

Raw Mid360 → 20-frame accumulation / DBSCAN / M1 → processed Mid360；
Raw Livox → zero removal / FPS → processed Livox；
fusion → candidate DBSCAN → M2 FULL → Kalman → longest-lived valid track
→ AR(3) → interpolation → B-spline → final XYZ → GT evaluation。M3 BYPASSED。
全部合法 candidate 进入系统推理；GT 不参与 candidate/track 选择、补全或平滑。
GT timestamp 只作评价时间坐标，GT XYZ 仅在预测生成后参与评分。

## 3. Public Code vs Reconstructed Components

| Component | Status |
|---|---|
| Mid360 / Livox / fusion / candidate clustering | Public-code logic, unchanged parameters; M1 tuple logits adapter |
| M0 classifier | Public 9D ordinary LSTM; not paper 7D Attention LSTM |
| M1 | Reconstructed 9D scalar attention over all hidden states |
| M2 | Reconstructed PointNet-like residual regressor, local points + observed center |
| Kalman / association | Public StoneSoup settings and update logic |
| Track selection / deleted-track archive | Reconstructed selection and engineering sidecar |
| AR3 completion | Reconstructed per-track OLS, no GT; fixed limits |
| Linear / B-spline | Public maths; bounded support / maximum1s gap reconstruction |
| M3 polynomial bias | Bypassed; unknown24D definition |

原仓库 dtc111111/Multi-Modal-UAV commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7。
Frozen config SHA256: `c192bc120b45d9851884c8ee21f6d1953d70d7293790a0d3e560197c0ef9b45f`。本地重建不等于公开代码 baseline，也不等于论文严格实现。

## 4. Module-level Results

Classifier dataset: train58,701 clusters/140 positives；validation12,036/46 positives。
Official public checkpoint F1=.9318；M0 retrained best F1=.9890；M1 best F1=1.0000。
这些是固定构建条件下的 classifier 指标，不是系统 candidate detection recall。
M2/M2.5 使用 GT-conditioned module-level 的2359个冻结 validation IDs：

| Model | MSE_coord | Mean 3D error (m) |
|---|---:|---:|
| Geometric baseline | 0.276017 | 0.795198 |
| M2_FULL | 0.062591 | 0.360731 |
| M2_POINTS_ONLY | 0.104097 | 0.467844 |
| M2_CENTER_ONLY | 0.068860 | 0.382692 |

FULL 最佳epoch3/停止epoch18。绝对观测位置提供大部分校正价值，局部结构提供额外收益；
不把 FULL 的改善全部归因于目标形状。

## 5. Validation Results

| Mode | Coverage | Matched / Missing | MSE_coord | MSE_3D | Mean / Median 3D error (m) |
|---|---:|---:|---:|---:|---:|
| GEOMETRIC | 80.88% | 4853 / 1147 | 0.377161 | 1.131484 | 0.898811 / 0.706902 |
| FULL | 80.88% | 4853 / 1147 | 0.058698 | 0.176093 | 0.348128 / 0.296206 |
| FULL_TEMPORAL | 90.75% | 5445 / 555 | 0.058620 | 0.175859 | 0.345912 / 0.303191 |

15/15 三模式成功，无 failure/empty。GEOMETRIC 与 FULL matched timestamp sets 完全一致。
FULL 的中心校正收益进入轨迹后仍存在；temporal主要增加coverage。

## 6. Frozen Heldout Results

| Mode | Coverage | Matched / Missing | MSE_coord | MSE_3D | Mean / Median 3D error (m) |
|---|---:|---:|---:|---:|---:|
| GEOMETRIC | 74.98% | 4499 / 1501 | 2.017344 | 6.052032 | 1.263041 / 0.881165 |
| FULL | 75.03% | 4502 / 1498 | 1.734186 | 5.202557 | 0.625553 / 0.305578 |
| FULL_TEMPORAL | 91.20% | 5472 / 528 | 1.398907 | 4.196720 | 0.548464 / 0.285503 |

15/15 三模式成功，无 failure/empty。GEOMETRIC 与 FULL 原始匹配集合不完全一致。
原始汇总降幅只作描述；严格中心校正证据以如下完全配对集合为准：

| Model | Same timestamps | MSE_coord | MSE_3D | Mean / Median 3D error (m) |
|---|---:|---:|---:|---:|
| GEOMETRIC | 4499 | 2.017344 | 6.052032 | 1.263041 / 0.881165 |
| FULL | 4499 | 1.735256 | 5.205769 | 0.625559 / 0.305491 |

共同时间戳=4499；geometric-only=0；full-only=3。
配对 MSE_coord下降13.98%，mean3D下降50.47%，median3D下降65.33%。
原始汇总 MSE_coord/mean/median 降幅分别为
14.04% /
50.47% /
65.32%，不是严格配对估计。
Temporal coverage提升16.17pp，新增970个匹配时间点。
其 available-prediction error 来自不同集合，不能将总体 MSE 变化解释成同样本精度提升。
辅助同时间戳比较见 heldout_temporal_paired_timestamp_comparison.csv，不替代coverage。
FULL∩FULL_TEMPORAL=4474；temporal新增998个，
另有28个FULL可用时间戳不在temporal可用集合内，净增加970个。
辅助配对 MSE_coord：1.701663 → 1.703046；
mean3D：0.621960 → 0.608943m。
不以总体MSE的下降证明同样本精度大幅提升；本轮不改变这些支持区间规则。
Error仅对matched计算，missing/coverage显式报告；不能隐藏无预测时间点，也不对missing伪造有限误差。

Performance contains heavy-tail failures。Top3按per-sequence MSE_3D排序，贡献为
MSE_3D × matched_count / overall squared-error sum，不作因果定位：

| Mode | Sequence | MSE_3D | Contribution to overall squared error |
|---|---|---:|---:|
| full | seq0065 | 80.260892 | 97.66% |
| full | seq0013 | 0.281881 | 0.29% |
| full | seq0066 | 0.278194 | 0.47% |
| full_temporal | seq0065 | 77.122072 | 97.39% |
| full_temporal | seq0066 | 0.254505 | 0.44% |
| full_temporal | seq0013 | 0.243193 | 0.42% |

## 7. Comparison with Paper

论文 §4.2 / Results table，参考既有 paper-code audit；原文见
[MMUAV technical report](https://arxiv.org/abs/2405.16464)。

| Center regression | Before | After | Reduction |
|---|---:|---:|---:|
| Paper | .270000 | .050000 | 81.48% |
| Our reconstruction (local MSE_coord) | 0.276017 | 0.062591 | 77.32% |

Before高于论文数值2.23%；after高于25.18%。
**NUMERICAL COMPARISON, not strict reproduction**：metric_alignment=unresolved；
sample_alignment=unresolved；architecture_alignment=partial。

| Final metric reference | Value | Status |
|---|---:|---|
| Paper official test Pose MSE | 2.21375 | NUMERICAL REFERENCE ONLY |
| Our FULL_TEMPORAL heldout MSE_coord | 1.398907 | Local metric |
| Our FULL_TEMPORAL heldout MSE_3D | 4.196720 | Local metric |

没有证据证明论文PoseMSE等于任一当地定义，而且challenge test与official-train留出集不同。
因此不计算与论文最终PoseMSE的百分比优劣，不声称达到或超过论文。

## 8. Key Findings

几何均值不是可靠的最终目标位置。学习式中心校正在module、validation和paired-heldout有效。
Observed absolute position 与局部结构具有互补价值；temporal主要验证coverage收益。
Heldout有明显长尾失败：典型点准确不代表所有sequence可靠；本轮只归因统计，不解决或调参。

## 9. Transferable Ideas for Radar+RGB

不要直接将 Radar/LiDAR cluster centroid 当最终3D表示。
Radar cluster geometry + observed spatial position prior + RGB target feature
→ learned target-center correction → corrected3D position；temporal独立负责连续性。
这是待验证研究假设，不是本次LiDAR结果已经证明Radar有效。

## 10. Known Limitations

1. 论文 7D dynamic feature 定义未完全恢复；本轮使用 9D。
2. 论文 center regression 的 PointNet/MLP 具体结构有歧义，本轮为明确标记的重建。
3. 论文 24D third-order polynomial 定义未恢复，M3 在本轮旁路。
4. Paper Pose MSE 与本地 MSE_coord / MSE_3D 口径未对齐，不能据此判定优劣。
5. 本地 heldout 是官方 train 的 sequence-level 固定留出集，不是官方 challenge test。


## 11. Final Status

Frozen configuration和模型保持不变；未进行收尾训练、拟合、重跑或heldout结果驱动的方法选择。
This is a completed reconstruction, not strict paper-level numerical replication.

**MMUAV 3D LOCALIZATION RECONSTRUCTION COMPLETE**
