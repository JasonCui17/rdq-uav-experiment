# Stage 4 定位失败分析路线图

本阶段目标不是追求最高 IoU，而是依次定位最小 RDQ 在 tiny-UAV 2D/3D 定位中的主要瓶颈。所有实验只使用 train/validation，不读取 test；每次只改变一个主要变量，并遵守上一阶段的 stop criteria。

## 决策顺序

1. Stage 4.3：tiny-batch BatchNorm / learning-rate 稳定性
2. Stage 4.4：background / trajectory shortcut
3. Stage 4.5：stride16 与 stride8 空间分辨率
4. Stage 4.6：single fused token 空间压缩
5. Stage 4.7：Radar 当前时刻对应性
6. Stage 4.8：dual-fisheye seam / edge 区域误差

## 总表

| Hypothesis | Experiment | Key metric | Result | Status | Interpretation | Next action |
|---|---|---|---|---|---|---|
| BN instability | normal BN vs frozen running statistics | tail Mean/Std/Min IoU、final IoU、train/eval gap | Freeze-BN tail=0.222±0.220，final=0.007 | rejected | 冻结 BN 提高单点峰值，但没有稳定 tail，也没有消除崩塌或 train/eval gap | 暂停；不继续调 BN |
| LR instability | new-module LR 1e-3 vs 1e-4 | tail Mean/Std/Min IoU、final IoU | Low-LR tail=0.234±0.157，min=0.091，final=0.139 | inconclusive | 稳定性相对 baseline 改善，但未达到 tail mean≥0.45 的支持阈值 | 经确认后进入 Stage 4.4，而非继续调 optimizer |
| background shortcut | 固定36×48低频双视图、冻结ResNet18、小MLP | 相对global mean的2D/3D gain | 2D +2.54%，3D +44.56% | rejected | joint shortcut较弱，但3D background prior为moderate | 允许Stage 4.5；后续3D必须保留控制基线 |
| trajectory shortcut | train-only mean/nearest/interpolation | 相对global mean的2D/3D gain | 插值：2D +3.25%，3D +44.75% | rejected | joint shortcut较弱，但3D trajectory prior为moderate | 允许Stage 4.5；不因shortcut修改split |
| stride16 bottleneck | out_index3 stride16 vs out_index2 stride8 | tail center error、tail Mean IoU | center −16.27%，tail IoU +0.1021 | supported | stride8使平均目标从不足1 cell提升至约1 cell，改善跨越完整tail | 暂停；高分辨率表示有依据，等待是否研究P2/P3/FPN |
| multi-scale semantic fusion | stride8-only vs minimal stride8+stride16 additive fusion | tail center error、tail Mean IoU | center恶化32.57%，IoU −0.114 | rejected | 更好best是孤立峰值；增加深层语义使tail更差且成本显著增加 | 保留stride8-only，等待single-token audit |
| single-token bottleneck | fused-token center vs真实attention中心 | tail center error | fused 9.67 px；mean-attention 204.44 px | inconclusive | attention本身没有找到UAV，不能归因于后续single-token压缩 | 不实现spatial decoder |
| Radar-to-visual spatial correspondence | per-head attention soft-argmax/peak/GT mass | attention center、entropy、GT mass | best fixed head 134.30 px；entropy 0.9946；GT mass低于均匀参考 | rejected | 当前Radar query没有形成目标附近的空间attention；fused head可经query residual/radar skip绕过它 | 只建议temporal-shift correspondence eval |
| Radar temporal correspondence | frozen checkpoint temporal shift eval | IoU/center/3D error shift curve | 未执行 | inconclusive | 空间attention correspondence已被拒绝，仍需区分当前Radar与session prior | 等待用户决定 |
| large-sample pathway attribution | Full/no-skip/Radar-only/Pure-Attention + modality interventions | full-val center error与attention空间诊断 | Full 32.61 px；no-skip 32.81；Radar-only 42.27；Pure 37.25 | supported | 视觉与Radar均有贡献，但空间attention未对齐；主要经标准query-residual/FFN与全局attended values路径 | 可讨论geometry/target-aware query，当前停止 |
| fisheye seam issue | fixed canvas-region grouped eval | seam/edge vs interior error | 未执行 | inconclusive | 只允许 read-only audit | 等待前置结论 |

## Stage 4.3：Tiny-Batch Optimization Stability

- **Hypothesis**：BatchNorm running statistics 或 new-module LR=1e-3 破坏已经学到的 tiny-box 解。
- **Controlled variable**：A 正常 BN/LR；B 仅冻结 BN running statistics；C 仅将 new-module LR 降为 1e-4。
- **Fixed variables**：RDQ、sigmoid CXCYWH、L1-only、20 samples、seed42、batch2、600 steps、AdamW、backbone LR=1e-4、增强、缓存样本和样本顺序。
- **Metrics**：best 与 tail IoU、final IoU、train/eval gap、BN running statistics、bbox/XYZ 误差和 head gradient norm。
- **Result**：A/B/C 均完成600 steps；完整记录位于 `outputs/stage4_optimization_stability_20260907_155837/`。
- **Status**：**inconclusive**。BN 子假设 rejected；Low-LR 有部分稳定性改善但未达到预设支持标准。
- **Stop criteria**：已满足“完成 A/B/C 后停止”；Stage 4.4 未执行。

### 严格配对结果

| Variant | Best IoU | Best step | Best Recall@0.5 | Tail mean | Tail sample std | Tail min | Final IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.512551 | 500 | 0.70 | 0.157891 | 0.210854 | 0.003006 | 0.003006 |
| Freeze-BN | **0.592107** | 450 | 0.65 | 0.221731 | 0.220086 | 0.007289 | 0.007289 |
| Low-LR | 0.439973 | 500 | 0.60 | **0.233693** | **0.157069** | **0.091471** | **0.139003** |

Tail 固定为 steps 400、450、500、550、600；标准差为 sample standard deviation。Freeze-BN 的单点峰值最高，但其 tail 标准差反而略高于 baseline，且 final 同样崩塌。Low-LR 改善了 tail mean、tail min、tail std 和 final IoU，但没有达到预先规定的 `tail_mean_iou >= 0.45`。

### Train/Eval mode gap

| Variant | Tail mean absolute gap | Tail signed gap | Final train IoU | Final eval IoU | Final gap (train-eval) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.075580 | -0.007636 | 0.094057 | 0.003006 | +0.091051 |
| Freeze-BN | 0.073912 | -0.046395 | 0.036842 | 0.007289 | +0.029553 |
| Low-LR | 0.110534 | -0.110534 | 0.120748 | 0.139003 | -0.018255 |

Freeze-BN 后仍存在明显 checkpoint-dependent gap；此时 BN 在 train/eval 中均固定，剩余差异主要包含 Dropout 与小样本随机前向效应。因此 BN running statistics 不是当前崩塌的充分解释。

### BN running statistics（step600）

| Variant | mean(abs(running mean)) | mean running var | min running var | max running var | num batches tracked |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.407958 | 0.703094 | 0.008822 | 7.459703 | 600 |
| Freeze-BN | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 0 |
| Low-LR | 0.408211 | 0.728999 | 0.010769 | 4.032652 | 600 |

共有15个 BatchNorm 层。Freeze-BN 的统计量严格保持初始化值，证明实现只冻结 running statistics；affine 参数仍参与反向传播。train-mode diagnostic 在计算前后备份并恢复 BN buffers 与 RNG，不改变训练轨迹。

### 结论与下一动作

- **Hypothesis**：tiny-batch BN 或 new-module LR 是不稳定性的主要来源。
- **Controlled variable**：仅切换 BN running-stat 更新，或仅将 new-module LR 从 `1e-3` 降至 `1e-4`。
- **Fixed variables**：RDQ、sigmoid CXCYWH、L1-only、20样本、seed42、batch2、600 steps、AdamW、backbone LR、增强、缓存样本与顺序。
- **Metrics**：best/tail/final IoU、train/eval gap、BN buffers、bbox/XYZ误差和 head gradient norm。
- **Interpretation**：BN 冻结不能阻止崩塌；Low-LR 只提供部分稳定性改善。当前问题更像 L1 在像素极敏感目标上的高方差更新、Dropout/mini-batch噪声或后续待审计的数据/空间表征问题，而非单一 BN running-stat 故障。
- **Next action**：保持暂停。若用户确认，严格按路线进入 Stage 4.4 Shortcut Audit；不继续调 optimizer。

## Stage 4.4：Localization Shortcut Audit

- **Hypothesis**：在全部样本均为UAV-positive frame时，仅凭session/time或固定低频背景，已经可以预测validation UAV的2D中心与3D位置。
- **Controlled variable**：global/sequence mean、同sequence最近train时间、同sequence train线性插值，以及不依赖GT bbox的低频RGB回归器。
- **Fixed variables**：`manifests_oracle_left_fixed256_bbox`的train/val、768×288虚拟画布、相同GT定义；不读取test。
- **Metrics**：2D中心mean/median pixel error，3D mean/median Euclidean error，MAE x/y/z，以及相对global mean的Gain2D/Gain3D。
- **Result**：完整结果位于 `outputs/stage4_shortcut_audit_20260907_182615/`。
- **Status**：trajectory joint shortcut为 **weak**；background joint shortcut为 **weak**。二者的3D分量均为 **moderate**。
- **Interpretation**：time/background无法有效预测2D中心，但可将3D error从8.25 m降低到约4.56 m，说明3D评估存在不可忽略的session/trajectory prior。
- **Next action**：不需要因joint shortcut暂停全部结构实验；可在用户确认后进入Stage 4.5，但所有后续3D结果必须同时报告global/time/background控制基线。

### A. Train-only trajectory baselines

| Method | 2D mean/median (px) | 3D mean/median (m) | MAE x/y/z (m) | Gain2D | Gain3D |
|---|---:|---:|---:|---:|---:|
| Global mean | 48.48 / 46.15 | 8.25 / 7.62 | 1.47 / 4.78 / 5.82 | 0.00% | 0.00% |
| Sequence mean | 54.25 / 53.98 | 8.51 / 7.69 | 1.54 / 4.97 / 5.77 | -11.89% | -3.12% |
| Nearest train time | 51.61 / 46.96 | 4.82 / 3.80 | 1.56 / 4.51 / 0.14 | -6.45% | +41.55% |
| Linear interpolation | **46.91 / 45.52** | **4.56 / 3.40** | **1.44 / 4.23 / 0.43** | +3.25% | +44.75% |

Nearest与interpolation只查询同`sequence_id`的train样本；区间外固定使用最近train端点。没有使用val label进行查找或拟合。按joint score `max(min(Gain2D, Gain3D))`，trajectory gain为3.25%，判为weak；若单独看3D，则44.75%属于moderate。

### Nearest train time gap

- Mean：8.32 s
- Median：7.00 s
- P95：20.66 s
- Max：24.80 s

| Gap | Samples | 2D mean error (px) | 3D mean error (m) |
|---|---:|---:|---:|
| 0–1 s | 0 | — | — |
| 1–2 s | 0 | — | — |
| 2–5 s | 129 | 26.44 | 2.72 |
| 5–10 s | 179 | 62.71 | 6.60 |
| 10–30 s | 107 | 63.38 | 4.39 |
| ≥30 s | 0 | — | — |

误差并非随gap严格单调，说明轨迹形态与被holdout temporal block的位置也有影响；不能把time gap单独当成充分解释。

### B. Fixed low-frequency RGB baseline

每个视图固定从288×384降采样为36×48，不使用GT bbox、Radar或自适应处理。使用本地缓存的ImageNet ResNet18并完全冻结，分别global average pooling后拼接，只训练`1024→128→5`小MLP。

| Method | 2D mean/median (px) | 3D mean/median (m) | MAE x/y/z (m) | Gain2D | Gain3D |
|---|---:|---:|---:|---:|---:|
| Background-only | 47.25 / 35.08 | 4.58 / 4.54 | 1.11 / 3.61 / 1.89 | +2.54% | +44.56% |

按joint score `min(Gain2D, Gain3D)`，background gain为2.54%，判为weak；单独3D gain为44.56%，属于moderate。由此不能声称低频背景足以完成联合定位，但也不能将未来3D性能全部归因于UAV目标证据。
