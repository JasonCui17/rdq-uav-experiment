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
| background shortcut | validation UAV erase diagnostic | retained localization performance | 未执行 | inconclusive | Stage 4.3 前禁止执行 | 等待确认 |
| trajectory shortcut | time-only XYZ MLP | validation 3D error relative to full model | 未执行 | inconclusive | Stage 4.3 前禁止执行 | 等待确认 |
| stride16 bottleneck | stride16 vs stride8 | Mean IoU、center error、runtime/memory | 未执行 | inconclusive | 需先完成 shortcut audit | 等待前置结论 |
| single-token bottleneck | fused-token regression vs center heatmap | center error thresholds | 未执行 | inconclusive | 需先判断空间分辨率 | 等待前置结论 |
| Radar correspondence | frozen checkpoint temporal shift eval | IoU/center/3D error shift curve | 未执行 | inconclusive | 需先获得稳定 baseline | 等待前置结论 |
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
