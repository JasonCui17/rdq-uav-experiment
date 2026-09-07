# Stage 4.5：Spatial Resolution Ablation

## 实验问题

验证stride16视觉特征是否是5–10 px tiny UAV二维定位的主要瓶颈。本实验只改变ResNet18的`out_index`，不改变Radar encoder、RDQ、fusion、bbox/XYZ head、loss、optimizer、学习率、样本或训练顺序。未读取test。

## 固定协议

- RDQ，sigmoid CXCYWH，L1-only，GIoU weight=0
- backbone LR=new-module LR=`1e-4`
- seed42，固定20样本，batch2，600 steps
- 每50 steps以`model.eval()+torch.no_grad()`评估完整20样本
- BN正常更新
- Primary checkpoint：最低mean center error
- Tail：steps 400/450/500/550/600

## 实际特征几何

以下数据来自训练前的真实forward，不由配置名推断。

| Variant | Per-view feature | Tokens（双视图） | Effective stride | Median bbox cells W×H | Mean bbox cells W×H |
|---|---:|---:|---:|---:|---:|
| stride16 / out_index3 | 18×24 | 864 | 16×16 | 0.413×0.337 | 0.585×0.529 |
| stride8 / out_index2 | 36×48 | 3456 | 8×8 | 0.825×0.675 | 1.171×1.057 |

stride16下平均目标在宽高方向都不足一个feature cell；stride8下平均目标约占1.17×1.06 cells。

## Center-primary结果

| Variant | Best center step | Center mean/median | P(<2px) | P(<4px) | IoU / Recall@0.5 | 3D error |
|---|---:|---:|---:|---:|---:|---:|
| stride16 | 550 | 7.450 / 6.829 px | 0.00 | 0.10 | 0.364 / 0.15 | 0.290 m |
| stride8 | 500 | **6.891 / 3.806 px** | **0.10** | **0.60** | **0.507 / 0.75** | 0.297 m |

stride16的最高IoU出现在step500：Mean IoU=0.440、Recall@0.5=0.60、P(<2px)=0、P(<4px)=0.55。stride8的center与IoU最佳点均为step500。

## Tail稳定性

| Variant | Tail center mean±sample std | Tail Mean IoU±sample std |
|---|---:|---:|
| stride16 | 11.614±3.708 px | 0.234±0.157 |
| stride8 | **9.725±2.855 px** | **0.336±0.149** |

- Tail center error降低：16.27%
- Tail Mean IoU提升：+0.1021

改善存在于固定五个tail checkpoints的聚合结果，不是单一checkpoint。center改善未达到30%，但tail Mean IoU超过预设的+0.10支持阈值。

## 运行成本

| Variant | Runtime（CPU） | Relative | Peak GPU memory |
|---|---:|---:|---:|
| stride16 | 253.97 s | baseline | N/A |
| stride8 | 224.09 s | -11.77% | N/A |

本次执行环境未暴露CUDA，因此不能给出可信GPU显存增量。stride8虽然token数为4倍，但backbone在更浅层提前退出，CPU总时间没有增加；该现象不能外推为GPU显存不会增加。

## 结论

- **Hypothesis**：stride16空间分辨率是tiny-UAV二维定位的主要瓶颈。
- **Controlled variable**：仅ResNet18 `out_index=3`与`out_index=2`。
- **Metrics**：center error及阈值命中率为主，IoU为辅；3D不作为决策依据。
- **Result**：stride8将平均目标覆盖从不足1 cell提升至约1 cell，tail center改善16.3%，tail IoU提升0.102。
- **Status**：**supported**。
- **Interpretation**：高分辨率视觉表示有明确依据，但当前实验只证明stride8优于stride16，不等于已经证明FPN或P2/P3融合必需。
- **Next action**：暂停。等待决定是否以stride8为高分辨率baseline，再单独设计P2/P3或FPN实验；不自动增加复杂模块。

3D限制：Stage 4.4已发现time interpolation与low-frequency background的Gain3D约44.7%，因此本阶段3D差异不能作为stride8有效的主要证据。
