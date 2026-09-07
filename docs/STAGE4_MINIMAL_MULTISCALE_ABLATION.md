# Stage 4.6：Minimal Multi-Scale Visual Fusion Ablation

## 实验问题

在stride8已证明优于stride16后，验证补充stride16深层语义是否还能持续改善tiny UAV二维定位。本轮只增加最小top-down additive fusion：stride8/stride16分别1×1投影到256，深层上采样到stride8后相加，再复用现有`DualViewTokenizer`最终投影与RDQ。未读取test。

## 固定协议

- RDQ，sigmoid CXCYWH，L1-only，GIoU weight=0
- backbone/new-module LR均为`1e-4`
- seed42，固定20样本与顺序，batch2，600 steps
- 每50 steps以eval模式评估固定全集
- 除视觉backbone的单层/双层输出外，Radar、fusion、heads、loss与optimizer完全相同
- Tail固定为steps 400/450/500/550/600

## 实际几何与成本

| Variant | Feature/视图 | 双视图tokens | Parameters | CPU runtime | GPU memory |
|---|---:|---:|---:|---:|---:|
| stride8-only | 36×48 | 3456 | 1,617,479 | 201.13 s | N/A |
| minimal_fpn | 36×48 | 3456 | 3,848,775 | 267.07 s | N/A |

minimal_fpn增加2,231,296参数（+137.95%），CPU运行时间增加32.78%。本次环境没有CUDA，不能给出可信GPU显存增量。两组最终token接口完全相同。

## Best checkpoint（按最低mean center error）

| Variant | Step | Center mean/median | P(<2px) | P(<4px) | Mean IoU | Recall@0.5 | 3D error |
|---|---:|---:|---:|---:|---:|---:|---:|
| stride8-only | 500 | 6.891 / 3.806 px | 0.10 | 0.60 | 0.507 | 0.75 | 0.297 m |
| minimal_fpn | 500 | **4.696 / 3.410 px** | 0.05 | **0.65** | **0.573** | **0.80** | 0.318 m |

minimal_fpn的单点best mean center改善31.85%、IoU提升0.065，但P(<2px)反而从0.10降到0.05。3D仅附带记录且略有恶化，不参与结论。

## Tail稳定性

| Variant | Tail center mean±sample std | Tail Mean IoU±sample std |
|---|---:|---:|
| stride8-only | **9.725±2.855 px** | **0.336±0.149** |
| minimal_fpn | 12.892±6.142 px | 0.222±0.236 |

相对stride8-only：

- minimal_fpn tail center error恶化32.57%；
- tail Mean IoU下降0.114；
- 两项tail标准差均明显增加；
- step500峰值后，IoU在step550/600降至0.056/0.027。

因此best checkpoint的改善是孤立峰值，不符合预设的持续改善要求。

## 结论

- **Hypothesis**：补充stride16深层语义可在stride8基础上持续提升tiny-UAV二维定位。
- **Controlled variable**：stride8-only与最小stride8+stride16 additive fusion。
- **Fixed variables**：RDQ、Radar、token接口、heads、loss、LR、optimizer、数据、seed和训练顺序。
- **Metrics**：center error及阈值命中率为主，Mean IoU为辅；使用best与固定tail共同判断。
- **Result**：minimal_fpn获得更好的孤立峰值，但tail center、tail IoU和稳定性均明显差于stride8-only。
- **Status**：**rejected**。
- **Interpretation**：当前证据不支持为最小RDQ增加stride16语义融合；高分辨率stride8本身更合适。不能以step500峰值为由继续堆叠正式FPN。
- **Next action**：保留stride8-only；按路线进入single-token spatial bottleneck audit，但需等待用户确认。
