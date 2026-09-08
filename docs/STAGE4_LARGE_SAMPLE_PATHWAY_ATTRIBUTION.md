# Stage 4.8：Large-Sample RDQ Pathway Attribution Audit

## 实验问题与固定协议

本轮验证Stage 4.7的空间attention结论是否只是20样本偶然现象，并区分当前center定位经由radar skip、Radar-only、query residual还是纯MHA attended value完成。四组共享seed42随机固定的512个train样本，验证使用完整415个val样本；batch8，ResNet18 stride8-only，center L1，backbone/new-module LR均为`1e-4`。每组最多150 epochs，val mean center error连续15 epochs不改善即停止，并回载validation-best checkpoint。未读取test。

固定512样本类别分布：Avata 161、Mavic2 91、Pham4 88、M300 75、Mavic3 97。

## 四条路径

| Pathway | Best / stopped epoch | Mean / median center error | P(<2px) | P(<4px) | P(<8px) |
|---|---:|---:|---:|---:|---:|
| Full RDQ | 12 / 27 | **32.61 / 29.42 px** | 0.000 | 0.019 | **0.140** |
| No Radar Skip | 15 / 30 | 32.81 / 29.46 px | **0.014** | **0.034** | 0.077 |
| Radar-only | 30 / 45 | 42.27 / 35.03 px | 0.000 | 0.002 | 0.089 |
| Pure Attention | 9 / 24 | 37.25 / 32.59 px | 0.017 | 0.027 | 0.051 |

相对Full RDQ，关闭radar skip只使mean error增加0.63%，因此radar skip不是主要bypass。Radar-only误差增加29.63%，不能解释Full RDQ能力。Pure Attention误差增加14.25%，说明标准attention block中的query residual/FFN residual路径提供重要信息；但Pure Attention仍比Radar-only低5.01 px，表明视觉attended value也有贡献。该对比同时删除query residual与post-attention FFN residual，不能进一步分离二者。

## Full RDQ validation intervention

| Input | Mean / median error | P(<8px) | 相对Normal误差变化 |
|---|---:|---:|---:|
| Normal | **32.61 / 29.42 px** | **0.140** | — |
| Shuffle image | 46.71 / 34.63 px | 0.017 | +43.24% |
| Zero image | 76.61 / 69.48 px | 0.000 | +134.94% |
| Same-class shuffle Radar | 39.54 / 29.54 px | 0.055 | +21.25% |
| Zero Radar | 198.05 / 202.77 px | 0.000 | +507.39% |

图像shuffle/zero均显著伤害center定位，故视觉不是可忽略输入；Radar shuffle也造成可见退化，zero Radar则使模型彻底失效。当前结果支持双模态依赖，而不是Radar-only/global prior完成定位。

## Full RDQ真实attention

Normal输入下实际shape为`[16,8,1,3456]`（最后一个batch尺寸不同），空间诊断如下：

| Readout | Mean center error |
|---|---:|
| Mean-head soft-argmax | 236.60 px |
| Mean-head peak | 235.06 px |
| Best fixed head | 146.14 px |
| Per-sample oracle-best head | 136.11 px |
| Stride8 grid oracle | 3 px量级（同一36×48网格） |

归一化attention entropy为`0.8285`。它明显低于20-sample实验的`0.9946`，所以“大样本attention仍近乎均匀”这一字面描述没有复现；但是其集中位置仍不是UAV：GT ±1 cell mass为`0.000943`，低于均匀参考`0.001159`；GT ±2 cells为`0.006158`，仅略高于均匀参考`0.004632`，而所有attention中心读出仍有136–237 px误差。

Shuffle image时attention soft-argmax几乎不变（236.55 px），但fused center恶化14.10 px。这说明视觉影响主要经由attended **values/全局视觉表征**进入head，而不能由“attention权重在目标附近”解释。

## 归因结论

- **Radar skip bypass**：rejected。关闭后mean error仅变化0.21 px。
- **Radar-only/global prior主导**：rejected。Radar-only比Full差9.66 px，图像干预明显伤害模型。
- **纯空间attention已经定位目标**：rejected。mean/peak/best-head全部远离GT。
- **主要有效路径**：标准RDQ的query-residual/FFN residual路径与全局visual attended values共同作用；其中radar skip不是关键。由于Pure Attention同时去除了query和FFN residual，本轮不能声称差异完全由query residual单独造成。
- **Stage 4.7复现情况**：核心结论“attention未对齐UAV”在512/415规模上得到验证；“attention近乎均匀”没有原样复现，它变得更集中，但集中在非目标区域。
- **下一步建议**：可以进入geometry-aware / target-aware Radar query讨论，目标应是建立显式Radar-to-image空间对应，而不是继续增强global fused head。本轮到此停止，不自动实现下一阶段。

完整结果：`outputs/stage4_large_sample_pathway_20260908_010459/`。
