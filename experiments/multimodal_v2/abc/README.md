# A/B/C 候选预测实验设计与执行

基于 `refactor-v2-standalone` 的 `6890f2e10d272c029b1dcdbb31e6eb03371b9ec8`。
本次实现针对 B0 点云候选生成、损失、排序。实际输入是 Avia/Mid360 LiDAR；本轮不更改 RGB、几何关联或多模态评分头，避免把多个因素混在一起。胜出方案再接回 B2/B3。

## 1. 已实现与未完成

- 已实现：A 软标签、A 独立质量头、B 距离加权、B OHEM、可选距离/密度特征、C query decoder。
- 旧配置不打开新开关时，保留旧模型参数键、原损失、候选筛选行为。
- 已提供：严格 A2 warmstart、独立输出目录、配置/权重/划分哈希、训练及评估调度、逐 query 配对比较。
- 当前机器没有 CUDA、MMAUD 数据或真实 A2 checkpoint。真实训练、独立测试及方案优劣结论尚未完成。
- CPU 合成训练只用于检验梯度、保存重载和端到端接口，不能作为性能证据。

## 2. 公平对照矩阵

全部实验从同一个原 A2 Lightning checkpoint 加载所有旧参数；新模块独立初始化；重新建立优化器。相同训练集、验证集、epoch、batch、学习率、FP32、候选输出上限。不能让基线不继续训练，而只让改进方案继续训练。

| 名称 | 与共同控制组的差异 | 目的 |
|---|---|---|
| baseline | 原 Focal + 原回归；无正支持 query 仍跳过 | 相同预算继续训练的旧基线 |
| control | 原损失 + 无正支持 query 负监督 | 单独量化共享修复收益 |
| a_soft | 正体素分类目标换为距离质量，BCE | 方案 A1 |
| a_bce | 二值目标 + BCE，其余同 control | 区分软标签收益与 Focal→BCE 收益 |
| a_head | 新增质量头；最终分数为 p×q_hat | 方案 A2 |
| b_range | 正分类与回归按体素距离加权 | 方案 B 距离因素 |
| b_ohem | 每个 query 的困难负例重加权 | 方案 B 难例因素 |
| b_both | b_range + b_ohem | 方案 B 主实验 |
| b_meta | 距离、点数两维元信息的零初始化残差 | 单独测显式元信息，不与 B 主实验混合 |
| c_query | 全 L0 特征 → 32 query decoder → 一对一监督 | 方案 C |

先跑 seed=42；筛出候选后，对控制组及候选方案用相同 seeds=42/43/44 复验。三个 seed 共用同一已训练 A2 checkpoint，因此衡量的是微调随机性，不是从零训练的完整方差。每组固定 12 epoch，无早停；C 若明显未收敛，应另立长预算实验，同时延长控制组，不能只给 C 更多训练后宣称公平胜出。

## 3. 方案 A：质量感知

### A1 软标签

文件：`radar_experiment_loss.py:dense_loss / quality_targets`。

- 原正/忽略/负划分保持：体素内原始点到当前 GT 的最近距离 ≤1m / (1,2]m / >2m。
- 正体素：q=exp(-d²/(2σ²))；默认 σ=1m；负体素 q=0；忽略体素不计损失。
- d 使用 `pred_xyz.detach()` 到 `target_xyz[outputs.batch_index]` 的距离，N0 个候选对应 N0 个 GT；不使用 N_point 的 point_gt。
- 分类采用 FP32 `binary_cross_entropy_with_logits`，不把连续目标直接塞进原二值 Focal 公式。
- 定位仍使用原正体素 SmoothL1，单位、权重不变；分类的标签路径不会反传到回归分支。
- 有正体素 query 按正体素数归一化；无正支持 query 按该 query 的负体素数归一化，乘 0.25；无有效 GT 完全屏蔽。
- “无正支持”是原始点支持划分，不是“预测误差超过 1m”。二者不可混用。

注意：早期坐标差会产生很低的质量目标，因此使用 A2 warmstart；通过 a_bce 控制组判断软标签的真实收益。

### A2 独立质量头

文件：`radar_model.py:LiDARUAVDetector.quality_head`，结构 LayerNorm(128) → Linear(128,64) → GELU → Linear(64,1)。输出 `quality_logits [N0]`，使用 logits 形式 BCE 避免 AMP 概率 BCE 问题。

正体素拟合 q；可靠负体素拟合 0；忽略区不监督。质量损失按正、负分别取均值：Lqual=mean_pos(BCE)+0.25×mean_neg(BCE)。负样本也受约束，避免只在正体素学习却在高分杂波上任意输出。无正支持 query 仍训练负质量。

总损失：L=Lcls+2×Lreg+1×Lqual。Lcls 和 Lreg 保持 control 逻辑。

**同一个 p×sigmoid(quality_logits) 同时用于预选 Top-K、NMS 顺序、候选分数和最终 Top1。**只修改 `_gather` 不足以实现这个方案。质量标注停止梯度，但质量头对共享特征正常反传。

## 4. 方案 B：距离与困难负例

文件：`radar_experiment_loss.py:range_weights / negative_weights / dense_loss`。

### 距离权重

以体素中心相对 `origin_xyz_m` 的欧氏距离计算；默认原点 [0,0,0]。必须确认这是当前点云坐标中的传感器原点。距离边界 [30,60]m，权重 [1,1.5,2.5]，均是待消融超参数，不是已证明最优值。

权重作用于正体素分类项和回归项；负体素不按距离放大，避免把海量远处杂波整体放大。分母仍是原正体素数，距离再平衡的含义不被归一化抵消。

### OHEM

每个 query 独立选取最多 200 个分类 logit 最高的可靠负体素，权重 2；其余负体素权重 0.25。保留所有正体素，忽略区继续忽略。没有固定 score>0.3 阈值，避免训练初期所有负例被筛空。排序只用于选样，不反传。

有正样本时维持正样本数归一化，无正样本时维持负样本数归一化，不因切换 OHEM 改成 `cls_mask.sum()`。OHEM 与 Focal 可能重复聚焦难例，所以必须检查 b_ohem，不能先认定组合一定更好。

### 元信息独立消融

b_meta 在最终 L0 特征 q 上加 `Linear([log1p(range),log1p(point_count)],128,bias=False)` 的残差；投影零初始化。该特征同时可被候选头和后续交互使用。未加入 RCS：当前输入只读取 XYZ，没有可验证的 RCS 字段。

## 5. 方案 C：小规模 set decoder

文件：`radar_query_decoder.py`，接在原 A2 骨干的最终 `q [N0,128]` 后。

| 项目 | 实现 |
|---|---|
| Memory | 每个样本全部 L0 特征，不经过 Top50/NMS |
| 空间位置 | (体素中心−场景体素中心均值)/50m → 位置 MLP，与 memory 相加 |
| Queries | 32×128 可学习向量 |
| Decoder | 1 层，4 heads，FFN=256，dropout=0，query self-attention + cross-attention |
| 分类 | `[B,K,2]`，0=UAV、1=背景 |
| XYZ | 场景体素中心均值 + 50m×MLP(query)，无 sigmoid、无坐标裁剪 |
| 批处理 | 对每个样本独立做 ragged attention，避免串样本及 padding 干扰 |
| 监督 | 单 GT 下按联合分类/位置代价 argmin，只有 1 个匹配 query 回归 |
| 推理 | 前景 softmax 概率排序，不做半径 NMS；所有 K 参与后再取最终报告 Top10 |

场景中心与位置编码只用输入点云，不使用 GT。直接预测坐标不受 [0,1] 范围限制。原 dense head 不参与 C 的输出、loss 或参数更新；本轮不添加 dense 辅助监督，避免把 C 与混合监督合在一个实验里。

匹配代价：`-p_uav + |pred_xyz−gt_xyz|_1 / 50m`。匹配在 no_grad 下进行。回归仍以米为单位做 SmoothL1(sum XYZ)，回归系数 2。分类 CE 类别权重 `[1,0.1]`，并非宣称背景梯度完全消失；需监测前景分数、匹配稳定性及收敛。

有效 GT 代表存在目标，即使原始点中没有 1m 支持也做一次匹配；缺失 GT 完全屏蔽。没有点云的样本不凭空生成 query 输出。当前标签协议没有显式“已确认无无人机”的样本，不能据此声称学会了全场景无目标拒识。

候选适配保留现有 `CandidateBatch`：two-class logits 差值对应 sigmoid 前景概率；query feature 保持 128D，可被后续 B2/B3 消费。C 的 source_index 是 query 编号，不再是体素编号；原体素故障审计脚本不能不加区分地用于 C。

## 6. 代码修改清单

| 文件 | 作用 |
|---|---|
| radar_model.py | 可选质量头、元信息残差、query decoder 接入；默认状态兼容 |
| radar_loss.py | 旧逻辑保留，显式开关分发新损失 |
| radar_experiment_loss.py | A/B 损失和单 GT query 匹配监督 |
| radar_query_decoder.py | 全体素 memory 的 decoder 与两任务头 |
| radar_selector.py | 统一质量排名；C 不做 NMS |
| radar_experiment_init.py | A2 严格 warmstart，仅允许显式新增模块缺键 |
| training.py | 新 warmstart、C dense head 冻结、质量项及监督数量日志 |
| train.py | 配置化最佳 checkpoint 指标；resume 不再次依赖原 warmstart |
| evaluate.py | 完整 checkpoint 独立加载；增加 raw/preselection Oracle 诊断 |
| abc/run_suite.py | 生成每组独立 YAML、哈希清单、gate/train/eval 调度 |
| abc/compare.py | 样本集合与 GT 一致性检查、逐 query 配对、距离分桶、seed 汇总 |
| tests/test_radar_abc.py | 标签梯度隔离、候选筛选、两步训练、重载与样本隔离 |

## 7. 执行命令

先在基于 6890f2e 的干净工作分支应用交付补丁；较新代码先 `git apply --check`，不能直接覆盖已有修改。

```bash
git apply --check /path/to/abc_candidates.patch
git apply /path/to/abc_candidates.patch
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests
```

从仓库根目录运行。下面只需要替换 A2 checkpoint 与 MMAUD 路径；不需要 detrex 权重来训练 B0。

```bash
python experiments/multimodal_v2/abc/run_suite.py \
  --action prepare \
  --checkpoint /absolute/path/to/A2/checkpoints/best_rmse.ckpt \
  --data-root /absolute/path/to/mmaud_official_train \
  --output outputs/abc_seed42 \
  --seeds 42 --epochs 12

python experiments/multimodal_v2/abc/run_suite.py \
  --action gate --output outputs/abc_seed42 --device cuda:0

python experiments/multimodal_v2/abc/run_suite.py \
  --action train --output outputs/abc_seed42 --device cuda:0

python experiments/multimodal_v2/abc/run_suite.py \
  --action eval --output outputs/abc_seed42 --device cuda:0

python experiments/multimodal_v2/abc/compare.py --suite outputs/abc_seed42
```

- 默认生成上表 10 组。预算紧张可在 prepare/gate/train/eval 一致传入 `--variants baseline control a_soft a_head b_both c_query`，后续再开新 suite 做 b_range/b_ohem/a_bce/b_meta 消融。
- `gate` 对前 32 条训练 query 做两次更新，只验证运行链路，不报告真实性能。正式 train 的输出、优化器与 gate 完全分开。
- 调度器串行运行，便于单张 GPU 控制内存；两个 GPU 可对互不重叠的 variants 分别指定 cuda:0/cuda:1。不要同时写相同 run。
- 输出中已有 last.ckpt 时默认拒绝覆盖；中断恢复使用 `--action train --resume --variants <中断组>`。已完成的组不需重跑。
- prepare 的 `--seeds 42 43 44` 也要在后续动作中传入相同 seeds。种子 43/44 不会自动由默认参数选中。
- 运行使用保存后的配置，子进程移除继承的 `RDQ_*` 覆盖，防止 RDQ_LIDAR_CONFIG 把全部方案静默替换成同一个模型；请在 prepare 的 base-config 或显式 data-root 中确定路径。
- 配置、split、候选配置、相机标定及 warmstart 被记录哈希；修改参数应建立新 suite，不覆盖旧结果。

## 8. 如何评价、如何选胜出方案

所有方案的最佳 checkpoint 统一按验证集 Top1@1m 选择，文件名 best.ckpt。原项目 B0 默认仍按 RMSE 选择；只有本 suite 改为上述协议。事先固定评价规则，不能每组挑不同指标最好的 checkpoint。

1. 比较 Top1@1m（主指标），Oracle@10/1m、3D RMSE、median、p95、coverage（辅助指标）。
2. 逐 query 对比 control：错→对、对→错。拒绝只比较两个结果的交集；样本或 GT 不一致时报错。
3. 按 GT range 统计 [0,30)、[30,60)、[60,∞)m 的样本量及指标；远距样本太少时不作稳定结论。
4. 全部预定运行完成后，覆盖率不低于控制组的方案才进入自动候选排序；Top1 相同再比较 RMSE。单 seed 标注 provisional，不能替代复验。
5. 多 seed 报告均值和标准差。相关时序 query 不宜当成大量独立样本来虚增显著性；最终论文再按序列做配对统计。
6. 冻结选择后，用 `--action eval --split heldout_test_sub --variants control <胜出组>` 做一次独立测试；compare 的 heldout 模式只报告，不重新选胜者。

无输出 query 计入 Top1 失败和 coverage；RMSE 仅对有输出样本计算，必须与 coverage 同看。当前数据没有确认的无目标标签，因此这里评价的是误定位/误选，不能把它直接称为部署虚警率。真正的误报评估需要额外的背景-only 数据和事先固定的阈值协议。

## 9. 接回多模态的后续验证

先用胜出 B0 的模型配置与 checkpoint 接入 B2/B3，RGB checkpoint、候选上限、训练预算保持相同。必须建立旧 B0+B2/B3 与新 B0+B2/B3 的配对对照，分别看单模态改善和额外跨模态收益。A2 质量乘积可能改变基础 logit 分布，原有 ±2 跨模态修正幅度是否够用要重新检查。本次未声称 B0 获益必然转化为多模态获益。
