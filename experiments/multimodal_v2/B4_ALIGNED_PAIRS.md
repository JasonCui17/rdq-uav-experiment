# B4：对齐特征、全局配对、联合重新评分

基于 `refactor-v2-standalone` 提交 `58943f31ee07ff649e4056d882679f17a41069f2`。
这是新的 B4 实验；B0–B3 的默认候选预算、注意力、评分和 loss 保留。

## 1. 实际前向链

1. 冻结已训练的点云检测器和 DINO，使用 eval 模式。
2. 从检测器取得 NMS 前 Top50：R `[N_R,128]`、V `[N_V,128]`，各自最多50个。
   V 的 `256→128 feature_proj` 在 B4 参与训练。原分数只用于 Top50 选择与 before 诊断。
3. 两个独立 adapter 执行 `Linear→LayerNorm→GELU→Linear→LayerNorm`，输出仍为128维。
4. 对 adapter 输出做 L2 归一化，计算 `S=U_R @ U_V.T`。双方各50个时，S 是 `[50,50]`。
5. 构造几何与语义代价，按 source ID 对矩阵两轴排序，调用 SciPy 匈牙利算法。
6. 每对拼接双方特征、绝对差、逐元素乘积、6个几何量及2个模态存在 mask，共520维。
   PairEncoder 将其映射到128维，JointHead 输出一个新的、无硬上限的 logit。
7. 每个 RV 的 `score_3d_after` 和 `score_2d_after` 都是同一个 `sigmoid(joint_logit)`。
   最终排序直接使用原始 joint logit，避免 sigmoid 饱和引起错误并列。
8. XYZ 和框保持源候选值，没有位置或框回归，没有局部图像 token 读取，没有邻域双向注意力。

## 2. 匹配代价与拒绝规则

`m(r)=clamp(8+160/r,8,48)` 仍用来归一化像素距离。

```text
C_ij = geometry_weight * log(1 + d_box_ij / m(r_i))
     + center_weight * log(1 + d_center_normalized_ij)
     + similarity_weight * (1 - cosine_ij) / 2
```

默认权重为 `1,0.01,1`。`reject_outside_gate=false`：几何距离作为软代价，不再硬拒绝框外远配对；
因此双方都有50个可投影候选时形成50对。配置为 true 可恢复门限拒绝，但配对数可能不足50。
无效投影、非有限代价仍然不能匹配。求解优先最大化合法配对数，再最小化总代价。

不足50个时使用真实数量，不补造候选。无效投影、数量不等或模态缺失时，未配对项保留 R/V-only，
由同一新评分头按缺失特征置零和显式存在 mask 评分，不使用旧分数回退；只有对应模态的预测字段有效。
因此整个输出表可能超过50行，但 RV 证据对始终最多50个。
Hungarian 的离散分配不反向传播；联合评分和完整相似度矩阵提供 adapter 的梯度。

## 3. GT 监督规则

当前数据合同是每个 query 一个 UAV 的3D位置及可选2D框，不能直接用于多目标身份监督。

| 配对标签 | 规则 | 用途 |
|---|---|---|
| 正样本 | R、V 分别可靠匹配本 query 的 GT | 高联合分数、特征拉近 |
| 困难负样本 | 一方可靠正确，另一方明确错误 | 低联合分数、特征分离 |
| 普通负样本 | 双方明确错误 | 低联合分数，不做背景间对比排斥 |
| 忽略 | 任一标签缺失或处于边界带 | 不贡献此配对的监督 |

评价固定使用 `3D误差≤1m` 和 `2D IoU≥0.5`。
训练增加可配置边界带：默认 R 正样本 `≤0.95m`、负样本 `>1.05m`；
V 正样本 `IoU≥0.52`、负样本 `IoU<0.48`。中间带忽略，旧1–2m大范围监督缺口不再保留。
R-only/V-only 在自身标签存在时单独监督；RV 缺少任一标签时不把不确定的联合质量标成负样本。
匈牙利预测结果绝不充当正标签。

## 4. Loss

```text
L = lambda_cls * L_joint_cls + lambda_align * L_align + lambda_rank * L_joint_rank
```

- `L_joint_cls`：直接对 joint logits 计算 BCE 等价的 softplus，正样本、困难负样本和普通负样本
  各自组内平均后求和，再对有监督 query 平均；默认困难负样本权重2，普通负样本权重1。
- `L_align`：可靠正对惩罚 `1-cos`，等价于归一化特征平方距离的一半；困难负对惩罚
  `2*relu(cos-0.2)`。分别组内平均。监督同一 query 完整的50×50矩阵，而不只监督已分配的50对，
  使早期配错时真实正对仍能训练映射。多个对应同一 GT 的正确候选均可成为正样本。
  不使用跨时间 in-batch negatives，避免把同一 UAV 的其他观测错误推远。
- `L_joint_rank`：`softplus(1 + max_negative_logit - max_positive_logit)`，直接推动至少一个正确
  联合候选超过最强错误竞争者。没有正对或没有负对的 query 不计算该项。
- 默认三个系数为 `1,1,0.5`，是实验起始值，未宣称最优。

几何坐标固定时，直接加入 GT 位置距离或投影距离的常数罚项不会产生模型梯度。
本版几何用于匹配代价和评分输入；可训练的“距离惩罚”作用于特征空间，不伪装成位置回归 loss。

## 5. 训练参数、指标与文件

训练参数：`pair_fusion.*` 与 `vision.builder.feature_proj.*`。
冻结：点云骨干和检测头、Swin/DINO、原交互和原评分模块；B4 不实例化旧交互/残差评分模块。

| 文件 | 职责 |
|---|---|
| `src/rdq_uav/multimodal_v2/pair_fusion.py` | adapter、相似度、匈牙利、联合编码与新评分 |
| `src/rdq_uav/multimodal_v2/pair_loss.py` | GT 标签、困难负对、全矩阵对齐和联合排序 |
| `src/rdq_uav/multimodal_v2/model.py` | B4 前向路由，绕过旧交互 |
| `src/rdq_uav/multimodal_v2/training.py` | B4 权重加载、冻结、优化、日志及联合评价 |
| `src/rdq_uav/multimodal_v2/contracts.py` | 联合 logit/similarity 字段和排序 |
| `src/rdq_uav/multimodal_v2/candidate_builders.py`、`radar_selector.py`、`lidar.py`、`vision.py` | B4 NMS 前 Top50 |
| `experiments/multimodal_v2/configs/b4_aligned_pairs.yaml` | 新配置 |
| `experiments/multimodal_v2/tests/test_v2_pair_fusion.py` | 匹配、评分、梯度、标签及 Lightning 更新测试 |

日志包括 `loss_joint_cls/loss_align/loss_joint_rank`、配对正/困难负/普通负/忽略数量。
`val/joint_success` 在双 GT 有效 query 上计算：同一个 RV 行赢得两项 Top1，且误差≤1m、IoU≥0.5。
`val/joint_oracle` 是已配对证据中同时满足两项标准的召回，反映配对后的能力上限。
新配置以 `val/joint_success` 选择 `best_joint.ckpt`，同时保留单独3D/2D指标和AP。
验证集必须包含双 GT query；没有时联合指标没有统计意义，不应启动正式训练。

## 6. 运行

先修改 B4 配置中的这3个字段：

```yaml
initialization:
  lidar_config: <与实际B0权重结构一致的雷达配置>
  b0_checkpoint: <实际已训练B0 checkpoint>
  b1_checkpoint: <实际已训练B1 checkpoint>
```

配置继承的默认路径是占位参考，不代表当前A2权重可用；不匹配会 strict-load 报错，不会静默忽略。
本版必须新建训练目录，不从旧 B2/B3 resume；B4 后续可以 resume 自身的 last.ckpt。

```bash
python -m pip install 'scipy>=1.11,<2'
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests

# 服务器真实样本两次更新：检查日志至少有配对标签和有限梯度。
PYTHONPATH=src python experiments/multimodal_v2/train.py \
  --config experiments/multimodal_v2/configs/b4_aligned_pairs.yaml \
  --output outputs/b4_two_update_gate \
  --max-updates 2 --train-limit 32 --val-limit 8 \
  --accumulate 1 --num-workers 0

# 验证流程与最佳权重保存：
PYTHONPATH=src python experiments/multimodal_v2/train.py \
  --config experiments/multimodal_v2/configs/b4_aligned_pairs.yaml \
  --output outputs/b4_small_gate --epochs 1 \
  --train-limit 32 --val-limit 8 --num-workers 0

# 正式训练：
PYTHONPATH=src python experiments/multimodal_v2/train.py \
  --config experiments/multimodal_v2/configs/b4_aligned_pairs.yaml

# 验证集评价：
PYTHONPATH=src python experiments/multimodal_v2/evaluate.py \
  --config experiments/multimodal_v2/configs/b4_aligned_pairs.yaml --mode B4 \
  --checkpoint outputs/own_multimodal_research/multimodal_v2/b4_aligned_pairs_seed42/checkpoints/best_joint.ckpt \
  --split validation_sub --output outputs/b4_eval_validation
```

`--max-updates` 是训练更新检查，不替代完整验证。若前32条没有双模态标注，使用已有
`--train-indices/--val-indices` 选择双 GT 样本，否则不能据此验证对齐路径。

## 7. 验证与实验边界

本地使用 CPU PyTorch 2.5.1，验证 FP32/BF16 梯度、两次优化更新、完整合成数据前向、
无 GT 泄漏、真实 Lightning 训练与 best_joint checkpoint 保存、旧 B0–B3 回归。
CUDA/FP16、真实 DINO+MMAUD 权重和真实训练收益尚未验证。

B4 把 NMS 前50个候选保留到报告，旧 B0–B3 通常报告 NMS 后最多10个；
比较AP/Recall时必须注明预算与NMS差异。`before` 诊断是同一 Top50 池的旧分数排序，
不能冒充原 NMS10 单模态正式结果。重复候选未做最终NMS，可能影响AP。
联合分数不使用 GT，GT 仅进入训练 loss 与离线评价。
