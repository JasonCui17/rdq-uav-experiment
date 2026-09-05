# 当前问题与外部方案迁移决策

## 1. 当前真正的问题

当前 V1 数据不是一个普通的“小样本过拟合”问题，而是以下因素叠加：

1. **类别与采集 session 完全耦合**：当前每个无人机型号来自一段独立长采集，因而
   `class_name == sequence_id`。模型无法仅凭现有数据区分“型号特征”和“session 特征”。
2. **帧数不等于独立样本数**：相邻视频帧高度相关。数千帧的有效独立信息远少于帧数。
3. **目标极小**：整幅鱼眼图降采样后，无人机可能只剩数个甚至不足一个输出像素。
4. **背景捷径已被实验证实**：固定 256 px ROI 中遮掉中心宽高各 80% 后，Test Macro-F1
   仍为 83.97%。完整图为 93.15%，说明目标区域有增益，但外围背景已能完成大部分分类。
5. **单帧任务与数据天然的序列结构不匹配**：真正可辨认的通常只是少量近距离、清晰帧。
6. **Radar 也可能产生捷径**：整帧点云可编码静态环境、距离和轨迹。若不先定位动态目标，
   Radar-only 或 RDQ 的提升也未必来自 UAV 回波。

其中第 1 点是**数据可识别性限制**，不能靠更复杂的网络、GroupDRO、IRM 或更强增强从根本上
消除。严格的跨 session 型号泛化需要每个型号至少有两个独立 session，并按 session 切分。

## 2. MMAUD 官方与获胜方案怎样处理

MMAUD 原论文把约 1700 秒数据切成 50 个短序列，并采用 60/20/20 划分；论文也明确承认
地理覆盖有限。CVPR 2024 UG2 获胜方案没有直接对每个完整帧做型号分类，而采用：

```text
重建 real sequence
    -> YOLOv9-e 零样本检测 airplane
    -> 裁剪 UAV ROI
    -> 按检测置信度选关键帧，并设置最小时间间隔
    -> EfficientNet-B7 对关键帧分类
    -> 对同一序列的 softmax 概率做软投票
```

其训练阶段每个 real sequence 最多随机采样 300 个 ROI，以减轻类别和序列长度不平衡。该方案
直接针对两个事实：绝大多数帧没有足够目标信息，以及测试标签本质上属于整个序列。

本地可参考实现：

- `/home/jasoncui/projects/open_source/Multi-Modal-UAV/visual_processing/yolov9/detect.py`
- `/home/jasoncui/projects/open_source/Multi-Modal-UAV/README.md`

仓库根许可证为 MIT。迁移时保留来源与许可证声明，不整体复制其数据读取脚本，因为其同步代码
使用“各模态最近时间戳再求平均”，不如当前项目的 GT anchor 和误差门限严格。

## 3. 决定迁移的方案

### 3.1 检测后分类，而非完整帧分类

迁移 UG2 获胜方案的接口设计：

```text
TargetProposer -> RoiExtractor -> KeyframeSelector
               -> FrameClassifier -> SequenceAggregator
```

第一阶段使用官方 2D bbox 作为 Oracle proposer，建立可信上限；第二阶段再替换为 YOLOv9 或
Radar 投影 proposer。分类器优先复用 `timm` 的 EfficientNet/ResNet，不自行实现 backbone。

### 3.2 精确 bbox 反事实诊断与背景不变增强

迁移 CVPR 2021《Towards Robust Classification Model by Counterfactual and Invariant Data
Generation》的思想。该方法利用 bbox 区分因果候选区域与非因果背景，并分别构造：

- `Full`：完整输入；
- `Counterfactual / bbox-erased`：擦除目标框，检验无目标时还能预测多少；
- `Foreground-only`：保留目标及固定 context，移除背景；
- `Background-randomized`：保留目标，随机化或跨 session 替换外围背景。

当前官方 2D 标注对固定 ROI manifest 的精确覆盖率为 Train 91.6%、Val 99.5%、Test 98.8%，
足以完成这组实验。先做无生成模型、可复现的灰色填充和跨样本背景替换，不优先引入扩散模型。

### 3.3 关键帧与序列级聚合

迁移获胜方案的两个规则：

1. 按清晰度/检测置信度/bbox 面积选 top-k；
2. 关键帧间设置最小时间间隔，避免选到近重复帧。

输出以序列为单位，将关键帧 logits 或 softmax 概率做 mean/weighted mean。先复现 soft vote，
再把它作为时序 Transformer 的控制组；没有超过 soft vote 前不引入自研时序结构。

### 3.4 分组评估，而非直接套用域泛化算法

迁移 WILDS/DomainBed 的评估原则：样本带 `domain/session/block` 元数据，报告每组指标、最差组
指标和多折均值。暂不迁入完整 DomainBed 训练框架，也不使用 GroupDRO/IRM 作为主修复手段：
这些方法需要多个环境中都观察到各类别，当前“一类一 session”不满足前提。

## 4. RDQ 主线怎样调整

保留 Radar Dynamic Query，但把它放进“定位后分类”流程，而不是让单个 Radar 全局 token 在
整幅鱼眼背景中自由寻找类别线索：

```text
Radar 连续帧动态点/轨迹
    -> 投影到鱼眼图像并产生带不确定度的 proposal
    -> 提取局部多尺度视觉 tokens
    -> Radar target features 生成 Dynamic Query
    -> Cross-Attention
    -> frame logits
    -> sequence soft vote
```

实验仍保留 Learned Query、Concat、Radar Zero/Shuffle/Temporal Shift，但所有模型必须共享同一
proposal、同一关键帧和同一序列聚合器。这样 `RDQ - Learned Query` 才能主要归因于 Query 来源。

## 5. 执行门槛

按以下顺序推进：

1. 官方 bbox 的 `Full / bbox-erased / foreground-only` 配对实验；
2. 复现 ROI 分类 + top-k 关键帧 + soft vote；
3. 为 Radar 做动态点/时间一致性诊断，确认 proposal 中确有目标回波；
4. 建立 Oracle proposal 与 Radar proposal 两套结果，上限和实际系统分开报告；
5. 最后运行 RGB、Radar、Concat、Learned Query、RDQ。

若 foreground-only 接近随机水平，而 bbox-erased 仍很高，则 V1 五分类只可作为 session-conditioned
工程实验，不应继续用它论证通用 UAV 型号识别。若后续没有新增独立 session，最终论文结论必须
明确限制在 MMAUD V1 的现有采集条件内。

## 6. 主要参考

- MMAUD: https://arxiv.org/abs/2402.03706
- UG2 获胜方案: https://arxiv.org/html/2405.16464v1
- 获胜方案代码: https://github.com/dtc111111/Multi-Modal-UAV
- Counterfactual and Invariant Data Generation: https://openaccess.thecvf.com/content/CVPR2021/html/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.html
- WILDS: https://proceedings.mlr.press/v139/koh21a.html
- DomainBed: https://github.com/facebookresearch/DomainBed
