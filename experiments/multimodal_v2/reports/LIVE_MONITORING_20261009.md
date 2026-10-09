# 实时训练显示与视觉三维定位边界

基版本：a3eb881b35c7e4fa5efb37a6e92eb873f8a7cefe。

## 已实现

1. B0：逐 batch 显示实际分类/加权回归 loss、本轮训练累计三轴 RMSE、三维
   RMSE、Coverage、Success@1m、样本数、学习率及峰值预留显存。
2. B1：逐 batch 显示实际原生 DINO loss 及互斥分组 cls/bbox/giou/dn/aux/enc/other；
   验证中每50个 batch 刷新累计图像 COCO AP/AP50/AP75/AP_small/AR100，轮末计算
   完整结果。训练时显示 lastV/ 上次完整验证值，不冒充当前训练指标。
3. 累计雷达指标使用平方误差总和/计数，不平均各 batch 的 RMSE；COCO AP
   使用累计去重图像重算，不平均各 batch AP。缺失输出不记为零误差。
4. T/ 为变化中模型的训练诊断；V~/ 为当前验证前缀；V/ 为完整验证结果。
   单独打印完整验证行，以免终端宽度导致进度条字段截断。
5. --val-check-interval 0.25 可在一轮约每四分之一处执行完整验证；默认仍一轮一次。
   --ap-every-val-batches 控制验证过程中的 AP 刷新频率，默认50。
6. B1 best checkpoint 改按 val/AP 选择；val/2d_iou50 保留日志。B0/B2/B3仍按
   val/success_1m 选择。重新训练 B1 使用新输出目录，避免混用旧的选择历史。

没有修改模型结构、损失数学、标定、候选筛选、第三方源码，也没有新增视觉三维头。

## 测试证据

CPU 环境 Python3.12 / torch2.14.1+cpu / Lightning2.6.6 / pycocotools2.0.11。
完整 V2 tests：115 passed, 1 skipped；12条警告包含模拟训练无logger/worker提示、
框架弃用提示和现有 attention mask 提示。
新增验证覆盖累计 RMSE 与离线 pooled RMSE 一致、原生 loss 分组互斥且总和不变、
compact vision 到原 batch 映射，以及 B0/B1 的真实 Lightning CPU 训练循环：
一轮两次验证、状态重置、实际参数更新和对应监控指标的 best.ckpt 保存。
检测器前向在该循环中使用模拟返回值，不是实际 DINO/真实数据 GPU smoke。
CLI、编译、差异及补丁检查通过；服务器实际实时显示速度与真实 CUDA 训练待验证。

## 视觉 Dx/Dy/Dz 的含义和后续设计建议（未实现）

二维框预测与三维位置预测是不同任务。当前 B1 只有像素 xyxy 与二维分数，
没有 XYZ 输出；二维中心误差只能得到像素 dx/dy，不能直接生成米制 dz。
论文表中的 Dx/Dy/Dz 表示三维坐标轴误差，不能用框坐标代替。

核实资料：
- Unsupervised UAV 3D Trajectories Estimation with Sparse Point Clouds 的 Table I
  确有 VisualNet/DarkNet/YOLOv5s 的视觉三维误差；已读正文未给出其回归头足够细节。
  https://arxiv.org/html/2412.12716v5
- A3PRL 主方法是 LiDAR 感知，搜索索引中有视觉基线；完整PDF被403阻断，不能据此
  确认其视觉基线结构与米制坐标恢复方法。
  https://openaccess.thecvf.com/content/CVPR2026/papers/Yuan_Adaptive_3D_Perception_for_Small_Aerial_Targets_Under_Sparse_Sampling_CVPR_2026_paper.pdf
- 相关作者的 Label-Free Long-Horizon 3D UAV Trajectory Prediction via Motion-Aligned
  RGB and Event Cues，IV-B 明确说明给原本二维的视觉基线添加三维回归头。
  这是有明确文字证据的相关实例，不是前两篇实现的直接证明。
  https://arxiv.org/html/2507.03365v1

若需要独立视觉三维基线，建议独立配置/输出：DINO目标query特征 + 归一化bbox
→ 小MLP → XYZ或距离 + 标定视线方向，使用真实3D GT监督并报告三轴/三维RMSE。
不能输入雷达预测、sequence/sample id/绝对时间或GT作为推理特征。
匹配必须来自正常检测/Hungarian流程，不能评估时使用GT挑选最近候选。
单目几何只提供视线方向，距离需要学习、尺寸先验、双目或额外测量；鱼眼模型
必须使用现有标定，不能照搬未校正针孔公式。

时间对齐必须先定：当前图像可能比query_time早1秒，现有3D GT属于query_time。
“图像时刻的视觉三维定位”与“用历史图像预测查询时刻位置”不可混写。
前者需要图像时刻GT；后者要显式设计相对时间/运动建模并记录任务语义。
建议保留 B1 纯二维检测，把视觉三维定位作为独立实验分支/配置；视觉三维输出
不自动进入当前 B2/B3 的雷达三维候选池，避免改变既定多模态边界。
