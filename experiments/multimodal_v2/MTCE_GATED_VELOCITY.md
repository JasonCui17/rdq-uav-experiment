# 固定体素的可靠性门控MTCE与速度引导预测

基于 `mmv2-b4-aligned-pairs` / `01f2ac5427a92a944376fbba13a76ee2f4a73cac`。
目标是增强可靠运动证据、利用观测运动先验修正query_time位置；本轮没有跨体素关联、点云重定位或新增目标轨迹。

## 实现结构

`radar_temporal_centroid.py` 统一拥有统计、MotionContext、三级条件调制与L0位置补偿。
复用原MTCE时间箱统计，使用已有L0/L1/L2层级的原始点归属，各尺度独立计算。
`radar_model.py` 只组织数据流：编码/聚合 → 条件调制 → 原Transformer → 回传 → 原预测头 → 位置补偿。
旧模式省略mode时仍是legacy，不改变已有MTCE权重键与行为。

## 运动估计与动态判定

使用过去1秒的5个时间箱。每箱计算观测质心、实际平均时间、点数及Mid360占比。
选择最长连续支持区间，并列时取较新区间；不跨空箱或过小时间间隔。
权重为sqrt(min(count,8))，对质心与真实时间作加权直线拟合；t_ref为同权重平均观测时间。
速度来自观测质心，不代表已识别物体的真实速度，也不是多普勒速度。

默认：至少3个连续箱、每箱至少2点、时间跨度≥0.3s、质心拟合RMS≤0.1m、
最新有效箱平均时间距query≤0.3s、传感器占比变化≤0.5、速度≤15m/s。
可靠性为支持箱比例、时间跨度因子、拟合残差衰减、传感器组成稳定性之积。
有效且可靠性≥0.25、速度≥0.15m/s才判为动态。
这些是可复现实验起始阈值，尚未由真实静态数据标定；0.15m/s不是已证明的噪声界。
阈值应在训练/验证数据上校准后建立新suite，不能根据heldout结果反复调参。

MotionContext包含11维描述：signed log速度XYZ、log速度大小、支持箱比例、log支持点数、
归一化拟合误差、观测年龄、传感器占比变化、可靠性和有效标志。
运动不可靠是unknown，不等同于static。可靠静态的特征仍可被网络使用。

## 特征门控

三级各有一个小型条件器：concat(128维空间特征,11维描述) → 64 → 256，输出gamma/beta。
`f' = f + reliability * (tanh(gamma) * f + beta)`。
注入位置为原空间编码/聚合的输出归一化之后、各级Transformer之前。
这是通道条件调制，不是逐点动静分割，也不删除静态体素。
末层零初始化；无有效拟合时reliability=0，严格保留基础特征。

## 只对可靠动态体素的位置补偿

只用对应L0的运动估计，不用父L2速度替换L0，不跨体素传播速度。
`prior = velocity_xyz * (-reference_dt)`。
由最终128维特征与11维描述生成有界、零初始化系数a=tanh(MLP(...))。
`offset = dynamic_mask * reliability * a * prior`，补偿向量模长最多0.5m。
静态、无效或可靠性不足：offset严格为0，仍保留基础XYZ回归。

注意：a∈[-1,1]是可学习的残差修正系数，允许抑制或抵消先验。
原A2回归头已经预测query_time位置，所以不能直接硬加完整v*dt。
这里实现速度引导的残差，而非声称原回归头输出了真实t_ref位置。
没有真实速度标签，不添加伪速度监督；原SmoothL1监督最终总偏移。

`residual_xyz = base_residual + offset`
`pred_xyz = voxel_centers + residual_xyz`

核心接口不变：logits[N0]、residual_xyz/pred_xyz[N0,3]、fine_features[N0,128]、
voxel_centers[N0,3]、batch_index/source_token_id[N0]。体素数与编号不变。
额外motion_diagnostics仅作诊断，不拼入CandidateBatch，不改变B4的520维联合输入。

## 训练与权重

原A2严格warmstart：旧参数必须完整匹配，只允许显式新增temporal_motion与可选LoRA等参数缺失。
新增条件器和补偿末层零初始化，初始FP32输出与加载后的原A2完全一致。
heads/lora模式均会训练temporal_motion与原预测头；full训练全检测器。
旧motion_only仍只适用于legacy MTCE，不适用于新模式。
不与C query decoder合用。本轮不改变分类标签、质量头、NMS、B4 loss或选模定义。

## 四组消融

`mtce/run_suite.py`提供baseline/gate/velocity/joint；默认同一原A2、seed42、LoRA、3epoch、FP32。
baseline也使用相同LoRA，不拿12轮full与3轮LoRA直接比较。
使用原分类监督，不额外引入ABC control的negative-only修复，保持本轮归因单一。
按val/success_1m保存best.ckpt。对照结果用已有abc/compare.py逐query配对汇总。
prepare生成独立目录，记录模型/训练配置、源码、划分、标定、候选配置和原权重SHA256。
gate为独立输出目录的2-update检查；train不继承gate；eval从完整best.ckpt独立加载。
已有日志时默认拒绝重复运行；eval --resume只允许追加日志并重评，train --resume恢复last.ckpt。
训练中不要修改代码或生成配置，runner会拒绝配置/源码漂移。

## 诊断与局限

训练日志按L0/L1/L2记录有效拟合、动态体素、总数，并按原始点支持的正/负体素记录
`num_motion_positive_valid/dynamic/voxels`、`num_motion_negative_*`及非零补偿数。
这些是体素统计，不是逐点运动分割精度。只有dynamic数量不等于补偿起作用，初始化补偿为0。
必须检查目标附近支持率、悬停成功率、原始候选Oracle、Top1、RMSE与覆盖率。
固定体素内的采样漂移、遮挡和背景混入仍可产生伪运动；拟合可靠性无法消除所有错误。
高速度跨体素时可能没有足够连续L0支持，此时明确回退基础回归，不伪造速度。

## 接回B4

先完成B0运动模型训练。B4配置中的lidar_config必须使用获胜run保存的radar.yaml，
b0_checkpoint使用同run的train/checkpoints/best.ckpt；如果run含LoRA，仅切换本目录模板是不够的。
B4会冻结整个点云检测器，包括LoRA和MTCE，只训练原有配对融合参数。
XYZ、128维候选特征和B4输入维度不变；特征与分数分布变化后仍需评价/微调B4。

## 01f2ac5兼容适配

保留该提交的B4 LearnSBE/MTCE架构约束与AMP修复。radar_architecture支持legacy与gated_velocity两种MTCE，
新模式必须具备三个已训练条件器，启用速度补偿时也检查其输出层；只检查随机隐藏层范数是不够的。
velocity-only为B0消融，不能用作此B4要求的三尺度MTCE前端。迁移工具支持新模式checkpoint的严格加载与架构报告。
