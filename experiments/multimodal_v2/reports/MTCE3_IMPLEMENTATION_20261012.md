# B0 A2 → MTCE-3

## 实现范围

- 保留A2 Learned-SBE32、三级Transformer、Up、CandidateHead、Loss及selector。
- L0在544→128之后、LayerNorm之前加入3→128无偏置运动投影。
- L1/L2在265→128之后、LayerNorm之前分别加入3→128运动投影。
- 三个投影在通用初始化结束后置零，总计1152参数。
- 开关关闭不新增checkpoint参数，旧A0/A1/A2配置继续使用原路径。

## 统计定义

原始点在delta_t∈[-1,0]内分为5个0.2s bin，内部边界归较新的bin，0纳入最后一箱。每个bin计算原始点数量、XYZ质心、平均观测delta_t。仅原始编号相邻且均非空、平均时间差≥0.05s的bin对有效；不跨空箱。

每个尺度输出[log1p(平均质心变化率/1m/s), 有效对数量/4, log1p(有效对两端最小点数)]。无有效对三维全零。L1/L2使用组合点映射对原始点重新统计，不平均子层速度。原始点与GT无关。该量是观测质心变化，不能当作目标真实速度。

FP32统计；return_debug=True可导出各bin计数、质心、时间、各相邻对的有效性、变化率及最终三维特征。训练CSV记录三级支持对数、拒绝小间隔对数、有效对数、有效voxel数和每层运动投影梯度范数。拒绝比例由总拒绝数/总支持对数计算，分母0时记为未定义。

## 权重与训练

initialization.b0_warmstart_checkpoint严格加载Lightning checkpoint的network.lidar.detector.*；只允许三个motion_proj.weight缺失，拒绝旧参数遗漏、额外键及形状不匹配。

training.radar_train_mode=motion_only只解冻三个投影；full恢复原B0完整雷达解冻行为。冻结旧参数不关闭计算图；统计本身不需要梯度。warmstart只加载权重，不恢复旧优化器或学习率调度。

新增模型checkpoint重载严格加载所有参数，包括已经训练过的运动权重。当前通用运行入口仍先加载配置指定的A2 warmstart，再由evaluate/--resume恢复MTCE完整状态，因此需保留原A2 checkpoint路径。

默认4轮、lr=3e-4、batch=4、accumulate=1、FP32。其他字段复制A2实验配置；不引入YAML继承语法。

## 校验与限制

新增test_radar_mtce.py验证：已知变化率、样本隔离、三级计数、时间边界、空/缺bin、小间隔过滤、非法时间、父级原始点统计、A2严格加载及零初始化输出和selector等价、两次原CandidateLoss更新、只更新三个投影、重载等价、可靠性消融、build_runtime与优化器集成。

本地验证使用合成样本及CPU，无MMAUD真实数据或用户A2 checkpoint，未验证seq0026既有审计数值，也未执行GPU正式训练。check_mtce_gate.py用于在用户机器上验证真实A2零初始化等价并导出真实bin统计。

可靠性消融：复制radar_a2_mtce3.yaml设置motion_centroid.reliability_only=true，再复制B0配置指向它、改用独立输出目录，从同一个原A2重新warmstart训练。不能把训练完的MTCE第一维临时清零当成训练消融。

## 推荐顺序

1. WSL应用补丁、运行全量V2测试并提交。
2. 服务器git拉取，确认原A2 best.ckpt路径与数据路径。
3. check_mtce_gate.py在validation_sub的指定query上验证等价。
4. 训练入口--max-updates 2，两次更新Gate使用独立输出目录。
5. 4轮正式训练；验证集选择best；方案固定后测试heldout_test_sub。
6. 使用原有evaluate.py输出Top1/Top10、三轴/三维RMSE、Coverage及逐query结果，配对对照A2。
