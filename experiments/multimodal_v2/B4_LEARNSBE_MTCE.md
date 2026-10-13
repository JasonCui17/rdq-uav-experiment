# B4 雷达前端：LearnSBE + 三尺度 MTCE

## 配置边界

新的入口配置为 `configs/b4_learnsbe_mtce.yaml`。
`configs/b4_aligned_pairs.yaml` 是先前分发、可能已有服务器路径修改的配置，保留作为迁移输入。
用 `diagnostics/prepare_b4_learnsbe_mtce.py` 从它生成 `configs/b4_learnsbe_mtce_server.yaml`。

B4 必须选择 `voxel.embedding: learned_sbe` 与 `motion_centroid.enabled: true`。
SBELite、无 MTCE、`reliability_only=true` 和不同于 1 秒的历史窗会在运行时被拒绝。
旧 SBELite/Legacy 实现只保留给历史单模态消融，B4 不会实例化或回退到它们。

## 实际前向

LearnSBE 点输入为局部 XYZ、传感器嵌入和相对时间，共5维，经点 MLP 变成32维。
每个子槽聚合 mean32、max32、log点数1、占用1，共66维。
8槽拼接528维，与 VQSA 的16维摘要构成544维。
L0 投影至128维；MTCE 的3维统计经 `motion_proj: 3→128` 加入投影结果，再做 LayerNorm。
L1、L2 分别在父体素聚合投影之后加入各自的 `motion_proj(3→128)`，再做 LayerNorm。
三尺度随后进入既有 Transformer 和上采样路径，输出128维候选特征。

三维 MTCE 为：`log1p(平均质心速度/参考速度)`、`有效相邻时间对比例`、`log1p(有效对最小点数)`。
五个时间槽位于过去1秒；每个尺度均基于原始点统计，不能跨空时间槽计算速度。
模块实现位于 `lidar_learned_sbe.py`、`radar_temporal_centroid.py` 和 `radar_model.py`。
LearnSBE 复用 `radar_sbe.py` 内的 VQSA 工具，不等于使用其中的 SBELiteVoxelEmbed。
LearnSBE 不使用 SBELite 的11维固定物理描述或指数时间加权池化。

## 训练与初始化

必须加载包含三个已训练、非零 MTCE 投影的 B0 checkpoint。
默认候选为 `outputs/b0_a2_mtce3_seed42/checkpoints/best.ckpt`，此前服务器已列出此文件；具体权重需要迁移工具在服务器严格核实。
不能把不含 MTCE 的 A2 checkpoint 用 `strict=False` 加载后冻结：新投影的零初始化会让 MTCE 无贡献。
LearnSBE、MTCE、雷达 Transformer 和雷达检测头都参与前向，但在 B4 冻结。
B4 仅训练联合特征映射/联合评分与 RGB 的256→128投影。

迁移工具读取 B0 checkpoint 保存的配置路径，验证完整 state_dict 键和形状。
它不会把 A2 deep 的 `[4,2,2]` 权重混入 MTCE `[2,2,2]` 配置，且不自动改深度。
若需要 deep + MTCE，需先训练并提供对应 B0 权重。
迁移保留服务器的 data、B1权重、loss、训练和精度配置，写入一个新 YAML，原 YAML 不覆盖。
若因更换雷达权重开启新 B4 实验，正式输出为 `outputs/b4_learnsbe_mtce_seed42`，不恢复旧 B4 checkpoint。

## 检查

启动时打印 `LearnedSBEVoxelEmbed + MTCE L0/L1/L2` 及三个投影范数。
范数为0、非有限或缺少投影会报错。
训练日志记录 `train/mtce_l0/valid_pair_count`、L1、L2 等实际时间统计；没有时间支持的体素合法地贡献0。
`effective_config.yaml` 的 runtime 字段记录已加载的雷达架构。
严格结构检查不能证明模型精度提高；比较联合方法时应同时报告对应 LearnSBE+MTCE 单模态基线。
