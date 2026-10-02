# 全输入点雷达监督清理：2026-10-02

## 最终契约

取消按最后若干事件标记点的监督字段。Multimodal V2 和共享 LiDAR V2 数据构造、collate、模型输入准备与 loss 均不再生成、传递或读取该字段，不保留兼容分支。

Multimodal V2 输入仍为同 sequence 的完整配置历史时间窗。点级字段只包含 points、delta_t、sensor_id；变长点云索引、模态有效性、targets 和 Meta 保持不变。旧 LiDAR-only 的 max_events 读取定义保持原样；本次取消的是它内部额外的监督子集，并非其历史输入定义。

## 共享雷达 loss

每个 voxel 使用所属全部输入点到 query GT 的最小距离 d：

- d <= 1m：positive。
- 1m < d <= 2m：ignore。
- d > 2m：negative。
- GT 无效：上述监督 mask 均为 False。

不再按事件排序或事件数量限制正样本支持。Focal、SmoothL1、阈值、权重和 sample normalization 公式不变。Multimodal V2 candidate ranking loss 不变。检测器网络、候选生成、投影、交互和 scoring 不变。

## 评估清理与影响

共享 LiDAR V2 evaluate_batch 删除基于事件子集的近邻计数与分组；summarize_metrics 不再返回该分组统计。保留总体、支持/无支持和 per_sequence。支持判定现在也使用全部输入点，不再带额外时间子集含义。

影响不仅是删除 Batch 字段：从头训练旧 LiDAR baseline 的正/忽略样本范围会变化，冻结雷达诊断 loss 和支持分组也可能变化。已有 checkpoint 的检测网络和候选推理不变。比较旧训练结果时必须注明监督契约改变；需要重新运行诊断以获得当前契约的指标。

## 修改文件

- src/rdq_uav/multimodal_v2/data.py
- src/rdq_uav/lidar_v2/data.py
- src/rdq_uav/lidar_v2/loss.py
- src/rdq_uav/lidar_v2/runtime.py
- experiments/multimodal_v2/tests/test_v2_data.py
- tests/test_lidar_uav_v2_spatial_query_pipeline.py
- tests/test_multimodal_v1_p3_data_contract.py
- experiments/multimodal_v2/README.md
- experiments/multimodal_v2/reports/DATA_REFACTOR_PROGRESS_20261002.md
- experiments/multimodal_v2/reports/SYMMETRIC_MODALITY_BATCH_20261002.md
- 本报告

## 验证状态

按用户此前要求，本轮暂不运行 pytest、真实数据或 GPU 验证。新增测试覆盖六事件窗口最早事件的点可提供正监督、1m/2m 阈值、无 GT mask 和移除旧分组后的评估 schema；既有 V1/V2 fixtures 已清理。

静态检查：全仓库活动源码、测试、工具和实验文档不再含被删除字段或对应事件子集操作；Python AST 语法检查和增量 diff 空白检查。完整运行和 checkpoint 等价性仍待设备验证。

后续运行：

```bash
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests \
  tests/test_lidar_uav_v2_spatial_query_pipeline.py \
  tests/test_multimodal_v1_p3_data_contract.py
```

之后运行实际数据 smoke、B0/B1 gate 与两步训练。
