# Multimodal V2 对称模态 Batch：2026-10-02

本文件是当前数据契约，取代早期报告中的 placeholder、全 B 图像堆叠和 point_counts[B] 定义。

## 职责边界

`data.py` 保留三个明确层次：Dataset 构造 Sample、collate 构造 Batch、prepare_model_batch 准备模型输入。Radar/image 文件索引、时间选择、文件读取、YOLO 查找均在此文件完成。后续模型与 loss 只消费 Batch/准备后的张量，不再次读文件、匹配时间或查标签。

时间选择保持：同 sequence 雷达窗口内全部有效事件 `[t-radar_history_s,t]`；图像为 `[t-max_image_gap_s,t]` 内最新历史帧；默认都为 1 秒，无未来观测、无事件/点数上限。query_time 唯一性规则保持。

## Sample：保留扁平字段，缺失为 None

| 分组 | 字段 | 有效 Shape / 类型 | 缺失时 |
|---|---|---|---|
| Radar | points | float32 [Nr,3] | None |
| Radar | delta_t | float32 [Nr]，event_time-query_time | None |
| Radar | sensor_id | int64 [Nr]，0 Avia/1 Mid360 | None |
| Radar | m_R | bool，等价于有效点数>0 | False |
| Vision | image_uint8 | uint8 [3,H,W] | None |
| Vision | vision_delta_t | float，image_time-query_time | None |
| Vision | image_source_wh / image_view_wh | int64 [2] | None |
| Vision | image_scale_xy | float32 [2] | None |
| Vision | m_V | bool | False |
| Target | target_xyz / target_valid | float32 [3] / bool | GT 无效时 mask |
| Target | gt_box_xyxy_px / gt_2d_valid | float32 [4] / bool | 零框 / False |
| Diagnostic supervision | supervision_recent_mask | bool [Nr]，历史窗口中最后四事件 | None |
| Meta | sequence_id / sample_id | str | 保留 |
| Meta | query_time / target_timestamp | float | 保留 |
| Meta | event_timestamps / event_sequence_ids / event_count | list[float] / list[str] / int | 可为空 |
| Meta | image_time / left_image_path / image_sequence_id | float / str / str | None |
| Meta | radar_history_s / max_image_gap_s / num_samples | float / float / int | 保留 |

YOLO 标签仅按实际选中图像 stem 在 sequence-local label_directory（默认 2d_detect）下读取。目录中的合法单框都作为有效监督；缺文件/空文件不监督，错误格式或多框明确报错。框按 calibrated left crop 尺寸转换，保留 resize 前源图像像素坐标。缺标签不会关闭视觉分支。

训练在初始化阶段过滤无有效雷达点且无有效历史图像的 query。验证保留，并将无 3D 输出计入已有覆盖率/失败统计。

## Batch：有效模态局部批次 + 原样本索引

B 是全部 Sample 数；Br=sum(m_R)，Bv=sum(m_V)。

| 字段 | Shape / 语义 |
|---|---|
| points | [sum(Nr),3]，仅有效雷达点 |
| delta_t / sensor_id / supervision_recent_mask | [sum(Nr)] |
| point_counts | [Br]，每个有效雷达样本的点数 |
| point_batch_index | [sum(Nr)]，取值 0..Br-1，雷达局部索引 |
| radar_batch_index | [Br]，雷达局部样本 → 原 Sample 索引 |
| image_uint8 | [Bv,3,H,W]；Bv=0 时 None |
| vision_delta_t | [Bv] |
| image_source_wh / image_view_wh / image_scale_xy | [Bv,2] |
| vision_batch_index | [Bv]，视觉局部样本 → 原 Sample 索引 |
| m_R / m_V / target_valid / gt_2d_valid | [B] |
| target_xyz / gt_box_xyxy_px | [B,3] / [B,4] |

Br=0 时雷达 packed 张量是长度为零的空集合，不是一个占位样本；雷达检测器不会被调用。Bv=0 时不调用 DINO preprocessing，也不调用视觉检测器。

Meta 保留用于审计，但 prepare_model_batch 只移动白名单字段。绝对 query_time、sequence_id、sample_id、路径不进入模型。

## 模型与监督路由

1. radar_model_batch 从整理好的 Batch 提取雷达局部 targets，保留雷达局部 point_batch_index；不读文件。
2. 两检测器各运行一次有效局部 Batch，跳过为空的模态。
3. 候选 batch_index 分别通过 radar_batch_index / vision_batch_index 恢复成原 Sample 索引；source_index 保持不变。
4. 跨模态交互仅选择双模态样本中的候选；临时转换为视觉局部索引以读取 compact pyramid，完成后回填 evidence。无全 B 假图像/假 feature map。
5. 单模态 evidence 为无效零证据，因此现有评分保持单模态 score、XYZ、box。
6. ranking loss 使用全部 B 的 targets；冻结雷达诊断 loss 使用雷达局部 targets；冻结 DINO 诊断 loss 使用视觉局部 GT 和 transforms。无该模态时跳过其 loss，记录零值。

prepare_model_batch 的六元返回接口保持：prepared_batch、images 或 None、padding mask 或 None、ProjectionContext、MultimodalTargets、有效图像 transforms。ProjectionContext 用原 Sample 索引，固定 calibration 不变；无图样本的中性 resize=1 只用于几何张量布局，不是观测，跨模态处理不会读取这些位置。

## 修改范围与待验证

修改 data.py、contracts.py、model.py、training.py、数据测试、数据 smoke 脚本和文档。没有修改 LiDAR-only baseline、检测器网络、候选生成、attention 数学、标定投影公式、关联或 loss 数学。模型适配器对 V1 / LiDAR V2 的既有依赖仍存在；本次只解耦数据构造与缺失模态执行。

本次按用户要求暂不运行 pytest / 实际数据 smoke / GPU gate。新增测试代码覆盖 None 缺失、缺失分支不调用、四种样本混合且索引非连续、paired interaction、evidence 梯度回填、冻结 loss 的局部监督映射、错误 Batch 索引。此前 55 PASS 是旧版本结果，不能当作本版本验证。

已执行的检查仅为 Python AST 语法解析、增量 diff 空白检查和修改包完整性检查。运行逻辑、真实 DINO、AMP、单模态等价性以及 B0/B1/GPU 训练均待设备验证。

后续执行：

```bash
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/check_data_samples.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --sequence seq0001 --samples 3
```

之后继续原 README 的 B0/B1 identity、两步更新与完整评估流程。真实 sequence、数据路径与权重可用性仍需在设备上确认。
