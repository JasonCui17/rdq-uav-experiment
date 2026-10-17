# RDQ-UAV 最终实验结果快照（2026-10-17）

## 当前完整序列结果

`results/seq0020_full_sensor/` 是 seq0020 的完整传感器时间轴结果：935 次查询、620 张左鱼眼图像；588 张有效标注图像参与检测评分。

| 方案 | 3D RMSE（m） | GT时刻轨迹覆盖 | AP50 | AP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 实时 | 0.466518 | 393/400 | 86.7572% | 29.7693% | 86.1958% | 91.3265% |
| 0.5秒平滑 | 0.377363 | 398/400 | 86.8709% | 29.7526% | 86.4952% | 91.4966% |

轨迹和所有图片的预测框见该目录的 `seq0020/trajectory.jsonl`、`trajectory.csv` 和 `bboxes.jsonl`。以上仅为 seq0020，不能称为完整验证集结果。

## 历史验证与迁移检查

- `reference/` 保存历史三序列 query benchmark、人工修订标注评分和训练配置；与完整传感器时间轴评分口径不同。
- 人工GT参考过模型预测，不是与模型独立的测试标注。历史JSON中的overlay描述保留原始记录；当前源码仅读取MMAUD中的活动2d_detect标签，详见README。
- `results/v3_consolidation_smoke/comparison.json`：重构后的12条轨迹、8张图像bbox与重构前一致。这是兼容性检查，不是新的完整数据集指标。
- 109项测试、独立审查和完整性检查通过，详见VALIDATION.md。

## 复现与文件范围

本目录同时保存当前自有源码、配置、测试、运行说明和原工程MANIFEST，以绑定结果与代码版本。源码已经统一在 `src/rdq_uav/multimodal_v3/`。

这是一份结果与源码快照，不是完整运行资产包：MMAUD数据、模型权重、COCO初始化、third_party源码和CUDA二进制未上传。原工程MANIFEST记录这些资产的SHA256；执行需在完整rdq-uav-final工程环境中按README准备资产。未重新训练或改变模型。

`UPLOAD_SHA256.json`只校验本次实际上传文件；`MANIFEST.json`为完整本地工程的清单，二者范围不同。
