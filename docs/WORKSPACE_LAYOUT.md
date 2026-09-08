# RDQ-UAV 工作区规范

## 三类资产分离

| 类型 | 规范路径 | 内容 |
|---|---|---|
| 实验代码 | `/home/jasoncui/projects/rdq-uav-experiment` | Git、源码、配置、测试、manifest、报告 |
| 官方数据 | `/home/jasoncui/datasets/MMAUD/official` | V1、官方 2D 标注、官方鱼眼标定 |
| 原始数据 | `/home/jasoncui/datasets/MMAUD/raw/mavic2` | rosbag、CSV、手工提取 NPY/图像 |
| 下载归档 | `/home/jasoncui/datasets/MMAUD/archives` | 下载包和已确认重复副本 |
| 临时数据 | `/home/jasoncui/datasets/MMAUD/workspace` | 日志、缓存、临时样本 |

## 仓库内部职责

```text
rdq-uav-experiment/
├── configs/       # 可复现实验配置
├── src/           # 可复用模型、数据与训练代码
├── tools/         # 训练入口、审计和数据维护工具
│   ├── calibration/
│   └── data/
├── tests/         # 单元测试
├── manifests*/    # 数据配对与划分；大部分可再生成
├── calibration/   # 紧凑标定参数和审计结果，不放原始大文件
├── docs/          # 设计、阶段报告和决策记录
└── outputs/       # 本地运行产物；checkpoint 通常被 Git 忽略
```

## 历史 outputs 的处理

已有 `outputs/` 约 2.8 GB，其中包含 Stage 1–4 的 checkpoint、预测和审计产物。
这些目录是历史证据，路径已经写入 CSV/JSON，因此本次不重命名、不覆盖、不按阶段搬迁。
查找实验时以 `docs/` 中对应的阶段报告为入口。新实验继续采用：

```text
outputs/<stage>_<purpose>_<timestamp>/
```

每个正式运行目录至少保存 resolved config、指标、预测和 best checkpoint；结论写入
`docs/`，不要只留在终端日志中。

## 路径规则

- 新配置使用官方数据规范路径 `/home/jasoncui/datasets/MMAUD/official/v1`。
- 新命令从 `/home/jasoncui/projects/rdq-uav-experiment` 执行。
- 历史兼容软链接仅用于旧结果复现，不应写入新的配置或报告。
- 原始 Mavic2 数据不能替代五类官方 V1 数据参与公平对比。

## 本次迁移

本次整理只做路径移动和索引，没有删除数据。旧入口保留软链接；大压缩包重复项经过
SHA-256 校验后集中保存，等待未来单独确认清理策略。
