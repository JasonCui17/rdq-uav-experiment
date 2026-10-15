# 固定版本 target_kf_v1_20261015

此版本固定2026-10-15的目标关联代码与参数，同时保留以下两个方案：

| 模式 | 方法 | 输出数 | 三维RMSE | 1m成功率 | >10m错误 |
|---|---|---:|---:|---:|---:|
| `realtime` | 仅使用当前和过去观测的目标KF | 1190/1200 | 0.592836m | 93.50% | 0 |
| `lag05` | 同一目标KF + 最多0.5秒未来窗口RTS | 1200/1200 | 0.533081m | 97.92% | 0 |

两种模式共享同一份固定 `config.json`。`reference_summary.json` 保存实测参考指标、
环境信息与缓存指纹，`SHA256SUMS` 固定推理、评估、配置、测试及版本入口文件。
运行入口先校验文件，内容变化会停止执行。后续改进应使用新版本名及新的结果目录。

## 运行

仓库根目录，已安装本项目及 `experiments/multimodal_v3/requirements-tracking.txt` 的rdq环境：

```bash
conda activate rdq

# 实时因果方案
bash experiments/multimodal_v3/releases/target_kf_v1_20261015/run.sh realtime

# 0.5秒未来窗口平滑方案
bash experiments/multimodal_v3/releases/target_kf_v1_20261015/run.sh lag05
```

默认复用本机 `outputs/kf_v3_multi_run/cache`，输出目录自动包含模式与时间戳。
也可以指定其他缓存和全新的结果目录：

```bash
bash experiments/multimodal_v3/releases/target_kf_v1_20261015/run.sh \
  lag05 /path/to/cache outputs/my_frozen_run
```

从Python入口直接复现时，`--lag-s 0`为实时模式，`--lag-s 0.5`为平滑模式。
使用其他缓存时，输出指标可能变化；源码校验不会把数据替换误当成原实验复现。

## 读取输出

- `summary.json.target`：实时方案指标；两种运行模式均保留此项。
- `summary.json.lagged`：0.5秒窗口方案指标，仅`lag05`模式存在。
- `per_query.jsonl/csv` 的 `xyz` / `target_error_m`：实时位置与误差。
- `lagged_xyz` / `lagged_error_m` / `lookahead_s` / `lagged_event`：平滑位置、误差、
  实际使用未来观测的时间范围和补全类型，仅`lag05`模式存在。

参考结果来自validation_sub的三条序列、全部1200个标注查询时刻。
实时模式起轨前缺失10点；平滑模式的这10点是标记为`startup_backfill`的外推估计。
平滑入口是批处理评估，不是已实现的在线延迟队列发布接口。
结果不代表所有原始雷达帧均已追踪，也不代表独立测试集性能。
GT仅用于事后计分，不进入关联、KF或平滑。

## 手动上传GitHub

将上传包解压到仓库根目录，按目录层级上传其中的文件到`multimodal_v3`分支。
包内包含当前未提交的V3源码补丁及新方案，保留已有历史实验以支持复现。
缓存、模型权重和完整输出数据保留本地；参考摘要随源码上传。
`UPLOAD_MANIFEST.json`列出本包文件及SHA256，可用于核对手动上传内容。

推理核心仍为 `src/rdq_uav/multimodal_v3/target_tracker.py` 与 `target_smoothing.py`，
复用原有 `tracking.py` 的CVKalman；本次固定版本未调整算法或参数。
