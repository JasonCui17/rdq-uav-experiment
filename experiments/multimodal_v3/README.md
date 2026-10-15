# Multimodal V3：候选关联与 6D Kalman 跟踪

当前固定版本为 [target_kf_v1_20261015](releases/target_kf_v1_20261015/README.md)，保留实时因果方案和最多0.5秒未来窗口平滑方案。固定配置、源码校验、两种运行命令及参考指标见版本说明；优化过程见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。

新增的 Stone Soup 多候选轨迹方法、评分配置、缓存 v1/v2 兼容性和五组手动运行命令见 [MULTI_TRACK_DESIGN.md](MULTI_TRACK_DESIGN.md)。本页保留原三组实验的使用方式；`evaluate_tracking.py` 默认仍运行原三组，显式 `--methods all` 才运行五组。

本分支从 `mmv2-b4-aligned-pairs` 创建。V3 保留 V2 的数据、雷达网络、checkpoint、候选选择器和训练代码；新增的跟踪层位于 `src/rdq_uav/multimodal_v3/tracking.py`。它只消费冻结 B0 网络的一次前向缓存，不参与网络训练，也不改变候选分数或网络输出。原始实现来自 `KF_Candidate_Tracking_20261014.zip`，已适配仓库路径。

## 首轮运行

默认使用 `outputs/mtce_joint_lora_local_20261014_011406` 的 B0 `best.ckpt`，以及配置中的 `validation_sub`。在仓库根目录执行：

```bash
conda activate rdq
bash experiments/multimodal_v3/run_wsl.sh
```

可通过 `RUN`、`KF_RUN`、`DEVICE`、`NUM_WORKERS` 覆盖默认路径和运行设备，例如 `DEVICE=cpu NUM_WORKERS=0 bash experiments/multimodal_v3/run_wsl.sh`。脚本写入新的 `outputs/kf_candidate_时间戳/`，其中 `cache/` 是每个 query 的 dense 候选和原 NMS 候选，`results/` 与 `results.zip` 是离线跟踪比较结果。已存在的 cache 或结果目录会被拒绝覆盖。

若只需少量真实 query，可分别执行两步：

```bash
export PYTHONPATH="$PWD/src:$PWD"
python experiments/multimodal_v3/extract_observations.py \
  --repo "$PWD" \
  --config outputs/mtce_joint_lora_local_20261014_011406/config.yaml \
  --checkpoint outputs/mtce_joint_lora_local_20261014_011406/train/checkpoints/best.ckpt \
  --cache outputs/kf_v3_smoke/cache --device cpu --num-workers 0 --limit 6
python experiments/multimodal_v3/evaluate_tracking.py \
  --cache outputs/kf_v3_smoke/cache --output outputs/kf_v3_smoke/results
```

真实数据和权重必须存在；此处不会回退到随机初始化或合成数据。`extract_observations.py` 使用现有 `build_runtime` 和 `load_network` 严格加载完整 checkpoint，按 sequence 和 query_time 排序，在 FP32、`eval()`、`no_grad()` 下提取一次前向，并记录权重/配置 SHA256 与实际 selector 参数。若配置目录有 `eval_best/per_query.jsonl`，还会核对原 Top1 误差。GT 只保存在缓存中供离线计分，不进入跟踪器。

## 三组比较

| 方法 | 观测来源 |
|---|---|
| `independent_top1` | 原网络和选择器的 Top1，无时序跟踪 |
| `top1_kf` | 仅原 Top1 可用于 KF 更新；可拒绝或短时只预测 |
| `candidate_kf` | 在预测位置附近从全部 dense 体素保留局部候选，再并入原全局 NMS 候选；门控后更新 KF |

两种 KF 共用噪声、门限、起轨确认、失锁和重捕获规则。状态为 `[px,py,pz,vx,vy,vz]`，按真实 query 时间差预测；位置和速度是独立轨迹输出。仅预测时没有候选 token ID，不把预测冒充检测。默认参数见 `tracker_defaults.json` 或 `TrackerConfig`；离线评估可用 `--tracker-config path.json` 指定参数覆盖文件。

`results/summary.json` 包含三组成功率、覆盖率、RMSE、严重错误和跟踪事件；`per_query.csv/jsonl`、`tracking_events.jsonl`、`track_confirmations.csv`、`sequence_metrics.csv`、`failure_comparison.csv` 分别保留逐 query 和逐 sequence 的细节。可复用缓存重新比较参数，无需再次运行 GPU：

```bash
export PYTHONPATH="$PWD/src:$PWD"
python experiments/multimodal_v3/evaluate_tracking.py \
  --cache outputs/kf_candidate_实际时间戳/cache \
  --output outputs/kf_second_results \
  --tracker-config experiments/multimodal_v3/tracker_defaults.json
```

## 评价边界

当前是雷达 B0 的单目标跟踪对照。查询计划沿用带 GT 的原评估时间戳，GT 仅用于确定该计划与事后评分；它不是对所有原始传感器帧的连续追踪。查询的 1 秒雷达历史窗口会重叠，连续确认不等于独立观测。请同时比较全 query 成功率、覆盖率、预热缺失、仅预测比例和错锁记录；稳定背景也可能形成错误轨迹。默认参数尚未在真实数据上校准，validation 上探索后需用独立测试集评价泛化。
