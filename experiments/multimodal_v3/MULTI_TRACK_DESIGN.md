# V3 多候选轨迹实验设计与使用边界

## 实现来源与依赖

- 固定 [Stone Soup 1.9.1](https://github.com/dstl/Stone-Soup/tree/v1.9.1)，[MIT 许可证](https://github.com/dstl/Stone-Soup/blob/v1.9.1/LICENSE)；官方 [多目标关联教程](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/06_DataAssociation-MultiTargetTutorial.html) 描述了全局一对一关联。版本要求 Python >=3.10，基础依赖不包含 Torch。
- [Multi-Modal-UAV 的 fusion_tracking.py](https://github.com/dtc111111/Multi-Modal-UAV/blob/main/point_cloud_processing/tracker/fusion_tracking.py) 作为处理思路参考；该仓库采用 [MIT 许可证](https://github.com/dtc111111/Multi-Modal-UAV/blob/main/LICENSE)。V3 未复制其代码，也不沿用按时间戳覆盖文件或只保留存活轨迹的输出方式。
- V3 调用 Stone Soup 的 `CombinedLinearGaussianTransitionModel`、`KalmanPredictor`、`KalmanUpdater`、`GNNWith2DAssignment`、`MultiMeasurementInitiator`、`UpdateTimeDeleter` 和 `MultiTargetTracker.update_tracker()`。六维状态在 Stone Soup 中是 `[x,vx,y,vy,z,vz]`；对外转换为 XYZ 和速度 XYZ。V3 的适配层负责空间候选池、起轨候选筛选、数量/重复限制、轨迹历史和在线选择。原单轨迹 `tracking.py` 保留为对照，不参与新增方法的滤波。

安装前可检查 pip 计划：

```bash
python -m pip install --dry-run -r experiments/multimodal_v3/requirements-tracking.txt
python -m pip install -r experiments/multimodal_v3/requirements-tracking.txt
```

这里没有 Torch/CUDA 依赖或安装命令。若环境缺少 Stone Soup，运行 `--methods all` 会直接报错，不会退回自写 KF。原三组可继续用 `--methods original`，不要求 Stone Soup。

## 数据与方法

冻结 B0 网络只前向一次。缓存 v1 的 dense / 全局 NMS / 原 Top1 候选仍能评估五组。v2 最小增加每个 L0 token 的原始点数、运动拟合 valid/dynamic/reliability/speed，均来自当前网络 `lidar_raw` 的 GT-free 层级和诊断输出。缺少这些字段时，目标运动和静态背景项为“未知”，不是 0 速度或确定背景；旧缓存的 `multi_candidate_kf_select` 仍能运行，但这两项没有证据。

| 方法 | 职责 |
|---|---|
| `independent_top1` | 原网络 Top1，不变 |
| `top1_kf` | 旧单轨迹，只接受原 Top1 观测 |
| `candidate_kf` | 旧单轨迹，在预测附近补充 dense 候选 |
| `multi_candidate_kf` | Stone Soup 多轨迹；轨迹级目标分数、支持率、近期观测、创新一致性 |
| `multi_candidate_kf_select` | 同一多轨迹计算，另用可靠运动和可获得的静态背景提示在线选轨迹 |

每个 sequence 单独实例，按真实 query 时间差推进。每条现存轨迹从全 dense 预测中独立选局部候选，与原全局候选及空间分散的起轨候选按 token ID 去重。全局二分配确保一条观测至多更新一条轨迹；未关联的轨迹仍预测，超时由 Stone Soup 删除。新轨迹需要配置的多次实际观测才能确认；临时轨迹和确认轨迹分别限数。确认轨迹不会因为速度低而删除。时间断档重置状态；此前所有轨迹的历史仍导出。

## 选择分项

所有权重和阈值位于 `multi_tracker_defaults.json`。只使用当前与过去 `window_s` 秒的真实观测：

- `target_score`：窗口内分类分数均值，范围 [0,1]，仅作相对证据；分数未校准。
- `observation_support`：窗口内有更新的查询数 / 窗口内查询数；只预测不算观测。
- `recency`：`exp(-距上次观测秒数/window_s)`。
- `innovation_consistency`：`exp(-平均创新距离/innovation_scale_m)`；新起轨无创新时该项未知，贡献为 0。
- `reliable_motion_evidence`：只对 `valid && dynamic` 的诊断取 reliability 均值；无效拟合保持未知。
- `static_background_hint`：仅当拟合有效、reliability 足够、L0 点数达到阈值时，由较低拟合速度构成的有界启发式提示。它**不是背景真值**，单纯低速度或 `motion_invalid` 均不会给出提示；提示不会独自将一个原本可信的确认轨迹压到输出阈值以下，也不会删除悬停轨迹。

总分是前四项加权和；选择增强组加运动项、减背景提示项。无可信确认轨迹可以缺失输出。其他轨迹需连续 `switch_consecutive` 次超过当前轨迹 `switch_margin` 才切换；失锁后的重新选择单独记录，不计持续优势切换。整段离线最优轨迹拼接和 MHT 均未实现。

## 输出与分母

`summary.json` 的 `queries` 是有 3D GT 的 query 数。0.5/1/2 m 成功率及覆盖率以全部有效 GT query 为分母，缺失输出算失败；RMSE、均值和中位误差仅在有输出的有效 GT query 上计算。`updated_outputs`、`prediction_only_outputs`、各类缺失和切换数按全部缓存 query 计数；另列有 GT 的对应输出数与成功数。持续错锁只计算同 sequence、同 track ID 连续至少两个 query 的 >10m 错误。

`per_query.jsonl/csv` 保留五组 XYZ、误差、输出状态与轨迹 ID；`tracking_events.jsonl` 是逐 query 事件；`track_scores.jsonl` 保存临时/确认轨迹的评分分项；`all_track_history.jsonl` 保存包括已删除轨迹的每一步状态，不按时间戳命名单轨迹文件。`track_confirmations.csv` 保存首次确认位置的事后误差，`reacquisition_events.csv` 保存重新选中轨迹时的事后误差。GT 只在两种跟踪器完成状态更新后用于评价。

## 手动运行

先安装依赖，之后由用户手动执行以下真实流程。本任务没有运行真实数据或 GPU：

```bash
export PYTHONPATH="$PWD/src:$PWD"
python experiments/multimodal_v3/extract_observations.py \
  --repo "$PWD" \
  --config outputs/mtce_joint_lora_local_20261014_011406/config.yaml \
  --checkpoint outputs/mtce_joint_lora_local_20261014_011406/train/checkpoints/best.ckpt \
  --cache outputs/kf_v3_multi_run/cache --device cuda:0 --num-workers 2
python experiments/multimodal_v3/evaluate_tracking.py \
  --cache outputs/kf_v3_multi_run/cache \
  --output outputs/kf_v3_multi_run/results \
  --methods all \
  --multi-config experiments/multimodal_v3/multi_tracker_defaults.json
```

复用已有 v1/v2 缓存时，仅执行第二条命令，并指定新的输出目录。评估输出旁还生成 `results.zip`。默认只运行原三组；显式 `--methods all` 才运行五组。默认参数是未经真实验证的起点，不能依据 validation 的最优调参结果直接声称无偏泛化。
