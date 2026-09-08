# Stage 4.11 Radar Target Association Audit

## 目的与边界

本阶段只定量比较由已确认字段支持的 Radar target candidate 规则。脚本不训练模型、
不读取 test、不修改 RDQ，也不把缺失 channel 推测为可用数据。

正式五类 MMAUD V1 `radar_enhance_pcl/*.npy` 均为 `(N,3)`，只含发布版 XYZ。
因此 Power、Doppler、Alpha/Beta、SNR、cluster ID、track ID、速度、Class 和
Confidence 均不属于本轮五类可比较输入。

## Candidate 规则

| 规则 | 输入字段 | 性质 |
|---|---|---|
| `released_xyz_all` | 有限 XYZ | 发布版增强点云的全部点 |
| `finite_range_le_50m` | XYZ 推导的 `‖xyz‖` | 项目既有 50 m 范围过滤；50 m 不是官方目标标签 |
| `oracle_gt_range_gate` | XYZ、GT range | 可选 `±0.5 m` oracle；依赖 GT，不可部署 |

这里没有所谓“官方 target-filtered”规则：`radar_enhance_pcl` 是检测点的时序累积，
发布 NPY 不包含目标 ID 或 cluster/track channel。脚本会在报告中明确记录这些候选为
`unavailable_not_invented`，而不是静默构造替代字段。

## 固定坐标与投影

按本阶段实验约束，默认读取：

```text
calibration/radar_frame_resolution.json
  -> global_results.global_rigid_icp.transform
```

即 Stage 4.10 的约 `+90° yaw` Radar→GT 变换，再复用已验证的 GT→左鱼眼
OmniRadtan 投影。本工具不重新拟合坐标关系。

注意：后续语义审计提出了发布版 V1 可能应使用 `rot180z` 的新证据；这与本轮指定的
固定 Stage 4.10 变换存在冲突。本脚本忠实执行指定变换，并通过 `--radar-transform`
保留替换接口。使用结果时必须把结论限定为“在该固定变换条件下的 candidate audit”。

## 指标和 Null Control

Train、val 分开输出：

- candidate non-empty rate；
- candidate 点数/frame；
- 最近 candidate 到 GT 的 3D 距离；
- 投影后最近 candidate 到官方 GT bbox center 的 2D 距离；
- Coverage@8/16/32/64；
- same-sequence half-cycle target shuffle control。

Oracle range gate 根据当前真实 GT range 选择 candidate，但 shuffle 仅替换评分目标；
因此它只能表示“已知目标距离后”的 association 上界，不能作为模型输入结果。

## 输出

```text
outputs/stage4_target_association_<timestamp>/
├── association_predictions.csv
├── association_summary.csv
└── report.json
```

## 手动执行

```bash
conda activate rdq
cd /home/jasoncui/projects/rdq-uav-experiment
python tools/radar_target_association_audit.py --with-gt-range-oracle --gt-range-gate-m 0.5
```

本阶段脚本准备完成后不会由 Codex 自动运行完整审计。
