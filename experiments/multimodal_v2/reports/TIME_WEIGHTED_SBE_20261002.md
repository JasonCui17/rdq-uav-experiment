# 时间加权 SBE：2026-10-02

## 当前实现位置与职责

雷达当前复用的 `src/rdq_uav/lidar_v2/sbe.py` 已实现新统计语义；Multimodal V2 无需重新读取数据，直接消费 data.py 整理的 points、delta_t、sensor_id 和既有 hierarchy.point_to_l0。未改 Batch、几何层级、候选、交互或 loss。该 SBE 是共享模块，因此 LiDAR-only 也会使用新统计语义；尚未将雷达 backbone 独立迁入 multimodal_v2。

## 统计契约

每个 0.5m L0 voxel 保留八个 0.25m 子槽。位置标准差在子槽内计算，并非先算整个 voxel 再重复复制到八个槽。位置残差 r 是相对子槽中心的位置除以 0.5m，均值/标准差均无量纲。

每点年龄 a=-delta_t；权重 w=2^(-a/h)，h 是正数半衰期，单位秒。默认 h=0.2s。0/0.2/0.5/1s 的点权重分别为 1/0.5/约0.1768/0.03125。窗口内点不被删除。

| 维度 | 统计 |
|---|---|
| 0..2 | 加权位置均值：sum(w*r)/sum(w) |
| 3..5 | 加权逐轴位置标准差：sqrt(sum(w*(r-mean)^2)/sum(w))，总体统计，不使用无偏校正 |
| 6 | log(1+真实点数)，不按时间加权 |
| 7 | occupancy |
| 8 | Avia 点数/真实总点数，不按时间加权 |
| 9 | 平均绝对时间权重 sum(w)/N，替代最新点年龄 |
| 10 | 未加权 delta_t 总体标准差，保持原定义 |

同一子槽内近期点对几何统计贡献更大；全部观测较旧的子槽仍通过平均时间权重表示较低新鲜度。它不是测量可信概率，不补偿运动、不估计速度，也不会单靠权重删除旧 voxel。

FP32 统计在 autocast 外计算。几何归一化采用相对该子槽最新时间的指数权重，等价于原 w 的公共缩放，可避免全部权重下溢；方差使用中心化残差的二次归约。平均时间权重使用绝对权重，不随槽内归一化而丢失新鲜度。单点标准差严格为零，空槽全部统计为零。非法正时间差/非有限值、非正/非有限或超出 FP32 正常数范围的半衰期均报错。

## 内部 cross-attention 与输出

- slot_stats：[V0,8,11]。
- 拼接固定三维子槽中心：[V0,8,14]。
- Linear(14,16)+GELU：[V0,8,16]。
- 每 voxel 一个可学习 16 维 Query，双头 cross-attention 读取八个 Slot Token，mask 空槽，输出：[V0,16]。
- 原 88 维展平统计与 16 维注意力摘要拼接：[V0,104]。
- Linear(104,128)+LayerNorm：[V0,128]。

attention 只发生在各 voxel 内部的八个槽之间，不是全部 voxel 之间。既有 CUDA batch 分块保持。

## 配置

```yaml
model:
  voxel:
    sbe:
      statistics_version: time_weighted_v2
      time_half_life_s: 0.2
```

配置位于 configs/lidar_uav_v2.yaml。Multimodal V2 initialization.lidar_config 当前指向此文件。其余 slots=8、slot_dim=11、flattened_dim=88、output_dim=128 和 VQSA 配置保持。

## Checkpoint 影响

参数名称、维度、数量不变，因此旧同形状 SBE checkpoint 可以严格加载，但其权重是在不同统计语义下训练的。加载成功不代表新旧输出一致，也不代表原 baseline 指标可沿用。新特征需要重新训练或验证适配；B0/B1 数值相等仍不能证明新 SBE 保留原单模态精度。不能把此次更改报告为“冻结雷达行为不变”。

## 验证状态

按用户此前安排，本轮不执行 pytest、真实数据和 GPU 训练。已修改旧单点/时间统计期望，新增解析例子、等时间等价性、时间平移、新鲜度、半衰期、下溢、非法输入、内部 cross-attention Shape/mask/梯度测试代码。

只做 AST 语法、增量 diff 和安装包检查。后续运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_lidar_uav_v2_sbe.py \
  tests/test_lidar_uav_v2_vqsa.py experiments/multimodal_v2/tests
```

随后验证 checkpoint 适配、真实样本、AMP、单模态精度和训练可用性。
