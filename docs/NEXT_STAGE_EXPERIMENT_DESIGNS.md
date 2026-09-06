# 下一阶段实验设计（仅准备，尚未执行）

本文只固定接口、统计口径和实验矩阵。当前没有运行 validation soft voting、没有过滤 Radar 文件，也没有训练 Oracle Target-RDQ。

## 1. RGB 与 RDQ 的 validation sequence soft voting

### 目的

比较单帧分类与简单时序概率聚合，判断无需新时序模型时，soft voting 是否能提高时间块级分类稳定性。

### 冻结协议

- 仅使用 validation，不访问 test。
- 使用已有 RGB 与 RDQ 的最佳 checkpoint，不重新训练。
- 按 `(sequence_id, temporal_block)` 分组；当前 validation manifest 含 415 帧、9 个类别-时间块组。
- 聚合输入为每帧 softmax probability，不对 hard label 投票。
- 不依据 validation 结果改变 top-k 或时间间隔。

### 固定比较

| 方法 | 帧选择 | 聚合 |
|---|---|---|
| Frame-level | 全部单帧 | 不聚合 |
| All-frame soft vote | 时间块内全部帧 | 概率均值 |
| Top-5 keyframe soft vote | bbox area 最大且相邻选择至少间隔 1.5 s 的最多 5 帧 | 概率均值 |

### 已有接口

`tools/evaluate.py` 与 `src/rdq_uav/engine/sequence.py` 已支持以上口径。获准后，对 RGB/RDQ 的 seed 0、1、2 checkpoint 分别执行：

```bash
python tools/evaluate.py CHECKPOINT --split val --radar-mode normal
```

记录 `frame-level`、`temporal_block_soft_vote_all`、`temporal_block_soft_vote_top5_bbox_area_gap1.5s` 的 Accuracy、Macro-F1、每类 recall、groups 与 frames_selected，并汇总 mean ± sample std。由于组数很少，该结果只作为时序稳定性诊断。

## 2. Oracle Target-RDQ

### 科学问题

已观察到 full-frame RDQ 的 same-class shuffle 不掉点。Oracle Target-RDQ 用当前样本 GT xyz 在三维空间内筛选 Radar 点，以诊断原结果是否主要来自整帧 Radar 的 session/background shortcut。

该方法在训练和 validation 都使用 GT 位置，属于 oracle 因果诊断，不能作为可部署方法或公平的常规分类基线。

### 建议的数据接口（尚未实现）

```yaml
data:
  radar:
    target_filter:
      enabled: true
      center_source: gt_xyz
      metric: xyz_euclidean
      radius_m: null          # 只根据 train 统计预先固定
      apply_before_sampling: true
      empty_policy: zero_mask # 不回退到整帧 Radar
```

Dataset 处理顺序固定为：读取原始 `(N,3)` → 去除非有限点和 50 m 外点 → 计算 `||p_xyz - gt_xyz||₂ <= radius_m` → 采样/补零 → 使用既有 train normalization。额外返回：

```text
radar_target_count: int
radar_target_valid: bool
radar_target_radius_m: float
```

`empty_policy` 必须为 zero + invalid mask。若空帧回退 full-frame，会重新引入待检验的 session shortcut。

### Train-only radius 统计

统计数据严格来自 `manifests_oracle_left_fixed256_bbox/train.csv` 的 1,828 帧；没有读取 val/test。点云先应用与 Dataset 一致的 finite/50 m 过滤。

整体统计：

| Radius | 非空帧率 | 每帧点数中位数 | P95 点数 |
|---:|---:|---:|---:|
| 0.5 m | 14.11% | 0 | 97.3 |
| 1.0 m | 24.56% | 0 | 219.6 |
| 2.0 m | 38.29% | 0 | 290.6 |
| 3.0 m | 42.83% | 0 | 304.0 |
| 5.0 m | 49.78% | 0 | 345.6 |
| 7.5 m | 66.58% | 182 | 456.6 |
| 10.0 m | 72.48% | 238 | 572.6 |
| 15.0 m | 89.00% | 324 | 628.6 |
| 20.0 m | 99.78% | 330.5 | 628.6 |

各类别非空帧率：

| Radius | Mavic2 | Mavic3 | Avata | M300 | Pham4 |
|---:|---:|---:|---:|---:|---:|
| 0.5 m | 17.92% | 21.73% | 3.32% | 30.25% | 8.63% |
| 1.0 m | 20.23% | 34.82% | 14.34% | 36.13% | 27.48% |
| 2.0 m | 23.12% | 39.28% | 23.43% | 64.29% | 61.34% |
| 3.0 m | 25.72% | 41.78% | 30.07% | 68.91% | 66.45% |
| 5.0 m | 30.35% | 48.47% | 38.29% | 80.25% | 70.61% |
| 7.5 m | 54.34% | 55.15% | 66.26% | 83.61% | 80.83% |
| 10.0 m | 65.61% | 63.79% | 68.36% | 92.86% | 82.11% |
| 15.0 m | 67.92% | 74.93% | 100.00% | 100.00% | 100.00% |
| 20.0 m | 98.84% | 100.00% | 100.00% | 100.00% | 100.00% |

最近 Radar 点到 GT 的整体距离分位数为：P10=0.383 m、P25=1.028 m、median=5.032 m、P75=11.216 m、P90=16.033 m、P95=17.663 m。

### 半径统计的含义

当前数据不支持直接把 1–2 m 当作统一目标半径：多数帧会变为空，且空帧率本身与类别强相关，模型可能转而把“是否为空”作为类别 shortcut。把半径扩大到 15–20 m 虽可提高覆盖率，却已包含大量场景点，无法有效隔离 UAV 回波。

因此执行前必须先由用户确认以下二选一：

1. **严格目标诊断**：固定较小半径并保留真实空帧，同时报告各类 non-empty 子集与全量结果；接受高空帧率这一限制。
2. **暂缓 Target-RDQ**：先确认 Radar 与 GT 的坐标/目标回波含义，再选半径。

不能根据 validation 表现搜索 radius。若执行严格诊断，候选半径统计已固定为 0.5/1/2/3/5 m；正式训练只能依据 train 覆盖率预选一个主半径，其余最多作为预先声明的敏感性检查。

### 最小实验矩阵（尚未执行）

| ID | 训练输入 | Validation 输入 | 作用 |
|---|---|---|---|
| F0 | Full-frame Radar | Normal | 已有 RDQ 基准 |
| F1 | Full-frame Radar | Same-class shuffle | 已有 session-shortcut 现象 |
| T0 | GT-radius Target Radar | Normal，使用当前 GT 过滤当前 Radar | Oracle Target-RDQ 主结果 |
| T1 | 与 T0 同一 checkpoint | 先 same-class shuffle 原始 Radar，再用当前样本 GT 过滤 | 严格破坏当前空间对应关系 |
| T2 | 与 T0 同一 checkpoint | Zero Radar | 判断 Target-RDQ 是否实际依赖筛选点 |

主要证据是比较 `(T0 − T1)` 与已有 `(F0 − F1)`。若 Target-RDQ 在正确配对时明显优于严格 shuffle，而 full-frame 不下降，才支持“整帧 session/background Radar 掩盖了当前目标对应关系”。同时必须按 `radar_target_valid` 分层报告，避免类别相关空帧率造成错误结论。

以上实验均等待用户确认后再实现或运行。
