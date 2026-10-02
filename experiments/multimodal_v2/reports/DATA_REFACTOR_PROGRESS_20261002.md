# Multimodal V2 数据入口重构与进度记录

日期：2026-10-02（Asia/Shanghai）。范围：统一 Sample、Batch、模型输入准备；不修改网络、候选、关联、评分公式或 loss 数学。

## 0. 工作目录和版本边界

- 用户目标目录 `/home/jasoncui/projects/rdq-uav-experiment-multimodal-v2` 不存在于本执行环境。
- 本次修改发生在可访问的 Git 仓库 `/workspace/scratch/13fdc3aa8fb1/rdq-uav-experiment`。
- 基础提交：`6cbb240ec16ebe22df85ca418b885b397ff80458`。修改前工作区干净。
- 本副本没有用户后来完成的 AMP dtype 修复 `9c6b0bc` 和 BCE AMP 修复 `e887ba8`。本次不覆盖 `scoring.py`、`loss.py`；应用到最新分支时应保留这些修复。
- 未提交、未推送；用户本地路径尚未同步。

## 1. 截至本次修改前的实验进度（来自用户运行记录）

| 项目 | 已报告的进度 | 边界 |
|---|---|---|
| detrex 和资产 | 本地检查通过 | 本环境未重新验证 |
| CPU 合同测试 | 修复后 17 passed，CUDA AMP test skipped | 用户先前修复环境 |
| GPU 两次更新 | 到达 max_steps=2 | 用户本地运行 |
| 小 epoch | global_step=8；3D 评分头末层 weight/bias 非零 | 参数更新已由用户检查 |
| 32 条验证 | Success@1m=23/32；coverage=31/32；median=0.833 m | 工程门禁，不是正式效果 |
| 完整 B0/B1 | 两者 4004/4800，Success@1m=83.4167%；coverage=4781/4800 | 旧输入定义 |
| 候选 oracle | 4050/4800=84.375%；旧候选池可重排救回上限 46 条 | 数据入口变更后需重新计算 |
| 正式 B2/B3 | 未在当前上下文提供正式训练完成证据 | 不记为已完成 |

## 2. Step 1 源码审计（修改前）

| 环节 | 实际代码与字段 | 结论 |
|---|---|---|
| query record | `lidar_v2/data.py::LiDARUAVDataset.__init__` 枚举 split 中各 sequence 的 `ground_truth/*.npy`；文件 stem 转为 query_time | 没有独立 V2 query records |
| 雷达索引 | `LiDARQueryBuilder.stream` 合并 `livox_avia/*.npy` 与 `lidar_360/*.npy`，sensor 0/1 | 同 sequence 双传感器 |
| 雷达选择 | `merged_lidar.py::select_last_history` 取 timestamp<=t 的最近 max_events（默认20） | 无 1 秒下界 |
| 点加载 | `load_released_xyz` 取前3列，剔除非有限和 XYZ 全零行 | 未额外变换坐标；本次保持 |
| 点时间 | 每点使用所属事件 filename_timestamp-query_time | 没有使用点内硬件时间戳 |
| 图像选择 | V1 LeftImageIndex：按 image_time+offset 选最近 PNG；原 V2 config gap=0.04s | 已允许未来图像；旧 gap 不符合新定义 |
| 3D GT | ground_truth 文件数组 reshape(3)，target_timestamp=query_time | 对应查询时刻 |
| 2D GT | manifest 中 `(sequence_id, image_filename)` -> source-pixel xyxy | 已按所选图像匹配，迁移保留 |
| collate | V2 -> V1 collate -> LiDAR V2 collate | 间接依赖旧 Dataset 数据契约 |
| preprocess | image_uint8/source_wh/view_wh/scale、points/sensor/dt、监督与有效性字段 | 构造 DINO padding mask、targets 和旧 InteractionContext |
| 调用者 | `training.py::forward_step`；`train.py`；`evaluate.py`；`diagnostics/check_b0_b1.py` | 均同步改造 |
| 老诊断 | `evaluate_e0.py` 使用旧 LiDAR Dataset；`diagnose_e5.py` 使用 V1 入口 | 保留历史实现，避免破坏复现 |

`LiDARPyramidContext` 是 V1 分阶段 backbone adapter 的上下文。当前 V2 直接调用 LiDAR 检测器：`model.py::spatial_forward` 每次前向构建一次 hierarchy，并传给后续各层。本次保留该逻辑，不引入一个没有调用者的新 PyramidContext，也不改旧文件。

## 3. Step 2 最小改造及接口边界

| 文件 | 改动 |
|---|---|
| `src/rdq_uav/multimodal_v2/data.py` | 自建 GT query records、雷达/图像索引、时间窗口选择、Sample、完整性检查、独立 collate、预处理；本地 ViewTransform 和 image-label loader |
| `src/rdq_uav/multimodal_v2/geometry.py`（新） | 迁入 ProjectionContext、标定加载及 omni+radtan 投影；数学计算保持原样 |
| `src/rdq_uav/multimodal_v2/model.py` | 接收 ProjectionContext，直接读取 batch m_R/m_V，取消旧上下文包装 |
| `src/rdq_uav/multimodal_v2/interaction.py` | 接收 projection+m_R+m_V；候选跨模态注意力算法不变 |
| `src/rdq_uav/multimodal_v2/training.py` | 投影加载改用 V2，实现调用签名同步 |
| `experiments/multimodal_v2/train.py` | build_datasets 解包改为 train,val |
| `experiments/multimodal_v2/evaluate.py` | Dataset 与投影调用同步 |
| `experiments/multimodal_v2/diagnostics/check_b0_b1.py` | 门禁调用同步 |
| `experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml` | history=1.0，gap=1.0；max_events=20 仅供历史 E0 使用 |
| `experiments/multimodal_v2/tests/test_v2_core.py` | 原16项测试改用直接投影和 mask 参数 |
| `experiments/multimodal_v2/tests/test_v2_data.py`（新） | 新增18项数据、模型接口、投影等价测试 |
| `experiments/multimodal_v2/diagnostics/check_data_samples.py`（新） | 真实 sequence 的至少3条 Sample 审计输出 |
| README、IMPLEMENTATION_STATUS、本记录 | 数据定义与当前进度，区分已验证和待验证 |

保留：扁平 detector 字段、DINO preprocess、padding mask、resize transform、MultimodalTargets、模态有效性 mask。

废弃的 V2 接口：Dataset 的 `lidar_dataset` 参数、`query_uid` 身份、逐 Sample calibration_handle、image_query_gap_s（旧符号为 query-image）、V1 collate、V1 LeftImageIndex、旧 InteractionContext、build_datasets 三返回值。

新接口：`MultimodalV2Dataset(root, sequence_ids, manifest, *, camera_wh, short_edge, max_size, radar_history_s=1.0, max_image_gap_s=1.0, filter_empty=False)`；`build_datasets -> (train,val)`；`prepare_model_batch -> (model_batch,images,padding_mask,projection,targets,transforms)`。

## 4. 最终数据流

query_time（ground_truth 文件名）
→ 同 sequence 雷达 [t-history,t]（双 LiDAR 全部事件）
→ 同 sequence 最近历史 left PNG within [t-gap,t]
→ 查询时刻 XYZ + 所选图像的人工 box
→ 完整 Sample
→ independent collate（cat points / stack images）
→ prepare_model_batch（DINO、padding、投影、targets）
→ 当前 V2 模型。

- timestamp 索引在 Dataset 初始化时建立；范围查询使用 searchsorted，复杂度 O(log E + K)，不在 getitem 中重新扫描目录。
- query_time 唯一；发现同一 Dataset 或 train/val 间重复时间戳时显式报错，不改用 sequence/query_uid 掩盖冲突。
- 边界包含 t-history 和 t；future 雷达与 future 图像均禁止。
- 图像取不晚于query_time的最新帧；超差图像 path/time=None，placeholder 且 m_V=False；不会读取它的2D框。
- 训练初始化过滤双缺失，必要时检查事件是否有有效点并缓存每事件布尔值；不缓存全部点云。验证保留双缺失查询。
- selected event_count 包含窗口内文件，即使某事件清理后0点；m_R 只由最终有效点数决定。
- `supervision_recent_mask` 保留窗口内末四事件语义，供冻结 LiDAR loss 诊断使用，不限制输入，不作为编码器特征。
- 模态状态只有 m_R/m_V，一一对应 radar.valid/vision.valid，没有第二套可能冲突的布尔量。
- 原图为左视图 crop 后 resize，image_uint8 在0..255；DINO normalizer 在 preprocess 中处理。padding mask 描述图像空间布局；placeholder 在 m_V 处屏蔽候选和证据，避免全 mask DINO 引入 NaN。

## 5. 完整 Sample 与 Batch 字段/shape

N 为单条有效点数，P=sum(N_i)，B 为 batch size，H/W 为 resize 后尺寸。

| 类别 | 字段 | Sample shape/type | Batch shape/type |
|---|---|---|---|
| Radar | points | float32 [N,3]，发布的 XYZ | float32 [P,3] |
| Radar | delta_t | float32 [N]，event_time-t | float32 [P] |
| Radar | sensor_id | int64 [N]，0=Avia/1=Mid360 | int64 [P] |
| Radar | m_R | bool，N>0 | bool [B] |
| Vision | image_uint8 | uint8 [3,H,W] | uint8 [B,3,H,W] |
| Vision | vision_delta_t | float，image_time-t；missing存0 | float32 [B]，须结合m_V |
| Vision | m_V | bool | bool [B] |
| Vision resize | image_source_wh | int64 [2] | int64 [B,2] |
| Vision resize | image_view_wh | int64 [2] | int64 [B,2] |
| Vision resize | image_scale_xy | float32 [2] | float32 [B,2] |
| Target | target_xyz | float32 [3] | float32 [B,3] |
| Target | target_valid | bool | bool [B] |
| Target | gt_box_xyxy_px | float32 [4]，source pixels；missing全0 | float32 [B,4] |
| Target | gt_2d_valid | bool | bool [B] |
| Diagnostic supervision | supervision_recent_mask | bool [N] | bool [P] |
| Packing | num_samples | int=1 | int=B |
| Packing | point_counts | batch生成 | int64 [B] |
| Packing | point_batch_index | batch生成 | int64 [P] |
| Meta | sequence_id、sample_id | str | list[str]，长度B |
| Meta | query_time、target_timestamp | float，秒 | float64 [B] |
| Meta | event_count | int | int64 [B] |
| Meta | event_timestamps | list[float] | list[list[float]] |
| Meta | event_sequence_ids | list[str] | list[list[str]] |
| Meta | image_time | float或None | float64 [B]；missing NaN |
| Meta | left_image_path、image_sequence_id | str或None | list[str或None] |
| Contract/audit | radar_history_s、max_image_gap_s | float | Dataset配置；不进入模型 |

prepare_model_batch 明确白名单，仅传递观测/相对时间/resize/packing/监督；identity、绝对时间、路径、event metadata 留在原 batch 用于日志。图像预处理输出 [B,3,Hpad,Wpad]，padding mask bool [B,Hpad,Wpad]，ProjectionContext 为 batch 级对象，无 Sample 内标定复制。

vision_delta_t 已保留，但当前网络没有新增时间编码层来消费它；不因这次数据改造改变网络结构。

## 6. 删除依赖与保留依赖

- V2 data.py 不再导入或构建 LiDARUAVDataset、LiDARQueryBuilder、V1 LeftImageIndex、V1 collate。
- V2 data.py 没有 V1/LiDAR V2 import；双传感器低层读取继续使用公共 `multimodal/merged_lidar.py`，不依赖 LiDAR V2 Dataset。
- V2 模型链路不再构建/使用 InteractionContext 或 calibration_handle；改为 projection 与 batch 有效性 mask。
- V2 仍有必要的 candidate-level R←V/V←R；这不是 V1 的 pre-stage HCI，本次算法不变。
- LiDAR 检测器、候选 builders、DINO adapters 和冻结诊断 loss 的旧包依赖仍存在；本次没有宣称完成整个 V2 的代码独立化。
- Python 包级 __init__ 仍会因现存模型导入加载旧包；本次消除了 Dataset 构建依赖，不宣称所有旧模块禁止 import 后可运行。
- `lidar_v2/data.py` 和所有 LiDAR-only baseline 保留不动；历史 E0、E5 诊断继续调用旧链路。

## 7. 实际验证与限制

最终测试命令：

```bash
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests \
  tests/test_lidar_uav_v2_spatial_query_pipeline.py \
  tests/test_multimodal_v1_p3_data_contract.py \
  tests/test_multimodal_v1_p4_geometry_gate.py
```

结果：48 passed，1 warning（旧 attention 的 bool/float mask 类型提示；算法未修改）。其中 V2 原16项+新增18项=34项，其余14项验证旧数据/几何链路未受影响。

- 实际使用 CPU PyTorch 2.5.1+cpu、torchvision 0.20.1+cpu、Lightning 2.6.6，Python3.12。不是用户GPU环境的复现。
- 所需7项测试全部覆盖；额外覆盖完整时间窗口不截断、图像标签错配、重复query_time、索引无getitem重扫、可配置窗口、非零offset拒绝、元数据不入模型、projection数学等价。
- 新 Sample→collate→preprocess→当前 V2 模型测试使用 fake DINO/candidate producers；覆盖有视觉证据和placeholder屏蔽。
- 另有真正 LiDARUAVDetector（随机初始化、CPU）+冻结 CandidateLoss 接口测试，含空雷达查询；不代表 checkpoint 实验效果。
- compileall：PASS。git diff --check：PASS。
- 初次扩展测试因安装过程中的 torch/torchvision 版本混杂无法收集；固定一致的CPU版本后重跑上述最终命令通过。没有隐藏仍然失败的用例。

真实 smoke 已执行以下入口，结果为明确缺数据错误：

```bash
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/check_data_samples.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --sequence seq0001 --samples 3
```

`FileNotFoundError: real sequence GT unavailable: .../data/mmaud_official_train/seq0001`。

状态：BLOCKED_MISSING_REAL_DATA，不是PASS。没有伪造三条真实Sample，也没有用其他点云数据集冒充MMAUD。

## 8. 尚未解决和下一步

- [x] 统一query-time Sample、独立collate、预处理入口。
- [x] 雷达和图像完整历史窗口，测试覆盖。
- [x] 去掉V2 Dataset对LiDARUAVDataset依赖、旧InteractionContext包装。
- [x] 保留标定数学和一次hierarchy构建逻辑。
- [x] CPU与旧链路回归验证，进度记录。
- [ ] 将这些改动合入用户包含AMP/BCE修复的最新分支；用户本地目录未被本次环境修改。
- [ ] 完整真实sequence至少3条Sample smoke。
- [ ] 真实GPU B0/B1 identity、短训练门禁。
- [ ] 新数据定义的完整B0/B1 baseline和oracle统计，再训练B2；使用新输出目录，不续训旧输入定义的checkpoint作同一实验。
- [ ] 全V2代码独立化（候选、backbone和诊断loss）属于后续任务。

待验证：发布XYZ与GT/相机坐标的真实独立核查；代码沿用既有同坐标假设，测试只证明数学迁移等价。当前 calibration time_offset_s=0.0，符合原始timestamp匹配；非零offset显式拒绝，不猜测新时钟规则。2D标注实际覆盖率需真实数据重新统计，最近历史帧可能切换到未标注图像，box_valid=False应正常忽略。

## 9. 2026-10-02 后续修订：图像严格因果

按用户最新要求，图像规则由前后1秒最近帧改为历史1秒最近帧。
选择最大 image_time<=query_time，且 image_time>=query_time-max_image_gap_s；
默认gap=1.0，包含两个端点。vision_delta_t范围为[-gap,0]。
更近的未来图像也不可选；没有合格历史图像时，m_V=False，path/time=None，
使用placeholder，不产生2D监督。初始化双缺失过滤自动使用这一新规则。
源码审计表保留修改前实现事实；第7节48项结果为上一轮历史记录。
新增回归覆盖未来更近、仅未来帧、窗口端点、可配置gap、未来图像完整性拒绝。

本轮修订验证：同第7节测试命令，55 passed，1个原有attention mask类型warning；
其中41项V2测试与14项旧链路回归。compileall和git diff --check通过。
真实MMAUD数据仍不在当前环境，真实Sample smoke/GPU门禁仍待验证。


## 2026-10-02: direct YOLO supervision (implementation pending verification)

V2 no longer consumes annotation_manifest or a bbox mapping dictionary.
For the selected causal historical image, read
`<root>/<sequence>/<label_directory>/<image_stem>.txt`. Default directory:
`2d_detect`. All directory boxes are trusted supervision as instructed by
the user. Convert YOLO class/cx/cy/w/h using the calibrated left source image
size, consistent with gt_bbox_annotator (including side-by-side PNG crops).
Missing/empty files give zero box and gt_2d_valid=False, never implicit negative
supervision. Malformed/out-of-frame or multiple-box files raise explicit errors:
the current target contract supports one UAV box. Historical image selection,
modality masks and train-only both-missing filtering remain unchanged.
Tests and the smoke entry were updated; no tests/smoke were run for this revision
per the user's instruction to defer verification until equipment is available.
Earlier passing test totals apply to the preceding manifest-based revision.
