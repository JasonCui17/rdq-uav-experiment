# MMAUD V1 雷达动态查询实验

本项目是一套面向 MMAUD V1 五类无人机识别的模块化研究代码，用于验证一个明确的
研究问题：**由当前 Radar 测量生成的动态 Query，是否比固定 Learned Query 和普通
特征拼接更有效地选择双鱼眼图像中的无人机信息？** 整体流程为：

```text
双鱼眼 RGB + 增强雷达 XYZ 点云
    → 雷达条件动态查询（Radar-conditioned Dynamic Query）
    → 对视觉 Token 进行交叉注意力
    → 无人机型号分类
```

五个类别及其固定标签为：

```text
0 Mavic2
1 Mavic3
2 Avata
3 M300
4 Pham4（沿用官方目录拼写；论文中可写为 Phantom4）
```

## 当前已经验证的状态

- 数据集根目录：`/home/jasoncui/datasets/MMAUD/v1`
- 数据审计报告：`/home/jasoncui/datasets/MMAUD/v1/audit/AUDIT_REPORT.md`
- 在 40 ms 同步阈值内成功配对的 GT：5,207 / 5,212
- 加入 1 秒隔离带后的主时间块划分：训练集 3,308，验证集 790，测试集 779
- Radar 均值和标准差只使用训练集数据计算。
- 数据加载时删除距离超过 50 m 的 Radar 点。审计发现，只有 Mavic2 和 M300
  存在反复出现的约 395 m 静态点；如果保留这些点，模型可能利用它们识别采集
  session 或类别，而不是学习无人机本身。
- 每帧 Radar 默认最多保留 768 点；不足时使用零填充，并通过布尔 Mask 排除填充点。

重要的实验边界：MMAUD V1 中每个无人机型号都与一次独立采集 session 绑定。
因此，本实验衡量的是 V1 固定采集条件下的分类性能，不能直接证明跨 session 或
跨场景的型号泛化能力。

## 工程结构

```text
configs/                 实验配置，避免将参数隐藏在代码中
src/rdq_uav/data/        Manifest、数据划分、RGB 与 Radar 预处理
src/rdq_uav/models/      可替换的骨干、点编码器、注意力和预测头
src/rdq_uav/engine/      训练循环和轻依赖的指标实现
tools/                   构建、训练、评估、预检查和可视化入口
tests/                   合成数据正确性测试
```

所有实验模型都通过配置项切换：

- `rgb`：RGB 全局视觉 Token 基线
- `radar`：Radar-only 基线
- `concat`：池化后的 RGB 特征与 Radar Token 直接拼接
- `learned_query`：固定 Learned Query 对视觉 Token 做交叉注意力，并保留相同的 Radar skip
- `rdq`：由当前 Radar 测量生成 Query，再对视觉 Token 做交叉注意力
- RDQ no-skip：设置 `variant=rdq`、`radar_skip=false`

迁移的开源组件及许可证说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 鱼眼时空对齐与 Oracle ROI

低分辨率 RGB 捷径实验在 `32×24` 输入上仍达到 100%，说明完整画面分类主要记住了
类别绑定的采集背景。正式 A～E 对照实验应暂停，先使用三维 GT、严格鱼眼模型和经过
验证的相机外参生成目标区域。

新的标定模块支持：

- 官方 Kalibr `omni+radtan` 鱼眼投影；
- 5 Hz GT 到图像时间的 PCHIP 插值；
- 相机外参与相机–GT时间偏移的鲁棒联合估计；
- 按 UAV 真实尺寸和距离生成自适应 Oracle ROI；
- 拟合集/留出标注集重投影误差报告和原图叠加检查。

完整设计、已知参数、未知外参以及执行命令见
[docs/FISHEYE_ALIGNMENT_DESIGN.md](docs/FISHEYE_ALIGNMENT_DESIGN.md)。在外参通过留出帧
验证前，不允许将投影框用于正式实验。

### 官方二维框与当前标定结果

官方 `MMAUD_2D.zip` 已解压到：

```text
/home/jasoncui/datasets/MMAUD/official_2d_detection
```

审计确认其 4,425 张 `1280×960` 图片全部是本地时间戳双鱼眼图的**左半幅精确像素
裁剪**：精确匹配 4,425/4,425，无未匹配或多义帧。不要把 `b1_500` 等编号当成 GT
帧号；应使用解码后像素哈希恢复时间戳：

```bash
python tools/calibration/import_official_2d.py
python tools/calibration/build_official_center_annotations.py
python tools/calibration/bootstrap_axis_aligned.py \
  --annotations calibration/official_left_center_annotations.csv \
  --output calibration/official_left_initial_extrinsics.json \
  --max-observations-per-camera 80
python tools/calibration/fit_spatiotemporal.py \
  --annotations calibration/official_left_center_annotations.csv \
  --initial calibration/official_left_initial_extrinsics.json \
  --output calibration/official_left_fitted_calibration.json
python tools/calibration/evaluate_official_projection.py
python tools/calibration/render_official_overlays.py
```

当前左相机解算结果：

| 数据 | 数量 | 中位误差 | P95 | 10 px 内 | 投影中心落入官方框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train 时间块 | 2,732 | 1.88 px | 7.39 px | 97.1% | 99.9% |
| Val 时间块 | 666 | 2.57 px | 8.31 px | 97.3% | 99.7% |
| Test 时间块 | 722 | 2.46 px | 8.52 px | 97.2% | 90.2% |

拟合出的时间约定为 `GT查询时刻 = 图像时刻 - 0.12415 s`。详细参数和评估分别见
`calibration/official_left_fitted_calibration.json` 与
`calibration/official_left_projection_evaluation.json`；直观检查图见
`calibration/official_overlays/heldout_montage.jpg`。

当前只能严谨生成**左鱼眼**的 GT 投影锚点。官方二维数据不含右相机框，公开标定包也
没有左右相机外参，所以右鱼眼仍须人工点击少量中心、可靠的右图检测结果或官方 stereo
外参后再解算；代码不会用“平行相机”假设伪造右相机结果。

### 左鱼眼 Oracle ROI 数据集

已经使用统一 `1.0 m` 三维包围半径和 `1.5×` 上下文生成独立 manifests：

```text
manifests_oracle_left/
├── train.csv       2925
├── val.csv          671
├── test.csv         735
├── radar_stats.json
└── oracle_roi_summary.json
```

所有类别使用相同半径，ROI 大小不读取真实 class；Radar normalization 也已只使用剩余
2,925 个 Oracle 训练样本重新计算。专用配置为：

```bash
python tools/preflight.py --config configs/oracle_left.yaml
python tools/train.py --config configs/oracle_left.yaml --limit-per-class 16 \
  --set experiment.name=oracle_left_overfit80 --set train.epochs=100 \
  --set data.train_color_jitter=0.0
```

上述自适应 ROI 的目标遮挡 test Macro-F1（85.61%）高于未遮挡（78.81%），证明它仍受
背景、投影位置和缩放 shortcut 主导。因此正式推荐入口已经切换为固定 256 px、目标严格
居中、越界即删除的配置：

```bash
python tools/preflight.py --config configs/oracle_left_fixed256.yaml
python tools/train.py --config configs/oracle_left_fixed256_lowres.yaml
python tools/train.py --config configs/oracle_left_fixed256_lowres_masked.yaml
```

固定中心数据规模为 train/val/test = `1996/417/565`，其 Radar 统计也已单独重算。完整
数据协议、shortcut 控制和 A～E 命令见
[docs/ORACLE_LEFT_EXPERIMENT.md](docs/ORACLE_LEFT_EXPERIMENT.md)。

### 官方 bbox 反事实实验

固定 ROI 的目标遮挡实验确认外围背景仍能达到很高的分类性能。按照 UG2 获胜方案的
“检测后裁剪分类”流程，以及 bbox 引导的反事实数据生成方法，当前已建立完全共享样本的
官方二维框子集：

```text
manifests_oracle_left_fixed256_bbox/
├── train.csv       1828
├── val.csv          415
└── test.csv         558
```

四组输入分别是完整 ROI、擦除目标框、仅保留目标框，以及将目标框裁剪放大。所有框统一保留
`2×` bbox context：

```bash
python tools/train.py --config configs/oracle_left_bbox_full.yaml
python tools/train.py --config configs/oracle_left_bbox_erased.yaml
python tools/train.py --config configs/oracle_left_bbox_foreground_only.yaml
python tools/train.py --config configs/oracle_left_bbox_crop.yaml
```

前三组用于严格诊断目标和背景贡献；第四组复现检测后分类的实际数据流。四组均为 RGB-only、
256×256、10 epoch，模型和训练超参完全相同。seed 0/1/2 的验证集 Macro-F1 为：Full
`99.30±0.77%`、BBox erased `96.98±2.00%`、BBox crop `76.26±2.93%`。结果证明目标
区域具有类别信息，但即使擦除 UAV，session/background 仍足以完成接近 97% 的分类；当前
不能把整图接近满分的结果解释为 UAV 外观识别。详细逐类结果见
[docs/ORACLE_LEFT_EXPERIMENT.md](docs/ORACLE_LEFT_EXPERIMENT.md)。在完成 M300 裁剪审计和
validation 时序聚合验证前，不启动 A～E 融合比较。

问题定性、外部方案来源、迁移与不迁移的理由见
[docs/PROBLEM_AND_METHOD_MIGRATION.md](docs/PROBLEM_AND_METHOD_MIGRATION.md)。

## 运行环境

本项目使用独立 Conda 环境 `rdq`，不依赖也不修改其他项目的环境。已经验证的环境为：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.21 |
| PyTorch | 2.1.1 + CUDA 11.8 |
| torchvision | 0.16.1 + CUDA 11.8 |
| NumPy | 1.26.4 |
| Pillow | 10.1.0 |
| timm | 1.0.14 |
| OpenCV | 4.9.0（headless） |

### 从零创建环境

在终端执行：

```bash
cd /home/jasoncui/datasets/MMAUD/rdq_uav_experiment
conda create -n rdq python=3.10 pip -y
conda activate rdq
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

`requirements.txt` 是从实际通过测试的 `rdq` 环境中整理出的精确 pip 版本清单，而不是
宽泛的最低版本范围。`-e .` 会把本项目以 editable 方式注册到环境中，修改 `src/`
代码后无需重复安装。

如果机器不能访问 PyTorch 官方索引，可以先单独安装与本机 CUDA 匹配的 PyTorch，
然后执行其余依赖安装。正式实验必须记录实际的 PyTorch、CUDA 和 GPU 型号，不能在
CPU 版 PyTorch 与 CUDA 版 PyTorch 之间直接比较训练时间。

### 验证环境

激活环境并进入项目目录后执行：

```bash
conda activate rdq
cd /home/jasoncui/datasets/MMAUD/rdq_uav_experiment
python tools/preflight.py
python -m pip check
python -m unittest discover -s tests -v
python tools/smoke_test.py
```

正常情况下应分别看到 `PREFLIGHT_OK`、`No broken requirements found`、单元测试 `OK`
和 `SMOKE_TEST_OK`。`preflight.py` 还会检查三个 Manifest 的样本数、Radar 训练集统计量、
ResNet18 权重缓存和 CUDA 设备。正式训练前请确认输出中至少包含：

```text
torch OK 2.1.1+cu118
cuda_available True
gpu 0 NVIDIA GeForce RTX 3070 Laptop GPU
```

### 更新依赖清单

新增依赖后，先在 `rdq` 环境中完成安装和测试，再用下面的命令查看环境实际版本：

```bash
python -m pip list --format=freeze
```

将运行依赖的精确版本同步到 `requirements.txt`。其中不记录 `pip`、`setuptools`、
`wheel` 这三个环境构建工具，也不记录 editable 安装产生的本地 `rdq-uav` 条目；
本地项目始终通过前面的 `python -m pip install -e .` 单独安装。

ResNet18 预训练权重由 `timm` 自动从缓存读取；若新机器尚未缓存，第一次使用
`pretrained=true` 时需要联网下载。如果只想离线检查代码结构，可以执行：

```bash
python tools/train.py --config configs/base.yaml --limit-per-class 1 \
  --set model.backbone.pretrained=false --set train.epochs=1
```

随机初始化骨干网络的结果只能用于代码检查，不能作为正式实验结果。

## 重新生成 Manifest

当前已经生成好 Manifest。如果修改了数据划分、同步阈值或 Radar 距离过滤规则，
需要重新执行：

```bash
python /home/jasoncui/datasets/MMAUD/audit_v1.py
python tools/build_manifest.py --config configs/base.yaml
```

如果要生成更严格的时间顺序留出实验，请使用单独的输出目录，避免覆盖主实验的
时间块划分：

```bash
python tools/build_manifest.py --config configs/base.yaml \
  --set data.split.mode=temporal_holdout \
  --set data.manifest_dir=/home/jasoncui/datasets/MMAUD/rdq_uav_experiment/manifests_holdout
```

## 数据和模型冒烟测试

```bash
python -m unittest discover -s tests -v
python tools/smoke_test.py
python tools/verify_images.py
```

冒烟测试配置使用小型 CNN 和低分辨率，只用于检查数据与模型管线，不能作为实验基线。
`verify_images.py` 会完整解码 train/val/test 实际使用的全部图像，而不只是检查 PNG 文件头。
实测 Pillow 在持久化 DataLoader 子进程中会偶发报告数据流错误，但同一图像在主进程
连续完整解码 100 次均正常。因此训练、验证和测试目前默认使用 `num_workers=0`，并在
单张图像读取失败时重新打开尝试三次。以后切换到经过验证的解码后端后，再恢复多 worker。

## 正式训练前：80 个样本过拟合测试

每类选择 16 个样本、关闭数据增强，并检查训练准确率能否超过 95%，然后才能开始
正式实验：

```bash
python tools/train.py --config configs/base.yaml --limit-per-class 16 \
  --set experiment.name=overfit80 \
  --set train.epochs=100 \
  --set data.train_color_jitter=0.0
```

如果无法下载预训练权重，应先解决权重问题，不要把随机初始化骨干网络得到的结果
当作正式实验。

## 对照实验命令

A～E 必须使用完全相同的 Manifest、随机种子和训练配置。

```bash
# A：仅使用 RGB
python tools/train.py --config configs/base.yaml \
  --set experiment.name=A_rgb --set model.variant=rgb

# B：仅使用 Radar
python tools/train.py --config configs/base.yaml \
  --set experiment.name=B_radar --set model.variant=radar

# C：RGB 与 Radar 直接拼接
python tools/train.py --config configs/base.yaml \
  --set experiment.name=C_concat --set model.variant=concat

# D：Learned Query 交叉注意力控制组
python tools/train.py --config configs/base.yaml \
  --set experiment.name=D_learned_query --set model.variant=learned_query

# E：Radar Dynamic Query 核心模型
python tools/train.py --config configs/base.yaml \
  --set experiment.name=E_rdq --set model.variant=rdq

# E-no-skip：去掉分类头前的 Radar 直接连接
python tools/train.py --config configs/base.yaml \
  --set experiment.name=E_rdq_no_skip --set model.variant=rdq \
  --set model.radar_skip=false
```

正式结果需要分别使用 `experiment.seed=0`、`1`、`2` 重复实验并报告均值和标准差。
在检查完 RGB 基线和捷径诊断结果之前，不建议一次性启动全部实验。

### 低分辨率背景捷径诊断

下面的设置把每个相机缩小到 32×24，此时无人机目标几乎消失，但全局背景和 session
外观仍然存在：

```bash
python tools/train.py --config configs/base.yaml \
  --set experiment.name=rgb_lowres_shortcut \
  --set model.variant=rgb \
  --set data.image_size='[24,32]'
```

如果这个实验仍得到很高的分类分数，说明数据中存在明显的背景/session 捷径，五分类
结果的科学解释范围必须相应收窄。

## 评估与机制验证

`evaluate.py` 默认读取 checkpoint 内保存的原始训练配置：

```bash
python tools/evaluate.py outputs/RUN/best.pt --radar-mode normal
python tools/evaluate.py outputs/RUN/best.pt --radar-mode zero
python tools/evaluate.py outputs/RUN/best.pt --radar-mode shuffle_same_class
python tools/evaluate.py outputs/RUN/best.pt --radar-mode shift --radar-shift-seconds 1
python tools/evaluate.py outputs/RUN/best.pt --radar-mode shift --radar-shift-seconds 3
python tools/evaluate.py outputs/RUN/best.pt --radar-mode shift --radar-shift-seconds 5
```

评估、注意力可视化和断点续训会直接从 checkpoint 恢复完整 Backbone 权重，不会再次
访问 Hugging Face 下载初始化权重，因此可以离线运行。

评估生成的 JSON 包含整体 Accuracy、Macro-F1、混淆矩阵、各类别指标、距离分桶指标和
sequence-wise accuracy。同类别 Shuffle 使用半个序列的时间偏移，而不是使用相邻帧。

对于 Learned Query 或 RDQ checkpoint，可以导出交叉注意力热图：

```bash
python tools/visualize_attention.py outputs/RUN/best.pt \
  --split test --index 0 --output outputs/RUN/attention_000.png
```

V1 没有提供 2D 检测框，因此当前注意力定位结果主要用于定性分析。除非后续加入
几何投影或独立标注，否则不能将热图当作严格的定量定位指标。

## A～E 实验中必须固定的数据协议

- 以 GT 为时间同步锚点，`|dt_image|` 和 `|dt_radar|` 都必须不超过 40 ms。
- 必须按连续时间块划分数据，禁止随机按帧划分。
- 如果相邻时间块属于不同 split，在边界两侧删除 1 秒隔离带。
- Radar 均值和标准差只从训练集计算，验证集和测试集必须复用相同统计量。
- RGB 第一阶段只进行光度增强，禁止对 RGB 单独执行几何变换。
- 所有使用 Radar 的模型必须采用相同的距离过滤、点数上限和采样规则。
- D 和 E 除“固定 Learned Query”与“当前 Radar 生成 Query”之外，其余结构和训练设置
  必须保持一致。
