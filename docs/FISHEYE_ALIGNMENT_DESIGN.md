# MMAUD V1 三维目标到双鱼眼图像的时空对齐设计

## 1. 目标

本模块不是用 Ground Truth 直接替代视觉检测，而是先建立可验证的几何标定，从而得到：

1. 每个图像时刻的 UAV 三维中心；
2. 三维中心在已完成外参标定的鱼眼图像中的像素位置；
3. 随目标距离、方向和物理尺寸变化的 Oracle ROI；
4. 投影误差、时间误差和不可见原因；
5. 可供 RGB、Radar 和 RDQ 模型共同使用的同一份 ROI Manifest。

生成的框必须标记为 **GT-projected / Oracle ROI**。它适合研究“已知大致位置后，Radar
Query 是否改善型号分类”，不能冒充无需定位器的端到端检测结果。

## 2. 已确认的官方信息

- 原图为两个同步相机横向拼接的 `2560×960` 图像；单相机分辨率为 `1280×960`，
  双相机基线约 `17.8 cm`。
- 两台相机均朝上，单相机视场约 `180°`。
- Radar 是 Oculii Eagle ETH04，官方论文给出的视场为水平 `120°`、垂直 `30°`。
- Leica Ground Truth 为 `5 Hz`，z 轴与重力方向相反，即 z 向上。
- ROS bag 中相机帧名为 `head_camera`，Radar 点云帧名为 `base_link`；bag 没有 `/tf`
  或 `/tf_static`，不能从消息中直接恢复外参。
- 官方论文说明：Radar 与音频相对顶部中央 Mid360 的外参来自 CAD；camera–LiDAR
  使用目标less标定；公开的 `fisheye_calibration.zip` 只提供左右相机各自的内参，未提供
  camera–LiDAR、camera–Radar 或 camera–GT 的数值外参。

对当前五类训练 Manifest 的数据检验显示：正确时间配对时，GT 周围 2 m 内出现 Radar
点的比例约为 11%～35%；把 Radar 循环错开半条序列后，五类该比例全部降为 0%。这强烈
支持官方文件中的 Radar XYZ 和 GT 已经处于同一发布坐标约定，但它仍属于数据推断，
不是官方外参声明。可用 `tools/calibration/audit_radar_gt_alignment.py` 重现并查看
`calibration/radar_gt_audit.json`。

官方相机模型是 Kalibr 的统一全向模型：

```text
camera_model: omni
distortion_model: radtan
intrinsics: [xi, fu, fv, pu, pv]
distortion_coeffs: [k1, k2, p1, p2]
```

配置已经原样写入 `configs/calibration/mmaud_v1_omni.yaml`，没有改成近似针孔模型。

## 3. 为什么不能忽略 z

设 GT 坐标中的目标为 `p_g(t)=[x,y,z]`，相机外参为：

```text
p_c = R_cg p_g + t_cg
```

这里的 z 同时影响：

- 目标相对相机的三维方向；
- 鱼眼球面投影的位置；
- 目标距离以及给定真实尺寸对应的像素尺寸；
- 目标是否进入某一台相机的有效成像圆。

只有在求平面方位角等非常有限的场景中才能暂时忽略 z，不能用于生成可靠的图像锚点。

## 4. 严格鱼眼投影

对于相机坐标点 `p_c=[X,Y,Z]`，令：

```text
d  = sqrt(X²+Y²+Z²)
xn = X / (Z + xi*d)
yn = Y / (Z + xi*d)
```

之后应用 Kalibr 文件中的 radtan 畸变，再乘 `fu/fv` 并加主点 `pu/pv`。实现位于
`src/rdq_uav/calibration/omni.py`。这与 OpenCV 的 `cv2.fisheye` 等距模型不同，二者
不能混用。

## 5. 时间对齐

不再使用“GT 找最近图像后直接套用原 GT 坐标”的近似。对于图像时间 `t_i`：

```text
p_i = interpolate_GT(t_i + delta_t)
```

- 使用 PCHIP 对 5 Hz GT 轨迹进行形状保持插值，避免普通高阶样条过冲；
- `delta_t` 是待估计的相机–GT固定时延；
- 优化范围默认 `[-0.25 s, +0.25 s]`；
- 最终必须报告该时延以及不补偿时的重投影误差对照。

如果不同 bag 的最优时延明显不同，应把时延改为 per-session 参数，但相机外参仍应共享；
不能让每个类别拥有完全独立的外参，否则外参本身会再次成为类别标签。

## 6. 外参和时延如何求

需要一组鱼眼图像中的 UAV 中心 `q_i=[u_i,v_i]`。优先级如下：

1. MMAUD 官网提供的官方 2D detection 数据；
2. 本地 YOLOv9 产生的候选框，经人工复核后取框中心；
3. 人工点击少量跨方向、跨距离和跨速度的关键帧。

拟合目标为：

```text
min Σ robust( project_omni(R p_gt(t_i+delta_t)+t) - q_i )
```

左右相机可分别估计 `R,t`，并共享一个 `delta_t`。使用 SciPy `least_squares` 的 `soft_l1`
损失降低错误检测和中心标注误差的影响。至少需要每台相机 6 个点，但正式标定建议每台
相机 50 个以上，并覆盖五类轨迹中的方位、距离和速度变化。

标注按 80% fit、20% validation 分开；validation 标注不参与拟合。验收建议：

- validation median reprojection error `< 10 px`；
- validation P95 `< 25 px`；
- 左右相机反推的中心间距接近官方 `0.178 m`；
- 五个 session 的误差分布相近；
- 时间偏移不应总是卡在搜索边界 `±0.25 s`。

这些是初始工程门限，不是官方宣称的精度；最终应根据人工中心标注噪声调整。

## 7. 框大小

单个三维中心只决定锚点，不决定框。项目采用“目标三维包围球”方法：

1. 为每个型号填入经过核实的物理半径；
2. 在 GT 中心周围采样三维球面点；
3. 逐点经过同一外参和 omni+radtan 模型投影；
4. 用投影点的包围矩形得到目标几何框；
5. 再乘固定 context scale，并设置最小/最大像素边长。

这样距离增大时框自然缩小，在鱼眼边缘还会自动体现非线性形变。未知无人机姿态会通过
上下文扩张和不确定性边界处理，而不是假装已知精确 3D bounding box。

## 8. 官方二维框的精确时间戳恢复

已下载的 `MMAUD_2D` 含 4,425 张单相机图和一一对应的 YOLO 框。官方 README 只公开了
`b1`～`b5` 的类别含义，没有公开编号到原始时间戳的映射。本地验证表明：

- 图片尺寸为 `1280×960`，每张只有一个 UAV 框；
- 4,425 张图全部与原始 `2560×960` 图像的左半幅逐像素相同；
- 精确像素哈希匹配率为 4,425/4,425，无多义匹配；
- `bN_index` 不是 GT 排序号，不能按编号直接配对。

使用下面的工具生成精确映射并继承本项目时间块 split：

```bash
python tools/calibration/import_official_2d.py
python tools/calibration/build_official_center_annotations.py
```

原始图哈希缓存在 `calibration/cache/raw_left_pixel_hashes.csv`，后续运行不再解码全部
33,200 张双鱼眼图。

## 9. 标定工具顺序

官方左相机框已经转换为训练观测；优先运行：

```bash
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

若需要补充右相机标注，再准备均匀覆盖五类轨迹的人工标注清单：

```bash
python tools/calibration/prepare_center_annotations.py --per-class 40
```

然后生成离线浏览器标注页：

如果需要人工点击，可以生成完全离线的浏览器标注页：

```bash
python tools/calibration/build_annotation_site.py
```

打开 `calibration/annotation_site/index.html`，在左右半图中点击 UAV；标注状态保存在浏览器
本地存储中，完成后点击“导出 CSV”，用导出的文件替换 `center_annotations.csv`。每个相机
使用自己的 `1280×960` 局部像素坐标，工具会自动完成左右半图坐标换算。

完成右相机点击后，再从 24 个轴对齐旋转中搜索其合理初值：

```bash
python tools/calibration/bootstrap_axis_aligned.py \
  --annotations calibration/center_annotations.csv
```

联合精调并计算留出标注误差：

```bash
python tools/calibration/fit_spatiotemporal.py \
  --annotations calibration/center_annotations.csv \
  --initial calibration/initial_extrinsics.json
```

在原始双鱼眼图像上绘制时间补偿后的投影中心与物理包围球：

```bash
python tools/calibration/project_overlay.py IMAGE.png \
  --sequence-id Mavic2 \
  --calibration calibration/fitted_calibration.json \
  --radius-m VERIFIED_RADIUS \
  --output calibration/overlays/example.png
```

在相同外参下投影整帧 Radar，并仅对 GT 邻域的运动目标点做帧间运动补偿：

```bash
python tools/calibration/project_radar_overlay.py IMAGE.png RADAR.npy \
  --sequence-id Mavic2 \
  --calibration calibration/fitted_calibration.json \
  --output calibration/overlays/radar_example.png
```

## 10. 当前结果与剩余边界

左相机已经使用官方二维框完成时空标定：

- 时间偏移：`-0.1241537 s`，约定为 `GT查询时刻 = image_time + offset`；
- 训练拟合中位/P95：`1.89 / 7.26 px`；
- 训练内部验证中位/P95：`1.85 / 7.84 px`；
- 分类 test 时间块中位/P95：`2.46 / 8.52 px`；
- test 中 `97.2%` 在 10 px 内，`90.2%` 的投影中心落入官方框。

所以左相机 GT 中心投影已经通过定量和叠加图验证，可以用于生成明确标记为 Oracle 的
左图 ROI。右相机仍是阻塞项：官方二维数据仅覆盖左半幅，公开 calibration zip 未提供
stereo 外参。必须补充右图中心观测或取得官方 stereo 外参后才能严谨投影右图。

另外，二维中心只能产生锚点；按真实大小生成 Oracle 框还需要核实五个 UAV 的物理尺寸，
不能用类别相关的任意固定像素框。

## 11. 依据

- MMAUD 论文：https://arxiv.org/abs/2402.03706
- MMAUD 官方仓库：https://github.com/ntu-aris/MMAUD
- 官方 2D detection 数据入口：https://drive.google.com/drive/folders/1_LpPyIfETQS-k2vlSsbzI9pzyVzZScSx
- Kalibr 支持的相机模型：https://github.com/ethz-asl/kalibr/wiki/supported-models
- Mei–Rives omni 模型：C. Mei and P. Rives, ICRA 2007
- 3D Radar–Camera 时空标定：https://arxiv.org/abs/2211.01871
