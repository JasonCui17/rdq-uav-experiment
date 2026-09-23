# P4 — LiDAR→Left Image Geometry Correctness Gate

Status: ready-for-agent
Blocked by: 04（P3 数据契约——audit 需要时间绑定与 calibration handle）
Type: task（audit 性质：结论由数据决定，不根据期望选择）

## Goal

为 GeometryLocal 主方法建立可验证的空间映射 contract：验证 LiDAR coordinate frame、LiDAR→camera transform、left fisheye 投影、时间对齐、有效投影区域、feature-map 坐标映射、3×3/K=9 邻域 coverage。输出明确的 PASS/FAIL。

**身份：prerequisite，不是算法选择器。**

## Non-goals

- 不实现 HCI。
- 不因结果调整任何算法决策（PASS → 走主方法；FAIL → 修标定）。
- 不复用旧 radar_enhance_pcl 的 Radar→Image 审计结论（传感器不同，V1.1 §2 已明确）。

## Files / modules expected to touch

- 新建 `tools/audit_lidar_v2_to_left_image.py`（或 `src/rdq_uav/multimodal_v1/` 下的 audit 模块 + CLI）
- 只读复用：`src/rdq_uav/calibration/omni.py`、`spatiotemporal.py`
- 参考（不复用结论）：`tools/radar_image_geometry_audit.py`、`docs/STAGE4_RADAR_IMAGE_GEOMETRY_AUDIT.md`

## Dependencies

- 04（P3）。
- 阻塞下游：**P4 PASS → 06（P5 GeometryLocal HCI）才能开工**；FAIL → 05 保持 ready，先修前置。

## Frozen constraints

- V1.1 §6/§21-25：correctness gate 是实现前置条件；**FAIL ≠ 换 LatentBridge**，只能修 calibration/data semantics。
- 协议：train 拟合/确定参数，validation 独立验证，test 禁止参与。
- 正式 audit 必须遵循两阶段流程门：先完成 audit 实现和 dry-run/test，再在配置或报告头中写入全部 PASS/FAIL 固定数值阈值并提交；只有阈值提交后才能首次正式运行 audit。正式运行后不得根据观察到的结果修改阈值。

## Implementation notes

- 必含 null control：real correspondence vs **same-sequence shuffled** correspondence。
- 指标必须按物理单位和语义拆开报告，不得合并成笼统的 `pixel projection error`：
  - **3D nearest distance**：单位为米（m）。
  - **Image reprojection distance** 以及 nearest-center / nearest-bbox distance：单位为像素（px）；实际使用 center、bbox 或两者均报告时，字段名必须明确区分。
  - **Coverage@8/16/32/64**：以 8/16/32/64 px 为像素半径的覆盖率。
  - **Feature-neighborhood coverage**：在 `/4`、`/8`、`/16` 三个 feature stride 下分别统计 3×3 feature neighborhood coverage，不与像素 Coverage@8/16/32/64 合并。
- real correspondence 与 same-sequence shuffle 的对照结果须按上述各适用指标分别报告。
- `/4`、`/8`、`/16` 的 3×3 feature-neighborhood coverage 直接用于判断 GeometryLocal 的 K=9 邻域是否物理可行。
- audit 实现完成后、首次正式运行前，必须在配置或报告头中写入每项 PASS/FAIL 的具体固定数值阈值。阈值定义必须随代码提交，并记录对应 commit；未冻结阈值时，工具不得给出正式 PASS/FAIL，也不得启动正式 audit。
- 正式 audit 一旦开始，禁止依据已观察结果调整阈值。若阈值契约本身确需修订，当前运行作废，并以新的预注册 commit 开启独立 audit，不得覆盖原结果。

## Acceptance criteria

- audit 工具可重复运行，输出结构化报告（JSON + markdown）。
- 配置或报告头包含完整的数值阈值、阈值冻结 commit，以及阈值冻结状态；未满足阈值冻结流程门时正式 audit 必须拒绝运行。
- 正式报告分别输出米制 3D nearest distance、像素制 image reprojection/nearest-center/nearest-bbox distance、像素 Coverage@8/16/32/64，以及 `/4`、`/8`、`/16` 的 3×3 feature-neighborhood coverage。
- 使用预先提交的固定阈值得出明确 PASS 或 FAIL；运行后阈值保持不变。

## Tests

- audit 判定本身作为被测契约：固定小输入 → 确定性结论（真实配对 vs shuffle 的方向性在合成数据上可控）。
- 投影函数单元测试：合成标定下已知 3D 点 → 已知像素。

## Artifacts / reports expected

- `results/lidar_v2_to_left_image_geometry_audit.md`（+ JSON）：全部指标、PASS/FAIL、判据原文。

## Stop condition

- FAIL：本 ticket 关闭并产出 **calibration/data-semantics prerequisite ticket**（修坐标系/外参/同步），主方法 ticket（06）保持阻塞。禁止把 FAIL 解释为切换 LatentBridge。

## Comments
