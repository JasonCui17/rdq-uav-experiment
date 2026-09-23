# Spec: Multimodal V1 — Hierarchical Geometry-aware Multimodal UAV Detection & Localization

Status: ready-for-agent
Source of truth: 冻结版实验方案 V1.1 修正版（`小目标雷达多模态_冻结版实验方案_V1.1_修正版.docx`，2026-09-21）。**V1.1 §21 冻结决策清单（Q1–Q47 决策账本）为不可覆盖约束。** 旧 V1 文档已 superseded。
Branch: `lidar-uav-v2`

## 变更纪律（V1.1 §0，约束本 spec 的一切后续条目）

- GrillMe Freeze > 后续工程推测。任何实现期发现（标定问题、显存约束、数据接口、checkpoint 管理问题）**只能**记录为 prerequisite / engineering risk / fallback baseline，不得修改冻结算法。
- 若主方法因数据前置条件暂时无法运行，修正前置条件；不得用 fallback 悄悄替换 frozen method。

## Problem Statement

LiDAR V2 单模态分支已审计但从未正式训练冻结（无 formal checkpoint），多模态 V1.1 设计已冻结但零代码实现。研究者需要在 MMAUD 上实现"Geometry-guided 层级双向 HCI + DINO-Swin-T 共享视觉链路 + 双候选共享假设解码"的主模型，跑完 E0–E5 取得 strongest model，再补严谨消融，形成可写论文的证据链——全程不破坏已审计的 LiDAR V2 基线，不因工程困难漂移冻结架构。

## Solution

实现 `multimodal_v1` 包（YAML + Registry + 独立积木类）：

1. **LiDARV2PyramidAdapter** 以 Stage Adapter 暴露 R0/R1/R2（LiDAR V2 数学零改动，只读复用）。
2. **共享视觉链路 DINO-Swin-T**：Swin-T backbone 同时服务 HCI 与 DINO 检测头，R→V 交互必须能影响 RGB candidate generation；禁止第二套在线视觉 backbone。
3. **Geometry Bi-HCI（主方法）**：每级 stage 前（Pre-Stage），Radar token XYZ 经已验证投影 Π 到图像取局部邻域（3×3, K=9），双向但物理不对称的 cross-attention（V→R 读局部视觉；R→V 只更新 Radar-supported 的稀疏视觉位置），token-wise near-zero gate，残差回原模态。
4. **双候选**：Radar（复用 CandidateHead/Selector）+ RGB（DINO query/proposal feature → 128D）→ 各自 Top-K/NMS → geometry hard gate + cosine feature cost + Hungarian → H^RV/H^R/H^V 三类全保留。
5. **Hypothesis Reliability Gate → Typed Shared Query（128D）→ 标准 2-layer Transformer Decoder**（memory 只读当前 alignment 最终交互层），输出 2D box + 3D XYZ + fused score + c_2D + c_3D，有先验做 residual。
6. 训练 T0 correctness → T1 fusion warm-up → T2 fusion training → T3 final fine-tune（解冻最后 2 个 Swin stages，RGB 低 LR）；实验 E0–E5。

## User Stories

### Baseline 与等价性（P0–P2）

1. As a 研究者，I want 先正式训练并冻结 E0 LiDAR V2 baseline（checkpoint/config/metrics/commit 全记录），so that 多模态实验有不可悄悄改变的 3D 单模态对照。
2. As a 研究者，I want LiDARV2PyramidAdapter 在同一 batch/state_dict/FP32/eval 下与原 LiDARUAVDetector 数值等价（logits、residual_xyz、pred_xyz、fine_features、voxel_centers 一致），so that Stage 拆分被证明未触碰已审计数学。
3. As a 研究者，I want SwinPyramidAdapter + DINO 共享 backbone adapter 在 identity 路径下保持原 Swin/DINO 行为与 shape，so that 多模态代码链路先被验证不破坏单模态视觉能力。
4. As a 研究者，I want Identity 模块作为正式注册组件，so that E2 late-fusion 对照零特殊代码。
5. As a 研究者，I want 多模态实验不得通过"顺手优化 LiDAR baseline"获得隐性收益，so that 对照的公平性不受侵蚀。

### 几何前置条件（P4，prerequisite 而非算法选择器）

6. As a 研究者，I want LiDAR→Left Camera geometry correctness gate 作为实现前置条件（坐标系、外参、鱼眼投影、image_time/query_time 对齐、投影有效区域、局部窗口 coverage），so that 主方法的投影取邻域建立在可验证 contract 上。
7. As a 研究者，I want audit 用 train 拟合、validation 独立验证、test 不参与，且报告真实配对 vs same-sequence shuffle 的投影误差与 Coverage@8/16/32/64，so that 几何结论可信且无泄漏。
8. As a 研究者，I want gate FAIL 时流程是"继续修 calibration/data semantics"而不是"自动降级 LatentBridge"，so that 冻结算法不被工程困难无痕替换。

### HCI 主方法（P5）

9. As a 研究者，I want Geometry Bi-HCI 在每级 stage 前执行：Projection → Geometry Sampling → 局部 CA → Gate → Residual（不含额外 FFN/self-attention），so that 稀疏 3D 空间先验直接引导跨模态特征交互且停止发散边界被执行。
10. As a 研究者，I want 双向从同一组原始 R/V 并行计算（无顺序偏置），但物理不对称（V→R 局部读取、R→V 稀疏更新），so that 交互结构与两模态密度差异一致。
11. As a 研究者，I want interaction space 统一 128D、视觉保留 native channels 仅在 HCI 内投影/反投影、token-wise near-zero gate 残差，so that 预训练特征初始不受破坏。

### 视觉候选（R→V→candidate 链路）

12. As a 研究者，I want RGB candidate 由 HCI-enhanced Swin/DINO 路径产生（RGB → Swin stages → HCI 增强 → DINO queries/boxes → CandidateSet），so that Radar→Vision 交互真正帮助视觉发现 tiny UAV。
13. As a 研究者，I want YOLO11 只作为 RGB-only baseline / 工程对照，so that 外部 detector 不混入主方法。
14. As a 研究者，I want L_V 复用 DINO 成熟 detection supervision 而不另造视觉损失，so that 视觉分支训练稳定。

### 双候选与 Association（P6）

15. As a 研究者，I want Radar 与 RGB 各自独立 Top-K（初始 K≈50，由 val 定）+ 对象级去重/NMS，so that Hungarian one-to-one 假设成立。
16. As a 研究者，I want association 为 geometry hard gate → cosine feature cost → Hungarian（无 learned matcher、无 association loss、无 score cost），so that 匹配可解释且冻结于 Q31。
17. As a 研究者，I want H^RV/H^R/H^V 三类全部保留、不用高置信阈值提前删单模态候选（最终阈值只作用于 s_fusion），so that Radar miss 时 RGB 仍可独立恢复、反之亦然。

### 融合解码（P7）

18. As a 研究者，I want Hypothesis Reliability Gate 以 softmax(MLP) 学习 [w_R, w_V, w_J]（注意与 HCI 内 Interaction Residual Gate 是两个不同组件），so that 高高/高低/低低由学习决定而非 if/else。
19. As a 研究者，I want Typed Shared Query 用 missing/type embedding 显式编码证据来源，so that decoder 不需要知道候选来自哪条路径也能正确利用来源信息。
20. As a 研究者，I want 有先验做 residual refinement（H^RV: B+ΔB/P+ΔP；H^R: 以 Π(P_R) 为参考预测 box；H^V: 从 query+memory 补 XYZ），so that 不从零重猜已有证据。
21. As a 研究者，I want c_2D/c_3D 的监督标签明确区分"GT 是否存在 / 模态是否支持 / 输出是否可靠"三个语义，so that validity supervision 不被混为一个 presence flag。

### 数据与训练

18. As a 研究者，I want Hypothesis Reliability Gate 以 softmax(MLP) 学习 [w_R, w_V, w_J]（注意与 HCI 内 Interaction Residual Gate 是两个不同组件），so that 高高/高低/低低由学习决定而非 if/else。
18a. As a 研究者，I want 缺失模态时 gate 的行为按 contract 写死（presence mask 进输入、缺失 feature 用显式 missing embedding、缺失侧权重 mask 为 0），so that 实现者不会自填 0/假 score/复制 feature 来掩盖缺失。
23. As a 研究者，I want left RGB 时间绑定显式记录 image_time/query_time/gap、无跨 sequence 泄漏，so that 时间语义可信。
24. As a 研究者，I want modality dropout（p≈0.1 可调超参，禁双 drop）防止 Reliability Gate 塌缩到单一模态，so that 单模态失效鲁棒性可被评估。
25. As a 研究者，I want T0–T3 分阶段训练按 Q38 冻结策略执行，so that 新模块先学会不破坏预训练特征再冲上限。

### 实验与评估

26. As a 研究者，I want E0–E5 按冻结矩阵运行（E0 LiDAR-only / E1 RGB-only / E2 late fusion / E3 single-level HCI / E4 3-stage HCI / E5 Full V1），so that 每级回答一个因果问题。
27. As a 研究者，I want E3/E4 的单向方向在实现前固定一个参考方向并保持一致（双向与另一方向留作后续消融），so that E3→E4→E5 差异可归因。
28. As a 研究者，I want 评估覆盖 Radar candidate / RGB candidate / association / final 2D / final 3D / joint / 效率七层指标，so that 论文表格完整。
29. As a 研究者，I want 后续消融（alignment/direction/stage count/gate/router/candidate source/hypothesis type/modality dropout）在 strongest model 之后补齐，so that 第一优先级是跑通主结果。

## Implementation Decisions

- **代码边界**：新包 `src/rdq_uav/multimodal_v1/`（registry/contracts/model/data/loss + radar/vision/interaction/candidate/decoder 子包；`geometry_local.py` 为主方法，`latent_bridge.py` 为 baseline/fallback）；旧 `lidar_v2/` 只读复用，多模态逻辑不得反向污染。
- **YAML 默认语义**即 V1.1 §16.1 冻结配置：GeometryBiHCI（pre, dim=128, local_window=3）、DINOWithSwinPyramidAdapter（backbone: swin_tiny）、DINOQueryCandidateAdapter、GeometryGateFeatureHungarian、CandidateReliabilityMLP、TypedSharedQuery(128)、StandardTransformerDecoder(128, 2层)。
- **LatentBridge 身份**：仅 baseline/fallback/debug，用于标定未就绪时验证多模态代码链路；不替换主方法。
- **Reliability Gate（含缺失模态接口澄清，不改架构）**：z=[s_R, s_V, f_R, f_V, association_info, presence/type]；[w_R, w_V, w_J]=softmax(MLP(z))；s_fusion=w_R·s_R+w_V·s_V+w_J·s_joint。缺失模态行为必须写死：
  - presence mask m_R, m_V ∈ {0,1} 进入 gate 输入；
  - 缺失 feature 使用显式 missing embedding（或 zero+mask），**禁止**自填 0 / 假 score / 复制对侧 feature；
  - 缺失侧 expert 权重必须 mask：m_V=0 ⇒ w_V=0，m_R=0 ⇒ w_R=0（w_J 仍可存在）。
- **Shared Query**：H^RV: MLP([f_R; f_V; association_info])+e_RV；H^R: MLP([f_R; e_missing_V])+e_R；H^V: MLP([e_missing_R; f_V])+e_V；统一 128D。
- **Loss**：L_total=λ_R·L_R+λ_V·L_V+λ_F·L_F；L_F=λ_cls·L_cls+λ_2D·L_box+λ_3D·L_xyz+λ_valid·L_valid；L_cls Focal、L_box L1+GIoU、L_xyz SmoothL1；第一版无 contrastive/consistency。
- **L_valid 的实现门**：损失形式暂保留，但 **P7 实现前必须先冻结 c_2D/c_3D 的 target contract**（明确区分"GT 是否存在 / 模态是否支持 / 输出是否可靠"三语义的标签构造规则）；target contract 未定义前不得实现 generic BCE——防止代码把"有 GT=1"偷偷当作"预测可靠=1"。
- **alignment**：默认 same-level（R0↔V0, R1↔V1, R2↔V2）；shifted/full-hierarchy 为后续结构消融；R3 沿现有规则最小延伸（SparseMerge23 + Stage3），仅 full_hierarchy 配置生效，且需同时报告 Params/FLOPs。
- **后处理**：最终重复候选只用简单 2D NMS + 3D radius suppression（Q45），不发明 learned duplicate remover。
- **实现顺序**：严格按 V1.1 §17 P0→P8；P0（E0 baseline）与 P1（adapter 等价性）可并行（服务器训练 / 开发侧实现）。
- **测试 seam（两层，严格区分）**：
  - **主 model seam**：完整 `HierarchicalMultimodalUAV` 的输入输出 contract——shape/语义、finite、gradient（synthetic backward）、candidate provenance（RGB 候选必须来自 DINO-Swin 链路）。**不要求** HCI=Identity 时与 LiDAR-only 数值等价——完整模型仍有双候选、SharedQuery、Decoder，等价性不成立。
  - **numerical identity seam（仅限 adapter 层）**：严格数值等价只属于 P1（LiDARV2PyramidAdapter ≡ LiDARUAVDetector，五项输出逐项一致）和 P2（Swin/DINO adapter 在 identity 路径下保持原行为与 shape）。

## Testing Decisions

- 好测试只测外部行为。**两层 seam 严格区分**：完整模型的 contract 测试（shape/finite/gradient/candidate provenance）；adapter 层的严格数值等价（P1 五项输出逐项一致、P2 Swin/DINO 行为与 shape）——完整模型不做数值等价断言。其次 CandidateSet 构造（H^RV/H^R/H^V 可正确构造并保留）、association 输出类型分布、UQP 唯一性、时间 gap 无泄漏、geometry gate 确定性 pass/fail。
- R→V→candidate 链路关键断言：identity 路径下 RGB candidate 也必须来自（未增强的）DINO-Swin 链路而非外部 detector；candidate source 在 config 中必须显式，禁止训练时静默切换到 YOLO。
- 不测内部 attention 权重等实现细节。
- Prior art：`tests/test_lidar_uav_v2_*` 系列的 correctness-gate 测试范式直接沿用。
- T0 通过条件：synthetic backward、真实 batch finite、tiny subset 可 overfit。

## Out of Scope

- 右鱼眼、音频及其他新模态；deformable cross-attention；contrastive/consistency loss；learned association Transformer；新 decoder 范式；重新发明 RGB detector；更复杂多级 reliability gate；为创新点数量堆注意力模块（V1.1 §18 全部禁止，E0–E5 跑完前一律 backlog）。
- LatentBridge / YOLO / late fusion / feature-only association 的主方法化（永远只属于 baseline/fallback/ablation）。
- 修改 V1.1 §21 任何冻结项——发现工程问题的唯一出口是 prerequisite/risk/fallback ticket + 重新决策流程。

## Further Notes

- **变更纪律是本 spec 的元规则**：工程问题（显存、标定、数据接口、checkpoint 管理）→ 只生成 prerequisite/risk/fallback ticket；不得触碰 frozen architecture。
- E0 无 formal checkpoint 是已确认事实，P0 不可跳过；P0 与 P1 可并行推进。
- 资源约束：WSL 12GB RAM / 3070 8GB——UQP、gradient accumulation、AMP 是 V1.1 §20 预案内的应对，属工程手段而非算法变更。
- 立即行动项（V1.1 §23）：P0 E0 正式训练 ‖ P1 adapter identity equivalence → P2 共享 Swin/DINO adapter → P3 数据/时间绑定 → P4 geometry gate → P5 GeometryLocal HCI → …

## Comments

- 2026-09-21: spec 按用户三条修正更新（seam 两层区分、Reliability Gate 缺失模态行为、L_valid 实现门）。
- 2026-09-21: 拆分为 tickets（`issues/01–13`，12 号已移除），依赖图与首批并行建议见 `issues/README.md`。各 ticket 经用户逐个审查修订（P1 hierarchy 贯穿、P2 reference 固定与真 stage boundary、P3 时间公式与复合键、P4 两阶段阈值冻结、P5 缺失模态 residual 归零、P6 hard gate 公式/selector config/feature 来源、P7a 契约细化、P7b s_joint 与 decoder memory mask、F1 debug 规则、A1 anchor=E5）。
