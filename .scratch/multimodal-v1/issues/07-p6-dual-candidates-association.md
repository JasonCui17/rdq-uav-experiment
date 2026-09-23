# P6 — Dual Candidates + Frozen Association

Status: ready-for-agent
Blocked by: 03（P2 DINO 候选源）、06（P5 HCI-enhanced 视觉特征）
Type: task

## Goal

装配双候选系统：Radar CandidateSet（复用 LiDAR V2 CandidateHead/Selector）+ RGB CandidateSet（来自 HCI-enhanced Swin→DINO 路径），经冻结 association 流程产生并保留 H^RV / H^R / H^V 三类 hypothesis。

## Non-goals

- 不实现 Reliability Gate / SharedQuery / Decoder（08/P7b）。
- 不做 learned matcher 或 association loss（冻结禁止）。
- 不把 YOLO 引入任何主路径 candidate。

## Files / modules expected to touch

- 新建 `src/rdq_uav/multimodal_v1/candidate/candidate_set.py`（统一 CandidateSet contract，若 P2 已建 contracts 则补全）
- 新建 `src/rdq_uav/multimodal_v1/candidate/association.py`
- 只读复用：`src/rdq_uav/lidar_v2/selector.py`（Radar 候选）
- 新建 `tests/test_multimodal_v1_p6_candidates.py`

## Dependencies

- 03（P2：DINO query/proposal 输出存在）
- 06（P5：HCI-enhanced 视觉特征——R→V 影响候选生成的前提）
- 阻塞下游：08（P7b 需要 hypothesis 输入）

## Frozen constraints

- V1.1 §8/§21-11/12：Radar 与 RGB 双 candidate pool；RGB 候选 feature 默认取 DINO object query/proposal feature 并投影 128D。
- V1.1 §21-13：Association = Geometry Hard Gate → cosine feature cost → Hungarian one-to-one。无 learned matcher、无 association loss、**无 score cost**（Q31 明确覆盖 Q24 的 score-consistency 方案）。
- V1.1 §21-14：fusion 前只有各模态独立 Top-K（初始 K≈50，占位值由 val 决定），不用高置信阈值提前删单模态候选；最终阈值只作用于 s_fusion。
  - **K 的落地方式（写死）**：多模态 P6 **复用 CandidateSelector 实现，但使用独立 multimodal selector config**；Radar CandidateSet 取其 nms 输出，`final_topk = K_R`。**不得修改 E0 baseline 的 selector config**（E0 冻结记录原样保留；multimodal config 是新文件）。RGB 侧同理：DINO 输出侧独立去重/NMS + Top-K（K_V），不与 Radar 共享参数。
- V1.1 §9：Radar/RGB 在 association 前各自先做对象级去重/NMS，使 Hungarian one-to-one 假设成立。
- 三类 hypothesis（H^RV/H^R/H^V）全部保留（§21-12）。

## Implementation notes

- CandidateSet contract 字段（spec）：score、feature[128]、xyz + xyz_valid、box + box_valid、batch_index、source ∈ {radar, rgb}。
- **Geometry hard gate 规则（写死，不给 implement 选择余地）**：

  d(Π(P_R), B_V) ≤ τ_geo

  其中 d 是**投影点到 2D box 的最短像素距离**（点在框内则为 0），Π 是 P4 audit 通过的投影，τ_geo 由 val 决定并记录。不使用 center distance / IoU / 其他替代规则。
- **Association feature 来源（写死，禁止临时换 pre-HCI feature）**：
  - Radar：P5 交互后经 LiDAR decode-to-fine 的 selected fine_features → 128D；
  - RGB：同一次 HCI-enhanced DINO forward 的 object query/proposal feature → 128D。
  - 两者即 C_feat 的输入。
- cosine feature cost：C_feat(i,j) = 1 − cosine(f_R_i, f_V_j)。
- Hungarian：标准 scipy 实现，逐 batch 独立求解。
- H^R 候选自然携带 xyz（box invalid）；H^V 候选携带 box（xyz invalid）——valid flag 不得伪造。

## Acceptance criteria

- 同一 batch 能构造出三类 hypothesis 且数量守恒（H^RV 配对数 + H^R 数 = Radar 候选数；H^RV + H^V = RGB 候选数）。
- Geometry hard gate 不可行 pair 绝不进入 Hungarian。
- Top-K/NMS 后候选数可控且可复现。
- H^R/H^V 不会被任何置信阈值提前删除（断言存在低置信单模态候选存活到最后）。

## Tests

- 合成候选集：已知配对关系 → Hungarian 正确恢复（matched-pair recall = 100%）——合成数据有 GT pair，此断言仅限合成场景。
- 类型分布：构造 Radar-only / RGB-only / 双模态混合场景，断言 H^RV/H^R/H^V 数量与预期一致。
- hard gate：不可行 pair（投影远离 box）被过滤的断言；**hard gate 规则断言**：点在框内 d=0、框外为最短像素距离（合成几何用例）。
- feature 来源断言：置乱 pre-HCI 特征不影响 cost（证明用的不是 pre-HCI feature）。
- 无 score cost：association cost 矩阵不依赖 score 字段（可用置乱 score 不改变匹配结果来断言）。
- CandidateSet 契约：字段完整、valid flag 与 source 一致。

## Artifacts / reports expected

- 真实 batch 上的 association 统计：**三类 hypothesis（H^RV/H^R/H^V）比例与匹配数量**（本 ticket 完成门仅此两项——GT pair 标签机制未建立前，matched-pair recall/precision/correct-match rate 不作为完成门；待后续有 GT pair 标签后再补）。

## Stop condition

- 发现 DINO 候选与 Radar 候选在真实数据上几乎无法 geometry-gate 配对 → 停止，开 prerequisite ticket 上报（可能是标定或候选质量问题），**不得**自行改成 feature-only association 主方法。
- 想加 learned matcher / association loss → §18/Q31 冻结禁止，直接拒绝。

## Comments
