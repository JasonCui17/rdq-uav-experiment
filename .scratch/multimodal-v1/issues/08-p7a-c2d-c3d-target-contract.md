# P7a — Freeze c2D/c3D Target Contract

Status: ready-for-agent
Blocked by: 07（P6——hypothesis 结构定型后才能定 target 语义）
Type: task（语义/契约设计，**不写模型代码**）

## Goal

在实现任何 `L_valid`（c_2D/c_3D supervision）之前，冻结其 target contract：明确三种语义——

1. **GT annotation availability**（这个 query 有没有 2D/3D GT）
2. **modality evidence/support**（该 hypothesis 有没有对应模态的证据，即 m_R/m_V）
3. **predicted output reliability**（模型输出的 box/XYZ 是否真的可靠）

分别是什么变量、取值空间、如何构造监督标签、损失如何按语义分 mask。

## Non-goals

* 不实现 Reliability Gate / SharedQuery / Decoder / loss 代码（09/P7b）。
* 不新增模型模块、不新增损失类型（只定义既有 L_valid 的 target 语义）。
* 不改变冻结输出接口（c_2D/c_3D 本来就在 §21-19 里）。

## Files / modules expected to touch

* `.scratch/multimodal-v1/issues/08-p7a-c2d-c3d-target-contract.md` 本身（契约写在这里，作为 P7b 的引用源）
* 可选：`docs/agents/` 或 spec 的 Further Notes 追加一段（若内容稳定）

## Dependencies

* 07（P6：hypothesis 类型与 valid flag 结构是 target 语义的输入）。
* **阻塞 09（P7b）：L_valid 的实现以此 contract 为准。**

## Frozen constraints

* V1.1 §11：输出含 c_2D/c_3D validity/reliability confidence；三语义不得混为一个 presence flag。
* V1.1 §21-19/20：输出接口与 residual 规则不变——本 ticket 是**实现语义澄清，不改变冻结输出接口**。
* L_valid 小权重（Q36）不变，但“小权重 BCE”的具体 target 构造必须由本 contract 先定义。

## Implementation notes

契约至少要回答：

| 语义                 | 变量                        | 取值          | 由谁决定                     | 损失角色                                     |
| ------------------ | ------------------------- | ----------- | ------------------------ | ---------------------------------------- |
| GT availability    | gt_2d_valid / gt_3d_valid | 0/1         | 数据集标注                    | 决定对应 validity loss 是否存在                  |
| modality support   | m_R / m_V                 | 0/1         | hypothesis provenance    | Gate / SharedQuery 输入，不是 validity target |
| output reliability | target_c2D / target_c3D   | 本 ticket 定义 | prediction + assigned GT | L_valid 的监督目标                            |

### 1. 先冻结 hypothesis → GT assignment

在构造 `target_c2D / target_c3D` 之前，必须先定义每个 hypothesis 使用哪个 GT 作为质量参照。

* 若当前 MMAUD 训练协议中每个 query 只有唯一 UAV GT，则所有有效 hypothesis 直接使用该 query 对应的有效 2D/3D GT。
* 若数据协议允许同一 query 存在多个目标，则本 ticket 必须先定义 deterministic hypothesis→GT matching 规则，再计算 IoU / 3D error。
* GT assignment 未定义前，不得实现 reliability target。

### 2. GT availability 与 modality support 必须解耦

`m_R / m_V` 只描述当前 hypothesis 的证据来源：

* H^RV：m_R=1, m_V=1
* H^R：m_R=1, m_V=0
* H^V：m_R=0, m_V=1

它们**不是 validity target，也不得决定 validity loss 是否存在**。

对应监督 mask 固定为：

```text
L_c2D mask = gt_2d_valid
L_c3D mask = gt_3d_valid
```

因此：

* H^V 即使没有 Radar evidence，只要 `gt_3d_valid=1`，其预测 XYZ / c_3D 仍应接受监督；
* H^R 即使没有 RGB evidence，只要 `gt_2d_valid=1`，其预测 box / c_2D 仍应接受监督；
* 无对应 GT 时，该 validity loss 不计算，而不是把 target 强行设为 0。

### 3. output reliability target 构造

`target_c2D / target_c3D` 必须描述“当前预测是否可靠”，不得直接使用 annotation presence。

候选方案由本 ticket 最终裁决并写死，例如：

* binary：

  * 2D：assigned GT 存在且预测 IoU 达到固定阈值 → `target_c2D=1`，否则 0；
  * 3D：assigned GT 存在且 XYZ error 小于固定阈值 → `target_c3D=1`，否则 0；
* 或 graded：

  * 根据 IoU / 3D localization error 构造连续 reliability target。

无论选择哪一种，必须明确：

* target 定义；
* threshold / mapping；
* sample mask；
* 对 H^RV / H^R / H^V 三类 hypothesis 的统一处理规则。

**禁止：**

```text
GT exists = 1
```

直接等价为：

```text
prediction reliable = 1
```

若 reliability target 由当前模型预测的 box/XYZ 质量计算，target 构造过程应作为监督标签使用，不允许通过 target 本身形成额外梯度路径。

## Acceptance criteria

* 契约明确覆盖：

  * H^RV
  * H^R
  * H^V
    与 2D/3D GT availability 的全部组合。
* hypothesis→GT assignment 规则明确，无歧义。
* `L_c2D` / `L_c3D` 的 sample mask 明确，并与 `m_R/m_V` 解耦。
* `target_c2D / target_c3D` 的构造规则、阈值或映射方式明确。
* 用户（研究者）批准本契约后 ticket 才能关闭；**P7b 只实现被批准的版本**。

## Tests

本 ticket 无代码测试。

P7b 后续测试必须逐条断言本 contract，例如：

* 无 2D GT → 不产生 c_2D loss，而不是 target_c2D=0；
* H^V + 有 3D GT → 仍产生 c_3D supervision；
* H^R + 有 2D GT → 仍产生 c_2D supervision；
* modality support mask 不被当成 reliability target。

## Artifacts / reports expected

* 本 ticket 正文即最终 contract：

  * 三语义表格；
  * hypothesis→GT assignment；
  * hypothesis × GT availability 规则；
  * c2D/c3D target 构造规则；
  * validity loss mask 规则。

## Stop condition

* 若定义 target 语义时发现必须改变输出接口（例如新增输出头）→ 触碰 §21-19 冻结项，停止并上报记录冲突，不得自行改 spec。
