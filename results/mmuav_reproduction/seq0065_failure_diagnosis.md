# seq0065 finite failure diagnosis

只读 saved candidate / tracker / selected / completed / final CSV 和 GT；无模型推理、拟合或重跑。
诊断阈值2m仅用于标记异常，不参与任何算法选择。

- 首次 FULL error >2m：1706258442.4468682。
- 峰值时间：1706258447.837367；FULL error=24.033m。
- 包含峰值的连续>2m区间：1706258442.446868—1706258447.837367（缺测会切断区间）。
- 区间内 candidate timestamps=54；selected states=54；
  prediction-only states=54。
- 最近原始候选到GT的误差中位数：0.9159190157486369m；
  最近corrected candidate误差中位数：0.475373411120752m。
- 最大M2 correction norm：0.6957023597917887m。
- FULL / temporal最大误差：24.033 / 23.980m。

GT用于事后比较所有候选，不参与生成或选择。最接近GT的候选只是oracle诊断，
不能证明tracker实际关联了哪个点；原记录没有保存hypothesis/detection IDs。
Raw tracker与selected的对应可由track_id核对；selected仅取固定最长存活track，
不把多个track连成一条，也不因结果差重选。Temporal是否放大只能比较已有FULL与final，
不从整体平均误差强行推断因果。见seq0065_stage_evidence.csv和fig4全时轴。

Most likely failure stage = B. tracker / trajectory selection 与可用目标候选不一致

Confidence = medium
