# seq0065 Root Cause Analysis

所有新case-study CSV、JSON和PNG位于
`outputs/mmuav_paper_reproduction/posthoc_robustness/seq0065/`。
V1评价不调用AR、interpolation或spline，不进行参数拟合；使用保存的measurement-supported Kalman状态，以冻结50ms最近时间匹配评分。Original与V1的覆盖集合不同，整体误差不是完全paired精度对照；V1 median略升（0.4584→0.4713m），保留此结果。
关联hypothesis与detection ID没有保存，因此无法确认旧track失去measurement的具体gate或抢占机制；“TRACK_ASSOCIATION_FAILURE”表示已观察到旧track丢失更新，不声称已证明某个Kalman参数有bug。

## 1. Symptom

旧selected轨迹在转向附近与候选云分叉，Y预测持续离开目标。

## 2. First divergence time

1706258442.446868，相对首GT 9.609s；规则为连续3个GT评价点error>2m，仅用于post-hoc diagnosis。

## 3. Candidate evidence

完整候选和逐timestamp最近GT误差见candidate_timestamp_audit.csv；不以GT删除任何候选。

## 4. Center-regression evidence

CENTER REGRESSION NOT PRIMARY FAILURE；存在近GT的corrected候选和measurement-supported新track，而旧轨迹prediction-only发散。

## 5. Tracker evidence

旧track最后measurement=1706258442.295600；prediction-only tail=5.501s；divergence之后54个状态全部prediction-only。track_candidates()的CV预测无法追随该转向，而旧track仍存活。

## 6. Track-selection evidence

select_track()以raw lifespan优先，选中了旧track。符合后续正确track证据的IDs：['97178949-5b55-47d6-bd9e-275abea23d93', '6874a06b-eabe-432f-8609-f430b89ccf6a']。GT支持“正确”的判断仅用于诊断；v1不读取这些oracle IDs。

## 7. Temporal evidence

TEMPORAL MODULE INHERITS TRACKING FAILURE；旧FULL已严重发散，ar_complete/resample沿用其selected输入。未调整AR或spline，也没有重新拟合。

## 8. Root cause

MULTIPLE。PRIMARY FAILURE = TRACK_SELECTION_FAILURE；secondary = TRACK_ASSOCIATION_FAILURE（旧track失去measurement）。Confidence = HIGH。新track仍跟随候选，所以不是整个tracker完全不能重建后续目标。

## 9. Minimal robustness fix

独立track_robustness.py/select_track_v2：measurement update count优先，supported duration其次，prediction-only tail惩罚；仅measurement-supported状态；按未来端点速度预测连续性拼接，1s/3m沿用现有尺度。RECONSTRUCTED_ROBUSTNESS_DESIGN；不是论文算法。现有模型、Kalman、select_track、AR、spline都未修改。[{"model": "ORIGINAL", "coverage": 0.7125, "matched": 285, "MSE_coord": 26.753630599605142, "mean_3d_error": 5.1485065630101285, "median_3d_error": 0.45840610483979694}, {"model": "ROBUSTNESS_V1", "coverage": 0.965, "matched": 386, "MSE_coord": 0.11089679087962609, "mean_3d_error": 0.5090069508067263, "median_3d_error": 0.4713113255088719}]

## 10. Scientific status

This is POST-HOC analysis on previously evaluated heldout data. Original frozen results remain the official reconstruction result. V1仅seq0065一次离线case study，没有正式heldout重评分；NaN缺测保留。
