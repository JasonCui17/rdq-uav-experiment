# MMUAV 论文—公开代码完整方法审计

## 1. 审计范围与证据等级

- 论文：*Multi-Modal UAV Detection, Classification and Tracking Algorithm—Technical Report for CVPR 2024 UG2 Challenge*，arXiv:2405.16464，重点为 §3.2、Fig. 2、Fig. 3、§4.2。[arXiv 原文](https://arxiv.org/abs/2405.16464)
- 本地论文文件：`/home/jasoncui/projects/open_source/1.Multi-Modal UAV Detection, Classification and Tracking Algorithm—Technical Report for CVPR 2024 UG2 Challenge.pdf`，SHA256 `f09c36ea3684b07c92c56d5338138bf3b6124d6c09f9039517c33c05d2b57003`。
- 公开源码：`/home/jasoncui/projects/open_source/Multi-Modal-UAV`，审计 commit `f11b57390effbe9623ee2c7d561afddc8d0cdfa7`。
- 本文严格区分三类事实：论文明确陈述、公开代码实际行为、由两者都不能确定的事项。没有用 README 替代论文方法描述。
- 当前源码工作区是 dirty。算法文件中可见的本地代码差异仅为 `fusion_tracking.py` 全序列 CLI 分支变量名由不存在的 `args.result_folder_path` 改为 `args.result_folder`；不改变跟踪算法。若干示例 NPY 也有工作区变化，但不影响以下静态源码结论。

## 2. 总体结论

论文描述的 3D pose pipeline 是：

```text
20帧点云 → 无监督聚类 → 7D时序特征与增强
→ Attention LSTM + MLP二分类
→ PointNet-like cluster center regression
→ 24D三阶多项式 + linear regression bias correction
→ Kalman multi-object tracking
→ 三阶AR轨迹补全 → 插值与B-spline平滑 → Pose MSE
```

公开代码真正闭合且可运行的简化链是：

```text
Mid360 20帧 → DBSCAN → 9D特征
→ 普通LSTM最后时刻hidden state + Linear二分类
→ 保留cluster原始点
→ 与Livox FPS点做全序列DBSCAN fusion
→ 每帧DBSCAN算术中心
→ StoneSoup最近邻/EKF跟踪
→ 线性插值和B-spline画图
```

两条链之间的主要缺口不是参数小差异，而是 center regression、论文版 bias correction、AR completion 和最终评价/提交链缺失；Attention LSTM及特征维度也不一致。因此公开代码只能复现简化 pipeline，不能严格复现论文完整动态点分析或最终 3D 精度。

## 3. 主对照表

| 模块 | 论文方法 | 论文关键参数 | 公开代码位置 | 代码实际行为 | 一致性 | 当前可运行 | 复现缺口 | 严重程度 |
|---|---|---|---|---|---|---|---|---|
| A. 原始点云输入 | Pose 主要使用多类 LiDAR 与 radar；Fig. 6 讨论 conic LiDAR、peripheral LiDAR、radar互补 | 未给统一 tensor/schema | `tracker/preprocess.py:34-49,52-128,131-184` | Mid360 与 Livox 两支可执行；主跟踪代码未使用 radar | PARTIAL | 是 | 论文传感器使用范围与发布主链不完全一致 | MEDIUM |
| B. 20帧时间窗口 | 累积20帧形成 temporal window | 20帧 | `process_lidar_360():65-97`；训练 builder `:31-44` | 非滑窗20帧块；推理另加 sequence 最后20帧窗口 | MATCH | 是 | 末尾额外窗口行为论文未说明 | LOW |
| C. 无监督聚类 | 对累积点云做 unsupervised clustering | 未给 DBSCAN 数值 | 训练 `build_lidar_360_dataset.py:46-55`；推理 `preprocess.py:106-117` | 均为 DBSCAN，但训练 `eps=1,min_samples=10`，推理 `eps=2,min_samples=10` | PARTIAL | 是 | train/inference clustering domain mismatch | HIGH |
| D. Cluster feature | 论文称每时刻7维 feature，但未逐维定义 | 7D；具体组成未说明 | `tracker/extract_feature.py:64-96` | `[mean_xyz,std_xyz,range_xyz]`，每时刻9D，20帧输入为 `[M,20,9]` | MISMATCH | 是 | 无法由论文确定7D准确组成 | HIGH |
| E. 时序数据增强 | temporal dropout、temporal reverse、spatial/global rotation | 未给角度或概率 | `train_lidara_detector.py:27-70,119-125` | reverse；随机置零1–3帧；三种选择概率0.5/0.25/0.25；未找到 rotation | PARTIAL | 训练脚本可运行（需特征文件） | spatial rotation缺失；论文未给增强参数 | MEDIUM |
| F. Attention LSTM | 对全部 hidden states 做 attention 加权融合 | 未给层数/hidden/attention公式 | `tracker/lidar_360_detector.py:4-27`；训练副本 `train_lidara_detector.py:82-104` | 单层普通 LSTM，直接使用 `out[:,-1,:]`；无 attention | MISMATCH | 是 | Attention模块、公式和权重缺失 | HIGH |
| G. MLP cluster classifier | Attention LSTM后接MLP完成cluster二分类 | 未给MLP层数/宽度/threshold | 同上 | LSTM hidden=64 后仅 `Linear(64,2)`；CrossEntropy训练；argmax推理 | PARTIAL | 是 | 论文所称MLP若含隐藏层则缺失；阈值只在代码中表现为argmax | MEDIUM |
| H. UAV cluster标签 | GT pose与估计cluster center最近邻关联生成标签 | 未给明确阈值 | `extract_feature.py:41-55`; builder `:42,60-61` | 取20帧块首时间戳最近GT；若cluster任一帧算术中心距这一个GT `<1m`，整cluster标1 | PARTIAL | 是（路径需适配） | 不是逐帧GT关联；1m阈值论文未说明；依赖坐标已对齐 | HIGH |
| I. PointNet-like中心回归 | detected cluster raw points → PointNet-like模块 → center prediction | 未给输入点数、层结构 | 全仓库未找到实现 | Tracker只对每帧点做 `np.mean`；没有PointNet或回归head | MISSING | 否 | **CRITICAL MISSING COMPONENT**：模型、训练数据、checkpoint、inference全缺 | CRITICAL |
| J. Center regression loss/target | 论文称中心回归MSE由0.27降至0.05；目标应为UAV center | 报告MSE结果；无训练细节 | 无对应实现 | 无 loss、optimizer、target生成或checkpoint | MISSING | 否 | loss定义细节、target时序匹配和全部训练超参数缺失 | CRITICAL |
| K. Nonlinear bias correction | 3D center做三阶多项式扩展到24D，再线性回归预测bias | degree=3，24D；24维具体构成未列 | `train_bias_correction.py:33-57` | 实现的是 `3→64→64→3` ReLU/Dropout MLP，不是 PolynomialFeatures+LinearRegression | MISMATCH | 仅孤立训练脚本 | 论文模型、拟合代码、参数/checkpoint均缺 | CRITICAL |
| L. Bias训练数据 | 从训练数据学习center-dependent observed bias | 未给构造细节 | `estimate_offsets.m:20-110`; `train_bias_correction.py:7-31` | 用GT全序列包围盒±1m筛原始点，均值后插值到GT时间，残差=`GT-interp_lidar`；matrix=`[timestamp,GT_xyz,residual_xyz]` | PARTIAL | 当前不能直接运行 | MAT文件未保存/未发布；GT ROI选择与论文未说明 | HIGH |
| M. Corrected center推理 | `corrected = initial center + predicted bias`（论文语义） | 未给运行参数 | 全仓库搜索无推理代码 | 无 `depth_correction.pth` 加载；无 polynomial/linear model；无 corrected center 写入 tracker | MISSING | 否 | training code exists，inference code missing；公开MLP按现有matrix需GT输入 | CRITICAL |
| N. StoneSoup/Kalman tracker | 多目标tracker；linear Kalman作为骨干，平滑轨迹并过滤clutter | 论文只定性说明低measurement noise | `fusion_tracking.py:66-199` | 6D CV状态；StoneSoup ExtendedKalmanPredictor/Updater；三轴ConstantVelocity(0.15) | PARTIAL | 是 | 代码用EKF类而论文称linear KF；但模型本身是线性的 | MEDIUM |
| O. Data association | 未关联测量建新track；covariance过阈值删track；较低association threshold | 未给数值 | `fusion_tracking.py:145-182,213-225` | Euclidean NearestNeighbour；`missed_distance=3`; MultiMeasurementInitiator `min_points=1`; CovarianceBasedDeleter trace=30 | MATCH | 是 | 参数只由代码给出，论文无法独立核实 | LOW |
| P. 轨迹补全 | 三阶AR，用前三时刻预测下一时刻，处理missing/lost track | AR order=3 | 全仓库未找到 | 无AR模型、拟合、递推或checkpoint | MISSING | 否 | **paper-only / code-missing** | CRITICAL |
| Q. B-spline平滑 | 插值到指定test时间并以B-spline平滑 | 未给s | `postprocess.py:8-39,107-144` | 0.1秒时间网格；linear与`scipy.splrep/splev`；默认`s=0.5`，只输出图 | PARTIAL | 是 | 不保存正式completed轨迹；无AR输入 | HIGH |
| R. GT评价/Pose MSE | test集按GT计算Pose MSE；表中Ours=2.21375；center regression另报0.27→0.05 | Pose MSE=2.21375 | 全仓库无对应评估/提交脚本 | 没有raw/corrected/tracked/completed任一阶段的正式MSE evaluator | MISSING | 否 | 无法确认具体被评分的中间产物和时间对齐/缺帧规则 | CRITICAL |

## 4. 逐模块固定问题审计

### A. 原始点云输入

1. **论文描述**：§3.2称 pose estimation 主要利用 point cloud；§4.1明确列出 conic 3D LiDAR、peripheral 3D LiDAR 和 77GHz radar。  
2. **公式**：无。  
3. **输入维度**：未给。  
4. **输出维度**：未给。  
5. **网络结构**：此阶段无。  
6. **训练损失**：无。  
7. **训练超参数**：无。  
8. **增强参数**：无。  
9. **代码实现**：部分存在。  
10. **证据**：`tracker/preprocess.py::process_lidar_livox()`、`process_lidar_360()`、`process_fusion()`。  
11. **一致性**：PARTIAL。  
12. **差异**：可运行主链只显式处理 `livox_avia` 与 `lidar_360`，未把 radar 送入 tracker。  
13. **严格复现影响**：论文“多模态点云”的确切使用边界不完整。  
14. **能否运行**：LiDAR两支可以。  
15. **缺失**：论文级传感器选择/坐标约定和 radar 接入实现。

### B. 20帧时间窗口

1. **论文描述**：Fig. 2/3与§3.2.1均明确先累积20帧。  
2. **公式**：无。  
3. **输入维度**：20帧点云，单帧点数未指定。  
4. **输出维度**：累积点集，未指定点数。  
5–8. **结构/损失/超参数/增强**：除窗口20外未说明。  
9–10. **代码**：`process_lidar_360():65-79` 每满20帧清空；`:81-97` 再处理最后20帧。训练 builder `:31-44` 也按非重叠20帧。  
11. **一致性**：MATCH。  
12. **差异**：推理“额外最后20帧”可能与末个整块重复/重叠，论文未说明。  
13. **影响**：低，主要影响边界帧重复。  
14. **能否运行**：是。  
15. **缺失**：窗口边界、末尾策略的论文定义。

### C. 无监督聚类

1. **论文描述**：先对20帧累积点云进行 unsupervised clustering。  
2–8. **公式/维度/结构/loss/train/augment参数**：论文未给聚类算法或参数。  
9–10. **代码**：训练 builder `DBSCAN(eps=1,min_samples=10)`；正式 `process_lidar_360()` 为 `eps=2,min_samples=10`。  
11. **一致性**：PARTIAL。  
12. **差异**：论文无eps；代码 train/inference eps 不一致。  
13. **影响**：高，因为 cluster几何和LSTM特征分布直接随eps改变。  
14. **能否运行**：是，但大稠密点云可能高内存。  
15. **缺失**：论文最终使用eps、坐标单位和距离预处理。

### D. Cluster feature extraction

1. **论文描述**：§3.2.1明确称“extract seven-dimensional features”。  
2. **公式**：未给。  
3. **输入维度**：cluster跨20帧；具体点格式未给。  
4. **输出维度**：论文为每时刻7D；**论文未明确给出7维具体组成**。  
5–8. **结构/loss/train/augment**：未给。  
9–10. **代码**：`extract_feature_set_predict():64-96` 每帧计算3D mean、3D std、3D coordinate range，无点填9个零。  
11. **一致性**：MISMATCH。  
12. **差异**：论文7D；代码9D=`[mean_x,mean_y,mean_z,std_x,std_y,std_z,range_x,range_y,range_z]`。  
13. **影响**：高，论文Attention LSTM的输入和发布checkpoint的9D输入不是同一网络。  
14. **能否运行**：代码9D版本可运行。  
15. **缺失**：论文7D逐维定义、归一化方式。

### E. 时序数据增强

1. **论文描述**：temporal dropout、temporal reverse、spatial/global rotation。  
2–4. **公式/输入/输出**：未给；形状应保持时序feature形状，但这是语义推断，不作为参数。  
5. **结构**：无网络。  
6–7. **loss/训练超参数**：未单列。  
8. **增强参数**：论文未给概率、drop帧数或rotation角度。  
9–10. **代码**：`train_lidara_detector.py::reverse_sequence()`；`random_replace_with_zeros(max_replace=3)`；`augment_data()`选择概率0.5 original/0.25 reverse/0.25 random_replace。训练前`:69-70`增强一次，每个batch又在`:121`增强。  
11. **一致性**：PARTIAL。  
12. **差异**：实现了reverse和1–3帧置零；未找到spatial rotation。  
13. **影响**：中，影响泛化但不阻止简化推理。  
14. **能否运行**：有feature文件时可以。  
15. **缺失**：rotation实现和论文所有增强参数。

### F. Attention LSTM

1. **论文描述**：不只使用最后hidden state，而是对全部hidden states分配动态attention权重并加权融合。  
2. **公式**：论文正文未给可复现attention公式。  
3–4. **输入/输出维度**：论文未给。  
5. **结构**：Attention LSTM + 后续MLP，未给层宽。  
6–8. **loss/超参数/增强参数**：未完整给。  
9–10. **代码**：`tracker/lidar_360_detector.py::MyLSTMClassifier.forward()` 调用普通`nn.LSTM`，随后 `self.fc(out[:,-1,:])`。训练脚本同样如此。  
11. **一致性**：MISMATCH。  
12. **差异**：**论文：Attention over all hidden states；代码：普通LSTM，只取最后hidden state。**  
13. **影响**：高；不能将简单LSTM checkpoint称为论文Attention LSTM。  
14. **能否运行**：简化普通LSTM可运行。  
15. **缺失**：attention结构、权重、论文版checkpoint和训练超参数。

### G. MLP cluster classification head

1. **论文描述**：Attention LSTM后由MLP head分类dynamic/UAV cluster。  
2. **公式**：无。  
3–4. **输入/输出维度**：论文未给；输出语义为二分类。  
5. **结构**：只称MLP，无层数。  
6–8. **loss/超参数/增强**：论文未给。  
9–10. **代码**：训练代码参数为input=9、hidden=64、layers=1、classes=2；head仅`Linear(64,2)`；CrossEntropy、Adam(lr=0.001)、batch64、20 epochs，按validation loss保存。推理为logits argmax，无概率阈值。  
11. **一致性**：PARTIAL。  
12. **差异**：代码没有独立多层MLP，只有一个Linear。  
13. **影响**：中；分类功能存在，但不是论文所述完整网络。  
14. **能否运行**：原`lstm_model.pth`可运行。  
15. **缺失**：论文head结构、threshold及Attention输入。

### H. UAV cluster label generation

1. **论文描述**：通过UAV pose label与估计cluster center做nearest-neighbor association生成GT。  
2. **公式**：无。  
3–4. **输入/输出**：cluster centers + pose labels → binary cluster label；维度未给。  
5–8. **网络/loss/hyperparameter/augmentation**：标签阶段无；阈值未给。  
9–10. **代码**：builder用20帧块首timestamp找最近一个GT；`extract_feature_set()`对每个cluster逐帧算center，只要任一center与该固定GT小于1m，整cluster为1。  
11. **一致性**：PARTIAL。  
12. **差异**：代码不是每帧分别匹配最近GT；1m规则未见于论文。训练脚本路径名为`gt`，official数据为`ground_truth`。  
13. **影响**：高，标签可能受目标运动、时间差和坐标系影响。  
14. **能否运行**：修正数据路径后可运行，算法本身存在。  
15. **缺失**：论文精确label association与阈值；已生成`feature_train.npy/label_train.npy`也未发布。

### I. PointNet-like center regression

1. **论文描述**：对检测出的UAV cluster使用PointNet-based module进行center regression，优于不完整点云的几何均值。  
2–8. **公式/维度/结构/loss/超参数/增强**：除了“PointNet-based”和回归center外均未给。  
9–10. **代码**：在整个`point_cloud_processing`递归搜索PointNet/regression没有对应实现；`fusion_tracking.py::point_cloud_detector()`仅`np.mean(cluster_points)`。  
11. **一致性**：MISSING。  
12. **差异**：论文学习回归，代码算术平均。  
13. **影响**：**CRITICAL**，这是论文从cluster到高精度measurement的核心。  
14. **能否运行**：不能。  
15. **缺失**：输入采样、网络、target、loss、训练脚本、checkpoint、推理接入全部缺失。

### J. Center regression loss/target

1. **论文描述**：§4.2仅报告center regression MSE从0.27降到0.05。  
2. **公式**：没有明确MSE定义。  
3–5. **输入/输出/网络**：应与I对应，细节未给。  
6. **loss**：由结果文字可确定为MSE量度，但训练loss是否完全相同未明确。  
7–8. **训练/增强参数**：未给。  
9–10. **代码**：无对应代码。  
11. **一致性**：MISSING。  
12–13. **差异/影响**：无法重建论文center回归训练或0.05结果，影响严重。  
14. **能否运行**：否。  
15. **缺失**：全部监督训练定义及数据。

### K. Nonlinear bias correction

1. **论文描述**：回归误差与cluster center强相关；将3D坐标通过三阶多项式特征扩展为24D，再用linear regression拟合bias，修正initial center。  
2. **公式**：只给文字关系，没有列24个基函数；**24维具体构成未给**。  
3–4. **输入/输出**：3D center → 3D bias；中间24D。  
5. **结构**：Polynomial feature transformer + linear regression。  
6–8. **loss/训练/增强**：未给。  
9–10. **代码**：`train_bias_correction.py::MLP` 是3→64→64→3，两层ReLU、dropout0.2，MSE、Adam。  
11. **一致性**：MISMATCH。  
12. **差异**：论文 polynomial+linear；代码 neural MLP。  
13. **影响**：CRITICAL，不能把公开MLP当成论文最终bias correction。  
14. **能否运行**：缺MAT，不能直接运行。  
15. **缺失**：论文模型实现、基函数定义、训练参数和checkpoint。

### L. Bias correction训练数据生成

1. **论文描述**：从训练数据中观察并学习center-dependent bias；未详细描述数据制作。  
2–8. **公式/维度/结构/loss/hyperparameter/augment**：论文未给。  
9–10. **代码**：`estimate_offsets.m`先用全sequence GT min/max加1m构造3D ROI；筛选每帧原始LiDAR点并求均值；插值到GT timestamp；计算`residual=GT-interpolated_lidar`；写入内存矩阵。  
11. **一致性**：PARTIAL。  
12. **关键列语义**：`gt=[timestamp,gt_x,gt_y,gt_z]`；`data_matrix_360=[gt,residual_360]`，所以列为`[timestamp,GT_x,GT_y,GT_z,residual_x,residual_y,residual_z]`。Python `X=matrix[:,1:4]`因此确实是**GT xyz**，不是measured cluster center。  
13. **影响**：高；且用GT ROI抽点并非可部署inference。  
14. **能否运行**：脚本硬编码Windows路径/`gt`，也没有保存MAT；现状不能闭环。  
15. **缺失**：`depth_correction_filtered.mat`的保存/过滤过程、论文真实训练输入。

### M. Corrected center inference

1. **论文描述**：用predicted bias调整initial center后再进入tracker。  
2–8. **公式/维度/结构/loss/train/augment**：除3D bias语义外未进一步给出。  
9–10. **代码**：全仓库没有加载`depth_correction.pth`，没有`PolynomialFeatures`/`LinearRegression`，也没有把bias加回center的调用。  
11. **一致性**：MISSING。  
12. **差异**：只有孤立training script，没有inference。  
13. **影响**：CRITICAL。公开Python脚本按其matrix列定义还要求GT作为输入，故标记 **NOT DEPLOYABLE AS WRITTEN**。  
14. **能否运行**：否。  
15. **缺失**：真实模型、checkpoint、输入预处理、加回符号、tracker接口。

### N. StoneSoup/Kalman tracker

1. **论文描述**：linear Kalman filter backbone；以低measurement noise利用center regression；处理clutter和锯齿轨迹。  
2–4. **公式/输入/输出维度**：论文未给；代码可确定measurement=XYZ、state=6D。  
5. **结构**：非学习型tracker。  
6–8. **loss/训练/增强**：无需训练。  
9–10. **代码**：`fusion_tracking.py::process_sequence()`：state `[x,vx,y,vy,z,vz]`；measurement mapping `(0,2,4)`；prior covariance diag `[.01,.1,.01,.1,.01,.1]`；noise covariance `.001 I`; 三轴`ConstantVelocity(.15)`；ExtendedKalman predictor/updater。  
11. **一致性**：PARTIAL。  
12. **差异**：论文称linear KF，代码调用EKF类；由于transition和measurement均线性，数值结构仍是线性高斯滤波。  
13. **影响**：中低；更大的断层是输入未经过论文center regression/correction。  
14. **能否运行**：是。  
15. **缺失**：无训练缺口；论文未给所有数值。

### O. Data association

1. **论文描述**：未关联measurement初始化track；covariance过大删除；较严格/较低association threshold降低clutter影响。  
2–8. **公式/维度/网络/loss/train/augment**：未给。  
9–10. **代码**：每fusion frame先DBSCAN(`eps=1,min_samples=1`)并取均值；Euclidean `DistanceHypothesiser` + `NearestNeighbour`，`missed_distance=3`; `MultiMeasurementInitiator(min_points=1)`；`CovarianceBasedDeleter(trace=30)`。  
11. **一致性**：MATCH（定性行为）。  
12. **差异**：论文没有数值，无法证明这些就是提交参数。  
13. **影响**：低至中。  
14. **能否运行**：是，无训练。  
15. **缺失**：正式提交参数证据及多轨迹输出不覆盖的工程实现。

### P. Trajectory completion

1. **论文描述**：third-order autoregressive model，用前三个时刻预测下一步，补missing observation/lost tracker。  
2. **公式**：没有给AR系数估计公式。  
3–4. **输入/输出**：历史3个3D位置→下一3D位置；更细维度未给。  
5. **结构**：三阶AR，是否逐轴独立未说明。  
6–8. **loss/训练/增强**：未给。  
9–10. **代码**：递归搜索无AR实现。  
11. **一致性**：MISSING。  
12. **差异**：paper-only / code-missing。  
13. **影响**：CRITICAL于完整时间序列和最终Pose MSE，但不阻止raw tracker运行。  
14. **能否运行**：否。  
15. **缺失**：AR拟合、缺口判定、外推长度、稳定约束和输出。

### Q. B-spline smoothing

1. **论文描述**：按test timestamps插值，再以B-spline平滑。  
2. **公式**：无。  
3–4. **输入/输出**：轨迹timestamp+XYZ → 完整timestamp上的XYZ。  
5. **结构**：非学习型。  
6–8. **loss/train/augment**：无。  
9–10. **代码**：`postprocess.py::interpolate_trajectory()`、`interpolate_trajectory_spline()`；0.1s网格；`splrep(...,s)`默认s=0.5；最终`plot_trajectories()`只写PNG。  
11. **一致性**：PARTIAL。  
12. **差异**：代码没有前置AR completion，也不保存submission XYZ。  
13. **影响**：高于最终评价复现。  
14. **能否运行**：是，已有raw track时可画图。  
15. **缺失**：论文使用的s、目标时间表、completed轨迹文件输出。

### R. GT evaluation / Pose MSE

1. **论文描述**：challenge test以GT Pose MSE评分；论文表1报告2.21375；§4.2报告center regression MSE 0.27→0.05。  
2. **公式**：正文没有明确按轴/样本聚合公式。  
3–5. **输入/输出/结构**：未明确指出用于表1的是哪个中间文件。  
6–8. **loss/train/augment**：不适用。  
9–10. **代码**：没有Pose MSE evaluator、submission writer或GT对齐评估脚本。`postprocess.py`只画图。  
11. **一致性**：MISSING。  
12. **差异**：无法从公开代码复现2.21375。  
13. **影响**：CRITICAL。  
14. **能否运行**：否。  
15. **缺失**：评价阶段、时间对齐、缺失帧处理、输出选择。

**最终比较阶段判断：UNCERTAIN / MOST LIKELY**。论文的方法顺序和§4.2语境表明，表1的Pose MSE最可能来自“corrected center → tracker → AR completion → timestamp interpolation → B-spline”的最终完整轨迹，而不是raw cluster center或raw StoneSoup track。证据是§3.2.2明确把corrected center送入tracker，§3.2.3再把tracker输出补全和平滑，然后§4.2报告最终test performance。仍然不确定，因为论文未显式命名被评分文件，仓库也没有evaluation/submission代码。

## 5. 重点冲突核查结论

1. **Feature维度**：论文明确为7D，但未定义七项；代码明确为9D mean/std/range三轴。MISMATCH。
2. **Attention LSTM**：论文对全部hidden states加权；代码`out[:,-1,:]`。MISMATCH。
3. **分类head**：论文称MLP；代码只有`Linear(64,2)`，CrossEntropy，argmax。PARTIAL。
4. **增强**：论文有drop/reverse/rotation；代码有1–3帧置零和reverse，没有rotation。PARTIAL。
5. **PointNet center regression**：论文核心组件；公开代码完全缺失。CRITICAL MISSING COMPONENT。
6. **Bias correction**：论文3D→24D三阶多项式→线性回归；代码3→64→64→3 MLP。MISMATCH。
7. **Bias数据列**：MATLAB逻辑证明Python的X为GT xyz，故公开MLP **NOT DEPLOYABLE AS WRITTEN**。
8. **Bias inference**：training code exists，inference code missing。
9. **Tracking**：非学习型，公开参数完整可执行；其输入却不是论文回归/修正中心。
10. **Postprocess**：线性/B-spline存在，三阶AR缺失。

## 6. 复现阻塞项

| 优先级 | 缺失项 | 为什么重要 | 是否可从论文重建 | 是否需要自己重新实现 |
|---|---|---|---|---|
| P0 | PointNet-like center regression完整实现与checkpoint | 论文把它作为cluster几何中心到高精度measurement的关键，直接影响tracker输入与0.05 center MSE | 否；论文没有结构、采样、target和训练参数 | 是；若无作者补充材料只能设计近似版，不能宣称严格复现 |
| P0 | 论文版24D polynomial + linear bias correction及inference | 决定送入tracker的center；公开MLP不是同一模型且按现有数据列不可部署 | 部分；degree和24D已知，但24维构成、预处理、系数未知 | 是，或向作者索取实现/模型 |
| P0 | 三阶AR trajectory completion | 最终完整时间序列和Pose评分很可能依赖它 | 部分；order=3已知，其余缺口策略/拟合细节未知 | 是，或向作者索取 |
| P0 | 最终submission/evaluation链 | 无法确定表1 MSE评估的是哪一阶段，也无法复现缺帧/时间对齐规则 | 否 | 是，最好先取得challenge evaluator/提交格式 |
| P0 | 论文Attention LSTM和7D feature定义/checkpoint | 严格动态点分析的核心表征与公开9D普通LSTM不同 | 否；7D组成和attention公式均缺 | 是，或向作者索取；简化pipeline可暂用公开LSTM |
| P1 | 训练/推理DBSCAN eps不一致的权威解释 | 会改变cluster与LSTM输入分布，但公开checkpoint仍可直接运行 | 不能从论文判断 | 不宜自行“修正”；先保留并报告 |
| P1 | Spatial rotation增强实现 | 影响论文分类器泛化，不阻止已有checkpoint推理 | 否，角度/概率未给 | 若重训论文版分类器则需要 |
| P1 | Cluster标签时间关联定义 | 当前代码用窗口首帧的一个GT标注全部20帧，与论文nearest association不够一致 | 部分 | 重训前需要明确/实现 |
| P1 | Bias MAT生成与过滤步骤 | `estimate_offsets.m`没有保存训练脚本所需MAT，且Python输入列有语义问题 | 否 | 需要澄清后重做，不能照错脚本推断论文结果 |
| P2 | 多track同timestamp覆盖 | 不改变算法，但会丢失公开tracker输出 | 可从代码直接修复 | 是，旁路track_id CSV已可规避 |
| P2 | Windows硬编码路径与`gt`/`ground_truth`命名 | 阻碍脚本直接执行，不改变方法 | 是 | 薄数据adapter即可 |

说明：training label generation和training dataset generation并非完全缺失，因此没有机械列为P0；它们属于可见但有语义/路径/eps冲突的P1。Attention LSTM对“严格复现论文动态模块”是P0，但对“运行公开简化pipeline”不是阻塞项。

## 7. 明确复现判断

### A. 当前公开代码能否复现论文完整动态点分析模块？

**不能。** 公开代码实现了20帧、DBSCAN、9D feature和普通LSTM二分类，但论文要求7D feature、Attention LSTM、MLP head、PointNet center regression及完整增强；其中多项缺失或冲突。

### B. 当前公开代码能否复现论文最终3D轨迹精度？

**不能。** 论文版center regression、bias correction、AR completion及最终evaluator/submission链缺失，因而无法复现Pose MSE 2.21375。

### C. 当前公开代码能否复现一个简化版可运行pipeline？

**能。** 已能运行Mid360/Livox preprocessing、公开普通LSTM、fusion、StoneSoup tracker和B-spline可视化；当前seq0001已经生成一条122时间戳的连续raw track，但Mid360分支全空，实际轨迹来自Livox/fusion。

### D. 论文级完整复现还必须补什么？

至少必须取得或重建：**7D+Attention LSTM、PointNet-like center regression、论文版polynomial bias correction及推理、三阶AR completion、最终evaluation/submission链**。在作者未提供结构/参数时，任何自行实现只能称“paper-inspired reconstruction”，不能称严格复现。

