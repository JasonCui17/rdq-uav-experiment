# MMUAV 学习模块训练要求审计

## 1. 使用说明

本表只记录论文或 commit `f11b57390effbe9623ee2c7d561afddc8d0cdfa7` 公开代码能够证明的信息。`NOT SPECIFIED` 表示不能从公开材料安全恢复，不能用常见默认值补齐。

## 2. Attention LSTM + cluster classifier

| 项目 | 论文 | 公开代码 | 严格复现判断 |
|---|---|---|---|
| training input | 20帧cluster temporal feature；论文称每时刻7D | `feature_train.npy`，实际shape语义为`[cluster,20,9]`，9D=`mean_xyz,std_xyz,range_xyz` | MISMATCH；论文7D组成NOT SPECIFIED |
| training target | UAV pose与cluster center最近邻关联得到binary label | `label_train.npy`；任一帧cluster mean距窗口首时刻最近GT `<1m`则label=1 | PARTIAL；论文threshold NOT SPECIFIED |
| model | Attention LSTM聚合全部hidden states + MLP分类head | 普通单层`nn.LSTM(9,64)`，只取`out[:,-1,:]`，随后`Linear(64,2)` | MISMATCH |
| loss | NOT SPECIFIED | `CrossEntropyLoss` | 仅简化代码可确定 |
| optimizer | NOT SPECIFIED | Adam | 仅简化代码可确定 |
| learning rate | NOT SPECIFIED | 0.001 | 仅简化代码可确定 |
| epochs | NOT SPECIFIED | 20 | 仅简化代码可确定 |
| batch size | NOT SPECIFIED | 64 | 仅简化代码可确定 |
| temporal reverse | 论文有 | 代码有；augmentation选择概率0.25 | PARTIAL |
| temporal dropout | 论文有 | 随机将1–3个frame feature置零；augmentation选择概率0.25 | PARTIAL；论文参数NOT SPECIFIED |
| spatial/global rotation | 论文有 | 未找到 | MISSING；角度/概率NOT SPECIFIED |
| original sample probability | NOT SPECIFIED | 0.5 | 仅代码行为 |
| augmentation schedule | NOT SPECIFIED | 全train tensor预增强一次；训练batch内再次增强 | 论文无法核实 |
| train/val/test split | NOT SPECIFIED | `feature_train.npy`训练，`feature_val.npy`被变量名视为test；没有独立第三split | PARTIAL |
| checkpoint selection | NOT SPECIFIED | validation/test loss最低 | 仅代码行为 |
| checkpoint output | NOT SPECIFIED | `lstm_model.pth` | 仓库已发布一个可加载checkpoint，但对应普通LSTM |
| inference input | 论文7D cluster sequence | 代码`[M,20,9]` float32 | MISMATCH |
| inference output | cluster UAV/background | 两类logits，argmax class；无显式概率threshold | PARTIAL |

### 代码证据

- Feature与标签：`point_cloud_processing/codes for training lidar detector/extract_feature.py::extract_feature_set()`，第12–60行。
- Dataset builder：`build_lidar_360_dataset.py`第31–73行；训练DBSCAN为`eps=1,min_samples=10`。
- 增强：`train_lidara_detector.py::reverse_sequence()`、`random_replace_with_zeros()`、`augment_data()`，第27–70行。
- 模型与训练：同文件第82–160行；`out[:,-1,:]`证明没有attention。
- 推理模型：`tracker/lidar_360_detector.py::MyLSTMClassifier`，结构与训练副本一致。
- 正式推理DBSCAN：`tracker/preprocess.py:112-119`，`eps=2,min_samples=10`，与训练不一致。

## 3. PointNet-like center regression

| 项目 | 论文 | 公开代码 | 严格复现判断 |
|---|---|---|---|
| training input | detected UAV cluster的raw points | NOT IMPLEMENTED | Point count/sampling/normalization NOT SPECIFIED |
| training target | UAV cluster center/pose | NOT IMPLEMENTED | 时间关联和坐标定义NOT SPECIFIED |
| model | PointNet-based center regression module | NOT IMPLEMENTED | **CRITICAL MISSING COMPONENT** |
| output | 3D center | `fusion_tracking.py`只用点算术均值，不是回归 | MISMATCH |
| loss | 论文只报告center regression MSE由0.27降到0.05 | NOT IMPLEMENTED | 训练loss细节NOT SPECIFIED |
| optimizer | NOT SPECIFIED | NOT IMPLEMENTED | MISSING |
| learning rate | NOT SPECIFIED | NOT IMPLEMENTED | MISSING |
| epochs | NOT SPECIFIED | NOT IMPLEMENTED | MISSING |
| batch size | NOT SPECIFIED | NOT IMPLEMENTED | MISSING |
| augmentation | 论文在动态点分析上下文提到point cloud augmentation，但未明确center head是否共享 | NOT IMPLEMENTED | NOT SPECIFIED |
| train/val/test split | NOT SPECIFIED | NOT IMPLEMENTED | MISSING |
| checkpoint output | NOT SPECIFIED | 无checkpoint | MISSING |
| inference input | UAV cluster points | 无接口 | MISSING |
| inference output | regressed 3D initial center | 实际tracker measurement来自`np.mean(filtered_data[class_mask],axis=0)` | MISMATCH |

### 代码证据

- 全仓库`point_cloud_processing`没有PointNet或center-regression模型实现。
- `tracker/fusion_tracking.py::point_cloud_detector()`第50–63行只以算术平均生成measurement。
- 论文§3.2.1明确描述PointNet-based module；§4.2只给0.27→0.05结果，未给训练配置。

## 4. Bias correction

| 项目 | 论文 | 公开代码 | 严格复现判断 |
|---|---|---|---|
| training input | initial 3D cluster center；论文称bias与center强相关 | Python脚本实际读取`filtered_data_matrix_360[:,1:4]` | 由MATLAB列构造可证这是GT xyz，不是measured center |
| training target | observed 3D bias | `matrix[:,4:]`，为`GT-interpolated_lidar`残差xyz | PARTIAL |
| model | degree-3 polynomial feature transformer：3D→24D；linear regression：24D→bias | MLP：3→64→64→3，ReLU，两个Dropout(0.2) | MISMATCH |
| 24D具体构成 | NOT SPECIFIED | 未实现 | 不能猜是否含bias项及具体monomial排列 |
| loss | NOT SPECIFIED | MSELoss | 仅孤立MLP代码可确定 |
| optimizer | 论文linear regression拟合方法NOT SPECIFIED | Adam | MISMATCH |
| learning rate | NOT SPECIFIED | 0.001 | 仅MLP代码 |
| epochs | NOT SPECIFIED | 100 | 仅MLP代码 |
| batch size | NOT SPECIFIED | 32 | 仅MLP代码 |
| augmentation | NOT SPECIFIED | 无 | UNCERTAIN |
| split | NOT SPECIFIED | 按现有顺序70%/15%/15%；仅train loader shuffle | 仅MLP代码 |
| checkpoint output | 论文模型参数NOT SPECIFIED | `depth_correction.pth` | 文件未发布 |
| inference input | initial cluster center | 无推理代码；公开MLP按现有X语义需GT | **NOT DEPLOYABLE AS WRITTEN** |
| inference output | predicted bias xyz；加到initial center得到corrected center | 无加回实现 | MISSING |

### 训练数据列反推

`estimate_offsets.m`明确构造：

```text
gt = [timestamp, gt_x, gt_y, gt_z]
interp_360 = LiDAR mean center interpolated to GT timestamps
residual_360 = gt_xyz - interp_360
data_matrix_360 = [gt, residual_360]
```

所以矩阵列是：

| MATLAB/Python列 | 语义 |
|---|---|
| 1 / index 0 | timestamp |
| 2:4 / index 1:4 | GT x,y,z |
| 5:7 / index 4:7 | residual x,y,z |

`train_bias_correction.py`的`X=[:,1:4]`因此是GT xyz。该结论来自生成逻辑，不来自变量名猜测。

### 数据生成的其他限制

- `estimate_offsets.m:20-35`先用整个sequence GT min/max加`margin=1m`建立ROI。
- `:51-88`只保留落在GT ROI内的原始LiDAR点，再取均值。
- `:94-110`插值并计算残差。
- 脚本没有保存`depth_correction_filtered.mat`；仓库也没有该MAT文件。
- 仓库无`depth_correction.pth`、`PolynomialFeatures`、`LinearRegression`或bias inference调用。

## 5. 训练信息完整性结论

| 模块 | 是否有训练代码 | 是否与论文一致 | 是否有checkpoint | 是否有inference | 当前状态 |
|---|---|---|---|---|---|
| Attention LSTM + classifier | 有简化版 | 否：9D普通LSTM vs 7D Attention LSTM | 有简化版`lstm_model.pth` | 有简化版 | 可运行简化版；不可严格复现论文版 |
| PointNet center regression | 无 | 无法对应 | 无 | 无 | P0阻塞 |
| Bias correction | 有孤立MLP脚本 | 否：MLP vs polynomial+linear | 无 | 无 | P0阻塞；公开脚本按现状不可部署 |

## 6. 在任何重新训练前必须取得的信息

1. 论文7D feature的逐维定义及归一化。
2. Attention计算公式、LSTM hidden/layer参数及MLP head结构。
3. PointNet-like center regression的输入采样、网络、target、loss和checkpoint。
4. 三阶多项式24D基函数顺序、linear regression参数与bias加回约定。
5. 动态cluster label的逐帧/逐窗口GT关联规则。
6. spatial rotation以及所有augmentation参数。
7. 作者实际使用的train/val划分和checkpoint选择标准。

这些信息没有公开时，可以构建新的可解释实现，但必须命名为“paper-inspired reconstruction”，不能称为论文严格复现。
