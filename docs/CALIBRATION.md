# 坐标、时间与标定

- 相机模型：`configs/calibration/mmaud_v1_omni.yaml`，左鱼眼Unified/Omni模型，畸变和内参均来自原工程固定配置；1280×960左视图，不做假定的针孔线性映射。
- 投影实现：`src/rdq_uav/multimodal_v3/modeling/geometry.py`；使用打包的相机模型和`calibration/official_left_p4_current_geometry.json`外参。
- `time_offset_s=0.0`，沿用raw image_time与query_time约定；不在最终验证中重新搜索时间偏移。
- 输入point XYZ使用现有released MMAUD坐标约定。模型输出3D单位米，沿用原GT坐标系。Avia/Mid360历史点云仍按既有sensor_id区分，不重新拼接未经确认的新坐标变换。
- 冻结几何标定的validation mean reprojection error为2.756px，median1.848px，p95 7.838px。小目标仅十余像素时，投影中心与框中心可以明显偏离；裁剪门控采用原inverse-range margin，不能将投影点自动当成像素级GT。
- 原标定结果包含2183 fit correspondences和549 validation observations的汇总。原拟合CSV不在已有源码快照，工程仅携带固定结果；重新拟合必须另收集对应点，不使用最终验证GT偷偷重拟合。
- 标定JSON仅保留time_convention、time_offset_s及左相机旋转/平移。拟合状态、重投影误差统计、相机中心冗余值及来源描述不进入运行文件；本说明保留必要背景。数值参数保持原值，精简前后投影结果逐值一致。最终checksum以本工程MANIFEST.json为准。
