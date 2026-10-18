# 发布验收（2026-10-17）

## 已执行

- 58项现有跟踪、起轨、图像推理、裁剪训练、提取测试通过；3项最终工程测试通过，覆盖实时评分不读未来状态、不确定图片不作为负例、只读当前MMAUD标签且保留无标签图像输入。
- 独立只读审查完成。修复了无GT传感器推理被旧评分器阻断、完成状态不一致和实时评分误用未来插值等问题。
- 本地RTX3070 Laptop GPU：12传感器查询端到端smoke通过；包含8张左图，明确标为smoke，不当作全覆盖指标。
- GPU雷达训练入口fast_dev_run（batch1）完成，成功加载打包A2资产。该一批为skip=1，因此仅证明装载/循环/验证入口，不证明本次发生有效参数更新。
- GPU视觉裁剪训练入口fast_dev_run（batch1）完成，loss约19，skip=0，显存allocated约3.19GiB。验证首张属于不确定排除项，V/images=0，符合不将其作为负例的约定。不将单批验证作AP结论。
- 全sequence实际运行：`runs/seq0020_full_sensor/`；无GT传感器推理935查询，整图预测620张，最终bbox620/620。

## seq0020完整sensor时间轴结果

| 分支 | RMSE(m) | 在GT时刻可评分输出 | AP50 | AP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| realtime | 0.46652 | 393/400 | 86.76% | 29.77% | 86.20% | 91.33% |
| lag05 | 0.37736 | 398/400 | 86.87% | 29.75% | 86.50% | 91.50% |

传感器时间轴实时输出916/935，平滑935/935；GT时刻评分另有边界/起轨缺失，不能混为一个coverage。左图620张全部处理，32张human uncertain整图排除，588张计算图像指标；其中含未人工复审的原数据标签。

这条sequence的全图新指标不等于历史3条sequence查询关联1158张人工GT的89.01% AP50；不能把89.01%承诺给所有帧。最终全验证集/全数据集仍由用户接下来的完整验证确认。未重新训练最佳权重，未修改数据/标注/标定数值来抬高分数。

## 独立性

- `python tools/audit_portability.py`检查工程文本及所有软链接，没有旧工程路径依赖；引用源码已内置。
- `python run.py doctor`检查实际导入RDQ、detrex、detectron2、Stone Soup及detrex扩展都来自本文件夹，核对四份checkpoint、COCO初始化与MANIFEST SHA。
- vendor CUDA扩展携带当前环境版本；换平台需要从本目录源码重编译。没有在全新Conda环境重新安装所有依赖，该项不宣称通过。
- 历史标定CSV未提供，冻结标定数值可推断，但不声称可从缺失CSV复现标定优化。

## 最后验证建议

先运行`bash run_final_validation.sh MMAUD_ROOT benchmark runs/user_benchmark`复核历史查询协议，再运行`bash run_final_validation.sh MMAUD_ROOT sensor runs/user_sensor`验证完整帧覆盖和传感器时间轴指标。两个目录独立保存，保持发布权重不变。额外sequence可用run.py infer --sequences指定，或--split all处理全部。

## 删除工程内GT覆盖逻辑（2026-10-17）

用户要求统一数据入口后，训练、单sequence和全局池化评分均只读取MMAUD当前2d_detect；工程内annotations目录及其读取配置已移除。37个未确认txt移入MMAUD审核归档，文件原内容保留，活动数据按无可靠GT处理。14项相关测试通过。此前GPU预测不变，新的纯数据集标签评分结果见runs/dataset_labels_check/，不重新推理或训练。

## 正式流程迁出experiments（2026-10-17）

正式推理模块迁至src/rdq_uav/pipeline，训练模块迁至src/rdq_uav/training；在用配置统一configs/final，测试统一根tests。run.py以python -m调用正式模块，正式入口及配置/测试不再依赖experiments。

109项测试通过，独立只读审查通过。迁移后的本地GPU端到端smoke见runs/migrated_pipeline_smoke：12条轨迹及8张图片bbox与迁移前完全一致；候选ID和标记一致，雷达坐标缓存最大绝对差约7.63e-6m，logit最大差约1.57e-5，为浮点复算微小差异。没有重训或改模型权重。历史实验剩余内容先保留，不作为当前运行入口。

迁移验收后已删除根目录experiments及其历史配置/辅助脚本；正式源码、配置和测试不依赖该目录。

## 2026-10-17：源码统一到 multimodal_v3

- 将模型/数据/几何实现迁入 `multimodal_v3/modeling/`，正式流程和训练入口迁入其 `pipeline/`、`training/`，路径解析也迁入 v3。
- 删除未使用的旧版源码目录和无调用方的轨迹池、框合并及 trace 模块。仅保留 `rdq_uav/__init__.py` 作为父包标识；不提供旧路径兼容副本。
- 更新入口、测试、文档和来源哈希路径。模型类名、参数键、权重及算法数值配置保持原样。
- 109 项测试通过；独立只读审查通过；可移植性审计通过。
- RTX 3070 Laptop GPU 实测 `runs/v3_consolidation_smoke`：12 个 sensor 查询；12 条轨迹、8 张图片 bbox 与 `runs/migrated_pipeline_smoke` 逐条 JSON 完全一致。此次小规模检查验证迁移兼容性，不作为新的完整数据集指标。
