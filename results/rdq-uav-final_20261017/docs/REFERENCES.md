# 引用与工程来源

- detrex（包含DINO与Swin实现）：third_party/detrex，保留LICENSE、CITATION及完整源码。官方工程 https://github.com/IDEA-Research/detrex 。
- detectron2：third_party/detrex/detectron2，保留LICENSE及源码。官方工程 https://github.com/facebookresearch/detectron2 。
- Stone Soup：Python依赖stonesoup==1.9.1，用于固定窗口Kalman RTS平滑； https://github.com/dstl/Stone-Soup 。其完整安装源码见third_party/stonesoup（标准库依赖由requirements安装）。
- BYTE关联思想：高分候选起轨，低分候选只续轨；本工程是单目标3D适配，未声称完整ByteTrack复现。参考源码快照已收在third_party/reference_only/ByteTrack。 https://github.com/ifzhang/ByteTrack 。
- CenterFusion：投影几何关联思想；本工程视觉支持起轨及裁剪门控为本项目适配，未使用其学习架构或训练代码。参考源码快照已收在third_party/reference_only/CenterFusion。 https://github.com/mrnabati/CenterFusion 。
- DINO： https://github.com/IDEA-Research/DINO 。实际运行代码已收在detrex内。

推理不加载上述URL或其它项目目录。网络仅用于首次安装Python依赖。研究阶段尝试的QFL/MAL等非最终默认方法，不作为最终方案依赖。

参考快照的固定commit记录在reference/reference_projects.json，许可证保留。reference_only不是最终可执行模块；CenterFusion的nuScenes/DCNv2子模块不用于本MMAUD工程，未下载数据集或其预训练权重。
