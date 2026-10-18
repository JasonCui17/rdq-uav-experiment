# 训练、权重选择与推理

## 已验证训练资产的来历

1. 雷达：learned SBE A2初始化 → 时间/运动质心、LoRA适配 → B0目标候选。最终radar.ckpt来自原best checkpoint，30epoch配置、lr1e-4、batch4，按val/success_1m选取。
2. 整图：原B4训练的DINO/Swin-T分支，best_joint.ckpt严格载入；最终只使用其图像分支。历史B4多模态联合训练不在最后这个系统中继续优化。
3. 裁剪：从同一B4视觉权重混合整图与256×256正裁剪训练，概率0.5、中心抖动≤64px，lr1e-5、batch4、AMP，最多12epoch，原实验早停于epoch3。实测最佳双模型融合采用last.ckpt。原损失/QFL/MAL等后续试验没有替换最终选定权重。
4. KF/起轨/几何筛选不参与梯度训练，参数冻结于configs/final/。

精确的历史配置在reference/；只描述资产来源，不依赖其中历史输出目录。模型hash见MANIFEST。使用修正GT重新训练会形成新的实验，不能承诺复现历史数值；保持已验证权重作基线。

## 训练入口

```bash
# 雷达：从打包A2初始化训练
python -u run.py train --stage radar \
  --data-root /home/jasoncui/datasets/MMAUD/official/train \
  --output runs/train_radar_v2 --epochs 30 --batch-size 4

# 视觉：从提供的COCO初始化开始训练单类别整图检测器
python -u run.py train --stage image-coco \
  --data-root /home/jasoncui/datasets/MMAUD/official/train \
  --output runs/train_image_coco --epochs 12 --batch-size 4

# 视觉整图：从已验证整图权重继续训练
python -u run.py train --stage image-full \
  --data-root /home/jasoncui/datasets/MMAUD/official/train \
  --output runs/train_image_full_v2 --epochs 12 --batch-size 4

# 裁剪模型：从已验证整图权重开始混合裁剪训练
python -u run.py train --stage image-crop \
  --data-root /home/jasoncui/datasets/MMAUD/official/train \
  --output runs/train_image_crop_v2 --epochs 12 --batch-size 4
```

4090历史训练batch4峰值约17.6GiB；本地3070 Laptop建议batch1、--accumulate 4。显存与图像大小/软件版本相关。训练配置保持14/3/3 sequence划分，query schedule沿用GT时间文件名；训练监督允许读取训练GT。训练和验证2D监督均直接读取MMAUD的2d_detect文件；缺失/空标签不提供2D监督。当前37张未确认标注已移出活动标签目录，不变为背景负样本。测试split不参与训练/选checkpoint。

训练结果：output/checkpoints/及logs/csv，保存effective_config.yaml。雷达best按success_1m；图像best_ap按整图AP，因此裁剪模型应另做固定双模型融合验证，不能只因整图best_ap高就替换历史last基线。

```bash
# 用新训练模型做端到端验证（路径均在本工程内）
python run.py infer --data-root /home/jasoncui/datasets/MMAUD/official/train \
  --split validation_sub --schedule benchmark \
  --radar-checkpoint runs/train_radar_v2/checkpoints/best.ckpt \
  --full-checkpoint weights/vision_full.ckpt \
  --crop-checkpoint runs/train_image_crop_v2/checkpoints/last.ckpt \
  --output runs/new_models_validation
```

COCO→单类别整图→混合裁剪是可训练链路；打包的B4最佳视觉分支则保留历史联合训练成果。没有自动声称这条重训链会生成完全相同的B4权重。若要续训optimizer状态，使用正式训练模块`PYTHONPATH=src python -m rdq_uav.multimodal_v3.training.train --config configs/final/radar.yaml --resume`；默认warmstart只加载模型参数，不恢复epoch/优化器。

正式配置：radar.yaml与radar_architecture.yaml用于雷达，vision_coco.yaml用于COCO初始化，vision_train.yaml用于已训练视觉权重微调，candidates.yaml配置候选预算；均位于configs/final。
