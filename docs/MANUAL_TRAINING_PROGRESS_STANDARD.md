# 手动训练与进度显示规范

## 执行权限

正式训练只由用户在终端手动启动。Codex 只负责实现、测试训练脚本并提供命令；smoke/unit/audit 不属于正式训练，但也不得读取 test，除非用户明确授权。

## tqdm 输出

训练 batch：

```text
Epoch x/N | batch/total | % | GPU_mem | loss | lr | it/s | ETA
```

验证 batch：

```text
Val x/N | batch/total | % | mean center error | P<8px | it/s | ETA
```

epoch 结束：

```text
val metric | best metric | patience current/limit | checkpoint
```

定位训练默认用 validation `bbox_center_error_px_mean`，`mode=min`。最大 epoch 由配置决定，Early Stopping 默认 patience=15。best checkpoint 只能由 validation metric 选择。

## 标准手动命令

```bash
conda activate rdq
cd /home/jasoncui/projects/rdq-uav-experiment

python tools/train_localization.py \
  --config configs/localization/rdq.yaml \
  --set train.epochs=150 \
  --set train.early_stopping_patience=15 \
  --set train.checkpoint_metric=bbox_center_error_px_mean \
  --set train.checkpoint_mode=min
```

运行前必须检查 resolved config 和 split，禁止使用 test 调参。训练中断后只能使用同一次 run 保存的 `last.pt` 恢复，不能用 test 结果选择 checkpoint。

