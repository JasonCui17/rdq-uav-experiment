# Stage 4 Runtime Parameter Tuning

## 目标与执行边界

本工具只测当前模型在本机上的数据加载、Host-to-Device、前向、反向与 optimizer 吞吐，不改变模型、loss、optimizer 或学习率。每次命令只测试一个 `(workers, batch_size)` 组合；正式训练仍只允许用户手动启动。

入口：

```text
tools/training_speed_benchmark.py
```

每次默认执行 10 个 warmup batch 和 100 个 measure batch。计时边界调用 `torch.cuda.synchronize()`，输出：

- data time
- Host-to-Device time
- forward time
- backward + optimizer time
- total batch time
- batch/s 与 samples/s
- CUDA peak allocated/reserved memory
- 预计单 epoch 和 150 epoch 时间

结果逐次追加到：

```text
outputs/runtime_tuning/benchmark_results.csv
outputs/runtime_tuning/recommended_runtime_config.yaml
```

## 严格执行顺序

第一轮固定 batch=8，分别手动运行 workers=0/2/4/8。workers>0 时固定使用：

```text
pin_memory=true
persistent_workers=true
prefetch_factor=2
```

从吞吐距离最高值不超过 5% 的配置中选择最小 workers。

第二轮固定选出的 workers，依次测试 batch=8/16/32/64。出现 CUDA OOM、reserved memory 超过总显存 85%，或 batch 翻倍而 samples/s 提升小于 5% 时停止增大。

第三轮只对吞吐最好的两个 batch 做用户手动短训练：30 epochs、patience=8、完整 val、seed=42。比较 validation center error、best epoch、samples/s、epoch time 和 GPU memory。吞吐结果只是候选，不能直接替代质量验证。

## 启动信息核对

benchmark 与 localization trainer 都会打印：GPU 型号、CPU 核数、batch、workers、pin memory、persistent workers、prefetch factor、AMP，以及每个 optimizer parameter group 的真实 LR。

当前默认配置的真实 LR 是：

```text
backbone:    1e-4
new_modules: 1e-3
```

吞吐调优不会自动修改它们。

