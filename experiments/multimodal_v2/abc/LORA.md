# A2/ABC 的 LoRA 效率预实验

## 目标与范围

在上一版 ABC 包上新增 full / heads / lora 三种训练方式。先对 control、a_head 各跑三种方式，共六组；所有组独立加载同一个原 A2 checkpoint。原有正在运行的 ABC 输出及配置不改。

LoRA 原理是冻结 W、训练低秩增量 `(alpha/r)BA`。参考原论文 https://arxiv.org/abs/2106.09685 。论文中大语言模型的提速和显存比例不能直接套到本项目。

当前骨干只有约 110 万参数，LoRA 的优化器节省未必是主要提速来源；原始点加载、体素化、注意力前向、候选选择、训练指标和验证仍然存在。本交付不预报每轮 20 分钟会缩短到多少分钟。

## 实际模块配置

- 适配层：`encoder1.blocks.{0,1}.{qkv,proj}`、`encoder2.blocks.{0,1}.{qkv,proj}`，共 8 个 nn.Linear。
- `qkv` 是融合投影，适配 Q/K/V 全部通道，不是只适配 Q、V。
- 默认 rank=8、alpha=16、dropout=0；A 用 Kaiming 初始化，B 全零，初始增量严格为零。
- 冻结体素嵌入、encoder0、merge、位置编码、相对偏置、FFN、上采样等原参数。
- full：骨干、原分类头、原回归头、若存在则质量头均训练。
- heads：仅原分类/回归头、若存在则质量头训练，整个骨干冻结。
- lora：8 个目标层的 A/B + 上述预测头训练。LayerNorm/bias 不额外解冻。
- 一般 ABC 接口也支持新增 metadata_proj、C 的 decoder 全量训练；本轮六组不包含 C，不混合改变算法因素。

### 参数量（代码实测）

| 方案 | 全量可训练 | 仅预测头 | LoRA＋预测头 | LoRA A/B 本身 |
|---|---:|---:|---:|---:|
| control | 1,102,746 | 16,772 | 41,348 | 24,576 |
| a_head | 1,111,323 | 25,349 | 49,925 | 24,576 |

约 96% 的可训练参数减少不是 96% 的训练耗时减少。

## 梯度、权重与推理

`LoRALinear` 继承 nn.Linear，保留原 weight/bias 参数键，新增 lora_A/lora_B。加载 A2 时严格允许新增模块和适配参数缺失，旧参数缺失或尺寸不匹配立即报错；LoRA 注入在质量头等初始化之后，且隔离随机数状态，使不同训练方式的新增头初始化一致。

**没有对整个骨干使用 no_grad。**没有可训练参数和可微输入的冻结前缀自然不会保留反向图；从 encoder1 第一个适配器开始，后续冻结层仍需对输入求导，把梯度传回适配器。测试验证 encoder0 输出不需要梯度，encoder1/上采样输出在 LoRA 方式下需要梯度。

checkpoint 保存完整基础权重和 LoRA 参数；独立评价、resume 不要求原 A2 文件仍存在，但必须使用同一个模型配置。这样首先保证训练/评价兼容性；没有另做 adapter-only 权重打包。

推理保留未合并 LoRA 计算，存在少量额外矩阵乘；没有声称零推理开销。后续若要部署，可单独实现合并导出，并验证等价。

## 代码文件

| 文件 | 改动 |
|---|---|
| radar_lora.py | LoRALinear、明确目标层匹配、冻结策略、参数统计 |
| radar_model.py | 显式启用时注入适配器，默认行为不变 |
| radar_experiment_init.py | A2 warmstart 允许缺失适配器参数 |
| training.py | `radar_train_mode: heads/lora`，只将指定参数交给优化器 |
| efficiency.py / train.py | 参数清单、训练/验证墙钟时间、GPU 峰值显存记录 |
| abc/run_suite.py | 新训练方式开关及独立 200-update benchmark |
| abc/efficiency_pilot.py | 六组独立调度 |
| abc/efficiency_report.py | 相同 query 的精度及速度对照 |
| tests/test_radar_lora.py | 初始恒等、冻结参数不变、LoRA 梯度、严格重载、runtime 集成 |

## 第一步：应用补丁

包中两份补丁二选一，先检查再应用：

- 已应用上一版 ABC 包：使用 `lora_on_abc.patch`。
- 仍是原始 `6890f2e`：使用 `abc_with_lora_full.patch`，包含 ABC 和本次新增内容。
- 如果你已另外修改这些文件，`git apply --check` 可能拒绝；应保留已有改动并合并冲突，不能直接覆盖或 reset。

```bash
git apply --check /path/to/lora_on_abc.patch
git apply /path/to/lora_on_abc.patch
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests
```

不需要安装 PEFT。使用现有 rdq/PyTorch 环境即可；CUDA 实测仍需要在你的服务器执行。

## 第二步：先测速度，避免直接再跑几十小时

下面 prepare 生成六组独立配置，统一学习率 1e-4、相同 batch/精度/epoch，默认保存最佳 Top1@1m。LoRA 不单独提高学习率，若后续调参，需要按相同验证预算比较。

```bash
python experiments/multimodal_v2/abc/efficiency_pilot.py \
  --action prepare --output outputs/lora_pilot_seed42 \
  --checkpoint /absolute/path/to/A2/checkpoints/best_rmse.ckpt \
  --data-root /absolute/path/to/mmaud_official_train \
  --epochs 3 --num-workers 2

python experiments/multimodal_v2/abc/efficiency_pilot.py \
  --action gate --output outputs/lora_pilot_seed42 --device cuda:0

python experiments/multimodal_v2/abc/efficiency_pilot.py \
  --action benchmark --output outputs/lora_pilot_seed42 \
  --benchmark-updates 200 --device cuda:0

python experiments/multimodal_v2/abc/efficiency_report.py \
  --pilot outputs/lora_pilot_seed42 --phase benchmark
```

- gate 只做两次更新，检查可运行；不能当速度结论。
- benchmark 每组做 200 次更新，保留正常训练指标开销，但关闭每轮验证；只用来测训练阶段速度和显存，不评价准确率。
- benchmark 不接着 gate，也不被后面的 train 继承。每次独立 A2 warmstart。
- 六组按顺序占用同一个 GPU；不要同时在该 GPU 上跑其他训练再做速度对比。
- 如果预算更紧，可在各命令统一加 `--variants control`，先测三种训练方式；报告会明确缺失的预定组。
- benchmark 若未完成要求的更新次数，报告拒绝把它当完整对照；增加准备配置的 epochs 或检查跳过批次原因。

## 第三步：只有确有提速，再跑短精度对照

```bash
python experiments/multimodal_v2/abc/efficiency_pilot.py \
  --action train --output outputs/lora_pilot_seed42 --device cuda:0

python experiments/multimodal_v2/abc/efficiency_pilot.py \
  --action eval --output outputs/lora_pilot_seed42 --device cuda:0

python experiments/multimodal_v2/abc/efficiency_report.py \
  --pilot outputs/lora_pilot_seed42 --phase train
```

默认六组各三轮，是速度与早期收敛筛选，不是最终的收敛精度排名。后续对有希望的方式用相同更长预算及多 seed 复验，然后才把选定方式统一推广到 A/B/C 正式对照。不要拿原有 12 轮 full 的最好结果与 3 轮 LoRA 做最终公平结论。

新增 `--regimes full heads lora`、`--variants control a_head` 可限制互不重叠的运行；中断恢复指定同一 run 并加 `--resume`。默认禁止覆盖已有训练。

## 如何读结果

每组 `train/efficiency/` 或 `benchmark/efficiency/` 下生成：

- `parameters.json`：总参数、训练参数、LoRA 参数、训练参数名称、设备、软件版本、batch/精度。
- `timing.jsonl`：epoch 总墙钟时间、去掉验证后的训练时间、验证时间、实际更新次数、样本数、samples/s、显存。
- `efficiency_<phase>_comparison.csv/json`：每种方案相对同方案 full 的训练提速、整轮提速、显存与逐 query 精度变化。

计时在 epoch/验证边界同步 CUDA，不在每个 batch 强制同步；训练时间包括数据读取、模型、候选选择、训练日志和训练指标，不冒充纯 GPU kernel 时间。epoch 峰值显存包含验证。独立评价和部分 checkpoint 回调开销不在 epoch 计时范围，不能用该数精确推断整场任务墙钟耗时。

多 epoch 报告优先排除第 0 轮，取后续 epoch 的中位数；若仅一轮/截断 benchmark，明确标记未排除首轮。首次 kernel 初始化可能影响短 benchmark，必要时重复同一配置的独立 benchmark。

相同硬件、软件版本、batch、精度、数据划分、A2 checkpoint、样本与更新次数才计算提速；不同 query 集或 GT 直接报错。缺少真实数据时不宣布最快且最好方案。

选择时先看“训练速度是否改善”，再看“整轮速度是否改善”，最后看 Top1@1m、RMSE、coverage 与错→对/对→错。若 LoRA 并不比 full 快，或只训练头已经兼顾速度精度，就不应仅因可训练参数少而强行使用 LoRA。
