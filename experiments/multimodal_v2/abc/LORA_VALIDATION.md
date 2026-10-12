# LoRA 交付验证

代码基线：refactor-v2-standalone 的 `6890f2e10d272c029b1dcdbb31e6eb03371b9ec8`，叠加上一版 ABC 实验补丁。

## 已执行

- `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q experiments/multimodal_v2/tests`：149 passed，1 skipped（CUDA），12 warnings。
- Python compileall 与 git diff --check 通过。
- 真实训练入口的 CPU 合成数据端到端检查：control / a_head × full / heads / lora，共六组；各运行两轮，完成保存、独立重载评价及汇总，报告六行且无缺失。
- 六组分别执行独立的 4-update benchmark，实际更新次数均为 4；计时汇总通过，未复用精度实验的训练结果。生产默认计时预算为 200 updates。
- 检查零初始化 LoRA 与原模型输出恒等，冻结参数在优化后不变，LoRA A/B 和预测头获得梯度，严格 checkpoint 重载输出一致，冻结前缀与适配器之后的梯度路径符合预期。

运行环境：Python 3.12、PyTorch 2.14.1+cu130、Lightning 2.6.6；实际计算设备为 CPU。

## 尚未验证

没有真实 MMAUD 数据、用户原 A2 checkpoint 或可用 GPU。合成检查使用随机初始化 A2 权重，只验证代码路径；合成数据的精度和耗时均不能用于判断实验优劣。尚不能给出 LoRA 在服务器上的真实每轮耗时、显存节省、Top1@1m 或最终获胜方案。

先按 LORA_Design_and_Run.md 运行六组计时，再进行同预算短精度筛选；最终用足够训练预算与多 seed 复验。
