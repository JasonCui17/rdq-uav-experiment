# 验证记录：2026-10-12

代码基线：6890f2e10d272c029b1dcdbb31e6eb03371b9ec8。

## 已执行

- Python 编译检查：通过。
- git diff --check：通过。
- 完整 experiments/multimodal_v2/tests：142 passed, 1 skipped。跳过项需要 CUDA FP16。
- 新增 A/B/C 测试覆盖 10 个配置，每种配置执行两次 AdamW 更新，验证有限梯度及严格模型重载。
- 六组端到端流程：baseline、control、a_soft、a_head、b_both、c_query。
- 每组通过实际 train.py 的 2-update gate，独立正式输出目录的 1 epoch 训练、best checkpoint 保存，以及实际 evaluate.py 的完整 checkpoint 重载与逐 query 导出。
- compare.py 成功读取六组输出，严格配对同一批 6 个验证 query，missing_runs 为空。

## 测试环境与数据

- CPU；Python 3.12；PyTorch 2.14.1+cpu、torchvision 0.29.1+cpu、Lightning 2.6.6。
- 原项目服务器依赖是 Python 3.10/PyTorch 2.1.1+cu118；尚未在该环境验证 CUDA。
- 合成数据采用仓库的 20-sequence 划分，每序列 2 个 query；28 个训练 query、6 个验证 query。目标点簇加远处杂波，无 RGB。
- warmstart 由未训练 A2 结构生成，仅验证参数映射；不是用户真实 checkpoint。
- 不将合成数据的方案排序或数值写成真实实验结果。

## 尚未执行

- MMAUD 全量训练、真实 A2 warmstart、GPU 显存/吞吐测量。
- 真实 validation_sub 的 A/B/C 优劣比较、三 seed 重复实验。
- heldout_test_sub 最终测试及接回 B2/B3 后的多模态比较。

当前不能声明 A、B、C 中任何方案效果最好。下一步需要在含 MMAUD 和 A2 checkpoint 的服务器执行 README 的 prepare/gate/train/eval/compare 命令。
