# Radar L0 embedding ablation

仅切换 L0 embedding；radar_sbe.py、几何 hierarchy、encoder0/1/2、merge、up、CandidateHead、loss、data 不改。

| 方案 | 文件 | 输入处理 | 输出 |
|---|---|---|---|
| A0 | radar_sbe.py（原实现保留） | 8×11 统计 + VQSA16 | [V0,128] |
| A1 | lidar_legacy.py | learned sensor + local xyz + delta_t；5→32→64；whole-voxel max/mean/logcount | [V0,128] |
| A2-16 | lidar_learned_sbe.py | 5→16；8 slots，每槽 mean/max/logcount/occupancy；VQSA16；288→128 | [V0,128] |
| A2-32 | lidar_learned_sbe.py | 5→32；8 slots，每槽 mean/max/logcount/occupancy；VQSA16；544→128 | [V0,128] |

Python 模块使用下划线，配置 embedding 分别为 sbe_lite、legacy、learned_sbe。
A2 为新的实验方案，不能声称是历史 Legacy 的精确恢复。A1 按当前仓库旧 LegacyVoxelEmbed 数学和网络实现恢复；这不自动恢复旧训练协议。
A2 的 delta_t 是 learned point 输入，没有 A0 的显式指数时间加权。A2 相对 A0 改了统计、时间处理与前端容量，结果只能归因于前端整体，不能单独归因于 Point MLP。
输出为每 voxel，而非每 point；slot 统计是 [V0,8,Dslot]。

新增配置启用 matched_frontend_init 和 frontend_seed=42，隔离前端随机数消耗，使相同全局 seed 下三种模型的下游初始参数相同。原默认配置不变。
所有实验随机初始化，不使用旧 checkpoint，不使用 --resume。原 75.92% 来自包含恢复操作的运行，应重新跑 A0，先比较 A0/A1，然后 A2。
训练相同 split、batch=4、accumulate=1、precision=32-true、seed=42、epochs=12 和优化器。保持相同 workers、设备及数据。比较验证集 best，同时报告 last、Top1@1m 和 oracle recall@10；不依据 heldout 选择结构。

## WSL 应用
解压到仓库外，运行：
```bash
cd /home/jasoncui/projects/rdq-uav-experiment
git status --short
git rev-parse HEAD
python ~/Downloads/radar_embedding_ab/apply_to_repo.py --repo "$PWD"
python ~/Downloads/radar_embedding_ab/apply_to_repo.py --repo "$PWD" --apply
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests/test_radar_embedding_ab.py
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests
git diff --check
git add src/rdq_uav/multimodal_v2/radar_model.py src/rdq_uav/multimodal_v2/lidar_legacy.py src/rdq_uav/multimodal_v2/lidar_learned_sbe.py experiments/multimodal_v2/configs/radar_a*.yaml experiments/multimodal_v2/configs/b0_a*.yaml experiments/multimodal_v2/tests/test_radar_embedding_ab.py experiments/multimodal_v2/reports/RADAR_EMBEDDING_AB_20261005.md
git commit -m "feat: add controlled radar voxel embedding ablations"
git push origin refactor-v2-standalone
```
安装前会检查全部文件和 Python 语法，有冲突则停止，不覆盖冲突文件。实际本地旧代码备份在 outputs/backup_radar_embedding_ab_时间/，含 lidar_11.py。备份不需要提交；Git 历史保留原提交。包 original_files 另含下载时原版本。

## 服务器
```bash
cd /root/autodl-tmp/rdq-uav-experiment-v2
git status --short
git pull --ff-only origin refactor-v2-standalone
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH=src python experiments/multimodal_v2/train.py --config experiments/multimodal_v2/configs/b0_a0_sbe.yaml --output outputs/b0_a0_sbe_seed42 --accelerator gpu --devices 1 --precision 32-true --batch-size 4 --accumulate 1 --num-workers 2
PYTHONPATH=src python experiments/multimodal_v2/train.py --config experiments/multimodal_v2/configs/b0_a1_legacy.yaml --output outputs/b0_a1_legacy_seed42 --accelerator gpu --devices 1 --precision 32-true --batch-size 4 --accumulate 1 --num-workers 2
```
A2 用 b0_a2_learned_sbe16.yaml 或 b0_a2_learned_sbe32.yaml，输出分别用新的同名目录。顺序运行；输出目录若已存在旧实验，请换目录，不自动恢复。
服务器 data/mmaud_official_train 必须仍正确指向真实数据。

## 验证状态
本交付环境未安装 PyTorch。仅进行了静态语法、下游源码一致性、安装器 dry-run/实际应用/重复应用/本地非冲突修改保留检查。新增单元测试涵盖 shape、梯度、旧前端数学、slot 空位/批次隔离、点排列不变性、下游初始化一致性及配置一致性，但尚未执行。没有启动 GPU 训练，也没有性能结论。
