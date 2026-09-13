# Known limitations

1. 论文 7D dynamic feature 定义未完全恢复；本轮使用 9D。
2. 论文 center regression 的 PointNet/MLP 具体结构有歧义，本轮为明确标记的重建。
3. 论文 24D third-order polynomial 定义未恢复，M3 在本轮旁路。
4. Paper Pose MSE 与本地 MSE_coord / MSE_3D 口径未对齐，不能据此判定优劣。
5. 本地 heldout 是官方 train 的 sequence-level 固定留出集，不是官方 challenge test。
