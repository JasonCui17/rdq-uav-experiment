# Method takeaways

1. **Geometric center 不是可靠的最终目标位置。** 稀疏/残缺 cluster 的均值不能直接视作 UAV center；本轮系统轨迹也含明显长尾失败。
2. **学习式 center correction 在三个层次有效。** M2 模块级、validation 系统级，以及 heldout 共同时间戳对照均显示收益；不是仅凭数量近似进行比较。
3. **Observed absolute position 具有很强的校正价值。** M2.5 中 CENTER_ONLY 接近 FULL，POINTS_ONLY 也改善，FULL 最佳；支持位置先验和局部结构互补，不证明已经学到 UAV 特有形状或全部因果来源。
4. **Temporal processing 主要改善完整性。** Coverage 必须与 available-prediction error 分开报告；共同时间戳误差只是辅助，不能替代 coverage，更不能将组合收益单独归因于 AR。
5. **向 Radar+RGB 迁移的是研究假设，不是已证实的雷达结论。** 不直接使用雷达/LiDAR cluster centroid 作为最终 3D 表示：Radar cluster geometry + observed spatial position prior + RGB target feature → learned target-center correction → corrected 3D position。Temporal module 独立负责连续性。Radar target association、坐标关系与可观测性仍需在自己的任务中验证。

本轮仅完成重建和固定留出评价；不做新实验、不调参，不以未知口径匹配论文数值。
