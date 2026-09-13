# M2 frozen / M2.5 contribution ablation

Evaluation mode: `gt_conditioned_module_level`. Same 2359 frozen validation IDs;
heldout remains unread. M2_FULL uses existing best_val_loss.pth (epoch 3), not last
(epoch 18). Frozen file hashes and offline diagnostics are recorded in
results/mmuav_reproduction/m2_frozen_summary.json. Original M2 files are unchanged.

Variants share data, association 2m, train-frozen timestamp tolerance, 64 points,
batch 64, Adam 0.001, MSE residual target, seed 42, 100 epochs and patience 15.
FULL: point MLP 3-64-128-256, max pool + observed center, head 259-128-64-3.
POINTS_ONLY: same point MLP, head 256-128-64-3; absolute center never enters forward.
CENTER_ONLY: observed center, head 3-128-64-3; dataset does not open point shards.
All outputs are residual XYZ added to immutable full-cluster geometric center.
GT is only the target, never a neural-network input. FULL checkpoint names and
parameter shapes remain compatible. Different variant capacities are explicit;
these are input-contribution ablations, not parameter-count-matched models.

Offline diagnostics: outputs/mmuav_paper_reproduction/center_regression/m2_diagnostics/.
Unchanged sample tolerance: 1e-8 m. Sequence rankings use relative coordinate-MSE
reduction; degradation list includes only genuinely negative reductions.
Paper 0.27 to 0.05 is numerically close, not strict reproduction: metric alignment,
association and architecture are unresolved. No M3 is implemented or trained.

Each manual training run saves its own same-ID geometric/regressed comparison.
The offline m2_5_comparison.csv is a template: untrained variants remain blank,
not fabricated results. Compare precision of regression only on this accepted
oracle subset, never interpret as system detection performance.
