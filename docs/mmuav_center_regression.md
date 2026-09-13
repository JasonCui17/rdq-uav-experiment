# M2: GT-conditioned module-level center regression

`evaluation_mode = gt_conditioned_module_level`. Both training and validation
use GT nearest-candidate association to study center accuracy conditional on a
correctly associated target cluster. These are NOT system-level detection or
trajectory results. All candidate points, rejected candidates and unsuccessful
timestamp pairings remain stored. GT never enters clustering or M1 classification.

RECONSTRUCTED DESIGN: regression is inserted after public LiDAR fusion and
per-frame DBSCAN, before tracker measurements. The paper's exact insertion
location remains unresolved.

Mid360 inference: eps=2/min_samples=10, non-overlap 20-frame accumulation plus
the source final-20 overwrite behavior, no motion compensation. M1 was trained
with eps=1: its perfect classifier validation score does not establish M2
candidate recall. Only M1 logits select clusters, never attention weights.
Avia retains zero removal and source FPS(max_pts=100). Public fusion retains
its timestamp merging, eps=1/min_samples=10 and exclusive 50,000 point ceiling.
Candidate DBSCAN uses eps=1/min_samples=1. Valid empty results are legal;
missing input/model/shape/processing failures must stop, not imply no target.

M2 first pairing ceiling: 2m, selected from train_sub seq0001 pilot nearest
candidate distances (median 1.160m, maximum 1.485m), with margin for partial
returns, before full validation statistics are examined. This is a reconstructed
quality ceiling, not proof of true target identity. Pilot validation was already
smoke-tested but was NOT used to choose this ceiling. Full validation uses it
unchanged. All excluded candidates remain available for later system evaluation.

All valid candidate points determine the geometric mean before any sampling.
Candidates are stored in per-sequence numeric point buffers and offsets, and
all_candidates.csv records oracle associations and rejection reasons. Public
fusion does not retain source identity; this is explicitly marked unknown.

Timestamp tolerance is frozen from training P95 nearest-GT gaps. Spatial pairing
quality threshold is an explicit training-only decision; cumulative 0.5/1/2/3m
tables are audits, not experiments or paper-score matching. Accepted-subset MSE
is conditioned on that ceiling. Validation cannot select or alter thresholds.

PointNet-like PAPER-INSPIRED RECONSTRUCTION: local XYZ -> shared 3/64/128/256
MLP/ReLU -> max pooling -> concat observed geometric center -> 259/128/64/3.
Absolute geometric center is an observed auxiliary input, not GT; it may permit
learning device/location-dependent bias. Model outputs a residual; target is
GT minus the full-cluster center. Default 64 points, random training sampling,
deterministic validation sampling, replacement for small clusters.

Adam 1e-3, batch64, max100 epochs, patience15, validation residual MSE stopping
are RECONSTRUCTED training parameters. No heldout data are read. Geometric and
PointNet scores use exactly the frozen validation sample IDs; invalid model
predictions raise rather than being removed.

MSE_coord averages all XYZ coordinates; MSE_3D averages squared Euclidean error
(three times MSE_coord). Paper 0.27 -> 0.05 is recorded only in paper_reference.json;
metric and selection alignment remain unresolved. Range grouping is disabled:
DISABLED_REFERENCE_FRAME_UNVERIFIED. No calibration transform is fitted/applied.
