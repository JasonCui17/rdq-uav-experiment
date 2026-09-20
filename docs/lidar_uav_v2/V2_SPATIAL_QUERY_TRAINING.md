# LiDAR V2 Spatial Query Training

The formal Spatial-only training and evaluation unit is one query. Each dataset
item contains one `query_time`, its latest 20 causal Avia/Mid360 events, and its
optional spatial target. Point-level `delta_t` and latest-four supervision
metadata remain unchanged.

The formal loaders directly consume `LiDARUAVDataset` and
`LiDARUAVValidationDataset` through `collate_lidar_samples`. They do not build
temporal clips, padding slots, occurrence restoration maps, or UQP layouts.
`TemporalQueryClipDataset`, temporal collate, and UQP remain only as archived
helpers and regression coverage for earlier experiments.

EQS now selects sequence-local query ordinals directly. With stride four,
epoch offsets 0, 1, 2, and 3 form a disjoint partition of all training queries.
Every four epochs therefore complete one coverage cycle. Full FP32 validation
runs at the end of each coverage cycle (epochs 4, 8, ..., 100), while `last.pt`
continues to update after every epoch.

The model and loss are unchanged:

`points -> hierarchy -> SBE/VQSA -> multiscale spatial encoder -> SparseUp -> CandidateHead -> candidate set`

`loss = loss_cls + 2 * loss_reg`

Queries without a positive recent-support voxel continue to skip spatial loss.
The VQSA CUDA batch-dimension chunk remains an execution-only correctness fix.
