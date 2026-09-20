# LiDAR UAV V2 Spatial-Only Refactor

LiDAR V2 now ends at a dense set of spatial candidates. The query-level
`CandidateAwareQueryPool`, continuous query-time embedding, presence embedding,
causal temporal transformer, absolute temporal XYZ head, and temporal position
loss were removed after the learnability audit showed that spatial Top1 could
fit the target while soft pooling placed almost no weight on the GT-near
candidate.

The spatial architecture is unchanged: HierarchyBuilder, SBE-Lite, VQSA,
SpatialTransformer, EOOE SparseMerge, SparseUp, CandidateHead, residual codec,
and CandidateSelector retain their existing definitions. Query time and
point-level `delta_t` still define the causal latest-20-event input. Recent4
remains supervision/evaluation metadata only.

EQS, clip batching, and UQP remain training compute mechanisms. They select
and deduplicate spatial queries without creating a query-level temporal model.

The current public data flow is:

`points -> hierarchy -> SBE/VQSA -> multiscale spatial encoder -> SparseUp -> CandidateHead -> candidate set`

Historical Query-Causal specifications, smoke reports, and diagnostic
checkpoints remain archived as evidence for this removal decision.
