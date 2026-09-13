# Convergent MMUAV 3D branch reconstruction

M0/M1/M2/M2.5 are frozen. M3 is bypassed. This is a reconstructed runnable branch,
not a claim of strict paper replication or challenge-metric alignment.

Public preprocessing reused unchanged: Mid360 nonoverlap20 + final20 overwrite,
DBSCAN(2,10), original 9D feature, M1 logits argmax; Livox zero removal/FPS100;
public fusion DBSCAN(1,10), then candidate DBSCAN(1,1). All candidates enter M2
FULL (immutable full-cluster mean, deterministic64-point sample) and tracker.
Existing fusion cache is read-only, checked against M1, split and data-root hashes.
GT-conditioned accepted IDs, candidate tables and sequence_records are NOT read.

StoneSoup tracker mirrors fusion_tracking.py defaults: six-state [x,vx,y,vy,z,vz],
ConstantVelocity(.15) x3, ExtendedKalmanPredictor/Updater, measurement noise .001,
Euclidean NearestNeighbour/missed_distance3, covariance deleter30, initiator1.
Engineering adaptations: initialize first nonempty frame, archive deleted tracks,
CSV instead of timestamp-file overwrites. Original association/update maths stays.

RECONSTRUCTED_DESIGN: select valid track with longest duration, then more measurement
updates; exact ties use initial XYZ. No GT. Public source has no final UAV selection.
AR3 per coordinate: OLS fitted solely on the selected predicted trajectory,
uniformized to .1s; no GT. Formal fitting runs only in the user's manual
full_temporal command. Fewer than8 observations fall back to fixed [2,-1,0,0].
Only interior gaps, maximum1s, are completed. Smoke disables AR fitting explicitly.
These coefficients and limits are reconstruction assumptions, not paper parameters.

Interpolation: public scipy interp1d linear math. B-spline: public splrep/splev,
fixed s=.5, cubic if >=4 states; short tracks retain linear interpolation.
Safety reconstruction: no unbounded extrapolation, support and gap1s masks.
Geometric/FULL have NO extra interpolation or smoothing: nearest state within .05s
is evaluated. FULL_TEMPORAL resamples at the SAME GT timestamp coordinates, without
reading GT XYZ until final predictions have been saved. This coverage change is
explicitly reported rather than comparing only good samples.

Outputs per sequence: all_candidate_measurements.csv, raw_tracker_trajectory.csv,
selected_trajectory.csv, completed_trajectory.csv, interpolated_trajectory.csv,
smoothed_final_trajectory.csv, metrics.json, and temporal AR coefficients.
Run outputs: run_config.json, overall_metrics.json, per_sequence_metrics.csv,
processing_failures.csv. Empty results count as zero coverage, not failure.
Failures are listed and cause nonzero exit; pooled metrics exclude failed sequences
explicitly. MSE_coordinate=mean across samples and XYZ; MSE_3D=3*MSE_coordinate.
Error is matched-only with missing counts and coverage alongside. Overall is
frame-pooled; sequence mean coverage also reported. Nonfinite predictions fail.

Only validation_sub is supported now. Heldout reading is prohibited by CLI.
No smoothing/tracker/AR search, no M3 or additional experiment matrix.
Single-sequence smoke is functional only, never a formal research result.

Functional smoke completed on validation seq0007 for all three modes. The source
fusion empty shape (0,) is accepted as valid empty, while nonempty invalid shapes
remain processing failures. Seven necessary tests passed. Smoke did not fit AR
parameters. Formal validation and AR fitting are manual-only; no background monitoring.
Frozen final reproduction config is deliberately NOT created before formal A/B/C
validation has been reviewed.
