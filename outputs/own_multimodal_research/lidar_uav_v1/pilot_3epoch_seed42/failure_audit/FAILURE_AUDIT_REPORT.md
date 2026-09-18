# LiDAR UAV V1 Failure Audit

## Scope and consistency

Read-only audit of all 1600 exported validation samples. The exact production `ValidationReferenceAdapter`, merged-stream ordering, causal last-20 selection, released XYZ cleaning, sensor IDs, and exported candidates were reused. No model inference, fitting, or training was performed.

- Future-event violations: 0
- GT/export mismatches: 0
- Recomputed/exported current-support mismatches: 0
- Empty outputs in all validation samples: 23

`current_support` has its original V1 meaning: at least one point within 1 m of current GT among the last four selected LiDAR events. Support in older events does not make a sample current-supported.

## Catastrophic failures (>5 m)

Finite Top1 errors above 5 m: **90**. Empty outputs are separate and therefore contribute **0** rows to this finite-error table.

- History contains no 1 m GT support: **20**
- History has support, but the last four events do not: **6**
- A correct <=1 m candidate exists in Raw Top10 while Top1 is >5 m: **49**
- Raw Top10 has a correct candidate that NMS Top10 removes: **0**
- Current support exists but Raw Top10 has no <=1 m candidate: **21**

**64/90 (71.1%)** catastrophic samples still have current support, and **70/90** have support somewhere in the 20-event history. Therefore temporal observation absence explains a material minority, not the majority, of the finite catastrophic errors.

For the 49 ranking failures, the first correct Raw candidate has median rank **5** and the median Top1-minus-correct score gap is **0.0645**. NMS removes none of these 1 m candidates from NMS Top10.

Primary types use this precedence: EMPTY_INPUT, NO_HISTORY_SUPPORT, NMS_REMOVED_GOOD_CANDIDATE, CANDIDATE_EXISTS_RANKING_FAIL, CURRENT_SUPPORT_BUT_GENERATION_FAIL, STALE_HISTORY_SUPPORT, then unclassified generation failure. Counts: `{"CANDIDATE_EXISTS_RANKING_FAIL": 49, "CURRENT_SUPPORT_BUT_GENERATION_FAIL": 21, "NO_HISTORY_SUPPORT": 20}`.

## Last support age

Only non-empty samples with at least one historical 1 m support event enter this table.

| Last support age | Samples | Top1@1m | >5m rate | Median error | P90 error |
|---|---:|---:|---:|---:|---:|
| 0-0.05s | 614 | 0.948 | 0.046 | 0.304 | 0.720 |
| 0.05-0.10s | 642 | 0.953 | 0.036 | 0.289 | 0.725 |
| 0.10-0.20s | 133 | 0.902 | 0.083 | 0.335 | 0.926 |
| 0.20-0.50s | 34 | 0.529 | 0.235 | 0.944 | 29.947 |
| 0.50-1.00s | 0 | nan | nan | nan | nan |
| >1.00s | 0 | nan | nan | nan | nan |

The bucket statistics show no degradation between 0 and 0.10 s. Performance begins weakening in 0.10-0.20 s and changes sharply in the observed 0.20-0.50 s bucket: Top1@1m falls from 0.902 to 0.529 and the catastrophic rate rises from 0.083 to 0.235. There are no eligible samples above 0.50 s, so this audit cannot support claims beyond that range. Age is not sufficient by itself: 1256 samples below 0.10 s still include 51 catastrophic outputs.

## Focus sequences

| Sequence | Current support | Any history support | Mean support age | Median points | Avia point fraction | Top1@1m | >5m count | Ranking fail | Current-support generation fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq0004 | 0.640 | 0.820 | 0.137 | 46012 | 0.000 | 0.670 | 26 | 10 | 0 |
| seq0005 | 0.950 | 0.990 | 0.091 | 44759 | 0.000 | 0.700 | 29 | 20 | 9 |
| seq0006 | 0.990 | 0.990 | 0.036 | 48988 | 0.028 | 0.850 | 14 | 11 | 3 |
| seq0008 | 0.930 | 0.970 | 0.053 | 48769 | 0.000 | 0.780 | 15 | 4 | 9 |

The four focus sequences contain **84/90** catastrophic errors. Their mechanisms differ:

- `seq0004` is observation-limited: current support is only 64%, any-history support is 82%, and 16/26 failures have no historical 1 m support. Its other 10 failures are ranking failures.
- `seq0005` is not primarily support-limited: current support is 95%; all 29 catastrophic samples have current support. Twenty are ranking failures and nine are generation failures despite current support.
- `seq0006` has 99% current support; all 14 failures have current support, split into 11 ranking and three generation failures.
- `seq0008` has 93% current support; among 15 failures, nine are current-support generation failures, four are ranking failures, and two have no history support.

The failures occur in contiguous temporal runs rather than predominantly isolated points. Run lengths for the focus sequences are `seq0004: 16;1;4;5, seq0005: 2;14;1;1;3;4;4, seq0006: 14, seq0008: 2;13`. The complete 16-sequence comparison is in `per_sequence_failure_diagnosis.csv`, and run statistics are in `catastrophic_run_summary.csv`. The Avia fraction is close to zero in several good and bad sequences, so that stream composition alone does not separate the failures.

## Interpretation boundary

This report separates temporal observation absence, candidate generation, ranking, NMS, and empty input using exported facts. It does not claim a sensor detection rate and does not propose or evaluate model changes. Of the 90 catastrophic errors, the direct exported-output diagnoses are 49 ranking failures, 21 current-support generation failures, and 20 no-history-support cases. NMS accounts for none. The exact 2-5 m gap follows from this discrete selection behavior: 1487 samples select a candidate within 2 m, while every remaining non-empty Top1 selection is a distant background candidate above 5 m; no sample selects an intermediate candidate. The case studies show the corresponding GT-local cluster and distant Top1 locations.
