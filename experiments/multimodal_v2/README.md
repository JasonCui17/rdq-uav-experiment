# Multimodal V2 experiments

> 当前数据接口以 [对称模态 Batch 契约](reports/SYMMETRIC_MODALITY_BATCH_20261002.md) 为准：缺失观测在 Sample 为 None；collate 只收集有效模态，使用 radar_batch_index / vision_batch_index；模型跳过缺失分支。下文早期 placeholder 与全 B 图像 Shape 已被替代。本轮运行验证待设备可用后进行。 后续已移除事件数量限制的点级监督 mask，雷达监督使用全部输入点；详见 [全输入点监督清理](reports/ALL_INPUT_POINT_SUPERVISION_20261002.md)。 雷达特征现使用 [时间加权 SBE](reports/TIME_WEIGHTED_SBE_20261002.md)，旧雷达 checkpoint 的输入统计语义已改变，需重新验证精度。

All experiment-only entry points, configurations, tests, diagnostics, and
reports live here. The reusable model is confined to
`src/rdq_uav/multimodal_v2/`; no new root-level `tools/` or `tests/` files are
required.

After checking out this branch in a fresh clone, initialize the pinned detrex
gitlink before any DINO command:

```bash
git submodule update --init --recursive
```

## Scientific boundary

The first trainable experiment is B2:

```text
LiDAR V2 candidates (XYZ is immutable) ─┐
                                        ├─ candidate geometry ─ cross evidence ─ bounded score delta
Swin-DINO candidates (2D box only) ─────┘
```

The old three-stage pre-Swin HCI, Fusion Decoder, and absolute XYZ prediction
head are absent. Association is a deterministic, rejectable one-to-one greedy
match over calibrated image geometry: pairs farther than 16 source pixels from
the box are rejected and ties use the original R/V source indices.

The output is intentionally two-task:

- R has immutable LiDAR `xyz_m`, `has_xyz=True`, `has_box=False`, and only a
  `score_3d_*` value.
- V has the DINO box, `has_box=True`, `has_xyz=False`, and only a
  `score_2d_*` value.
- RV carries the LiDAR XYZ and DINO box while retaining independent 3D and 2D
  before/after scores.

`top3d_indices()` sees only R/RV and `top2d_indices()` sees only V/RV. Zero
storage never establishes validity. Missing 2D labels are masked, never
converted to negatives.

## B2 and B3 boundary

B2 freezes LiDAR V2, Swin-DINO, both candidate builders, V-query radar
attention, and the 2D score head. It trains only the R-query visual attention,
the V0/V1 feature projections, and the zero-initialized 3D score head. A
projected R candidate reads a 3x3 feature-cell neighborhood from both V0 and V1
even when it is not associated with a DINO box. Invalid projection, padding,
or missing RGB produces an exact base-score identity.

B3 is implemented as a dormant interface: every V/RV box can read at most 16
R candidates within 16 pixels of the box, but its attention and 2D score head
remain frozen and disabled in the B2 config.

## Reused audited components

- LiDAR V2 and `CandidateSelector` preserve the independent spatial checkpoint,
  candidate XYZ, feature, score, and source-token identity.
- V1 `SwinPyramidAdapter`, `DINOAdapter`, and `RGBCandidateBuilder` preserve the
  audited DINO execution, SSOD weights, padding masks, and object-query
  candidate semantics.
- V2 owns query-time Sample construction and collate in `data.py`. Radar uses
  the inclusive `[t-radar_history_s,t]` window; RGB uses the latest historical raw timestamp
  within `[t-max_image_gap_s,t]` (future RGB is forbidden). `max_events` remains only
  for the historical LiDAR-only E0 diagnostic.
- `geometry.py` owns ProjectionContext and the migrated, unchanged calibrated
  projection mathematics. V2 no longer uses the V1 InteractionContext wrapper.

V2 owns association, bidirectional candidate evidence, task-separated scoring,
and ranking loss. It does not call V1 candidate association.

## Assets

The default config expects these local, Git-ignored assets:

- `data/mmaud_official_train`
- `checkpoints/lidar_v2/best_spatial.pt`
- `checkpoints/dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth`
- `checkpoints/multimodal_v1/e5_last.ckpt`

The E5 Last checkpoint contributes only its SSOD-trained DINO weights and RGB
candidate projection; its failed fusion output is never loaded. LiDAR always loads the independent V2 spatial
checkpoint strictly. No E5 HCI, fusion, XYZ head, or radar parameters are
loaded.

## Gates before training

Run the isolated CPU contract tests:

```bash
PYTHONPATH=src python -m pytest -q experiments/multimodal_v2/tests
```

Run a real GPU B0/B1 identity gate (no optimizer step):

```bash
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/check_b0_b1.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --device cuda:0 --samples 2 \
  --output outputs/own_multimodal_research/multimodal_v2/b0_b1_gate
```

`status` must be `PASS`; XYZ, boxes, both task scores, and both task orderings
must be exactly equal.

Run the independent E0 baseline on all 4,800 validation queries:

```bash
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/evaluate_e0.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --device cuda:0 \
  --output outputs/own_multimodal_research/multimodal_v2/e0_full4800
```

Run the E5 Best/Last full candidate diagnosis. Do not pass `--limit` for the
formal 4,800-query audit:

```bash
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/diagnose_e5.py \
  --config outputs/own_multimodal_research/multimodal_v1/e5_annotated20_4090_seed42/effective_config.yaml \
  --best outputs/own_multimodal_research/multimodal_v1/e5_annotated20_4090_seed42/checkpoints/best.ckpt \
  --last outputs/own_multimodal_research/multimodal_v1/e5_annotated20_4090_seed42/checkpoints/last.ckpt \
  --device cuda:0 --expected-best-success-1m 0.839583 \
  --expected-last-success-1m 0.205833 \
  --output outputs/own_multimodal_research/multimodal_v2/e5_full4800_diagnosis
```

## Manual short training gate

The assistant does not execute training. Run two optimizer updates manually:

```bash
mkdir -p outputs/own_multimodal_research/multimodal_v2
PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/multimodal_v2/train.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --max-updates 2 --train-limit 32 --val-limit 32 \
  --output outputs/own_multimodal_research/multimodal_v2/b2_two_update_gate
```

Evaluate B0 and B1 before a formal B2 run:

```bash
for MODE in B0 B1; do
  PYTHONPATH=src python experiments/multimodal_v2/evaluate.py \
    --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
    --mode "$MODE" --device cuda:0 \
    --output "outputs/own_multimodal_research/multimodal_v2/${MODE,,}_full4800"
done
```

## Manual formal B2 training

After the gates pass:

```bash
mkdir -p outputs/own_multimodal_research/multimodal_v2
PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/multimodal_v2/train.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --output outputs/own_multimodal_research/multimodal_v2/b2_seed42 \
  2>&1 | tee outputs/own_multimodal_research/multimodal_v2/b2_seed42_train.log
```

Resume with `--resume auto`. Then evaluate the checkpoint in FP32:

```bash
PYTHONPATH=src python experiments/multimodal_v2/evaluate.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --mode B2 --checkpoint outputs/own_multimodal_research/multimodal_v2/b2_seed42/checkpoints/best.ckpt \
  --device cuda:0 --output outputs/own_multimodal_research/multimodal_v2/b2_seed42/evaluation
```

B2 loss uses positive candidates at <=1m, ignores (1m,2m], and treats >2m as
negative. Queries with negatives but no positive contribute a separately
normalized focal loss at weight 0.25; no-candidate and missing-GT queries are
reported and skipped. Batches without a trainable loss return `None` through
Lightning rather than aborting the epoch.

B3 remains disabled until an independently labeled 2D validation set is
large enough to support AP50, AP50:95, and small-object recall claims. The
current five valid validation boxes are exploratory only.

There is no trained checkpoint from the earlier V2 implementation. New V2
checkpoints use one canonical DINO prefix, `vision.dino.detector.*`; E5 source
checkpoints are still imported by the explicit compatibility loader.

## Query-time data contract (2026-10-02)

See [data refactor audit and progress](reports/DATA_REFACTOR_PROGRESS_20261002.md)
for field shapes, compatibility changes, verification and remaining gates.
`build_datasets()` now returns `(train, val)`, not a wrapped LiDAR Dataset.
The training Dataset removes both-missing queries during initialization; the
validation Dataset retains them. Absolute time and file identities stay in the
audit batch and do not enter the model input dictionary.

Before new training, run the read-only real-data smoke on a sequence that is
present locally:

```bash
PYTHONPATH=src python experiments/multimodal_v2/diagnostics/check_data_samples.py \
  --config experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml \
  --sequence seq0001 --samples 3 \
  --output outputs/own_multimodal_research/multimodal_v2/data_1s_smoke.json
```

Run B0/B1 identity and full B0/B1 again with separate output directories. The
previous 83.4167% B0 result used the old input definition and is historical;
new B2 results require a B0 comparison under this same one-second definition.
Historical E0 retains its latest-20 definition and is not an input-aligned
replacement for the new B0 baseline.


## 2026-10-02: direct YOLO supervision (implementation pending verification)

V2 no longer consumes annotation_manifest or a bbox mapping dictionary.
For the selected causal historical image, read
`<root>/<sequence>/<label_directory>/<image_stem>.txt`. Default directory:
`2d_detect`. All directory boxes are trusted supervision as instructed by
the user. Convert YOLO class/cx/cy/w/h using the calibrated left source image
size, consistent with gt_bbox_annotator (including side-by-side PNG crops).
Missing/empty files give zero box and gt_2d_valid=False, never implicit negative
supervision. Malformed/out-of-frame or multiple-box files raise explicit errors:
the current target contract supports one UAV box. Historical image selection,
modality masks and train-only both-missing filtering remain unchanged.
Tests and the smoke entry were updated; no tests/smoke were run for this revision
per the user's instruction to defer verification until equipment is available.
Earlier passing test totals apply to the preceding manifest-based revision.

Current Dataset constructor removes the manifest argument. Configuration uses
`data.label_directory: 2d_detect`; label changes take effect at the next read.

## 2026-10-03 Query and geometry update

Dataset receives `QueryRecord` objects and an optional independent exact-time
3D target index. `build_datasets` still uses GT filenames as the offline query
source; manually supplied queries work without GT. See
[implementation report](reports/QUERY_CANDIDATE_ADAPTIVE_GATE_20261003.md).
Current candidates: pre-NMS 50, final at most 10 per modality/sample. The shared
V1 p6 candidate configuration was also updated. V2 uses `model.geometry_gate`
for both V<-R admission and association (inverse range, 20m=16 source pixels,
8..48px); old `box_margin_px`/V1 `geometry_gate_px` do not configure V2 anymore.
Time-weighted SBE is already implemented with 0.2s half-life; it was not changed
in this patch. Runtime and real-data validation remain pending.
