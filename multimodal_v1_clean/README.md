# Multimodal V1.1 — clean training project

This directory is the compact, runnable extraction of the final Multimodal
V1.1 E5 model from the parent `rdq-uav-experiment` workspace. Historical
audits, intermediate experiments, generated outputs, patches and unrelated
models are intentionally excluded.

## Final data flow

```text
causal Avia + Mid360 history
  -> frozen LiDAR V2 spatial pyramid (SBE-Lite / VQSA / EOOE)
RGB left image
  -> DINO Swin-T visual pyramid
three pre-stage geometry HCI blocks
  -> LiDAR and RGB candidate sets
  -> geometry association and reliability gate
  -> typed shared queries and fusion decoder
  -> fused 2D box + 3D XYZ candidates
```

Training uses the frozen V1.1 objective `L_R + L_V + L_F`. Samples without a
verified 2D box are removed from DINO's supervised criterion; they are not
treated as background negatives. LiDAR supervision and eligible fusion
supervision remain active.

## Kept structure

- `src/rdq_uav/multimodal_v1/`: P1–P8 multimodal implementation.
- `src/rdq_uav/lidar_v2/`: exact LiDAR spatial backbone dependency.
- `src/rdq_uav/multimodal/merged_lidar.py`: causal Avia/Mid360 event loader.
- `tools/train_multimodal_v1_full.py`: full E5 trainer and two-update gate.
- `configs/`: final E5, candidate, LiDAR and camera configuration.
- `calibration/`: geometry calibration consumed by projection/HCI.
- `manifests/`: verified 2D supervision records.
- `splits/`: reproducible train/validation/test sequence split.
- `tests/`: focused P1–P8 and LiDAR geometry regression tests.

## External assets

Large assets are linked instead of copied:

- `third_party/detrex`
- `checkpoints/dino_swin_t`
- `checkpoints/lidar_v2/best_spatial.pt`
- `data/mmaud_official_train`

Run the checker before training:

```bash
cd /home/jasoncui/projects/rdq-uav-experiment/multimodal_v1_clean
python tools/check_assets.py
```

If this directory is moved to another machine, recreate these links or replace
them with real directories/files at the same paths. Dataset and checkpoints
are runtime assets and are not part of `SOURCE_SHA256.txt`.

## Environment

The verified environment is Python 3.10 with PyTorch 2.1.1 + CUDA 11.8. The
parent repository's detrex/detectron2 source is used directly. To install this
clean package into the existing environment:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests
```

## Required two-update gate

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 \
python tools/train_multimodal_v1_full.py \
  --config configs/multimodal_v1/e5_full_v1.yaml \
  --output outputs/e5_gate2 \
  --device cuda:0 \
  --max-updates 2
```

The gate output must report `status=PASS` and `optimizer_updates=2`.

## Full E5 training

Start the full run from the audited initial checkpoints, using a different
output directory from the gate:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 \
python tools/train_multimodal_v1_full.py \
  --config configs/multimodal_v1/e5_full_v1.yaml \
  --output outputs/e5_full_v1_seed42 \
  --device cuda:0
```

Resume an interrupted completed-epoch checkpoint with:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 \
python tools/train_multimodal_v1_full.py \
  --config configs/multimodal_v1/e5_full_v1.yaml \
  --output outputs/e5_full_v1_seed42 \
  --device cuda:0 \
  --resume auto
```

The trainer writes effective configs, `training_log.csv`, validation JSONL,
`last.pt`, `best.pt`, and `run_summary.json` under the selected output path.

## Scope

This extraction is a snapshot, not a second source of truth. Changes made here
do not automatically propagate to the parent workspace. No historical outputs
or earlier architecture drafts are included.
