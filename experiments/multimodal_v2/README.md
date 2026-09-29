# Multimodal V2 experiments

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

The first experiment is B2:

```text
LiDAR V2 candidates (XYZ is immutable) ─┐
                                        ├─ candidate geometry ─ cross evidence ─ bounded score delta
Swin-DINO candidates (2D box only) ─────┘
```

The old three-stage pre-Swin HCI, Fusion Decoder, and absolute XYZ prediction
head are absent. A V-only hypothesis has `has_xyz=False` and is excluded from
3D ranking. Missing 2D labels are masked, never converted to negatives.

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

`status` must be `PASS`; all XYZ and score max differences must be zero.

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

B3 remains disabled until an independently labeled 2D validation set is
large enough to support AP50, AP50:95, and small-object recall claims. The
current five valid validation boxes are exploratory only.
