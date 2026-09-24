# Multimodal V1 Lightning training

This is an independent training path for the frozen E5 architecture. The
original `tools/train_multimodal_v1_full.py` remains the reference and is not
replaced. The Lightning wrapper calls the same dataset, batch preparation,
model construction, `L_R + L_V + L_F`, validation outcome, and T1/T2/T3 helper
functions.

The implementation follows the installed K-Dense PyTorch Lightning skill:

- `MultimodalV1LightningModule` owns train/validation steps, optimizer,
  warmup-cosine scheduler, FP32 validation, stage switching, and checkpoint
  metadata.
- `MultimodalV1DataModule` owns deterministic DataLoaders and generator state.
- Trainer owns AMP, accumulation, gradient clipping, seeds, and native resume.
- callbacks provide last/best checkpoints, LR logging, early stopping, and GPU
  peak-memory logging.

## Checkpoints

`checkpoints/last.ckpt` is a Lightning-native complete checkpoint and resumes
with `--resume auto`. It is refreshed every 250 optimizer updates and when
Lightning catches an exception. `checkpoints/best.ckpt` preserves the original
lexicographic policy: maximize 3D Success@1m, then minimize median error.

An original E5 `.pt` can initialize network weights with
`--legacy-checkpoint PATH`. This is intentionally weight-only migration;
optimizer/loop state resumes only from a native `.ckpt`, avoiding unsafe state
mapping between loop implementations.

## Gates

Run static/unit gates first:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_multimodal_v1_lightning.py tests/test_multimodal_v1_p8_training.py
```

On the RTX 3070, run a real one-batch train/validation gate:

```bash
PYTHONPATH=src python \
  tools/train_multimodal_v1_lightning.py --fast-dev-run --accelerator gpu
```

Then verify two real optimizer updates and FP16 stability in a fresh output:

```bash
PYTHONPATH=src python \
  tools/train_multimodal_v1_lightning.py --max-updates 2 --accelerator gpu \
  --precision 16-mixed \
  --output outputs/own_multimodal_research/multimodal_v1/lightning_two_update_gate
```

Run one complete epoch without changing the planned 12-epoch LR schedule, then
verify native resume from that epoch boundary:

```bash
PYTHONPATH=src python \
  tools/train_multimodal_v1_lightning.py --accelerator gpu --precision 16-mixed \
  --stop-after-epoch 1 \
  --output outputs/own_multimodal_research/multimodal_v1/lightning_resume_gate

PYTHONPATH=src python \
  tools/train_multimodal_v1_lightning.py --accelerator gpu --precision 16-mixed \
  --resume auto \
  --output outputs/own_multimodal_research/multimodal_v1/lightning_resume_gate
```

The complete single-epoch and old/new performance comparison must be run on
the same GPU and data limits. Do not infer GPU performance from CPU timings.

## Formal RTX 3070 command

After all gates pass:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 \
python tools/train_multimodal_v1_lightning.py \
  --config configs/multimodal_v1/e5_annotated20_lightning.yaml \
  --accelerator gpu --devices 1 --precision 16-mixed
```

Resume with the same command plus `--resume auto`.

The framework exposes a variant contract, but only E5 is currently verified.
E0-E4 must be registered after their frozen architecture/loss definitions and
equivalence fixtures are available; selecting them currently fails fast.

## Method reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
