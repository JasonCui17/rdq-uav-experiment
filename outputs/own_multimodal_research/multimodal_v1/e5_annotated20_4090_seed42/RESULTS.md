# Multimodal V1 E5 annotated-20 — RTX 4090 seed 42

This directory archives the reproducible outputs of the completed 12-epoch
Lightning E5 run performed from source commit
`d1fc723868602e8652fdaca5ce9a1d65effa1cd8`.

## Run status

- Epochs: 12/12
- Optimizer updates: 24,000
- Training precision: FP16 mixed precision
- Validation precision: FP32
- Validation queries: 4,800
- Best checkpoint: epoch 1 (`global_step=2000`)
- Best validation 3D Success@1m: 0.839583
- Best validation median 3D error: 0.371650 m
- Last validation 3D Success@1m: 0.205833
- Last validation median 3D error: 21.722412 m

The best checkpoint was selected lexicographically by maximizing final 3D
Success@1m and then minimizing median 3D error.  The final checkpoint is kept
for audit and diagnosis; it is not the recommended inference checkpoint.

## Versioned files

- `effective_config.yaml` and `effective_config.json`: exact effective run configuration.
- `logs/csv/version_0/metrics.csv`: step and epoch metrics.
- `logs/tensorboard/version_0/`: TensorBoard events and hyperparameters.
- `train.log.gz`: gzip-compressed complete console training log.
- `diagnosis_smoke/`: exploratory best-versus-last candidate diagnosis outputs.
- `CHECKPOINTS.sha256`: checkpoint identities and expected release asset names.

## Checkpoint release assets

The checkpoint binaries exceed GitHub's normal Git blob limit and are kept out
of repository history.  Attach these two local files to the GitHub Release
tagged `multimodal-v1-e5-4090-seed42`:

| Release asset | Local size | SHA256 |
|---|---:|---|
| `best.ckpt` | 212,337,371 bytes | `b33023490fdcd1c390a5126d1adefc4422ad1114c6059f4b2b526119afeefd9b` |
| `last.ckpt` | 593,646,195 bytes | `5c54d80bf194e05a42810983e9f138bbd3d6a0025f51a35c7356c7e9d8c80a98` |

After the release is published, the expected download URLs are:

- `https://github.com/JasonCui17/rdq-uav-experiment/releases/download/multimodal-v1-e5-4090-seed42/best.ckpt`
- `https://github.com/JasonCui17/rdq-uav-experiment/releases/download/multimodal-v1-e5-4090-seed42/last.ckpt`

Verify a downloaded checkpoint with `sha256sum` before evaluation or resume.

## Interpretation boundary

The run completed without NaN, Inf, OOM, or missing 3D outputs.  The best
checkpoint is the T1 model; after T2 unfreezing, final 3D Success@1m collapsed
while the small 2D validation subset improved.  Only five validation samples
had valid 2D annotations, so the recorded validation IoU is exploratory and
does not establish vision generalization.
