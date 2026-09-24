# Multimodal V1 E5 Nextflow pipeline

This DSL2 pipeline wraps the existing reviewed-label and Lightning E5 system
without changing its model, target, or loss semantics:

```text
reviewed labels + manifest validation
→ Lightning model training
→ independent FP32 checkpoint evaluation + report
```

Human annotation remains an interactive prerequisite. Stage 1 validates,
sequence-isolates, fingerprints, and freezes the reviewed manifest and split;
it never edits source labels or manifests.

## Reproducibility contract

- Nextflow is pinned by `.nxf-version` to `24.10.0`.
- WSL2 uses the local executor and the active Conda environment. Set
  `RDQ_CONDA_ENV` when the pipeline is launched outside an activated env.
- `run.sh` fingerprints source, model checkpoints, manifest, and split. That
  fingerprint participates in task hashes, so code/weight changes invalidate
  the training cache.
- Every run records Git commit/status, input SHA-256 values, data version,
  Python/Conda state, Nextflow version, platform, and GPU/driver information.
- Nextflow `-resume` reuses completed training when evaluation fails. If an
  interrupted training task is retried, its durable
  `<outdir>/.training_state/checkpoints/last.ckpt` is detected and Lightning
  receives `--resume auto`. Source and effective-config hashes must match;
  incompatible state fails closed instead of loading stale optimizer state.
- `--report_revision <value>` is an evaluation-only cache key. Bumping it
  reruns evaluation/report generation while preserving manifest preparation
  and training; the stub contract test verifies this behavior.
- `work/` is intentionally retained; deleting it removes resume capability.

## Install and inspect

```bash
conda install -n rdq -c conda-forge -c bioconda \
  'openjdk=17' 'nextflow=24.10.0'

cd /path/to/rdq-uav-experiment
NXF_VER=24.10.0 nextflow \
  -c nextflow/nextflow.config config nextflow/main.nf -profile test
```

## Lightweight test profile

The test profile uses exactly two disjoint sequences (`seq0001` for training,
`seq0007` for validation), one epoch, two training queries, two validation
queries, batch 1, and one RTX GPU.

First validate pipeline wiring and cache behavior without model execution:

```bash
bash nextflow/tests/test_pipeline.sh
```

Then execute the real two-sequence GPU test:

```bash
nextflow/run.sh -profile test
```

Resume the same run with:

```bash
nextflow/run.sh -profile test -resume
```

## Formal run

Only after the real test profile passes:

```bash
nextflow/run.sh -profile standard
```

Override protocol fields through config/CLI, for example
`--epochs 12 --batch_size 1 --accumulate 4 --num_workers 2`. Training outputs
are published under `02_training`; evaluation consumes the immutable training
directory and publishes `metrics.json` and `report.md` under `03_evaluation`.

## Outputs

- `01_manifest/prepared/`: frozen manifest/split/config, summary, provenance,
  and a package/environment snapshot.
- `02_training/training_run/`: Lightning effective config, logs, `last.ckpt`, `best.ckpt`.
- `03_evaluation/evaluation/`: independent FP32 metrics and Markdown report.
- `pipeline_info/`: Nextflow trace, timeline, execution report, and DAG.

The workflow procedures follow Kassis et al., *Scientific Agent Skills: A
Library of Procedural Knowledge for Research Agents* (2026),
https://doi.org/10.48550/arXiv.2609.00065.
