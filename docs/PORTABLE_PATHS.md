# Portable runtime paths

All tracked experiment configs use paths relative to the repository root.
Activate the target server's Python/Conda environment. Entry points derive the
repository root from `__file__`, so an absolute script path works from any
working directory; the shorter commands below assume `cd` into the repository.

Large datasets and checkpoints are not stored in Git. They may be provided at
the default repository paths or overridden without editing an experiment YAML:

| Environment variable | Effective config field |
|---|---|
| `RDQ_DATA_ROOT` | `data.root` |
| `RDQ_SPLIT_FILE` | `data.split_file` |
| `RDQ_ANNOTATION_MANIFEST` | `data.annotation_manifest` |
| `RDQ_CAMERA_CONFIG` | `data.camera_config` |
| `RDQ_GEOMETRY_CALIBRATION` | `data.geometry_calibration` |
| `RDQ_LIDAR_CONFIG` | `initialization.lidar_config` |
| `RDQ_LIDAR_CHECKPOINT` | `initialization.lidar_checkpoint` |
| `RDQ_DETREX_ROOT` | `initialization.dino_root` |
| `RDQ_DINO_CONFIG` | `initialization.dino_config` |
| `RDQ_DINO_CHECKPOINT` | `initialization.dino_checkpoint` |
| `RDQ_P6_CONFIG` | `initialization.p6_config` |
| `RDQ_OUTPUT_DIR` | `experiment.output_dir` |

The Nextflow wrapper additionally accepts `RDQ_CONFIG` and
`RDQ_NEXTFLOW_OUTDIR`; it also honors the manifest, split, dataset, and
checkpoint variables above when computing cache fingerprints.
Set `RDQ_DATA_VERSION` to a stable dataset release/hash when two revisions can
occupy the same dataset path; otherwise the wrapper fingerprints the effective
dataset-root string together with the manifest and split contents.

Example on a new server:

```bash
cd /path/to/rdq-uav-experiment
conda activate rdq
export RDQ_DATA_ROOT=/datasets/MMAUD/official/train
export RDQ_LIDAR_CHECKPOINT=/models/rdq/best_spatial.pt
python tools/check_assets.py
PYTHONPATH=src python tools/train_multimodal_v1_lightning.py \
  --config configs/multimodal_v1/e5_annotated20_lightning.yaml \
  --accelerator gpu --devices 1 --precision 16-mixed
```

CLI arguments such as `--output` take precedence over the corresponding
environment override. Effective configs record the paths that actually ran.
Machine-generated historical outputs retain their original absolute paths as
provenance and are not runtime inputs.

The reviewed annotation manifests store `image_path` relative to the dataset
root (for example `seq0001/Image/<timestamp>.png`). Current E5 loading joins
the record to `RDQ_DATA_ROOT`, and newly generated manifests follow the same
contract.

Detrex is vendored under `third_party/detrex`. Its internal
`detrex/config/configs` link is repository-relative. After a fresh clone, run
`python tools/check_assets.py --config <experiment.yaml>` before training.
