#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$PACKAGE_DIR/../.." && pwd)}"
RUN="${RUN:-outputs/mtce_joint_lora_local_20261014_011406}"
KF_RUN="${KF_RUN:-$REPO/outputs/kf_candidate_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
cd "$REPO"
mkdir -p "$KF_RUN"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "KF_RUN=$KF_RUN"
python "$PACKAGE_DIR/extract_observations.py" --repo "$REPO" --config "$RUN/config.yaml" --checkpoint "$RUN/train/checkpoints/best.ckpt" --cache "$KF_RUN/cache" --device "$DEVICE" --num-workers "$NUM_WORKERS" 2>&1 | tee "$KF_RUN/extract.log"
python "$PACKAGE_DIR/evaluate_tracking.py" --cache "$KF_RUN/cache" --output "$KF_RUN/results" "$@" 2>&1 | tee "$KF_RUN/tracking.log"
echo "Upload: $KF_RUN/results.zip"
