#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PIPELINE_DIR}/.." && pwd)"
export NXF_VER="$(tr -d '[:space:]' < "${PIPELINE_DIR}/.nxf-version")"
export NXF_OFFLINE="${NXF_OFFLINE:-true}"
export NXF_LOG_FILE="${NXF_LOG_FILE:-${PIPELINE_DIR}/.nextflow.log}"
export RDQ_CONDA_ENV="${RDQ_CONDA_ENV:-${CONDA_PREFIX:-}}"
if [[ -z "${RDQ_CONDA_ENV}" ]] && command -v python >/dev/null 2>&1; then
  RDQ_CONDA_ENV="$(python -c 'import sys; print(sys.prefix)')"
  export RDQ_CONDA_ENV
fi
if [[ -z "${RDQ_CONDA_ENV}" || ! -d "${RDQ_CONDA_ENV}" ]]; then
  echo "Activate the rdq Conda environment or set RDQ_CONDA_ENV to its prefix." >&2
  exit 2
fi
if [[ -z "${JAVA_HOME:-}" && -n "${RDQ_CONDA_ENV}" && -d "${RDQ_CONDA_ENV}/lib/jvm" ]]; then
  export JAVA_HOME="${RDQ_CONDA_ENV}/lib/jvm"
fi
if [[ -n "${JAVA_HOME:-}" ]]; then
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi
if [[ -n "${RDQ_CONDA_ENV}" ]]; then
  export PATH="${RDQ_CONDA_ENV}/bin:${PATH}"
fi
export RDQ_GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

resolve_from_project() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
  esac
}

CONFIG_PATH="$(resolve_from_project "${RDQ_CONFIG:-configs/multimodal_v1/e5_annotated20_lightning.yaml}")"
MANIFEST_PATH="$(resolve_from_project "${RDQ_ANNOTATION_MANIFEST:-manifests/multimodal_v1/vision_annotated20.jsonl}")"
SPLIT_PATH="$(resolve_from_project "${RDQ_SPLIT_FILE:-splits/mmaud_annotated20.json}")"
LIDAR_CHECKPOINT_PATH="$(resolve_from_project "${RDQ_LIDAR_CHECKPOINT:-checkpoints/lidar_v2/best_spatial.pt}")"
DINO_CHECKPOINT_PATH="$(resolve_from_project "${RDQ_DINO_CHECKPOINT:-checkpoints/dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth}")"
for required_path in "${CONFIG_PATH}" "${MANIFEST_PATH}" "${SPLIT_PATH}" "${LIDAR_CHECKPOINT_PATH}" "${DINO_CHECKPOINT_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required pipeline input does not exist: ${required_path}" >&2
    exit 2
  fi
done
export RDQ_CONFIG="${CONFIG_PATH}"
export RDQ_ANNOTATION_MANIFEST="${MANIFEST_PATH}"
export RDQ_SPLIT_FILE="${SPLIT_PATH}"
export RDQ_LIDAR_CHECKPOINT="${LIDAR_CHECKPOINT_PATH}"
export RDQ_DINO_CHECKPOINT="${DINO_CHECKPOINT_PATH}"

fingerprint_paths=(
  "${PROJECT_ROOT}/src/rdq_uav/multimodal_v1"
  "${PROJECT_ROOT}/src/rdq_uav/lidar_v2"
  "${PROJECT_ROOT}/tools/train_multimodal_v1_full.py"
  "${PROJECT_ROOT}/tools/train_multimodal_v1_lightning.py"
  "${PROJECT_ROOT}/tools/evaluate_multimodal_v1_lightning.py"
  "${PROJECT_ROOT}/configs/multimodal_v1"
  "${PROJECT_ROOT}/configs/calibration"
  "${PROJECT_ROOT}/calibration/official_left_p4_current_geometry.json"
  "${PROJECT_ROOT}/nextflow"
)
export RDQ_SOURCE_FINGERPRINT="$({
  find "${fingerprint_paths[@]}" -type f \
    -not -path '*/work/*' -not -path '*/results/*' -not -path '*/.conda_cache/*' \
    -not -path '*/__pycache__/*' -not -name '*.pyc' \
    -not -name '.stub_*.log' -not -name '.nextflow.log*' -print0 \
    | sort -z | xargs -0 sha256sum
  sha256sum \
    "${CONFIG_PATH}" \
    "${LIDAR_CHECKPOINT_PATH}" \
    "${DINO_CHECKPOINT_PATH}"
} | sha256sum | awk '{print $1}')"
if [[ -z "${RDQ_DATA_VERSION:-}" ]]; then
  RDQ_DATA_VERSION="$({
    sha256sum "${MANIFEST_PATH}" "${SPLIT_PATH}"
    printf 'dataset_root=%s\n' "${RDQ_DATA_ROOT:-data/mmaud_official_train}"
  } | sha256sum | awk '{print $1}')"
  export RDQ_DATA_VERSION
fi

NEXTFLOW_BIN="${NEXTFLOW_BIN:-$(command -v nextflow || true)}"
if [[ -z "${NEXTFLOW_BIN}" ]]; then
  echo "Nextflow is not installed; install pinned version ${NXF_VER}." >&2
  exit 127
fi

cd "${PROJECT_ROOT}"
exec "${NEXTFLOW_BIN}" run "${PIPELINE_DIR}/main.nf" \
  -c "${PIPELINE_DIR}/nextflow.config" "$@"
