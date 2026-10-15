#!/usr/bin/env bash
set -euo pipefail

release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$release_dir/../../../.." && pwd)"
mode="${1:-}"
case "$mode" in
  realtime) lag_s=0 ;;
  lag05) lag_s=0.5 ;;
  *)
    echo "Usage: bash $0 {realtime|lag05} [cache_directory] [new_output_directory]" >&2
    exit 2
    ;;
esac
if (( $# > 3 )); then
  echo "Expected at most three arguments" >&2
  exit 2
fi
cd -- "$repo_dir"
sha256sum --check "$release_dir/SHA256SUMS"
cache_dir="${2:-outputs/kf_v3_multi_run/cache}"
output_dir="${3:-outputs/target_kf_v1_20261015_${mode}_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python -u experiments/multimodal_v3/evaluate_target.py \
  --cache "$cache_dir" \
  --config "$release_dir/config.json" \
  --lag-s "$lag_s" \
  --output "$output_dir"
