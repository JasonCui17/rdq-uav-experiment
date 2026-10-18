#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
data_root="${1:?Usage: bash run_final_validation.sh MMAUD_ROOT [sensor|benchmark] [NEW_OUTPUT_DIR]}"
schedule="${2:-sensor}"
output="${3:-runs/final_validation_$(date +%Y%m%d_%H%M%S)}"
python run.py doctor
python -u run.py infer --data-root "$data_root" --split validation_sub --schedule "$schedule" --output "$output"
python tools/report_run.py "$output"
