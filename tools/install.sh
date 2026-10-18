#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt
export MAX_JOBS="${MAX_JOBS:-2}"
python -m pip install -e third_party/detrex/detectron2 --no-build-isolation --no-deps
python -m pip install -e third_party/detrex --no-build-isolation --no-deps
python -m pip install -e . --no-deps --no-build-isolation
python run.py doctor
