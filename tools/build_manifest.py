#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.manifest import build_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-aware MMAUD V1 manifests")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    summary = build_manifests(load_config(args.config, args.overrides))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
