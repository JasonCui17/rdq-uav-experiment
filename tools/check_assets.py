#!/usr/bin/env python3
"""Fail fast when a required external or reproducibility asset is missing."""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rdq_uav.runtime_paths import (
    apply_runtime_path_overrides,
    ensure_detrex_config_link,
    resolve_project_path,
)

DEFAULT_CONFIG = ROOT / "configs/multimodal_v1/e5_annotated20_lightning.yaml"


def required_assets(config: Path = DEFAULT_CONFIG) -> dict[str, Path]:
    cfg = apply_runtime_path_overrides(yaml.safe_load(config.read_text()))
    data, initialization = cfg["data"], cfg["initialization"]
    detrex_root = resolve_project_path(initialization["dino_root"], ROOT)
    ensure_detrex_config_link(detrex_root)
    return {
        "detrex root": detrex_root,
        "detrex packaged configs": detrex_root / "detrex/config/configs/common",
        "detectron2 packaged configs": detrex_root / "detectron2/detectron2/model_zoo/configs/common",
        "detrex config": resolve_project_path(initialization["dino_config"], ROOT),
        "DINO checkpoint": resolve_project_path(initialization["dino_checkpoint"], ROOT),
        "LiDAR checkpoint": resolve_project_path(initialization["lidar_checkpoint"], ROOT),
        "MMAUD train root": resolve_project_path(data["root"], ROOT),
        "split": resolve_project_path(data["split_file"], ROOT),
        "2D manifest": resolve_project_path(data["annotation_manifest"], ROOT),
        "camera calibration": resolve_project_path(data["camera_config"], ROOT),
        "geometry calibration": resolve_project_path(data["geometry_calibration"], ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = resolve_project_path(args.config, ROOT)
    rows = []
    for name, path in required_assets(config).items():
        rows.append({"name": name, "path": str(path), "exists": path.exists(), "resolved": str(path.resolve())})
    missing = [row for row in rows if not row["exists"]]
    print(json.dumps({
        "status": "PASS" if not missing else "FAIL",
        "project_root": str(ROOT),
        "config": str(config),
        "path_overrides": {
            key: value for key, value in os.environ.items() if key.startswith("RDQ_")
        },
        "assets": rows,
    }, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
