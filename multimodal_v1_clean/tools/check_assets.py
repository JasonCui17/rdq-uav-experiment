#!/usr/bin/env python3
"""Fail fast when a required external or reproducibility asset is missing."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "detrex config": ROOT / "third_party/detrex/projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py",
    "DINO checkpoint": ROOT / "checkpoints/dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth",
    "LiDAR checkpoint": ROOT / "checkpoints/lidar_v2/best_spatial.pt",
    "MMAUD train root": ROOT / "data/mmaud_official_train",
    "split": ROOT / "splits/mmaud_splits.json",
    "2D manifest": ROOT / "manifests/multimodal_v1/vision_manual161_train.jsonl",
    "camera calibration": ROOT / "configs/calibration/mmaud_v1_omni.yaml",
    "geometry calibration": ROOT / "calibration/official_left_p4_current_geometry.json",
}


def main() -> None:
    rows = []
    for name, path in REQUIRED.items():
        rows.append({"name": name, "path": str(path), "exists": path.exists(), "resolved": str(path.resolve())})
    missing = [row for row in rows if not row["exists"]]
    print(json.dumps({"status": "PASS" if not missing else "FAIL", "assets": rows}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
