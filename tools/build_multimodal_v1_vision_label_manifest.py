#!/usr/bin/env python3
"""Create a deterministic train-only 5% MMAUD image manifest for manual UAV bbox labeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/multimodal_v1/vision_ssod.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--total-limit", type=int, help="Diagnostic subset cap, e.g. 64; formal run omits this.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    output = args.output or ROOT / data_cfg["annotation_manifest"]

    from rdq_uav.lidar_v2.data import LiDARUAVDataset
    from rdq_uav.multimodal_v1.data import LeftImageIndex
    from rdq_uav.multimodal_v1.vision.ssod_data import assign_stratified_roles

    root = Path(data_cfg["root"])
    split_file = ROOT / data_cfg["split_file"]
    geometry_path = ROOT / data_cfg["geometry_calibration"]
    geometry = json.loads(geometry_path.read_text())
    image_index = LeftImageIndex(
        root,
        time_offset_s=float(geometry["time_offset_s"]),
        max_abs_gap_s=float(data_cfg["max_image_gap_s"]),
    )
    dataset = LiDARUAVDataset(root, split_file, data_cfg["train_split"])

    candidates = []
    for record in dataset.records:
        match = image_index.match(record["sequence_id"], record["query_time"])
        if not match.valid or match.path is None:
            continue
        xyz = np.load(record["target_path"], allow_pickle=False).reshape(3).astype(np.float64)
        if not np.isfinite(xyz).all():
            continue
        candidates.append({
            "sequence_id": record["sequence_id"],
            "query_uid": record["query_uid"],
            "query_time": float(record["query_time"]),
            "image_path": str(match.path),
            "image_time": float(match.image_time),
            "image_query_gap_s": float(match.gap_s),
            "gt_xyz_m": xyz.tolist(),
            "range_m": float(np.linalg.norm(xyz)),
        })

    roles = assign_stratified_roles(
        candidates,
        labeled_train_fraction=float(data_cfg["labeled_train_fraction"]),
        calibration_fraction=float(data_cfg["calibration_fraction"]),
        range_edges_m=tuple(float(x) for x in data_cfg["range_edges_m"]),
        seed=int(data_cfg["seed"]),
        total_limit=args.total_limit,
    )
    selected = []
    for record in candidates:
        key = (record["sequence_id"], record["query_uid"])
        if key not in roles:
            continue
        item = dict(record)
        item.update(
            role=roles[key],
            box_xyxy_px=None,
            gt_2d_valid=False,
            annotation_note="Fill box_xyxy_px=[x1,y1,x2,y2] in source 1280x960 left-camera pixels, then set gt_2d_valid=true.",
        )
        selected.append(item)
    selected.sort(key=lambda x: (x["role"], x["sequence_id"], str(x["query_uid"])))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = {role: sum(item["role"] == role for item in selected) for role in ("labeled_train", "labeled_calibration")}
    print(json.dumps({
        "status": "PASS",
        "source_split": data_cfg["train_split"],
        "eligible_train_queries": len(candidates),
        "selected": len(selected),
        "counts": counts,
        "output": str(output),
        "total_limit": args.total_limit,
    }, indent=2))


if __name__ == "__main__":
    main()
