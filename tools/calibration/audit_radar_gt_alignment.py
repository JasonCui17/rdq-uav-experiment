#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.utils.io import write_json  # noqa: E402


def nearest_distance(points: np.ndarray, target: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[finite, :3]
    return float(np.linalg.norm(points - target, axis=1).min()) if len(points) else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether official Radar XYZ and GT share a frame")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/train.csv")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/v1"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "calibration/radar_gt_audit.json")
    args = parser.parse_args()
    grouped = defaultdict(list)
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["class_name"]].append(row)
    report = {}
    for class_name, rows in grouped.items():
        clouds = [np.load(args.dataset_root / row["radar_path"], allow_pickle=False) for row in rows]
        targets = [
            np.asarray([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])])
            for row in rows
        ]
        aligned = np.asarray([nearest_distance(cloud, target) for cloud, target in zip(clouds, targets)])
        shift = len(rows) // 2
        shifted = np.asarray(
            [nearest_distance(clouds[(index + shift) % len(rows)], target) for index, target in enumerate(targets)]
        )
        report[class_name] = {
            "frames": len(rows),
            "aligned_nearest_m": {
                "min": float(aligned.min()), "median": float(np.median(aligned)),
                "p10": float(np.percentile(aligned, 10)), "p25": float(np.percentile(aligned, 25)),
            },
            "hit_rates": {
                f"within_{threshold:g}m": {
                    "aligned": float((aligned < threshold).mean()),
                    "half_sequence_shifted": float((shifted < threshold).mean()),
                }
                for threshold in (0.5, 1.0, 2.0)
            },
        }
    write_json(report, args.output)
    print(args.output.resolve())
    for class_name, values in report.items():
        print(class_name, values["hit_rates"])


if __name__ == "__main__":
    main()
