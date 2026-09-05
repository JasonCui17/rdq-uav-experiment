#!/usr/bin/env python3
"""Render representative full-left views beside their generated Oracle crops."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Oracle ROI audit montage")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "manifests_oracle_left/test.csv"
    )
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT.parent / "v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "calibration/oracle_left_roi_montage.jpg",
    )
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    selected = []
    for class_name, class_rows in sorted(grouped.items()):
        class_rows.sort(key=lambda row: float(row["image_time"]))
        indices = np.linspace(0, len(class_rows) - 1, args.per_class + 2, dtype=int)[1:-1]
        selected.extend(class_rows[index] for index in indices)

    panels = []
    for row in selected:
        image = cv2.imread(str(args.data_root / row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'Could not decode {args.data_root / row["image_path"]}')
        left = image[:, :1280].copy()
        box = tuple(int(float(row[key])) for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
        x1, y1, x2, y2 = box
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.drawMarker(
            left,
            (round(float(row["roi_center_u"])), round(float(row["roi_center_v"]))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            28,
            3,
        )
        overview = cv2.resize(left, (320, 240), interpolation=cv2.INTER_AREA)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (320, 240), interpolation=cv2.INTER_AREA)
        panel = np.hstack((overview, crop))
        cv2.putText(
            panel,
            f'{row["class_name"]} side={row["roi_side_px"]}px',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        panels.append(panel)
    montage_rows = []
    for index in range(0, len(panels), 2):
        pair = panels[index : index + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        montage_rows.append(np.hstack(pair))
    montage = np.vstack(montage_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
