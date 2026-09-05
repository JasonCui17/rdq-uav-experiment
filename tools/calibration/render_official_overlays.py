#!/usr/bin/env python3
"""Render official 2D boxes against time-compensated 3D GT projections."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402


def representative_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["experiment_split"] in {"val", "test"}:
            grouped[(row["class_name"], row["experiment_split"])].append(row)
    selected = []
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda row: float(row["image_time"]))
        selected.append(group[len(group) // 2])
    return selected


def make_panel(image: np.ndarray, row: dict[str, str], predicted: np.ndarray) -> np.ndarray:
    left = image[:, :1280].copy()
    u, v = float(row["u"]), float(row["v"])
    width, height = float(row["bbox_width_px"]), float(row["bbox_height_px"])
    x1, y1 = round(u - width / 2), round(v - height / 2)
    x2, y2 = round(u + width / 2), round(v + height / 2)
    center = (round(float(predicted[0])), round(float(predicted[1])))
    cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.drawMarker(left, center, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)

    overview = cv2.resize(left, (640, 480), interpolation=cv2.INTER_AREA)
    crop_radius = max(60, round(max(width, height) * 3.0))
    cx, cy = round(u), round(v)
    xa, xb = max(0, cx - crop_radius), min(1280, cx + crop_radius)
    ya, yb = max(0, cy - crop_radius), min(960, cy + crop_radius)
    crop = left[ya:yb, xa:xb]
    zoom = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((480, 960, 3), dtype=np.uint8)
    panel[:, :640] = overview
    panel[80:400, 640:] = zoom
    error = float(np.linalg.norm(predicted - np.asarray([u, v])))
    text = f'{row["class_name"]} {row["experiment_split"]}  error={error:.2f}px'
    cv2.putText(panel, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(
        panel,
        "green=official bbox  red=projected GT",
        (645, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Render held-out GT/fisheye overlay examples")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "calibration/official_overlays"
    )
    parser.add_argument("--max-manifest-dt", type=float, default=0.12)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fitted = json.loads(args.calibration.read_text(encoding="utf-8"))
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    camera_fit = fitted["cameras"]["left"]
    rotation = np.asarray(camera_fit["rotation_camera_from_gt"], dtype=np.float64)
    translation = np.asarray(camera_fit["translation_camera_from_gt_m"], dtype=np.float64)
    offset = float(fitted["time_offset_s"])
    with args.mapping.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["match_status"] == "exact"
            and row["experiment_split"] in {"val", "test"}
            and abs(float(row["dt_to_manifest_s"])) <= args.max_manifest_dt
        ]
    selected = representative_rows(rows)
    trajectories = {
        class_name: PositionTrajectory.from_directory(
            Path(config["dataset_root"]) / class_name / "ground_truth"
        )
        for class_name in sorted({row["class_name"] for row in selected})
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in selected:
        point, valid_time = trajectories[row["class_name"]].evaluate(
            float(row["image_time"]) + offset
        )
        pixel, valid_projection = camera.project(
            transform_points(point, rotation, translation), require_in_image=True
        )
        if not bool(valid_time) or not bool(valid_projection):
            continue
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'Could not decode {row["image_path"]}')
        panel = make_panel(image, row, pixel)
        filename = f'{row["experiment_split"]}_{row["class_name"]}_{row["official_index"]}.jpg'
        cv2.imwrite(str(args.output_dir / filename), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        panels.append(panel)
    if panels:
        rows_of_panels = []
        for index in range(0, len(panels), 2):
            pair = panels[index : index + 2]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            rows_of_panels.append(np.hstack(pair))
        montage = np.vstack(rows_of_panels)
        montage_path = args.output_dir / "heldout_montage.jpg"
        cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(montage_path.resolve())
    print(f"rendered {len(panels)} panels in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
