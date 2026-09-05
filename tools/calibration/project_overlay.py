#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402


def projected_sphere_box(
    center_gt: np.ndarray,
    radius_m: float,
    camera: OmniRadtanCamera,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[int, int, int, int] | None:
    directions = np.asarray(
        [direction for direction in np.ndindex(3, 3, 3) if direction != (1, 1, 1)],
        dtype=np.float64,
    ) - 1.0
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0)
    points_gt = np.concatenate((center_gt[None], center_gt[None] + radius_m * directions), axis=0)
    pixels, valid = camera.project(transform_points(points_gt, rotation, translation), require_in_image=True)
    pixels = pixels[valid]
    if len(pixels) < 2:
        return None
    low = np.floor(pixels.min(axis=0)).astype(int)
    high = np.ceil(pixels.max(axis=0)).astype(int)
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay time-compensated GT projection on a stereo image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--image-time", type=float, default=None)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument("--radius-m", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fitted = json.loads(args.calibration.read_text(encoding="utf-8"))
    image_time = args.image_time if args.image_time is not None else float(args.image.stem)
    trajectory = PositionTrajectory.from_directory(
        Path(config["dataset_root"]) / args.sequence_id / "ground_truth"
    )
    point_gt, valid_time = trajectory.evaluate(image_time + float(fitted["time_offset_s"]))
    if not bool(valid_time):
        raise ValueError("Image time plus fitted offset lies outside the GT trajectory")
    canvas = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if canvas is None:
        raise OSError(f"Could not decode {args.image}")
    split_x = int(config["image_layout"]["split_x"])
    camera_offsets = {"left": 0, "right": split_x}
    for camera_name, camera_fit in fitted["cameras"].items():
        x_offset = camera_offsets[camera_name]
        camera = OmniRadtanCamera.from_config(config["cameras"][camera_name])
        rotation = np.asarray(camera_fit["rotation_camera_from_gt"], dtype=np.float64)
        translation = np.asarray(camera_fit["translation_camera_from_gt_m"], dtype=np.float64)
        pixel, valid = camera.project(
            transform_points(point_gt, rotation, translation), require_in_image=True
        )
        if not bool(valid):
            continue
        center = (round(float(pixel[0])) + x_offset, round(float(pixel[1])))
        cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.putText(
            canvas, f"{camera_name} GT", (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
        if args.radius_m is not None:
            box = projected_sphere_box(point_gt, args.radius_m, camera, rotation, translation)
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(canvas, (x1 + x_offset, y1), (x2 + x_offset, y2), (0, 255, 0), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f"Could not write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
