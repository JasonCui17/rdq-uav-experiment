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


def main() -> None:
    parser = argparse.ArgumentParser(description="Project official Radar XYZ onto both fisheye images")
    parser.add_argument("image", type=Path)
    parser.add_argument("radar", type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--image-time", type=float, default=None)
    parser.add_argument("--radar-time", type=float, default=None)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument("--target-gate-m", type=float, default=2.0)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fitted = json.loads(args.calibration.read_text(encoding="utf-8"))
    image_time = args.image_time if args.image_time is not None else float(args.image.stem)
    radar_time = args.radar_time if args.radar_time is not None else float(args.radar.stem)
    trajectory = PositionTrajectory.from_directory(
        Path(config["dataset_root"]) / args.sequence_id / "ground_truth"
    )
    time_offset = float(fitted["time_offset_s"])
    gt_image, valid_image = trajectory.evaluate(image_time + time_offset)
    gt_radar, valid_radar = trajectory.evaluate(radar_time)
    if not bool(valid_image) or not bool(valid_radar):
        raise ValueError("Image or Radar time lies outside the GT trajectory")
    points = np.asarray(np.load(args.radar, allow_pickle=False), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Radar array (N,>=3), got {points.shape}")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    # Only the target-associated returns move between the two acquisitions.
    target_mask = np.linalg.norm(points - gt_radar, axis=1) <= args.target_gate_m
    points[target_mask] += gt_image - gt_radar

    canvas = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if canvas is None:
        raise OSError(f"Could not decode {args.image}")
    split_x = int(config["image_layout"]["split_x"])
    ranges = np.linalg.norm(points, axis=1)
    normalized_range = np.clip(ranges / max(float(np.percentile(ranges, 95)), 1e-6), 0, 1)
    colors = cv2.applyColorMap((255 * (1 - normalized_range)).astype(np.uint8), cv2.COLORMAP_TURBO)
    colors = colors[:, 0, :]
    camera_offsets = {"left": 0, "right": split_x}
    for camera_name, camera_fit in fitted["cameras"].items():
        x_offset = camera_offsets[camera_name]
        camera = OmniRadtanCamera.from_config(config["cameras"][camera_name])
        rotation = np.asarray(camera_fit["rotation_camera_from_gt"], dtype=np.float64)
        translation = np.asarray(camera_fit["translation_camera_from_gt_m"], dtype=np.float64)
        pixels, valid = camera.project(
            transform_points(points, rotation, translation), require_in_image=True
        )
        for pixel, color in zip(pixels[valid], colors[valid]):
            cv2.circle(
                canvas, (round(float(pixel[0])) + x_offset, round(float(pixel[1]))),
                args.point_radius, tuple(int(value) for value in color), -1,
            )
        gt_pixel, gt_valid = camera.project(
            transform_points(gt_image, rotation, translation), require_in_image=True
        )
        if bool(gt_valid):
            center = (round(float(gt_pixel[0])) + x_offset, round(float(gt_pixel[1])))
            cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.putText(
        canvas, f"motion-compensated radar target points: {int(target_mask.sum())}/{len(points)}",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f"Could not write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
