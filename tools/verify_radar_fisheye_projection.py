#!/usr/bin/env python3
"""
Verify theoretical mmWave Radar -> left fisheye projection on one MMAUD sample.

Inputs:
- dual fisheye RGB image (2560x960) or left image (1280x960)
- one GT 3D point, shape (3,)
- mmWave radar point cloud, shape (N,3)

This script:
1) projects GT -> left fisheye
2) projects Radar -> GT/world -> left camera -> left fisheye
3) overlays all valid Radar points
4) highlights the Radar point nearest to projected GT
5) writes per-point CSV and a short summary

No time-offset correction, no motion compensation, no GT-based Radar filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def find_project_root(start: Path) -> Path:
    """Walk upward until configs/calibration/mmaud_v1_omni.yaml is found."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "configs/calibration/mmaud_v1_omni.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate project root containing configs/calibration/mmaud_v1_omni.yaml"
    )


PROJECT_ROOT = find_project_root(Path.cwd())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402


def load_left_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.height != height:
            raise ValueError(f"Expected height={height}, got {im.height}")
        if im.width == width:
            return im.copy()
        if im.width >= 2 * width:
            return im.crop((0, 0, width, height))
        raise ValueError(
            f"Expected left image {width}x{height} or stitched image >= {2*width}x{height}, got {im.size}"
        )


def load_transforms(
    radar_calibration: Path,
    camera_calibration: Path,
):
    radar_payload = json.loads(radar_calibration.read_text(encoding="utf-8"))
    radar_transform = radar_payload["global_results"]["global_rigid_icp"]["transform"]

    R_gr = np.asarray(
        radar_transform["rotation_gt_from_radar"], dtype=np.float64
    ).reshape(3, 3)
    t_gr = np.asarray(
        radar_transform["translation_gt_from_radar_m"], dtype=np.float64
    ).reshape(3)

    camera_payload = json.loads(camera_calibration.read_text(encoding="utf-8"))
    left = camera_payload["cameras"]["left"]

    R_cg = np.asarray(
        left["rotation_camera_from_gt"], dtype=np.float64
    ).reshape(3, 3)
    t_cg = np.asarray(
        left["translation_camera_from_gt_m"], dtype=np.float64
    ).reshape(3)

    return R_gr, t_gr, R_cg, t_cg


def project_gt(
    gt_xyz: np.ndarray,
    camera: OmniRadtanCamera,
    R_cg: np.ndarray,
    t_cg: np.ndarray,
):
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64).reshape(1, 3)
    gt_cam = transform_points(gt_xyz, R_cg, t_cg)
    gt_pixel, gt_valid = camera.project(gt_cam, require_in_image=True)
    return gt_cam[0], gt_pixel[0], bool(gt_valid[0])


def project_radar(
    radar_xyz: np.ndarray,
    camera: OmniRadtanCamera,
    R_gr: np.ndarray,
    t_gr: np.ndarray,
    R_cg: np.ndarray,
    t_cg: np.ndarray,
):
    radar_xyz = np.asarray(radar_xyz, dtype=np.float64)
    finite = np.isfinite(radar_xyz[:, :3]).all(axis=1)
    radar_xyz = radar_xyz[finite, :3]

    points_gt = transform_points(radar_xyz, R_gr, t_gr)
    points_cam = transform_points(points_gt, R_cg, t_cg)
    pixels, valid = camera.project(points_cam, require_in_image=True)

    return radar_xyz, points_gt, points_cam, pixels, valid


def draw_cross(draw: ImageDraw.ImageDraw, u: float, v: float, color, size=10, width=3):
    draw.line((u - size, v, u + size, v), fill=color, width=width)
    draw.line((u, v - size, u, v + size), fill=color, width=width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--radar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manual_projection_verify"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml",
    )
    parser.add_argument(
        "--radar-calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration/radar_frame_resolution.json",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument("--radar-radius", type=int, default=2)
    parser.add_argument("--gt-radius", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])

    R_gr, t_gr, R_cg, t_cg = load_transforms(
        args.radar_calibration, args.camera_calibration
    )

    gt = np.load(args.gt, allow_pickle=False)
    radar = np.load(args.radar, allow_pickle=False)

    if gt.shape != (3,):
        raise ValueError(f"GT must have shape (3,), got {gt.shape}")
    if radar.ndim != 2 or radar.shape[1] < 3:
        raise ValueError(f"Radar must have shape (N,>=3), got {radar.shape}")

    gt_cam, gt_pixel, gt_valid = project_gt(gt, camera, R_cg, t_cg)
    if not gt_valid:
        raise RuntimeError(f"GT projects outside left image: {gt_pixel}")

    radar_xyz, radar_gt, radar_cam, radar_pixels, radar_valid = project_radar(
        radar[:, :3], camera, R_gr, t_gr, R_cg, t_cg
    )

    valid_indices = np.where(radar_valid)[0]
    valid_pixels = radar_pixels[radar_valid]

    if len(valid_indices) == 0:
        raise RuntimeError("No Radar points project into the left fisheye image")

    d_px = np.linalg.norm(valid_pixels - gt_pixel[None, :], axis=1)
    nearest_local = int(np.argmin(d_px))
    nearest_index = int(valid_indices[nearest_local])
    nearest_pixel = radar_pixels[nearest_index]
    nearest_distance_px = float(d_px[nearest_local])

    # Optional 3D diagnostic in GT frame.
    d3 = np.linalg.norm(radar_gt - gt.reshape(1, 3), axis=1)
    nearest_3d_index = int(np.argmin(d3))
    nearest_3d_distance_m = float(d3[nearest_3d_index])

    left = load_left_image(args.image, camera.width, camera.height)
    overlay = left.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # 1) draw all valid Radar points in red, small and semi-transparent
    for u, v in valid_pixels:
        r = args.radar_radius
        draw.ellipse(
            (u - r, v - r, u + r, v + r),
            fill=(255, 0, 0, 150),
        )

    # 2) GT in bright green
    gu, gv = gt_pixel
    draw_cross(draw, gu, gv, (0, 255, 0, 255), size=args.gt_radius, width=3)
    draw.ellipse(
        (gu - 5, gv - 5, gu + 5, gv + 5),
        outline=(0, 255, 0, 255),
        width=2,
    )

    # 3) nearest Radar-to-GT projected point in yellow
    ru, rv = nearest_pixel
    draw.ellipse(
        (ru - 9, rv - 9, ru + 9, rv + 9),
        outline=(255, 255, 0, 255),
        width=3,
    )
    draw.line((gu, gv, ru, rv), fill=(255, 255, 0, 220), width=2)

    # text panel
    font = ImageFont.load_default()
    lines = [
        f"GT pixel: ({gu:.1f}, {gv:.1f})",
        f"Radar valid: {int(radar_valid.sum())}/{len(radar_xyz)}",
        f"Nearest Radar pixel: ({ru:.1f}, {rv:.1f})",
        f"Nearest 2D distance: {nearest_distance_px:.2f} px",
        f"Nearest 3D Radar->GT distance: {nearest_3d_distance_m:.3f} m",
    ]
    y = 12
    for line in lines:
        bbox = draw.textbbox((12, y), line, font=font)
        draw.rectangle(
            (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
            fill=(0, 0, 0, 170),
        )
        draw.text((12, y), line, fill=(255, 255, 255, 255), font=font)
        y += 18

    overlay_path = args.output_dir / "radar_gt_projection_left.png"
    overlay.save(overlay_path)

    # Per-point CSV
    csv_path = args.output_dir / "radar_projection_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "radar_x",
                "radar_y",
                "radar_z",
                "gtframe_x",
                "gtframe_y",
                "gtframe_z",
                "camera_x",
                "camera_y",
                "camera_z",
                "u",
                "v",
                "valid",
                "pixel_distance_to_gt",
                "3d_distance_to_gt_m",
            ]
        )
        for i in range(len(radar_xyz)):
            pix_dist = (
                float(np.linalg.norm(radar_pixels[i] - gt_pixel))
                if radar_valid[i]
                else ""
            )
            writer.writerow(
                [
                    i,
                    *radar_xyz[i].tolist(),
                    *radar_gt[i].tolist(),
                    *radar_cam[i].tolist(),
                    radar_pixels[i, 0],
                    radar_pixels[i, 1],
                    int(radar_valid[i]),
                    pix_dist,
                    float(d3[i]),
                ]
            )

    # Summary JSON
    summary = {
        "image": str(args.image.resolve()),
        "gt_file": str(args.gt.resolve()),
        "radar_file": str(args.radar.resolve()),
        "gt_xyz": gt.tolist(),
        "gt_camera_xyz": gt_cam.tolist(),
        "gt_pixel": gt_pixel.tolist(),
        "radar_point_count": int(len(radar_xyz)),
        "valid_projected_radar_count": int(radar_valid.sum()),
        "nearest_radar_index_by_2d": nearest_index,
        "nearest_radar_xyz_by_2d": radar_xyz[nearest_index].tolist(),
        "nearest_radar_pixel": nearest_pixel.tolist(),
        "nearest_2d_distance_px": nearest_distance_px,
        "nearest_radar_index_by_3d": nearest_3d_index,
        "nearest_3d_distance_m": nearest_3d_distance_m,
        "R_gt_from_radar": R_gr.tolist(),
        "t_gt_from_radar_m": t_gr.tolist(),
        "R_camera_from_gt": R_cg.tolist(),
        "t_camera_from_gt_m": t_cg.tolist(),
        "overlay": str(overlay_path.resolve()),
        "csv": str(csv_path.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
