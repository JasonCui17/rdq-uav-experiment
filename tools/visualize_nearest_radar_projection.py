#!/usr/bin/env python3
"""Project the nearest train Radar frame onto one left fisheye image.

This is a visualization-only demo. It performs no GT point selection, target
association, time-offset correction, or motion compensation.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402


def read_train_rows(manifest: Path, sequence: str | None = None) -> list[dict[str, str]]:
    """Read train rows only; reject any unexpected split value."""
    if manifest.name != "train.csv":
        raise ValueError("This demo only accepts a train.csv manifest")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError(f"Manifest is empty or contains a non-train row: {manifest}")
    if sequence is not None:
        rows = [row for row in rows if row["sequence_id"] == sequence]
        if not rows:
            raise ValueError(f"No train rows found for sequence={sequence!r}")
    return rows


def nearest_row(rows: list[dict[str, str]], timestamp: float, key: str) -> dict[str, str]:
    return min(rows, key=lambda row: (abs(float(row[key]) - timestamp), float(row[key])))


def select_image_row(
    rows: list[dict[str, str]], image_time: float | None
) -> dict[str, str]:
    """Choose one deterministic smoke image or nearest requested train image."""
    if image_time is not None:
        return nearest_row(rows, image_time, "image_time")
    # Prefer a frame near the temporal middle instead of a boundary frame.
    ordered = sorted(rows, key=lambda row: float(row["image_time"]))
    return ordered[len(ordered) // 2]


def load_radar_xyz(path: Path) -> tuple[np.ndarray, int, int]:
    array = np.asarray(np.load(path, allow_pickle=False))
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected Radar array (N,>=3), got {array.shape}: {path}")
    raw_count = len(array)
    xyz = np.asarray(array[:, :3], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], raw_count, int((~finite).sum())


def load_transforms(
    radar_calibration: Path, camera_calibration: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    radar_payload = json.loads(radar_calibration.read_text(encoding="utf-8"))
    radar_transform = radar_payload["global_results"]["global_rigid_icp"]["transform"]
    rotation_gt_from_radar = np.asarray(
        radar_transform["rotation_gt_from_radar"], dtype=np.float64
    )
    translation_gt_from_radar = np.asarray(
        radar_transform["translation_gt_from_radar_m"], dtype=np.float64
    )

    camera_payload = json.loads(camera_calibration.read_text(encoding="utf-8"))
    camera_transform = camera_payload["cameras"]["left"]
    rotation_camera_from_gt = np.asarray(
        camera_transform["rotation_camera_from_gt"], dtype=np.float64
    )
    translation_camera_from_gt = np.asarray(
        camera_transform["translation_camera_from_gt_m"], dtype=np.float64
    )
    for name, value, shape in (
        ("rotation_gt_from_radar", rotation_gt_from_radar, (3, 3)),
        ("translation_gt_from_radar", translation_gt_from_radar, (3,)),
        ("rotation_camera_from_gt", rotation_camera_from_gt, (3, 3)),
        ("translation_camera_from_gt", translation_camera_from_gt, (3,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}: expected {shape}, got {value.shape}")
    provenance = {
        "radar_calibration": str(radar_calibration.resolve()),
        "radar_transform_entry": "global_results.global_rigid_icp.transform",
        "radar_euler_xyz_deg": radar_transform.get("rotation_euler_xyz_deg"),
        "camera_calibration": str(camera_calibration.resolve()),
        "camera_transform_entry": "cameras.left",
        "camera_time_offset_present_but_not_used_s": camera_payload.get("time_offset_s"),
    }
    return (
        rotation_gt_from_radar,
        translation_gt_from_radar,
        rotation_camera_from_gt,
        translation_camera_from_gt,
        provenance,
    )


def project_chain(
    radar_xyz: np.ndarray,
    camera: OmniRadtanCamera,
    rotation_gt_from_radar: np.ndarray,
    translation_gt_from_radar: np.ndarray,
    rotation_camera_from_gt: np.ndarray,
    translation_camera_from_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply Radar->GT->camera->OmniRadtan without consulting GT labels."""
    points_gt = transform_points(
        radar_xyz, rotation_gt_from_radar, translation_gt_from_radar
    )
    points_camera = transform_points(
        points_gt, rotation_camera_from_gt, translation_camera_from_gt
    )
    pixels, valid = camera.project(points_camera, require_in_image=True)
    return points_gt, points_camera, pixels, valid


def left_image(source: Path, expected_width: int, expected_height: int) -> Image.Image:
    with Image.open(source) as image:
        image.load()
        image = image.convert("RGB")
    if image.height != expected_height:
        raise ValueError(
            f"Expected image height {expected_height}, got {image.height}: {source}"
        )
    if image.width == expected_width:
        return image
    if image.width >= expected_width * 2:
        return image.crop((0, 0, expected_width, expected_height))
    raise ValueError(
        f"Expected a {expected_width}-px left image or stitched image, got {image.size}: {source}"
    )


def range_color(value: float, lower: float, upper: float) -> tuple[int, int, int]:
    fraction = 0.5 if upper <= lower else float(np.clip((value - lower) / (upper - lower), 0, 1))
    # Near: blue/cyan; far: yellow/red.
    hue = (1.0 - fraction) * (2.0 / 3.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.95, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def render_overlay(
    image: Image.Image,
    pixels: np.ndarray,
    valid: np.ndarray,
    ranges: np.ndarray,
    annotation: list[str],
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    valid_ranges = ranges[valid]
    if len(valid_ranges):
        lower, upper = (float(np.min(valid_ranges)), float(np.max(valid_ranges)))
        order = np.argsort(valid_ranges)[::-1]
        valid_pixels = pixels[valid]
        for index in order:
            u, v = valid_pixels[index]
            color = range_color(float(valid_ranges[index]), lower, upper)
            radius = 4
            draw.ellipse(
                (u - radius, v - radius, u + radius, v + radius),
                fill=(*color, 210), outline=(255, 255, 255, 230), width=1,
            )
    font = ImageFont.load_default()
    line_height = 17
    text_width = max((draw.textbbox((0, 0), line, font=font)[2] for line in annotation), default=0)
    panel = (8, 8, 24 + text_width, 20 + line_height * len(annotation))
    draw.rounded_rectangle(panel, radius=6, fill=(0, 0, 0, 185))
    for line_index, line in enumerate(annotation):
        draw.text((16, 14 + line_index * line_height), line, fill=(255, 255, 255, 255), font=font)
    return output


def write_point_csv(
    path: Path,
    radar_xyz: np.ndarray,
    points_gt: np.ndarray,
    points_camera: np.ndarray,
    pixels: np.ndarray,
    valid: np.ndarray,
) -> None:
    fields = [
        "radar_x", "radar_y", "radar_z", "gt_x", "gt_y", "gt_z",
        "camera_x", "camera_y", "camera_z", "range_m", "projected_u",
        "projected_v", "valid",
    ]
    ranges = np.linalg.norm(radar_xyz, axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for radar, gt, camera_point, pixel, point_range, is_valid in zip(
            radar_xyz, points_gt, points_camera, pixels, ranges, valid
        ):
            writer.writerow({
                "radar_x": radar[0], "radar_y": radar[1], "radar_z": radar[2],
                "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2],
                "camera_x": camera_point[0], "camera_y": camera_point[1],
                "camera_z": camera_point[2], "range_m": point_range,
                "projected_u": pixel[0], "projected_v": pixel[1],
                "valid": int(is_valid),
            })


def matrix_markdown(matrix: np.ndarray) -> str:
    return "\n".join("  [" + ", ".join(f"{value:.9f}" for value in row) + "]" for row in matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--image-time", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "manifests_oracle_left_fixed256_bbox/train.csv",
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml",
    )
    parser.add_argument(
        "--radar-calibration", type=Path,
        default=PROJECT_ROOT / "calibration/radar_frame_resolution.json",
    )
    parser.add_argument(
        "--camera-calibration", type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    rows = read_train_rows(args.manifest, args.sequence)
    image_row = select_image_row(rows, args.image_time)
    sequence = image_row["sequence_id"]
    sequence_rows = [row for row in rows if row["sequence_id"] == sequence]
    image_timestamp = float(image_row["image_time"])
    radar_row = nearest_row(sequence_rows, image_timestamp, "radar_time")
    radar_timestamp = float(radar_row["radar_time"])

    image_path = Path(image_row["image_path"])
    radar_path = Path(radar_row["radar_path"])
    if not image_path.is_absolute():
        image_path = dataset_root / image_path
    if not radar_path.is_absolute():
        radar_path = dataset_root / radar_path
    if not image_path.is_file() or not radar_path.is_file():
        raise FileNotFoundError(f"Missing image or Radar file: {image_path}, {radar_path}")

    radar_xyz, raw_count, nonfinite_count = load_radar_xyz(radar_path)
    radar_r, radar_t, camera_r, camera_t, provenance = load_transforms(
        args.radar_calibration, args.camera_calibration
    )
    points_gt, points_camera, pixels, valid = project_chain(
        radar_xyz, camera, radar_r, radar_t, camera_r, camera_t
    )
    delta_ms = abs(radar_timestamp - image_timestamp) * 1000.0
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" /
        f"nearest_radar_projection_{sequence}_{image_timestamp:.6f}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = output_dir / "radar_projection_left.png"
    csv_path = output_dir / "radar_projection_points.csv"
    summary_path = output_dir / "projection_summary.md"

    source_left = left_image(image_path, camera.width, camera.height)
    annotation = [
        f"sequence: {sequence}",
        f"image timestamp: {image_timestamp:.9f}",
        f"radar timestamp: {radar_timestamp:.9f}",
        f"|delta t|: {delta_ms:.3f} ms",
        f"raw radar points: {raw_count}",
        f"finite radar points: {len(radar_xyz)}",
        f"valid projected points: {int(valid.sum())}",
    ]
    render_overlay(
        source_left, pixels, valid, np.linalg.norm(radar_xyz, axis=1), annotation
    ).save(overlay_path)
    write_point_csv(csv_path, radar_xyz, points_gt, points_camera, pixels, valid)

    summary = f"""# Nearest-Timestamp Radar → Left Fisheye Projection

本结果只使用同 sequence 的 train 数据。没有使用时间偏移、运动补偿、GT 筛点或
target association。

## 输入

- Sequence: `{sequence}`
- Image: `{image_path}`
- Radar: `{radar_path}`
- Image timestamp: `{image_timestamp:.9f}`
- Radar timestamp: `{radar_timestamp:.9f}`
- Absolute timestamp gap: `{delta_ms:.3f} ms`
- Raw / finite / valid projected points: `{raw_count} / {len(radar_xyz)} / {int(valid.sum())}`
- Removed non-finite points: `{nonfinite_count}`

## 数学链

```text
p_gt  = R_gr @ p_r + t_gr
p_cam = R_cg @ p_gt + t_cg
(X, Y, Z)_cam -> OmniRadtan(xi, fu, fv, pu, pv, k1, k2, p1, p2) -> (u, v)
```

代码采用 row-vector 等价实现：`points @ R.T + t`。

### Radar → GT

```text
R_gr =
{matrix_markdown(radar_r)}
t_gr = [{', '.join(f'{value:.9f}' for value in radar_t)}] m
```

### GT → Left Camera

```text
R_cg =
{matrix_markdown(camera_r)}
t_cg = [{', '.join(f'{value:.9f}' for value in camera_t)}] m
```

标定来源：

- `{provenance['radar_calibration']}` → `{provenance['radar_transform_entry']}`
- `{provenance['camera_calibration']}` → `{provenance['camera_transform_entry']}`

标定文件中的 camera/GT 时间偏移 `{provenance['camera_time_offset_present_but_not_used_s']}` 秒
在本 demo 中明确不使用。
"""
    summary_path.write_text(summary, encoding="utf-8")
    run_info = {
        "scope": {"split": "train", "test_read": False, "single_sample": True},
        "sequence": sequence,
        "image_path": str(image_path.resolve()),
        "radar_path": str(radar_path.resolve()),
        "image_timestamp": image_timestamp,
        "radar_timestamp": radar_timestamp,
        "absolute_time_gap_ms": delta_ms,
        "raw_radar_point_count": raw_count,
        "finite_radar_point_count": len(radar_xyz),
        "removed_nonfinite_point_count": nonfinite_count,
        "valid_projected_point_count": int(valid.sum()),
        "rotation_gt_from_radar": radar_r.tolist(),
        "translation_gt_from_radar_m": radar_t.tolist(),
        "rotation_camera_from_gt": camera_r.tolist(),
        "translation_camera_from_gt_m": camera_t.tolist(),
        "outputs": {
            "overlay": str(overlay_path.resolve()),
            "point_csv": str(csv_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
