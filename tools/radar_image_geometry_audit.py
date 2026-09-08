#!/usr/bin/env python3
"""Audit raw Radar-to-fisheye spatial correspondence on train/val only.

The primary path deliberately uses every finite Radar XYZ point. GT is used
only after projection to score correspondence. The optional oracle path is
kept separate because it uses GT to associate and motion-compensate points.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402

RADII_PX = (8.0, 16.0, 32.0, 64.0)
RANGE_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, math.inf))
TIME_GAP_BINS_MS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, math.inf))


def load_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    if expected_split not in {"train", "val"}:
        raise ValueError("This audit permits train and val only")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["split"] != expected_split for row in rows):
        raise ValueError(f"Unexpected or empty split in {path}")
    return rows


def bbox_distance(pixels: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Euclidean point-to-rectangle distance (zero inside the rectangle)."""
    x1, y1, x2, y2 = bbox
    dx = np.maximum(np.maximum(x1 - pixels[:, 0], 0.0), pixels[:, 0] - x2)
    dy = np.maximum(np.maximum(y1 - pixels[:, 1], 0.0), pixels[:, 1] - y2)
    return np.hypot(dx, dy)


def score_pixels(pixels: np.ndarray, bbox: np.ndarray) -> dict[str, Any]:
    center = np.asarray(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
    valid_count = len(pixels)
    if valid_count:
        center_distances = np.linalg.norm(pixels - center[None, :], axis=1)
        rectangle_distances = bbox_distance(pixels, bbox)
        inside = rectangle_distances <= 0.0
        result: dict[str, Any] = {
            "valid_projected_point_count": valid_count,
            "nearest_center_distance_px": float(center_distances.min()),
            "nearest_bbox_distance_px": float(rectangle_distances.min()),
            "points_inside_bbox": int(inside.sum()),
        }
        for radius in RADII_PX:
            result[f"points_within_{int(radius)}px"] = int((center_distances <= radius).sum())
            result[f"coverage_{int(radius)}px"] = int(np.any(center_distances <= radius))
        return result
    result = {
        "valid_projected_point_count": 0,
        "nearest_center_distance_px": math.nan,
        "nearest_bbox_distance_px": math.nan,
        "points_inside_bbox": 0,
    }
    for radius in RADII_PX:
        result[f"points_within_{int(radius)}px"] = 0
        result[f"coverage_{int(radius)}px"] = 0
    return result


def numeric_bbox(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row[f"official_bbox_{key}"]) for key in ("x1", "y1", "x2", "y2")],
        dtype=np.float64,
    )


def deterministic_shuffle(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Half-cycle permutation within sequence; deterministic and distribution preserving."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["sequence_id"]].append(index)
    shuffled: list[dict[str, str] | None] = [None] * len(rows)
    for indices in grouped.values():
        offset = max(1, len(indices) // 2) if len(indices) > 1 else 0
        for local_index, target_index in enumerate(indices):
            shuffled[target_index] = rows[indices[(local_index + offset) % len(indices)]]
    return [row for row in shuffled if row is not None]


def bin_label(value: float, bins: tuple[tuple[float, float], ...], suffix: str) -> str:
    for lower, upper in bins:
        if lower <= value < upper:
            upper_text = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"{lower:g}-{upper_text}{suffix}"
    raise AssertionError(value)


def summarize(rows: list[dict[str, Any]], group_type: str, group_value: str) -> dict[str, Any]:
    nearest = np.asarray([row["nearest_center_distance_px"] for row in rows], dtype=np.float64)
    bbox_nearest = np.asarray([row["nearest_bbox_distance_px"] for row in rows], dtype=np.float64)
    finite = np.isfinite(nearest)
    bbox_finite = np.isfinite(bbox_nearest)
    output: dict[str, Any] = {
        "mode": rows[0]["mode"],
        "split": rows[0].get("split", "combined"),
        "group_type": group_type,
        "group_value": group_value,
        "frames": len(rows),
        "nearest_center_distance_px_mean": float(nearest[finite].mean()) if finite.any() else math.nan,
        "nearest_center_distance_px_median": float(np.median(nearest[finite])) if finite.any() else math.nan,
        "nearest_bbox_distance_px_mean": float(bbox_nearest[bbox_finite].mean()) if bbox_finite.any() else math.nan,
        "nearest_bbox_distance_px_median": float(np.median(bbox_nearest[bbox_finite])) if bbox_finite.any() else math.nan,
        "no_valid_projection_rate": float(np.mean([row["valid_projected_point_count"] == 0 for row in rows])),
        "valid_projected_points_mean": float(np.mean([row["valid_projected_point_count"] for row in rows])),
        "valid_projected_points_median": float(np.median([row["valid_projected_point_count"] for row in rows])),
        "inside_bbox_frame_coverage": float(np.mean([row["points_inside_bbox"] > 0 for row in rows])),
    }
    for radius in RADII_PX:
        output[f"coverage_{int(radius)}px"] = float(np.mean([row[f"coverage_{int(radius)}px"] for row in rows]))
    if "oracle_gate_point_count" in rows[0]:
        output["oracle_gate_nonempty_rate"] = float(np.mean([row["oracle_gate_point_count"] > 0 for row in rows]))
    return output


def grouped_summaries(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for mode in sorted({row["mode"] for row in frame_rows}):
        mode_rows = [row for row in frame_rows if row["mode"] == mode]
        for split in ("train", "val", "combined"):
            split_rows = mode_rows if split == "combined" else [row for row in mode_rows if row["split"] == split]
            if not split_rows:
                continue
            normalized = [{**row, "split": split} for row in split_rows]
            summaries.append(summarize(normalized, "overall", "all"))
        for group_type, key in (("sequence", "sequence_id"), ("range", "range_bin"), ("time_gap", "time_gap_bin")):
            groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in mode_rows:
                groups[(row["split"], row[key])].append(row)
            for (split, value), rows in sorted(groups.items()):
                summaries.append(summarize(rows, group_type, value))
    return summaries


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_radar(path: str) -> np.ndarray:
    points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Radar array (N,>=3), got {points.shape} at {path}")
    points = points[:, :3]
    return points[np.isfinite(points).all(axis=1)]


def project_raw_radar(
    points: np.ndarray,
    camera: OmniRadtanCamera,
    rotation_camera_from_radar: np.ndarray,
    translation_camera_from_radar: np.ndarray,
) -> np.ndarray:
    """Project all raw points without accepting or consulting any GT value."""
    pixels, valid = camera.project(
        transform_points(
            points, rotation_camera_from_radar, translation_camera_from_radar
        ),
        require_in_image=True,
    )
    return pixels[valid]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "manifests_oracle_left_fixed256_bbox")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--with-oracle", action="store_true")
    parser.add_argument("--oracle-gate-m", type=float, default=2.0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fitted = json.loads(args.calibration.read_text(encoding="utf-8"))
    if set(fitted["cameras"]) != {"left"}:
        raise ValueError("Stage 4.9 expects the verified left-camera fit only")
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    camera_fit = fitted["cameras"]["left"]
    rotation = np.asarray(camera_fit["rotation_camera_from_gt"], dtype=np.float64)
    translation = np.asarray(camera_fit["translation_camera_from_gt_m"], dtype=np.float64)
    time_offset = float(fitted["time_offset_s"])
    dataset_root = Path(config["dataset_root"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"stage4_geometry_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    all_frames: list[dict[str, Any]] = []
    trajectories: dict[str, PositionTrajectory] = {}

    for split in ("train", "val"):
        rows = load_rows(args.manifest_dir / f"{split}.csv", split)
        shuffled_rows = deterministic_shuffle(rows)
        distances_seen: list[float] = []
        valid_projected_total = 0
        valid_projection_frames = 0
        covered_32_frames = 0
        split_started = time.perf_counter()
        bar = tqdm(
            zip(rows, shuffled_rows), total=len(rows), desc=split, unit="frame",
            bar_format="{desc} | {n_fmt}/{total_fmt} | {postfix} | ETA {remaining}",
        )
        for processed, (row, shuffled_gt_row) in enumerate(bar, start=1):
            radar_path = Path(row["radar_path"])
            if not radar_path.is_absolute():
                radar_path = dataset_root / radar_path
            points = load_radar(str(radar_path))
            pixels = project_raw_radar(points, camera, rotation, translation)
            valid_projected_total += len(pixels)
            valid_projection_frames += int(len(pixels) > 0)
            image_radar_gap_ms = abs(float(row["image_time"]) - float(row["radar_time"])) * 1000.0
            common = {
                "sample_id": row["sample_id"], "split": split,
                "sequence_id": row["sequence_id"], "image_time": float(row["image_time"]),
                "radar_time": float(row["radar_time"]), "image_radar_time_gap_ms": image_radar_gap_ms,
                "raw_radar_point_count": len(points),
                "time_gap_bin": bin_label(image_radar_gap_ms, TIME_GAP_BINS_MS, "ms"),
            }
            real = {**common, "mode": "raw", "gt_source_sample_id": row["sample_id"],
                    "gt_range_m": float(row["distance_m"]),
                    "range_bin": bin_label(float(row["distance_m"]), RANGE_BINS, "m"),
                    **score_pixels(pixels, numeric_bbox(row))}
            all_frames.append(real)
            covered_32_frames += int(real["coverage_32px"])
            if math.isfinite(real["nearest_center_distance_px"]):
                distances_seen.append(real["nearest_center_distance_px"])

            shuffled_range = float(shuffled_gt_row["distance_m"])
            all_frames.append({
                **common, "mode": "shuffle_same_sequence",
                "gt_source_sample_id": shuffled_gt_row["sample_id"],
                "gt_range_m": shuffled_range,
                "range_bin": bin_label(shuffled_range, RANGE_BINS, "m"),
                **score_pixels(pixels, numeric_bbox(shuffled_gt_row)),
            })

            if args.with_oracle:
                sequence = row["sequence_id"]
                if sequence not in trajectories:
                    trajectories[sequence] = PositionTrajectory.from_directory(
                        dataset_root / sequence / "ground_truth"
                    )
                gt_radar, valid_radar = trajectories[sequence].evaluate(float(row["radar_time"]))
                gt_image, valid_image = trajectories[sequence].evaluate(float(row["image_time"]) + time_offset)
                target_points = np.empty((0, 3), dtype=np.float64)
                if bool(valid_radar) and bool(valid_image):
                    gate = np.linalg.norm(points - gt_radar, axis=1) <= args.oracle_gate_m
                    target_points = points[gate] + (gt_image - gt_radar)
                oracle_pixels, oracle_valid = camera.project(
                    transform_points(target_points, rotation, translation), require_in_image=True
                )
                oracle_range = float(row["distance_m"])
                all_frames.append({
                    **common, "mode": "oracle_gt_gate_motion", "gt_source_sample_id": row["sample_id"],
                    "gt_range_m": oracle_range,
                    "range_bin": bin_label(oracle_range, RANGE_BINS, "m"),
                    "oracle_gate_point_count": len(target_points),
                    **score_pixels(oracle_pixels[oracle_valid], numeric_bbox(row)),
                })
            median_distance = float(np.median(distances_seen)) if distances_seen else math.nan
            elapsed = max(time.perf_counter() - split_started, 1e-12)
            bar.set_postfix_str(
                f"valid%={100*valid_projection_frames/processed:.1f}% | "
                f"median nearest={median_distance:.1f}px | "
                f"coverage@32={covered_32_frames/processed:.3f} | "
                f"samples/s={processed/elapsed:.1f}",
                refresh=False,
            )
        bar.close()

    grouped = grouped_summaries(all_frames)
    write_csv(all_frames, output_dir / "frame_metrics.csv")
    write_csv(grouped, output_dir / "grouped_metrics.csv")
    overall = {
        row["mode"]: row for row in grouped
        if row["split"] == "combined" and row["group_type"] == "overall"
    }
    report = {
        "scope": {"splits_read": ["train", "val"], "test_read": False,
                  "train_frames": sum(row["split"] == "train" and row["mode"] == "raw" for row in all_frames),
                  "val_frames": sum(row["split"] == "val" and row["mode"] == "raw" for row in all_frames)},
        "primary_path": "all finite Radar XYZ -> fitted GT/radar-frame-to-left-camera transform -> OmniRadtan projection -> left half of 2560x960 stitched canvas",
        "gt_usage_primary": "scoring only; no GT gate, target selection, or motion compensation",
        "null_control": "deterministic half-cycle GT bbox permutation within each sequence and split",
        "calibration_limit": "Only the left camera has a fitted extrinsic; official 2D boxes are left-view only. The Radar/GT shared-frame convention is inferred, not supplied as a published camera-radar extrinsic.",
        "oracle": {"enabled": args.with_oracle, "gate_m": args.oracle_gate_m,
                   "description": "GT-gated target points only, motion-compensated from radar time to effective image time"},
        "overall": overall,
        "outputs": {"frame_metrics": "frame_metrics.csv", "grouped_metrics": "grouped_metrics.csv"},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
