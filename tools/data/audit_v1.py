#!/usr/bin/env python3
"""Audit MMAUD V1 data for the five-class RGB+Radar experiment.

The audit is intentionally read-only for the source dataset. It scans all GT,
image headers and enhanced-radar arrays, checks modality timestamps, and pairs
each GT timestamp with its nearest image and radar frame.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLASSES = ("Mavic2", "Mavic3", "Avata", "M300", "Pham4")
MODALITIES = (
    "ground_truth",
    "image",
    "radar_enhance_pcl",
    "lidar_360",
    "livox_avia",
)
EXPECTED_SUFFIX = {
    "ground_truth": ".npy",
    "image": ".png",
    "radar_enhance_pcl": ".npy",
    "lidar_360": ".npy",
    "livox_avia": ".npy",
}


def percentile(values: list[float] | np.ndarray, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def parse_timestamp(path: Path) -> float | None:
    try:
        value = float(path.stem)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def collect_files(
    directory: Path,
    suffix: str,
    class_name: str,
    modality: str,
    issues: list[dict[str, str]],
) -> tuple[list[Path], np.ndarray]:
    if not directory.is_dir():
        issues.append(
            {
                "class": class_name,
                "modality": modality,
                "path": str(directory),
                "issue": "missing_directory",
                "detail": "",
            }
        )
        return [], np.empty(0, dtype=np.float64)

    files: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != suffix:
            issues.append(
                {
                    "class": class_name,
                    "modality": modality,
                    "path": str(path),
                    "issue": "unexpected_extension",
                    "detail": f"expected {suffix}",
                }
            )
            continue
        timestamp = parse_timestamp(path)
        if timestamp is None:
            issues.append(
                {
                    "class": class_name,
                    "modality": modality,
                    "path": str(path),
                    "issue": "invalid_timestamp_filename",
                    "detail": path.name,
                }
            )
            continue
        files.append((timestamp, path))

    files.sort(key=lambda item: item[0])
    timestamps = np.asarray([item[0] for item in files], dtype=np.float64)
    paths = [item[1] for item in files]
    if len(timestamps) > 1:
        duplicate_indices = np.flatnonzero(np.diff(timestamps) == 0)
        for index in duplicate_indices:
            issues.append(
                {
                    "class": class_name,
                    "modality": modality,
                    "path": str(paths[index + 1]),
                    "issue": "duplicate_timestamp",
                    "detail": f"{timestamps[index + 1]:.9f}",
                }
            )
    return paths, timestamps


def timing_stats(timestamps: np.ndarray) -> dict[str, float | int]:
    if len(timestamps) == 0:
        return {
            "count": 0,
            "start": math.nan,
            "end": math.nan,
            "duration_s": math.nan,
            "interval_median_ms": math.nan,
            "interval_p95_ms": math.nan,
            "max_gap_s": math.nan,
            "segments_gap_gt_1s": 0,
        }
    delta = np.diff(timestamps)
    return {
        "count": int(len(timestamps)),
        "start": float(timestamps[0]),
        "end": float(timestamps[-1]),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "interval_median_ms": percentile(delta * 1000.0, 50),
        "interval_p95_ms": percentile(delta * 1000.0, 95),
        "max_gap_s": float(delta.max()) if len(delta) else math.nan,
        "segments_gap_gt_1s": int(np.count_nonzero(delta > 1.0) + 1),
    }


def read_png_ihdr(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature/header length")
    if header[12:16] != b"IHDR":
        raise ValueError("missing PNG IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    return width, height, bit_depth, color_type


def scan_images(
    paths: list[Path], class_name: str, issues: list[dict[str, str]]
) -> dict[str, object]:
    layouts: Counter[str] = Counter()
    invalid = 0
    for path in paths:
        try:
            width, height, bit_depth, color_type = read_png_ihdr(path)
            layouts[f"{width}x{height}/bd{bit_depth}/ct{color_type}"] += 1
        except Exception as exc:  # Keep auditing after a bad image.
            invalid += 1
            issues.append(
                {
                    "class": class_name,
                    "modality": "image",
                    "path": str(path),
                    "issue": "invalid_png_header",
                    "detail": str(exc),
                }
            )
    return {"invalid": invalid, "layouts": dict(layouts)}


def scan_gt(
    paths: list[Path], class_name: str, issues: list[dict[str, str]]
) -> tuple[dict[str, object], dict[Path, np.ndarray]]:
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    load_errors = 0
    nonfinite_files = 0
    values: dict[Path, np.ndarray] = {}
    distances: list[float] = []
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    for path in paths:
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            load_errors += 1
            issues.append(
                {
                    "class": class_name,
                    "modality": "ground_truth",
                    "path": str(path),
                    "issue": "npy_load_error",
                    "detail": str(exc),
                }
            )
            continue
        shapes[str(array.shape)] += 1
        dtypes[str(array.dtype)] += 1
        if array.shape != (3,):
            issues.append(
                {
                    "class": class_name,
                    "modality": "ground_truth",
                    "path": str(path),
                    "issue": "unexpected_shape",
                    "detail": str(array.shape),
                }
            )
            continue
        xyz = np.asarray(array, dtype=np.float64)
        if not np.isfinite(xyz).all():
            nonfinite_files += 1
            issues.append(
                {
                    "class": class_name,
                    "modality": "ground_truth",
                    "path": str(path),
                    "issue": "nonfinite_value",
                    "detail": repr(xyz.tolist()),
                }
            )
            continue
        values[path] = xyz
        distances.append(float(np.linalg.norm(xyz)))
        xyz_min = np.minimum(xyz_min, xyz)
        xyz_max = np.maximum(xyz_max, xyz)

    stats: dict[str, object] = {
        "load_errors": load_errors,
        "nonfinite_files": nonfinite_files,
        "shapes": dict(shapes),
        "dtypes": dict(dtypes),
        "valid": len(values),
        "distance_min": float(min(distances)) if distances else math.nan,
        "distance_p25": percentile(distances, 25),
        "distance_median": percentile(distances, 50),
        "distance_p75": percentile(distances, 75),
        "distance_p95": percentile(distances, 95),
        "distance_max": float(max(distances)) if distances else math.nan,
        "xyz_min": xyz_min.tolist() if distances else [math.nan] * 3,
        "xyz_max": xyz_max.tolist() if distances else [math.nan] * 3,
    }
    return stats, values


def scan_radar(
    paths: list[Path], class_name: str, issues: list[dict[str, str]]
) -> tuple[dict[str, object], dict[Path, int]]:
    shape_families: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    load_errors = 0
    invalid_shapes = 0
    nonfinite_files = 0
    empty_files = 0
    point_counts: list[int] = []
    valid_point_counts: dict[Path, int] = {}
    zero_xyz_points = 0
    total_points = 0
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    far_points_gt_100m = 0
    far_frames_gt_100m = 0

    for path in paths:
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            load_errors += 1
            issues.append(
                {
                    "class": class_name,
                    "modality": "radar_enhance_pcl",
                    "path": str(path),
                    "issue": "npy_load_error",
                    "detail": str(exc),
                }
            )
            continue
        dtypes[str(array.dtype)] += 1
        if array.size == 0:
            empty_files += 1
            shape_families["empty"] += 1
            point_counts.append(0)
            valid_point_counts[path] = 0
            continue
        if array.ndim != 2 or array.shape[1] != 3:
            invalid_shapes += 1
            shape_families[str(array.shape)] += 1
            issues.append(
                {
                    "class": class_name,
                    "modality": "radar_enhance_pcl",
                    "path": str(path),
                    "issue": "unexpected_shape",
                    "detail": str(array.shape),
                }
            )
            continue
        shape_families["N,3"] += 1
        count = int(array.shape[0])
        point_counts.append(count)
        valid_point_counts[path] = count
        total_points += count
        finite_rows = np.isfinite(array).all(axis=1)
        if not finite_rows.all():
            nonfinite_files += 1
            issues.append(
                {
                    "class": class_name,
                    "modality": "radar_enhance_pcl",
                    "path": str(path),
                    "issue": "nonfinite_value",
                    "detail": f"bad_rows={int(np.count_nonzero(~finite_rows))}",
                }
            )
        if finite_rows.any():
            finite_xyz = np.asarray(array[finite_rows], dtype=np.float64)
            xyz_min = np.minimum(xyz_min, finite_xyz.min(axis=0))
            xyz_max = np.maximum(xyz_max, finite_xyz.max(axis=0))
            far_mask = np.linalg.norm(finite_xyz, axis=1) > 100.0
            far_count = int(np.count_nonzero(far_mask))
            if far_count:
                far_points_gt_100m += far_count
                far_frames_gt_100m += 1
                issues.append(
                    {
                        "class": class_name,
                        "modality": "radar_enhance_pcl",
                        "path": str(path),
                        "issue": "radar_point_over_100m",
                        "detail": f"points={far_count}",
                    }
                )
        zero_xyz_points += int(np.count_nonzero(np.all(array == 0, axis=1)))

    stats: dict[str, object] = {
        "load_errors": load_errors,
        "invalid_shapes": invalid_shapes,
        "nonfinite_files": nonfinite_files,
        "empty_files": empty_files,
        "empty_ratio": empty_files / len(paths) if paths else math.nan,
        "shape_families": dict(shape_families),
        "dtypes": dict(dtypes),
        "total_points": total_points,
        "zero_xyz_points": zero_xyz_points,
        "points_min": int(min(point_counts)) if point_counts else 0,
        "points_p05": percentile(point_counts, 5),
        "points_median": percentile(point_counts, 50),
        "points_p95": percentile(point_counts, 95),
        "points_max": int(max(point_counts)) if point_counts else 0,
        "frames_gt_512": int(np.count_nonzero(np.asarray(point_counts) > 512)),
        "frames_gt_512_ratio": float(np.mean(np.asarray(point_counts) > 512))
        if point_counts
        else math.nan,
        "frames_gt_768": int(np.count_nonzero(np.asarray(point_counts) > 768)),
        "frames_gt_768_ratio": float(np.mean(np.asarray(point_counts) > 768))
        if point_counts
        else math.nan,
        "xyz_min": xyz_min.tolist() if total_points else [math.nan] * 3,
        "xyz_max": xyz_max.tolist() if total_points else [math.nan] * 3,
        "far_points_gt_100m": far_points_gt_100m,
        "far_frames_gt_100m": far_frames_gt_100m,
    }
    return stats, valid_point_counts


def nearest_index(sorted_values: np.ndarray, query: float) -> int | None:
    if len(sorted_values) == 0:
        return None
    index = int(np.searchsorted(sorted_values, query))
    candidates = []
    if index < len(sorted_values):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    return min(candidates, key=lambda item: abs(float(sorted_values[item]) - query))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path, output_dir: Path, sync_threshold_s: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    modality_rows: list[dict[str, object]] = []
    sync_rows: list[dict[str, object]] = []

    class_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    for unexpected in sorted(class_dirs - set(CLASSES) - {"audit"}):
        issues.append(
            {
                "class": "",
                "modality": "",
                "path": str(root / unexpected),
                "issue": "unexpected_class_directory",
                "detail": "",
            }
        )

    for class_id, class_name in enumerate(CLASSES):
        class_root = root / class_name
        print(f"Auditing {class_name} ...", flush=True)
        files: dict[str, list[Path]] = {}
        timestamps: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            paths, times = collect_files(
                class_root / modality,
                EXPECTED_SUFFIX[modality],
                class_name,
                modality,
                issues,
            )
            files[modality] = paths
            timestamps[modality] = times
            timing = timing_stats(times)
            modality_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "modality": modality,
                    **timing,
                }
            )

        expected_dirs = set(MODALITIES)
        actual_dirs = {path.name for path in class_root.iterdir() if path.is_dir()}
        extra_dirs = sorted(actual_dirs - expected_dirs)

        image_stats = scan_images(files["image"], class_name, issues)
        gt_stats, gt_values = scan_gt(files["ground_truth"], class_name, issues)
        radar_stats, radar_points = scan_radar(
            files["radar_enhance_pcl"], class_name, issues
        )

        # LiDAR is not used by the first-stage experiment. Spot-check its first,
        # middle and last arrays while still auditing all filenames/timestamps.
        lidar_samples: dict[str, str] = {}
        for modality in ("lidar_360", "livox_avia"):
            paths = files[modality]
            indices = sorted({0, len(paths) // 2, len(paths) - 1}) if paths else []
            descriptions = []
            for index in indices:
                path = paths[index]
                try:
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                    descriptions.append(f"{array.shape}/{array.dtype}")
                except Exception as exc:
                    descriptions.append(f"ERROR:{exc}")
                    issues.append(
                        {
                            "class": class_name,
                            "modality": modality,
                            "path": str(path),
                            "issue": "sample_npy_load_error",
                            "detail": str(exc),
                        }
                    )
            lidar_samples[modality] = ";".join(descriptions)

        class_sync_rows: list[dict[str, object]] = []
        for gt_index, (gt_path, gt_time) in enumerate(
            zip(files["ground_truth"], timestamps["ground_truth"])
        ):
            image_index = nearest_index(timestamps["image"], float(gt_time))
            radar_index = nearest_index(timestamps["radar_enhance_pcl"], float(gt_time))
            image_path = files["image"][image_index] if image_index is not None else None
            radar_path = (
                files["radar_enhance_pcl"][radar_index]
                if radar_index is not None
                else None
            )
            image_time = (
                float(timestamps["image"][image_index])
                if image_index is not None
                else math.nan
            )
            radar_time = (
                float(timestamps["radar_enhance_pcl"][radar_index])
                if radar_index is not None
                else math.nan
            )
            dt_image = image_time - float(gt_time)
            dt_radar = radar_time - float(gt_time)
            xyz = gt_values.get(gt_path)
            distance = float(np.linalg.norm(xyz)) if xyz is not None else math.nan
            num_radar_points = radar_points.get(radar_path, -1) if radar_path else -1
            valid_gt = xyz is not None
            valid_sync = (
                valid_gt
                and math.isfinite(dt_image)
                and math.isfinite(dt_radar)
                and abs(dt_image) <= sync_threshold_s
                and abs(dt_radar) <= sync_threshold_s
            )
            row: dict[str, object] = {
                "sample_id": f"{class_name}_{gt_index:06d}",
                "class_id": class_id,
                "class_name": class_name,
                "gt_time": f"{gt_time:.9f}",
                "image_time": f"{image_time:.9f}" if math.isfinite(image_time) else "",
                "radar_time": f"{radar_time:.9f}" if math.isfinite(radar_time) else "",
                "dt_image_s": dt_image,
                "dt_radar_s": dt_radar,
                "abs_dt_image_s": abs(dt_image),
                "abs_dt_radar_s": abs(dt_radar),
                "gt_path": str(gt_path.relative_to(root)),
                "image_path": str(image_path.relative_to(root)) if image_path else "",
                "radar_path": str(radar_path.relative_to(root)) if radar_path else "",
                "gt_x": float(xyz[0]) if xyz is not None else math.nan,
                "gt_y": float(xyz[1]) if xyz is not None else math.nan,
                "gt_z": float(xyz[2]) if xyz is not None else math.nan,
                "distance_m": distance,
                "radar_points": num_radar_points,
                "radar_empty": int(num_radar_points == 0),
                "valid_sync_40ms": int(valid_sync),
            }
            class_sync_rows.append(row)
            sync_rows.append(row)

        abs_dt_image = [float(row["abs_dt_image_s"]) for row in class_sync_rows]
        abs_dt_radar = [float(row["abs_dt_radar_s"]) for row in class_sync_rows]
        valid_sync_count = sum(int(row["valid_sync_40ms"]) for row in class_sync_rows)
        nearest_empty_count = sum(int(row["radar_empty"]) for row in class_sync_rows)

        summary: dict[str, object] = {
            "class_id": class_id,
            "class_name": class_name,
            "gt_count": len(files["ground_truth"]),
            "image_count": len(files["image"]),
            "radar_count": len(files["radar_enhance_pcl"]),
            "lidar_360_count": len(files["lidar_360"]),
            "livox_avia_count": len(files["livox_avia"]),
            "extra_directories": ";".join(extra_dirs),
            "image_invalid": image_stats["invalid"],
            "image_layouts": repr(image_stats["layouts"]),
            "gt_valid": gt_stats["valid"],
            "gt_shapes": repr(gt_stats["shapes"]),
            "gt_dtypes": repr(gt_stats["dtypes"]),
            "distance_min_m": gt_stats["distance_min"],
            "distance_p25_m": gt_stats["distance_p25"],
            "distance_median_m": gt_stats["distance_median"],
            "distance_p75_m": gt_stats["distance_p75"],
            "distance_p95_m": gt_stats["distance_p95"],
            "distance_max_m": gt_stats["distance_max"],
            "radar_empty_frames": radar_stats["empty_files"],
            "radar_empty_ratio": radar_stats["empty_ratio"],
            "radar_invalid_shapes": radar_stats["invalid_shapes"],
            "radar_nonfinite_files": radar_stats["nonfinite_files"],
            "radar_zero_xyz_points": radar_stats["zero_xyz_points"],
            "radar_points_p05": radar_stats["points_p05"],
            "radar_points_median": radar_stats["points_median"],
            "radar_points_p95": radar_stats["points_p95"],
            "radar_points_max": radar_stats["points_max"],
            "radar_frames_gt_512": radar_stats["frames_gt_512"],
            "radar_frames_gt_512_ratio": radar_stats["frames_gt_512_ratio"],
            "radar_frames_gt_768": radar_stats["frames_gt_768"],
            "radar_frames_gt_768_ratio": radar_stats["frames_gt_768_ratio"],
            "radar_xyz_min": repr(radar_stats["xyz_min"]),
            "radar_xyz_max": repr(radar_stats["xyz_max"]),
            "radar_far_points_gt_100m": radar_stats["far_points_gt_100m"],
            "radar_far_frames_gt_100m": radar_stats["far_frames_gt_100m"],
            "image_abs_dt_median_ms": percentile(abs_dt_image, 50) * 1000.0,
            "image_abs_dt_p95_ms": percentile(abs_dt_image, 95) * 1000.0,
            "image_abs_dt_max_ms": max(abs_dt_image) * 1000.0 if abs_dt_image else math.nan,
            "radar_abs_dt_median_ms": percentile(abs_dt_radar, 50) * 1000.0,
            "radar_abs_dt_p95_ms": percentile(abs_dt_radar, 95) * 1000.0,
            "radar_abs_dt_max_ms": max(abs_dt_radar) * 1000.0 if abs_dt_radar else math.nan,
            "valid_sync_40ms": valid_sync_count,
            "valid_sync_ratio": valid_sync_count / len(class_sync_rows)
            if class_sync_rows
            else math.nan,
            "nearest_radar_empty": nearest_empty_count,
            "nearest_radar_empty_ratio": nearest_empty_count / len(class_sync_rows)
            if class_sync_rows
            else math.nan,
            "lidar_360_samples": lidar_samples["lidar_360"],
            "livox_avia_samples": lidar_samples["livox_avia"],
        }
        summaries.append(summary)

    summary_fields = list(summaries[0].keys())
    modality_fields = list(modality_rows[0].keys())
    sync_fields = list(sync_rows[0].keys())
    distance_bin_rows: list[dict[str, object]] = []
    distance_bins = (("0-10", 0.0, 10.0), ("10-20", 10.0, 20.0), ("20-30", 20.0, 30.0), ("30+", 30.0, math.inf))
    for class_name in CLASSES:
        class_distances = [
            float(row["distance_m"])
            for row in sync_rows
            if row["class_name"] == class_name and math.isfinite(float(row["distance_m"]))
        ]
        for label, lower, upper in distance_bins:
            count = sum(lower <= value < upper for value in class_distances)
            distance_bin_rows.append(
                {
                    "class_name": class_name,
                    "distance_bin_m": label,
                    "count": count,
                    "class_ratio": count / len(class_distances) if class_distances else math.nan,
                }
            )
    write_csv(output_dir / "dataset_summary.csv", summaries, summary_fields)
    write_csv(output_dir / "modality_timing.csv", modality_rows, modality_fields)
    write_csv(output_dir / "sync_samples.csv", sync_rows, sync_fields)
    write_csv(
        output_dir / "distance_bin_counts.csv",
        distance_bin_rows,
        ["class_name", "distance_bin_m", "count", "class_ratio"],
    )
    write_csv(
        output_dir / "issues.csv",
        issues,
        ["class", "modality", "path", "issue", "detail"],
    )

    total_gt = sum(int(row["gt_count"]) for row in summaries)
    total_valid = sum(int(row["valid_sync_40ms"]) for row in summaries)
    total_empty = sum(int(row["radar_empty_frames"]) for row in summaries)
    total_radar = sum(int(row["radar_count"]) for row in summaries)
    report_lines = [
        "# MMAUD V1 五类数据审计报告",
        "",
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"- 数据根目录：`{root}`",
        f"- 同步规则：以 GT 为锚点，最近邻 Image/Radar，双侧阈值 `{sync_threshold_s * 1000:.0f} ms`",
        "- 图像检查：全量 PNG IHDR；解压过程已执行 ZIP 数据校验",
        "- GT/Radar 检查：全量加载；LiDAR 第一阶段不使用，仅审计时间戳并抽查首/中/末帧",
        "",
        "## 总览",
        "",
        "| 类别 | GT | Image | Radar | 空 Radar | 有效同步 | 有效率 | 距离 min/median/p95/max (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report_lines.append(
            f"| {row['class_name']} | {row['gt_count']} | {row['image_count']} | "
            f"{row['radar_count']} | {row['radar_empty_frames']} | {row['valid_sync_40ms']} | "
            f"{100 * float(row['valid_sync_ratio']):.2f}% | "
            f"{fmt(float(row['distance_min_m']))}/{fmt(float(row['distance_median_m']))}/"
            f"{fmt(float(row['distance_p95_m']))}/{fmt(float(row['distance_max_m']))} |"
        )
    report_lines.extend(
        [
            "",
            f"合计 GT `{total_gt}`，40 ms 内同时匹配图像和雷达 `{total_valid}`（`{100 * total_valid / total_gt:.2f}%`）。",
            f"Radar 空帧 `{total_empty}/{total_radar}`（`{100 * total_empty / total_radar:.2f}%`）。",
            "",
            "## 时间同步",
            "",
            "| 类别 | Image abs dt median/p95/max (ms) | Radar abs dt median/p95/max (ms) | GT 最近 Radar 为空 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        report_lines.append(
            f"| {row['class_name']} | {fmt(float(row['image_abs_dt_median_ms']))}/"
            f"{fmt(float(row['image_abs_dt_p95_ms']))}/{fmt(float(row['image_abs_dt_max_ms']))} | "
            f"{fmt(float(row['radar_abs_dt_median_ms']))}/{fmt(float(row['radar_abs_dt_p95_ms']))}/"
            f"{fmt(float(row['radar_abs_dt_max_ms']))} | {row['nearest_radar_empty']} |"
        )
    report_lines.extend(
        [
            "",
            "## Radar 点数",
            "",
            "| 类别 | P05 | Median | P95 | Max | >512 帧 | >768 帧 | >100m 点/帧 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        report_lines.append(
            f"| {row['class_name']} | {fmt(float(row['radar_points_p05']), 1)} | "
            f"{fmt(float(row['radar_points_median']), 1)} | {fmt(float(row['radar_points_p95']), 1)} | "
            f"{row['radar_points_max']} | {row['radar_frames_gt_512']} "
            f"({100 * float(row['radar_frames_gt_512_ratio']):.2f}%) | "
            f"{row['radar_frames_gt_768']} "
            f"({100 * float(row['radar_frames_gt_768_ratio']):.2f}%) | "
            f"{row['radar_far_points_gt_100m']}/{row['radar_far_frames_gt_100m']} |"
        )
    report_lines.extend(
        [
            "",
            "## 距离桶样本构成",
            "",
            "| 类别 | 0-10 m | 10-20 m | 20-30 m | 30+ m |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name in CLASSES:
        counts = {
            str(row["distance_bin_m"]): int(row["count"])
            for row in distance_bin_rows
            if row["class_name"] == class_name
        }
        report_lines.append(
            f"| {class_name} | {counts['0-10']} | {counts['10-20']} | "
            f"{counts['20-30']} | {counts['30+']} |"
        )
    report_lines.extend(
        [
            "",
            "## 目录与格式",
            "",
            "| 类别 | 图像布局 | GT shape/dtype | 额外目录 |",
            "|---|---|---|---|",
        ]
    )
    for row in summaries:
        report_lines.append(
            f"| {row['class_name']} | `{row['image_layouts']}` | "
            f"`{row['gt_shapes']}` / `{row['gt_dtypes']}` | "
            f"`{row['extra_directories'] or '-'}` |"
        )
    report_lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `dataset_summary.csv`：类别级质量、距离、Radar 点数和同步汇总",
            "- `modality_timing.csv`：每类每模态时间范围、采样间隔和时间断点",
            "- `sync_samples.csv`：逐 GT 样本最近邻配对，可作为 manifest 的输入",
            "- `distance_bin_counts.csv`：各类别在距离桶中的样本构成",
            "- `issues.csv`：格式、加载、非有限值和目录异常",
            "",
            "注意：`sync_samples.csv` 只是审计配对结果，尚未进行 temporal block 划分。",
        ]
    )
    (output_dir / "AUDIT_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"Report written to {output_dir / 'AUDIT_REPORT.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/official/v1"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sync-threshold-ms", type=float, default=40.0)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.root / "audit")
    audit(args.root.resolve(), output_dir.resolve(), args.sync_threshold_ms / 1000.0)


if __name__ == "__main__":
    main()
