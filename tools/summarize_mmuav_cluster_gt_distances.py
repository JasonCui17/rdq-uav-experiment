#!/usr/bin/env python3
"""Summarize raw Mid360 cluster-center distances to nearest-timestamp MMAUD GT.

This tool only reads existing source-faithful eps=2 cluster diagnostics. It does
not run DBSCAN/LSTM and deliberately applies no Mid360-to-GT extrinsic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration.trajectory import PositionTrajectory  # noqa: E402


COORDINATE_WARNING = "RAW Mid360 center vs GT coordinates; no verified extrinsic applied"
DIAGNOSTIC_PATTERN = "*/*/diagnostics/pre_lstm_clusters/cluster_diagnostics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=PROJECT_ROOT / "outputs/mmuav_eps_diagnostics_all_sequences",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/v1"),
    )
    parser.add_argument(
        "--single-file", type=Path, default=None,
        help="Analyze only one existing source-faithful cluster_diagnostics.json.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--failures-csv", type=Path, default=None)
    return parser.parse_args()


def nearest_gt(trajectory: PositionTrajectory, query_timestamp: float) -> tuple[float, np.ndarray]:
    insertion = int(np.searchsorted(trajectory.timestamps, query_timestamp))
    candidates = [
        index for index in (insertion - 1, insertion)
        if 0 <= index < len(trajectory.timestamps)
    ]
    if not candidates:
        raise ValueError("GT trajectory is empty")
    index = min(
        candidates,
        key=lambda item: (
            abs(float(trajectory.timestamps[item]) - query_timestamp),
            float(trajectory.timestamps[item]),
        ),
    )
    return float(trajectory.timestamps[index]), trajectory.positions[index].copy()


def diagnostic_identity(path: Path, input_root: Path) -> tuple[str, str]:
    relative = path.resolve().relative_to(input_root.resolve())
    if len(relative.parts) < 5:
        raise ValueError(f"Unexpected diagnostic path layout: {relative}")
    if tuple(relative.parts[-3:]) != (
        "diagnostics", "pre_lstm_clusters", "cluster_diagnostics.json"
    ):
        raise ValueError(f"Not a source-faithful pre_lstm diagnostic: {relative}")
    return relative.parts[0], relative.parts[1]


def parse_diagnostic(
    path: Path, input_root: Path, trajectory: PositionTrajectory,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    sequence_id, unit_id = diagnostic_identity(path, input_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    dbscan = payload.get("dbscan", {})
    if float(dbscan.get("eps")) != 2.0 or int(dbscan.get("min_samples")) != 10:
        raise ValueError(f"Expected source eps=2/min_samples=10, got {dbscan}")

    raw_timestamps = payload.get("frame_timestamps")
    if not isinstance(raw_timestamps, list) or len(raw_timestamps) != 20:
        raise ValueError(
            f"Expected exactly 20 frame_timestamps, got "
            f"{None if raw_timestamps is None else len(raw_timestamps)}"
        )
    timestamps = np.asarray([float(value) for value in raw_timestamps], dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError("frame_timestamps contains NaN/Inf")
    query_timestamp = float(np.median(timestamps))
    nearest_timestamp, gt_xyz = nearest_gt(trajectory, query_timestamp)
    if gt_xyz.shape != (3,) or not np.isfinite(gt_xyz).all():
        raise ValueError(f"Invalid nearest GT XYZ shape/value: {gt_xyz}")

    clusters_payload = payload.get("clusters")
    if not isinstance(clusters_payload, list):
        raise ValueError("clusters must be a list")
    clusters: dict[int, dict[str, Any]] = {}
    for cluster in clusters_payload:
        cluster_id = int(cluster["cluster_id"])
        if cluster_id < 0 or cluster_id in clusters:
            raise ValueError(f"Invalid/duplicate cluster_id: {cluster_id}")
        center = np.asarray(cluster["center_xyz"], dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"cluster {cluster_id} has invalid center_xyz: {center}")
        num_points = int(cluster["num_points"])
        if num_points < 0:
            raise ValueError(f"cluster {cluster_id} has negative num_points")
        clusters[cluster_id] = {
            "num_points": num_points,
            "center": center,
            "distance": float(np.linalg.norm(center - gt_xyz)),
        }

    nearest_id = min(clusters, key=lambda item: clusters[item]["distance"]) if clusters else None
    nearest = clusters.get(nearest_id) if nearest_id is not None else None
    row = {
        "sequence_id": sequence_id,
        "unit_id": unit_id,
        "diagnostic_file": str(path.resolve()),
        "query_timestamp": query_timestamp,
        "window_start_timestamp": float(timestamps[0]),
        "window_end_timestamp": float(timestamps[-1]),
        "nearest_gt_timestamp": nearest_timestamp,
        "gt_time_gap_ms": abs(nearest_timestamp - query_timestamp) * 1000.0,
        "gt_x": float(gt_xyz[0]),
        "gt_y": float(gt_xyz[1]),
        "gt_z": float(gt_xyz[2]),
        "dbscan_cluster_count": len(clusters),
        "coordinate_warning": COORDINATE_WARNING,
        "nearest_cluster_id": nearest_id,
        "nearest_cluster_distance_m": None if nearest is None else nearest["distance"],
        "nearest_cluster_center_x": None if nearest is None else float(nearest["center"][0]),
        "nearest_cluster_center_y": None if nearest is None else float(nearest["center"][1]),
        "nearest_cluster_center_z": None if nearest is None else float(nearest["center"][2]),
        "nearest_cluster_num_points": None if nearest is None else nearest["num_points"],
    }
    return row, clusters


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    fields = ("diagnostic_file", "sequence_id", "unit_id", "stage", "error")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or args.input_root / "cluster_gt_distance_summary.csv"
    failures_csv = args.failures_csv or args.input_root / "cluster_gt_distance_failures.csv"
    if args.single_file is not None:
        files = [args.single_file]
    else:
        files = sorted(args.input_root.glob(DIAGNOSTIC_PATTERN))

    trajectories: dict[str, PositionTrajectory] = {}
    rows_with_clusters: list[tuple[dict[str, Any], dict[int, dict[str, Any]]]] = []
    failures: list[dict[str, str]] = []
    maximum_cluster_id = -1
    for path in files:
        sequence_id = ""
        unit_id = ""
        stage = "path"
        try:
            sequence_id, unit_id = diagnostic_identity(path, args.input_root)
            stage = "gt"
            if sequence_id not in trajectories:
                trajectories[sequence_id] = PositionTrajectory.from_directory(
                    args.dataset_root / sequence_id / "ground_truth"
                )
            stage = "diagnostic"
            row, clusters = parse_diagnostic(
                path, args.input_root, trajectories[sequence_id]
            )
            rows_with_clusters.append((row, clusters))
            if clusters:
                maximum_cluster_id = max(maximum_cluster_id, max(clusters))
        except Exception as exc:  # Continue past corrupt/missing inputs by design.
            failures.append({
                "diagnostic_file": str(path.resolve()),
                "sequence_id": sequence_id,
                "unit_id": unit_id,
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
            })

    rows_with_clusters.sort(key=lambda item: (
        item[0]["sequence_id"], item[0]["query_timestamp"]
    ))
    base_fields = [
        "sequence_id", "unit_id", "diagnostic_file", "query_timestamp",
        "window_start_timestamp", "window_end_timestamp", "nearest_gt_timestamp",
        "gt_time_gap_ms", "gt_x", "gt_y", "gt_z", "dbscan_cluster_count",
        "coordinate_warning",
    ]
    cluster_fields = [
        f"cluster_{cluster_id}_{suffix}"
        for cluster_id in range(maximum_cluster_id + 1)
        for suffix in (
            "num_points", "center_x", "center_y", "center_z", "distance_to_gt_m"
        )
    ]
    nearest_fields = [
        "nearest_cluster_id", "nearest_cluster_distance_m",
        "nearest_cluster_center_x", "nearest_cluster_center_y",
        "nearest_cluster_center_z", "nearest_cluster_num_points",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=base_fields + cluster_fields + nearest_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row, clusters in rows_with_clusters:
            wide_row = dict(row)
            for cluster_id, cluster in clusters.items():
                prefix = f"cluster_{cluster_id}"
                wide_row[f"{prefix}_num_points"] = cluster["num_points"]
                wide_row[f"{prefix}_center_x"] = float(cluster["center"][0])
                wide_row[f"{prefix}_center_y"] = float(cluster["center"][1])
                wide_row[f"{prefix}_center_z"] = float(cluster["center"][2])
                wide_row[f"{prefix}_distance_to_gt_m"] = cluster["distance"]
            writer.writerow(wide_row)
    write_failures(failures_csv, failures)

    by_sequence: dict[str, list[float]] = defaultdict(list)
    sequence_units: dict[str, int] = defaultdict(int)
    for row, _ in rows_with_clusters:
        sequence_units[row["sequence_id"]] += 1
        distance = row["nearest_cluster_distance_m"]
        if distance is not None and math.isfinite(distance):
            by_sequence[row["sequence_id"]].append(float(distance))

    print(f"files_found={len(files)}")
    print(f"files_processed={len(rows_with_clusters)}")
    print(f"files_failed={len(failures)}")
    print(f"output_csv={output_csv.resolve()}")
    print("sequence | processed_units | min_raw_distance | median_min_raw_distance")
    for sequence_id in sorted(sequence_units):
        distances = np.asarray(by_sequence[sequence_id], dtype=np.float64)
        minimum = "N/A" if not len(distances) else f"{distances.min():.6f}"
        median = "N/A" if not len(distances) else f"{np.median(distances):.6f}"
        print(f"{sequence_id} | {sequence_units[sequence_id]} | {minimum} | {median}")


if __name__ == "__main__":
    main()
