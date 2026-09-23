#!/usr/bin/env python3
"""Visualize selected eps=1 MMUAV clusters from one 20-frame diagnostic unit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.baselines.mmuav_preprocess import (  # noqa: E402
    _accumulate_lidar_360_blocks,
    _dbscan_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir", type=Path, required=True,
        help="Unit diagnostics directory containing eps1_cluster_features.npz.",
    )
    parser.add_argument(
        "--raw-lidar-dir", type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/official/v1/Mavic2/lidar_360"),
    )
    return parser.parse_args()


def load_existing_diagnostic(diagnostic_dir: Path) -> tuple[np.lib.npyio.NpzFile, list[str]]:
    arrays = np.load(diagnostic_dir / "eps1_cluster_features.npz", allow_pickle=False)
    pre_lstm_path = diagnostic_dir / "pre_lstm_clusters/cluster_diagnostics.json"
    report = json.loads(pre_lstm_path.read_text(encoding="utf-8"))
    timestamps = [str(value) for value in report["frame_timestamps"]]
    if len(timestamps) != 20:
        raise ValueError(f"Expected 20 frame timestamps, got {len(timestamps)}")
    return arrays, timestamps


def replay_membership(
    raw_lidar_dir: Path, timestamps: list[str], arrays: np.lib.npyio.NpzFile,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = {
        timestamp: np.load(raw_lidar_dir / f"{timestamp}.npy", allow_pickle=False)[:, :3]
        for timestamp in timestamps
    }
    blocks, _ = _accumulate_lidar_360_blocks(frames)
    if len(blocks) != 1:
        raise RuntimeError(f"Expected one 20-frame block, got {len(blocks)}")
    accumulated = next(iter(blocks.values()))
    frame_indices = accumulated[:, 0].astype(np.int64)
    points = accumulated[:, 1:]
    # Diagnostic-only replay; never feeds fusion/candidate generation.
    labels = _dbscan_labels(points, eps=eps, min_samples=10)

    replay_ids, replay_counts = np.unique(labels[labels != -1], return_counts=True)
    expected_ids = arrays["cluster_ids"].astype(np.int64)
    expected_counts = arrays["num_points"].astype(np.int64)
    if not np.array_equal(replay_ids, expected_ids):
        raise RuntimeError(f"Replayed eps={eps} cluster IDs differ from saved diagnostics")
    if not np.array_equal(replay_counts, expected_counts):
        raise RuntimeError(f"Replayed eps={eps} cluster sizes differ from saved diagnostics")
    return points, frame_indices, labels


def replay_eps1_membership(
    raw_lidar_dir: Path, timestamps: list[str], arrays: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return replay_membership(raw_lidar_dir, timestamps, arrays, eps=1)


def cluster_trajectory(
    cluster_id: int, points: np.ndarray, frame_indices: np.ndarray, labels: np.ndarray,
    timestamps: list[str],
) -> tuple[list[dict[str, float | int | str | None]], dict[str, float | int]]:
    member_mask = labels == cluster_id
    rows: list[dict[str, float | int | str | None]] = []
    previous_center = None
    trajectory_length = 0.0
    steps = []
    for frame_index, timestamp in enumerate(timestamps, start=1):
        frame_points = points[member_mask & (frame_indices == frame_index)]
        center = frame_points.mean(axis=0) if len(frame_points) else None
        displacement = None
        if center is not None and previous_center is not None:
            displacement = float(np.linalg.norm(center - previous_center))
            trajectory_length += displacement
            steps.append(displacement)
        if center is not None:
            previous_center = center
        rows.append({
            "cluster_id": cluster_id,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "point_count": int(len(frame_points)),
            "center_x": None if center is None else float(center[0]),
            "center_y": None if center is None else float(center[1]),
            "center_z": None if center is None else float(center[2]),
            "center_displacement_from_previous_active_m": displacement,
        })
    return rows, {
        "total_points": int(np.count_nonzero(member_mask)),
        "active_frames": int(sum(row["point_count"] > 0 for row in rows)),
        "trajectory_length": float(trajectory_length),
        "mean_step": float(np.mean(steps)) if steps else 0.0,
        "max_step": float(np.max(steps)) if steps else 0.0,
    }


def write_trajectory_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_3d(points: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(35.0)
    elevation = np.deg2rad(25.0)
    horizontal = np.asarray([np.cos(azimuth), np.sin(azimuth), 0.0])
    vertical = np.asarray([
        -np.sin(azimuth) * np.sin(elevation),
        np.cos(azimuth) * np.sin(elevation),
        np.cos(elevation),
    ])
    return np.column_stack((points @ horizontal, points @ vertical))


def draw_trajectory(
    cluster_id: int, all_points: np.ndarray, cluster_points: np.ndarray,
    rows: list[dict[str, object]], output_path: Path, seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    background_count = min(5000, max(1, int(len(all_points) * 0.02)))
    background_indices = rng.choice(len(all_points), size=background_count, replace=False)
    background = all_points[background_indices]
    active_rows = [row for row in rows if row["point_count"] > 0]
    centers = np.asarray([
        [row["center_x"], row["center_y"], row["center_z"]] for row in active_rows
    ], dtype=np.float64)

    canvas = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    combined = np.vstack((background, cluster_points, centers))
    projected = project_3d(combined)
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    span = np.maximum(upper - lower, 1e-9)
    scale = min(1240.0 / span[0], 840.0 / span[1])
    projected_center = (lower + upper) / 2.0

    def to_pixel(projected_points: np.ndarray) -> np.ndarray:
        centered = (projected_points - projected_center) * scale
        return np.column_stack((700 + centered[:, 0], 500 - centered[:, 1]))

    background_px = to_pixel(project_3d(background))
    cluster_px = to_pixel(project_3d(cluster_points))
    center_px = to_pixel(project_3d(centers))
    for x, y in background_px:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(170, 170, 170))
    for x, y in cluster_px:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(220, 45, 45))
    if len(center_px) > 1:
        draw.line([tuple(point) for point in center_px], fill=(20, 75, 220), width=4)
    for point, row in zip(center_px, active_rows):
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(20, 75, 220))
        draw.text((x + 7, y - 7), str(row["frame_index"]), fill=(0, 0, 0), font=font)
    axis_origin = np.asarray([1260.0, 150.0])
    axis_vectors = project_3d(np.eye(3))
    for label, vector, color in zip(
        ("X", "Y", "Z"), axis_vectors, ((200, 0, 0), (0, 150, 0), (0, 70, 210))
    ):
        screen_vector = np.asarray([vector[0], -vector[1]])
        screen_vector *= 70.0 / max(np.linalg.norm(screen_vector), 1e-9)
        endpoint = axis_origin + screen_vector
        draw.line((tuple(axis_origin), tuple(endpoint)), fill=color, width=4)
        draw.text(tuple(endpoint + 4), label, fill=color, font=font)
    draw.text((30, 25), f"eps=1 cluster {cluster_id}: 3D trajectory", fill=(0, 0, 0), font=font)
    draw.text(
        (30, 45),
        "Gray: 2% sampled Mid360 background | Red: cluster points | Blue: per-frame centers",
        fill=(0, 0, 0), font=font,
    )
    draw.text((30, 65), "View rotation: azimuth 35 deg, elevation 25 deg; axes retain metric scale before projection", fill=(0, 0, 0), font=font)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    arrays, timestamps = load_existing_diagnostic(args.diagnostic_dir)
    points, frame_indices, labels = replay_eps1_membership(
        args.raw_lidar_dir, timestamps, arrays
    )
    positive_ids = arrays["cluster_ids"][arrays["predictions"] == 1].astype(np.int64)
    if set(positive_ids.tolist()) != {12, 14}:
        raise RuntimeError(f"Expected positive clusters 12 and 14, got {positive_ids.tolist()}")
    output_dir = args.diagnostic_dir / "positive_cluster_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cluster_id in positive_ids:
        rows, summary = cluster_trajectory(
            int(cluster_id), points, frame_indices, labels, timestamps
        )
        probability_index = int(np.flatnonzero(arrays["cluster_ids"] == cluster_id)[0])
        summary_row = {
            "cluster_id": int(cluster_id),
            "P_uav": float(arrays["probabilities"][probability_index, 1]),
            **summary,
        }
        summaries.append(summary_row)
        write_trajectory_csv(
            rows, output_dir / f"positive_cluster_{int(cluster_id)}_trajectory.csv"
        )
        draw_trajectory(
            int(cluster_id), points, points[labels == cluster_id], rows,
            output_dir / f"cluster_{int(cluster_id)}_trajectory.png",
            seed=int(cluster_id),
        )
    (output_dir / "positive_cluster_summary.json").write_text(
        json.dumps({"gt_used": False, "clusters": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
