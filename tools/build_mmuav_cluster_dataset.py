#!/usr/bin/env python3
"""Build the source-code-faithful MMUAV Mid360 [20,9] classifier dataset.

This first reproduction baseline deliberately uses the public training behavior:
non-overlapping 20-frame windows, DBSCAN(eps=1,min_samples=10), 9D
mean/std/range features, and one GT position selected at the first frame time.
GT is used only to create training labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
try:
    from tqdm import tqdm
except ImportError:  # The original MMUAV environment does not require tqdm.
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


DEFAULT_DATA_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")
DEFAULT_SPLITS = Path(
    "/home/jasoncui/projects/rdq-uav-experiment/outputs/"
    "mmuav_paper_reproduction/splits/splits.json"
)
DEFAULT_OUTPUT = Path(
    "/home/jasoncui/projects/rdq-uav-experiment/outputs/"
    "mmuav_paper_reproduction/datasets/public_9d"
)
FIELDNAMES = [
    "split", "sequence_id", "window_id", "window_index", "cluster_id",
    "window_start_timestamp", "window_end_timestamp", "frame_timestamps",
    "gt_timestamp", "gt_time_gap_ms", "gt_x", "gt_y", "gt_z",
    "label", "point_count", "per_frame_point_count", "cluster_center_xyz",
    "per_frame_cluster_centers", "feature_20x9", "dbscan_noise_points",
]
SEQUENCE_DISTRIBUTION_FIELDS = [
    "split", "sequence_id", "windows", "clusters", "positive", "negative",
    "positive_ratio",
]
WINDOW_DISTRIBUTION_FIELDS = [
    "split", "sequence_id", "window_id", "clusters", "positive", "negative",
]


def timestamp_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.npy"))
    try:
        return sorted(files, key=lambda path: float(path.stem))
    except ValueError as exc:
        raise ValueError(f"Non-numeric timestamp under {directory}") from exc


def load_xyz(path: Path) -> np.ndarray:
    data = np.asarray(np.load(path, allow_pickle=False))
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected [N,>=3] point cloud at {path}, got {data.shape}")
    data = np.asarray(data[:, :3], dtype=np.float64)
    return data[np.isfinite(data).all(axis=1) & np.any(data != 0, axis=1)]


def load_gt(sequence_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    paths = timestamp_files(sequence_dir / "ground_truth")
    if not paths:
        raise FileNotFoundError(f"No GT in {sequence_dir / 'ground_truth'}")
    timestamps = np.asarray([float(path.stem) for path in paths], dtype=np.float64)
    xyz = np.stack([
        np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)[:3]
        for path in paths
    ])
    if xyz.shape != (len(paths), 3) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid GT arrays under {sequence_dir / 'ground_truth'}")
    return timestamps, xyz


def nearest_gt(
    query_timestamp: float, gt_timestamps: np.ndarray, gt_xyz: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    index = int(np.argmin(np.abs(gt_timestamps - query_timestamp)))
    gap_ms = float(abs(gt_timestamps[index] - query_timestamp) * 1000.0)
    return float(gt_timestamps[index]), gt_xyz[index], gap_ms


def extract_window(
    frame_paths: list[Path], gt_timestamps: np.ndarray, gt_xyz: np.ndarray,
    split: str, sequence_id: str, window_index: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    if len(frame_paths) != 20:
        raise ValueError(f"Public training windows must contain 20 frames, got {len(frame_paths)}")
    frame_times = [float(path.stem) for path in frame_paths]
    point_frames = [load_xyz(path) for path in frame_paths]
    nonempty = [(index + 1, points) for index, points in enumerate(point_frames) if len(points)]
    if not nonempty:
        return (
            np.empty((0, 20, 9), dtype=np.float64),
            np.empty((0,), dtype=np.int64), [],
            {"clusters": 0, "positive": 0, "negative": 0, "noise_points": 0},
        )
    points = np.concatenate([item[1] for item in nonempty], axis=0)
    frame_indices = np.concatenate([
        np.full(len(item[1]), item[0], dtype=np.int16) for item in nonempty
    ])
    dbscan_labels = DBSCAN(eps=1.0, min_samples=10).fit(points).labels_
    cluster_ids = sorted(int(value) for value in np.unique(dbscan_labels) if value != -1)
    selected_gt_time, selected_gt, gt_gap_ms = nearest_gt(
        frame_times[0], gt_timestamps, gt_xyz
    )
    features: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []
    window_id = f"{sequence_id}_w{window_index:05d}"
    noise_points = int(np.count_nonzero(dbscan_labels == -1))
    for cluster_id in cluster_ids:
        member = dbscan_labels == cluster_id
        cluster_points = points[member]
        cluster_frame_indices = frame_indices[member]
        feature = np.zeros((20, 9), dtype=np.float64)
        frame_centers: list[list[float] | None] = []
        frame_counts: list[int] = []
        label = 0
        for frame_index in range(1, 21):
            frame_points = cluster_points[cluster_frame_indices == frame_index]
            frame_counts.append(int(len(frame_points)))
            if len(frame_points):
                mean = frame_points.mean(axis=0)
                std = frame_points.std(axis=0)
                span = frame_points.max(axis=0) - frame_points.min(axis=0)
                feature[frame_index - 1] = np.concatenate((mean, std, span))
                frame_centers.append(mean.tolist())
                # SOURCE-CODE LABEL: every frame compares to the same GT selected
                # using the first timestamp of the 20-frame window.
                if np.linalg.norm(mean - selected_gt) < 1.0:
                    label = 1
            else:
                frame_centers.append(None)
        features.append(feature)
        labels.append(label)
        rows.append({
            "split": split,
            "sequence_id": sequence_id,
            "window_id": window_id,
            "window_index": window_index,
            "cluster_id": cluster_id,
            "window_start_timestamp": frame_times[0],
            "window_end_timestamp": frame_times[-1],
            "frame_timestamps": json.dumps(frame_times, separators=(",", ":")),
            "gt_timestamp": selected_gt_time,
            "gt_time_gap_ms": gt_gap_ms,
            "gt_x": float(selected_gt[0]), "gt_y": float(selected_gt[1]),
            "gt_z": float(selected_gt[2]),
            "label": label,
            "point_count": int(len(cluster_points)),
            "per_frame_point_count": json.dumps(frame_counts, separators=(",", ":")),
            "cluster_center_xyz": json.dumps(
                cluster_points.mean(axis=0).tolist(), separators=(",", ":")
            ),
            "per_frame_cluster_centers": json.dumps(frame_centers, separators=(",", ":")),
            "feature_20x9": json.dumps(feature.tolist(), separators=(",", ":")),
            "dbscan_noise_points": noise_points,
        })
    feature_array = (
        np.stack(features).astype(np.float32, copy=False)
        if features else np.empty((0, 20, 9), dtype=np.float32)
    )
    label_array = np.asarray(labels, dtype=np.int64)
    return feature_array, label_array, rows, {
        "clusters": len(cluster_ids),
        "positive": int(label_array.sum()),
        "negative": int(len(label_array) - label_array.sum()),
        "noise_points": noise_points,
    }


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", newline="", encoding="utf-8", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_simple_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", newline="", encoding="utf-8", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def process_sequence(
    data_root: Path, sequence_id: str, split: str, max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    sequence_dir = data_root / sequence_id
    lidar_paths = timestamp_files(sequence_dir / "lidar_360")
    gt_timestamps, gt_xyz = load_gt(sequence_dir)
    available_windows = len(lidar_paths) // 20
    used_windows = min(available_windows, max_windows) if max_windows else available_windows
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_rows: list[dict[str, Any]] = []
    totals = Counter()
    per_window: list[dict[str, Any]] = []
    for window_index in tqdm(
        range(used_windows), desc=f"{split}/{sequence_id}", unit="window", leave=False
    ):
        start = window_index * 20
        features, labels, rows, stats = extract_window(
            lidar_paths[start:start + 20], gt_timestamps, gt_xyz,
            split, sequence_id, window_index,
        )
        all_features.append(features)
        all_labels.append(labels)
        all_rows.extend(rows)
        totals.update(stats)
        per_window.append({
            "split": split,
            "sequence_id": sequence_id,
            "window_id": f"{sequence_id}_w{window_index:05d}",
            "clusters": int(stats["clusters"]),
            "positive": int(stats["positive"]),
            "negative": int(stats["negative"]),
        })
    features = (
        np.concatenate(all_features, axis=0)
        if all_features else np.empty((0, 20, 9), dtype=np.float32)
    )
    labels = (
        np.concatenate(all_labels, axis=0)
        if all_labels else np.empty((0,), dtype=np.int64)
    )
    return features, labels, all_rows, {
        "sequence_id": sequence_id,
        "source_frames": len(lidar_paths),
        "available_complete_windows": available_windows,
        "processed_windows": used_windows,
        "clusters": int(totals["clusters"]),
        "positive": int(totals["positive"]),
        "negative": int(totals["negative"]),
        "noise_points_across_windows": int(totals["noise_points"]),
        "per_window": per_window,
    }


def load_split_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = ("train_sub", "validation_sub", "heldout_test_sub")
    sets = [set(payload[name]) for name in names]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Sequence leakage in frozen splits")
    return payload


def read_shard_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(args: argparse.Namespace) -> dict[str, Any]:
    split_payload = load_split_file(args.split_file)
    requested_splits = list(args.splits)
    selected: dict[str, list[str]] = {}
    rng = np.random.default_rng(args.seed)
    for split in requested_splits:
        sequences = list(split_payload[split])
        if args.smoke:
            count = min(args.smoke_sequences, len(sequences))
            sequences = sorted(rng.choice(sequences, count, replace=False).tolist())
        selected[split] = sequences

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = args.output_dir / ".shards"
    summaries: dict[str, list[dict[str, Any]]] = {}
    for split, sequences in selected.items():
        summaries[split] = []
        for sequence_id in sequences:
            shard_dir = shards / split
            npz_path = shard_dir / f"{sequence_id}.npz"
            csv_path = shard_dir / f"{sequence_id}.csv"
            summary_path = shard_dir / f"{sequence_id}.json"
            if args.resume and npz_path.is_file() and csv_path.is_file() and summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                print(f"reuse {split}/{sequence_id}: {summary['clusters']} clusters")
            else:
                features, labels, rows, summary = process_sequence(
                    args.data_root, sequence_id, split,
                    1 if args.smoke else args.max_windows_per_sequence,
                )
                atomic_npz(npz_path, features=features, labels=labels)
                write_metadata(csv_path, rows)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                print(
                    f"built {split}/{sequence_id}: windows={summary['processed_windows']} "
                    f"clusters={summary['clusters']} positive={summary['positive']}"
                )
            summaries[split].append(summary)

    final_names = {"train_sub": "train", "validation_sub": "val"}
    overall: dict[str, Any] = {
        "schema_version": 1,
        "mode": "smoke" if args.smoke else "full",
        "seed": args.seed,
        "split_file": str(args.split_file.resolve()),
        "split_assignment_sha256": split_payload.get("assignment_sha256"),
        "training_dbscan_eps": 1.0,
        "training_dbscan_min_samples": 10,
        "label_mode": "public_code_label",
        "feature_shape": [20, 9],
        "splits": {},
    }
    sequence_distribution: list[dict[str, Any]] = []
    window_distribution: list[dict[str, Any]] = []
    for split, output_name in final_names.items():
        if split not in selected:
            continue
        feature_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        metadata_rows: list[dict[str, Any]] = []
        for sequence_id in selected[split]:
            arrays = np.load(shards / split / f"{sequence_id}.npz", allow_pickle=False)
            feature_parts.append(arrays["features"])
            label_parts.append(arrays["labels"])
            metadata_rows.extend(read_shard_metadata(shards / split / f"{sequence_id}.csv"))
        features = np.concatenate(feature_parts) if feature_parts else np.empty((0, 20, 9), np.float32)
        labels = np.concatenate(label_parts) if label_parts else np.empty((0,), np.int64)
        np.save(args.output_dir / f"feature_{output_name}.npy", features)
        np.save(args.output_dir / f"label_{output_name}.npy", labels)
        write_metadata(args.output_dir / f"metadata_{output_name}.csv", metadata_rows)
        split_summary = {
            "sequences": len(selected[split]),
            "sequence_ids": selected[split],
            "windows": int(sum(item["processed_windows"] for item in summaries[split])),
            "clusters": int(len(labels)),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
            "positive_ratio": float(labels.mean()) if len(labels) else 0.0,
            "per_sequence": summaries[split],
        }
        overall["splits"][split] = split_summary
        for item in summaries[split]:
            clusters = int(item["clusters"])
            sequence_distribution.append({
                "split": split,
                "sequence_id": item["sequence_id"],
                "windows": int(item["processed_windows"]),
                "clusters": clusters,
                "positive": int(item["positive"]),
                "negative": int(item["negative"]),
                "positive_ratio": float(item["positive"] / clusters) if clusters else 0.0,
            })
            window_distribution.extend(item.get("per_window", []))
        print(
            f"{split}: sequences={split_summary['sequences']} windows={split_summary['windows']} "
            f"clusters={split_summary['clusters']} positive={split_summary['positive']} "
            f"negative={split_summary['negative']} positive_ratio={split_summary['positive_ratio']:.6f}"
        )
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="utf-8"
    )
    write_simple_csv(
        args.output_dir / "positive_distribution_by_sequence.csv",
        sequence_distribution, SEQUENCE_DISTRIBUTION_FIELDS,
    )
    write_simple_csv(
        args.output_dir / "positive_distribution_by_window.csv",
        window_distribution, WINDOW_DISTRIBUTION_FIELDS,
    )
    if args.smoke:
        smoke_positive = sum(item["positive"] for item in overall["splits"].values())
        if smoke_positive == 0:
            print("WARNING: NO_POSITIVE_IN_SMOKE_SAMPLE")
    else:
        train_summary = overall["splits"].get("train_sub")
        if train_summary is not None and int(train_summary["positive"]) == 0:
            raise RuntimeError("FULL_TRAIN_HAS_ZERO_POSITIVES")
        validation_summary = overall["splits"].get("validation_sub")
        if validation_summary is not None and int(validation_summary["positive"]) == 0:
            print(
                "WARNING: FULL_VALIDATION_HAS_ZERO_POSITIVES; positive recall/F1 "
                "cannot be evaluated normally"
            )
    print(f"dataset_output={args.output_dir}")
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--splits", nargs="+", choices=("train_sub", "validation_sub"),
        default=("train_sub", "validation_sub"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-windows-per-sequence", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-sequences", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
