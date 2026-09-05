from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rdq_uav.utils.io import write_json


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def _assign_blocks(
    rows: list[dict[str, Any]],
    num_blocks: int,
    mode: str,
    train_blocks: int,
    val_blocks: int,
    test_blocks: int,
    seed: int,
    class_id: int,
) -> tuple[np.ndarray, dict[int, str]]:
    if num_blocks < 3 or len(rows) < num_blocks:
        raise ValueError(f"Need at least {num_blocks} samples/blocks, got {len(rows)}")
    if train_blocks + val_blocks + test_blocks != num_blocks:
        raise ValueError("train_blocks + val_blocks + test_blocks must equal num_blocks")
    block_ids = np.minimum(
        np.arange(len(rows), dtype=np.int64) * num_blocks // len(rows), num_blocks - 1
    )
    if mode == "blocked_random":
        order = np.random.default_rng(seed + 1009 * class_id).permutation(num_blocks)
        mapping: dict[int, str] = {}
        for block in order[:train_blocks]:
            mapping[int(block)] = "train"
        for block in order[train_blocks : train_blocks + val_blocks]:
            mapping[int(block)] = "val"
        for block in order[train_blocks + val_blocks :]:
            mapping[int(block)] = "test"
    elif mode == "temporal_holdout":
        mapping = {}
        for block in range(num_blocks):
            if block < train_blocks:
                mapping[block] = "train"
            elif block < train_blocks + val_blocks:
                mapping[block] = "val"
            else:
                mapping[block] = "test"
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return block_ids, mapping


def _mark_guard_samples(
    rows: list[dict[str, Any]],
    block_ids: np.ndarray,
    mapping: dict[int, str],
    guard_seconds: float,
) -> np.ndarray:
    dropped = np.zeros(len(rows), dtype=bool)
    if guard_seconds <= 0:
        return dropped
    times = np.asarray([float(row["gt_time"]) for row in rows], dtype=np.float64)
    for index in range(len(rows) - 1):
        left_block = int(block_ids[index])
        right_block = int(block_ids[index + 1])
        if left_block == right_block or mapping[left_block] == mapping[right_block]:
            continue
        boundary = (times[index] + times[index + 1]) / 2.0
        dropped |= np.abs(times - boundary) < guard_seconds
    return dropped


def compute_radar_stats(
    rows: list[dict[str, Any]], root: Path, max_range_m: float
) -> dict[str, Any]:
    paths = sorted({str(row["radar_path"]) for row in rows if row["split"] == "train"})
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    dropped_nonfinite = 0
    dropped_range = 0
    for relative_path in paths:
        array = np.load(root / relative_path, allow_pickle=False)
        if array.size == 0:
            continue
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"Invalid radar shape: {relative_path}: {array.shape}")
        array = np.asarray(array, dtype=np.float64)
        finite = np.isfinite(array).all(axis=1)
        dropped_nonfinite += int(np.count_nonzero(~finite))
        array = array[finite]
        in_range = np.linalg.norm(array, axis=1) <= max_range_m
        dropped_range += int(np.count_nonzero(~in_range))
        array = array[in_range]
        if len(array) == 0:
            continue
        count += len(array)
        total += array.sum(axis=0)
        total_sq += np.square(array).sum(axis=0)
    if count == 0:
        raise RuntimeError("No valid training radar points remain after filtering")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "point_count": count,
        "unique_frame_count": len(paths),
        "max_range_m": max_range_m,
        "dropped_nonfinite_points": dropped_nonfinite,
        "dropped_out_of_range_points": dropped_range,
        "computed_from_split": "train",
    }


def build_manifests(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).expanduser().resolve()
    output_dir = Path(data_cfg["manifest_dir"]).expanduser().resolve()
    audit_path = root / "audit" / "sync_samples.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing audit pairing table: {audit_path}. Run audit_v1.py first."
        )
    classes = list(data_cfg["classes"])
    threshold = float(data_cfg["sync_threshold_ms"]) / 1000.0
    source_rows = _read_rows(audit_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_sync = Counter()
    for row in source_rows:
        class_name = row["class_name"]
        if class_name not in classes:
            continue
        if (
            abs(float(row["dt_image_s"])) > threshold
            or abs(float(row["dt_radar_s"])) > threshold
        ):
            rejected_sync[class_name] += 1
            continue
        grouped[class_name].append(dict(row))

    split_cfg = data_cfg["split"]
    all_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(classes):
        rows = sorted(grouped[class_name], key=lambda item: float(item["gt_time"]))
        if not rows:
            raise RuntimeError(f"No synchronized samples for class: {class_name}")
        block_ids, mapping = _assign_blocks(
            rows=rows,
            num_blocks=int(split_cfg["num_blocks"]),
            mode=str(split_cfg["mode"]),
            train_blocks=int(split_cfg["train_blocks"]),
            val_blocks=int(split_cfg["val_blocks"]),
            test_blocks=int(split_cfg["test_blocks"]),
            seed=int(split_cfg["seed"]),
            class_id=class_id,
        )
        guard = _mark_guard_samples(
            rows, block_ids, mapping, float(split_cfg["guard_seconds"])
        )
        for index, row in enumerate(rows):
            row["class_id"] = class_id
            row["class_name"] = class_name
            row["sequence_id"] = class_name
            row["temporal_block"] = int(block_ids[index])
            row["split"] = "guard_dropped" if guard[index] else mapping[int(block_ids[index])]
            all_rows.append(row)

    all_rows.sort(key=lambda item: (int(item["class_id"]), float(item["gt_time"])))
    fields = [
        "sample_id",
        "class_id",
        "class_name",
        "sequence_id",
        "temporal_block",
        "split",
        "gt_time",
        "image_time",
        "radar_time",
        "dt_image_s",
        "dt_radar_s",
        "abs_dt_image_s",
        "abs_dt_radar_s",
        "gt_path",
        "image_path",
        "radar_path",
        "gt_x",
        "gt_y",
        "gt_z",
        "distance_m",
        "radar_points",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "manifest_all.csv", all_rows, fields)
    for split in ("train", "val", "test"):
        _write_rows(
            output_dir / f"{split}.csv",
            [row for row in all_rows if row["split"] == split],
            fields,
        )

    radar_stats = compute_radar_stats(
        all_rows, root, float(data_cfg["radar"]["max_range_m"])
    )
    write_json(radar_stats, output_dir / "radar_stats.json")
    write_json(
        {"classes": {name: index for index, name in enumerate(classes)}},
        output_dir / "class_mapping.json",
    )

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test", "guard_dropped"):
        split_counts[split] = dict(
            Counter(
                str(row["class_name"])
                for row in all_rows
                if row["split"] == split
            )
        )
    summary = {
        "root": str(root),
        "audit_source": str(audit_path),
        "sync_threshold_ms": data_cfg["sync_threshold_ms"],
        "split": split_cfg,
        "split_counts": split_counts,
        "sync_rejected": dict(rejected_sync),
        "radar_stats": radar_stats,
    }
    write_json(summary, output_dir / "manifest_summary.json")
    return summary
