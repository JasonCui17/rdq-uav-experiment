#!/usr/bin/env python3
"""Run the source-faithful Multi-Modal-UAV LiDAR candidate baseline.

Default mode is dry-run: index train/val data and validate the original LSTM
checkpoint without executing DBSCAN or LSTM inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.baselines.mmuav_preprocess import (  # noqa: E402
    SOURCE_COMMIT,
    SOURCE_REPO,
    _accumulate_lidar_360_blocks,
    _dbscan_labels,
    extract_feature_set_predict,
    load_original_checkpoint,
    process_fusion,
    process_lidar_360,
    process_lidar_livox,
    read_lidar_files,
    write_candidate_sidecar,
)

SENSORS = ("lidar_360", "livox_avia")
ALLOWED_SPLITS = ("train", "val")


@dataclass
class SensorFrame:
    sequence_id: str
    sensor_type: str
    timestamp: str
    path: Path
    splits: set[str] = field(default_factory=set)
    manifest_units: set[tuple[str, int]] = field(default_factory=set)


@dataclass
class ProcessingUnit:
    sequence_id: str
    split: str
    temporal_block: int
    chunk_index: int
    time_start: float
    time_end: float
    sensor_frames: dict[str, list[SensorFrame]]

    @property
    def name(self) -> str:
        return f"{self.split}_block{self.temporal_block:02d}_chunk{self.chunk_index:03d}"

    @property
    def qualified_name(self) -> str:
        return f"{self.sequence_id}/{self.name}"


def read_manifests(manifest_dir: Path, splits: list[str]) -> tuple[list[dict[str, str]], list[Path]]:
    if any(split not in ALLOWED_SPLITS for split in splits):
        raise ValueError(f"Only train/val are allowed, got {splits}")
    rows: list[dict[str, str]] = []
    paths = []
    for split in splits:
        path = manifest_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            split_rows = list(csv.DictReader(handle))
        required = {"sample_id", "sequence_id", "temporal_block", "split", "gt_time"}
        missing = required - set(split_rows[0] if split_rows else {})
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if any(row["split"] != split for row in split_rows):
            raise ValueError(f"{path} contains a row with a different split")
        rows.extend(split_rows)
        paths.append(path)
    return rows, paths


def source_timestamp_index(directory: Path) -> tuple[np.ndarray, list[Path]]:
    entries = []
    for path in directory.glob("*.npy"):
        try:
            entries.append((float(path.stem), path))
        except ValueError:
            continue
    entries.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in entries]), [item[1] for item in entries]


# DATA-ADAPTER CHANGE:
# Manifest rows define time ranges only. Every original sensor NPY whose filename
# timestamp falls in that range is retained at its native rate.
def build_unique_sensor_frame_index(
    rows: list[dict[str, str]], dataset_root: Path, sequences: list[str] | None = None,
) -> tuple[
    dict[tuple[str, str], list[SensorFrame]],
    dict[str, Any],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    selected_sequences = sorted(set(sequences or [row["sequence_id"] for row in rows]))
    range_times: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    ignored_rows = 0
    for row in rows:
        sequence = row["sequence_id"]
        if sequence not in selected_sequences:
            ignored_rows += 1
            continue
        key = (sequence, row["split"], int(row["temporal_block"]))
        range_times[key].append(float(row["gt_time"]))
    time_ranges = {
        key: {
            "sequence_id": key[0], "split": key[1], "temporal_block": key[2],
            "t_start": min(times), "t_end": max(times), "manifest_rows": len(times),
        }
        for key, times in range_times.items()
    }
    for sequence in selected_sequences:
        ordered = sorted(
            (item for key, item in time_ranges.items() if key[0] == sequence),
            key=lambda item: item["t_start"],
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current["t_start"] <= previous["t_end"]:
                raise ValueError(
                    "Manifest temporal ranges overlap; strict split isolation cannot be "
                    f"guaranteed: {previous} vs {current}"
                )

    source_indexes: dict[tuple[str, str], tuple[np.ndarray, list[Path]]] = {}
    for sequence in selected_sequences:
        for sensor in SENSORS:
            directory = dataset_root / sequence / sensor
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            source_indexes[(sequence, sensor)] = source_timestamp_index(directory)

    selected_frames: dict[tuple[str, str], dict[Path, SensorFrame]] = {
        key: {} for key in source_indexes
    }
    block_sensor_counts: dict[tuple[str, str, int], dict[str, int]] = {
        key: {sensor: 0 for sensor in SENSORS} for key in time_ranges
    }
    for unit_key, time_range in sorted(time_ranges.items()):
        sequence, split, temporal_block = unit_key
        for sensor in SENSORS:
            key = (sequence, sensor)
            timestamps, paths = source_indexes[key]
            start = int(np.searchsorted(timestamps, time_range["t_start"], side="left"))
            stop = int(np.searchsorted(timestamps, time_range["t_end"], side="right"))
            selected_paths = paths[start:stop]
            block_sensor_counts[unit_key][sensor] = len(selected_paths)
            for path in selected_paths:
                if path not in selected_frames[key]:
                    selected_frames[key][path] = SensorFrame(
                        sequence_id=sequence, sensor_type=sensor,
                        timestamp=path.stem, path=path,
                    )
                frame = selected_frames[key][path]
                frame.splits.add(split)
                frame.manifest_units.add((split, temporal_block))

    result = {
        key: sorted(frames.values(), key=lambda frame: float(frame.timestamp))
        for key, frames in selected_frames.items()
    }
    audit = {
        "selection_policy": "all original sensor timestamps inside manifest GT time range",
        "ignored_manifest_rows": ignored_rows,
        "by_sequence_sensor": {},
        "time_ranges": [],
    }
    for unit_key, time_range in sorted(time_ranges.items()):
        audit["time_ranges"].append({
            **time_range,
            "selected_frames": block_sensor_counts[unit_key],
        })
    for key, frames in result.items():
        sequence, sensor = key
        source_frame_count = len(source_indexes[key][1])
        cross_split = sum(len(frame.splits) > 1 for frame in frames)
        unique_by_split = {
            split: sum(split in frame.splits for frame in frames) for split in ALLOWED_SPLITS
        }
        audit["by_sequence_sensor"][f"{sequence}/{sensor}"] = {
            "unique_sensor_frames": len(frames),
            "unique_sensor_frames_by_split": unique_by_split,
            "source_directory_frames": source_frame_count,
            "unselected_source_frames": source_frame_count - len(frames),
            "frames_selected_by_multiple_temporal_ranges": sum(
                len(frame.manifest_units) > 1 for frame in frames
            ),
            "frames_referenced_by_both_train_and_val": cross_split,
        }
    return result, audit, time_ranges


# DATA-ADAPTER CHANGE:
# Treat each manifest temporal block as an upstream-style short sequence. If a
# block is still large, split it chronologically without sampling or discarding
# frames. Source preprocessing remains unchanged inside every generated unit.
def build_processing_units(
    index: dict[tuple[str, str], list[SensorFrame]],
    time_ranges: dict[tuple[str, str, int], dict[str, Any]],
    max_frames_per_unit: int,
) -> tuple[list[ProcessingUnit], dict[str, Any]]:
    if max_frames_per_unit < 20 or max_frames_per_unit % 20 != 0:
        raise ValueError("max_frames_per_unit must be >=20 and divisible by 20")
    grouped: dict[tuple[str, str, int], dict[str, list[SensorFrame]]] = defaultdict(
        lambda: {sensor: [] for sensor in SENSORS}
    )
    for unit_key in time_ranges:
        grouped[unit_key]
    multi_unit_reference_frames = 0
    for (sequence, sensor), frames in index.items():
        for frame in frames:
            if len(frame.manifest_units) > 1:
                multi_unit_reference_frames += 1
                raise ValueError(f"Sensor frame belongs to overlapping ranges: {frame.path}")
            split, temporal_block = sorted(frame.manifest_units)[0]
            grouped[(sequence, split, temporal_block)][sensor].append(frame)

    units: list[ProcessingUnit] = []
    source_groups = []
    for (sequence, split, temporal_block), sensor_frames in sorted(grouped.items()):
        range_info = time_ranges[(sequence, split, temporal_block)]
        t_start = float(range_info["t_start"])
        t_end = float(range_info["t_end"])
        for sensor in SENSORS:
            sensor_frames[sensor].sort(key=lambda frame: float(frame.timestamp))
        mid360_frames = sensor_frames["lidar_360"]
        avia_frames = sensor_frames["livox_avia"]
        chunk_count = max(
            1, (len(mid360_frames) + max_frames_per_unit - 1) // max_frames_per_unit
        )
        group_audit: dict[str, Any] = {
            "sequence_id": sequence,
            "split": split,
            "temporal_block": temporal_block,
            "t_start": t_start,
            "t_end": t_end,
            "manifest_rows": int(range_info["manifest_rows"]),
            "selected_frames": {sensor: len(sensor_frames[sensor]) for sensor in SENSORS},
            "generated_chunks": chunk_count,
            "chunks": [],
        }
        source_groups.append(group_audit)
        mid360_times = np.asarray([float(frame.timestamp) for frame in mid360_frames])
        avia_times = np.asarray([float(frame.timestamp) for frame in avia_frames])
        for chunk_index in range(chunk_count):
            start = chunk_index * max_frames_per_unit
            stop = start + max_frames_per_unit
            mid360_chunk = mid360_frames[start:stop]
            if chunk_index == 0:
                chunk_start = t_start
            else:
                chunk_start = float((mid360_times[start - 1] + mid360_times[start]) / 2.0)
            if chunk_index == chunk_count - 1:
                chunk_end = t_end
                avia_stop = int(np.searchsorted(avia_times, chunk_end, side="right"))
            else:
                next_index = min(stop, len(mid360_times) - 1)
                chunk_end = float((mid360_times[stop - 1] + mid360_times[next_index]) / 2.0)
                avia_stop = int(np.searchsorted(avia_times, chunk_end, side="left"))
            avia_start = int(np.searchsorted(avia_times, chunk_start, side="left"))
            avia_chunk = avia_frames[avia_start:avia_stop]
            chunk_frames = {"lidar_360": mid360_chunk, "livox_avia": avia_chunk}
            if any(chunk_frames.values()):
                units.append(ProcessingUnit(
                    sequence_id=sequence,
                    split=split,
                    temporal_block=temporal_block,
                    chunk_index=chunk_index,
                    time_start=chunk_start,
                    time_end=chunk_end,
                    sensor_frames=chunk_frames,
                ))
                group_audit["chunks"].append({
                    "chunk_index": chunk_index,
                    "t_start": chunk_start,
                    "t_end": chunk_end,
                    "lidar_360_frames": len(mid360_chunk),
                    "livox_avia_frames": len(avia_chunk),
                })
    audit = {
        "strategy": (
            "manifest GT time range; Mid360 continuous-frame cap defines shared time chunks"
        ),
        "max_mid360_frames_per_unit": max_frames_per_unit,
        "source_groups": source_groups,
        "processing_unit_count": len(units),
        "multi_unit_reference_frames_assigned_once": multi_unit_reference_frames,
        "frames_are_sampled_or_discarded": False,
        "physical_frames_processed_more_than_once": False,
        "core_algorithm_changed": False,
    }
    return units, audit


def select_processing_units(
    units: list[ProcessingUnit], qualified_name: str | None,
) -> list[ProcessingUnit]:
    if qualified_name is None:
        return units
    selected = [unit for unit in units if unit.qualified_name == qualified_name]
    if not selected:
        examples = ", ".join(unit.qualified_name for unit in units[:5])
        raise ValueError(
            f"Unknown --unit {qualified_name!r}. Expected an exact name such as: {examples}"
        )
    return selected


def inspect_schemas(index: dict[tuple[str, str], list[SensorFrame]]) -> dict[str, Any]:
    result = {}
    for key, frames in index.items():
        dtypes: Counter[str] = Counter()
        column_counts: Counter[str] = Counter()
        row_counts: list[int] = []
        invalid_shapes: list[dict[str, Any]] = []
        failures = []
        for frame in frames:
            try:
                array = np.load(frame.path, mmap_mode="r", allow_pickle=False)
                dtypes[str(array.dtype)] += 1
                if array.ndim == 2:
                    row_counts.append(int(array.shape[0]))
                    column_counts[str(array.shape[1])] += 1
                else:
                    invalid_shapes.append({"path": str(frame.path), "shape": list(array.shape)})
            except Exception as exc:  # pragma: no cover - real corrupt-file reporting
                failures.append({"path": str(frame.path), "error": repr(exc)})
        example = frames[0] if frames else None
        example_array = (
            np.load(example.path, mmap_mode="r", allow_pickle=False) if example is not None else None
        )
        result[f"{key[0]}/{key[1]}"] = {
            "dtype_counts": dict(dtypes), "column_counts": dict(column_counts),
            "row_count": None if not row_counts else {
                "min": min(row_counts), "median": float(np.median(row_counts)),
                "max": max(row_counts),
            },
            "invalid_shapes": invalid_shapes, "load_failures": failures,
            "extra_columns_detected": any(int(columns) > 3 for columns in column_counts),
            "example": None if example is None else {
                "timestamp": example.timestamp, "path": str(example.path),
                "shape": list(example_array.shape), "dtype": str(example_array.dtype),
            },
            "xyz_policy": "columns[0:3]; extra columns reported and excluded from DBSCAN",
        }
    return result


def load_xyz_frames(frames: list[SensorFrame]) -> dict[str, np.ndarray]:
    data = {}
    for frame in frames:
        array = np.load(frame.path, allow_pickle=False)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Expected (N,>=3), got {array.shape}: {frame.path}")
        # DATA-ADAPTER CHANGE:
        # only changes how current dataset paths/timestamps are supplied.
        # Additional columns must not enter the source DBSCAN distance.
        data[frame.timestamp] = np.asarray(array[:, :3])
    return data


def _gt_oracle_status(calibration_path: Path) -> tuple[str, dict[str, Any]]:
    """Check transform credibility without reading any GT labels."""
    if not calibration_path.is_file():
        return "SKIPPED_UNVERIFIED_TRANSFORM", {
            "reason": f"calibration file does not exist: {calibration_path}",
        }
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "SKIPPED_UNVERIFIED_TRANSFORM", {"reason": repr(exc)}
    credibility = calibration.get("credibility", {})
    if credibility.get("credible") is not True:
        return "SKIPPED_UNVERIFIED_TRANSFORM", {
            "reason": "Mid360-to-GT calibration is explicitly not credible",
            "calibration_path": str(calibration_path.resolve()),
            "credibility": credibility,
        }
    return "AVAILABLE_BUT_NOT_IMPLEMENTED", {
        "reason": (
            "A credible transform was found, but this audit intentionally refuses to "
            "silently choose a sensor-to-GT timestamp association convention."
        ),
        "calibration_path": str(calibration_path.resolve()),
    }


def _prepare_single_diagnostic_block(
    lidar_360_data: dict[str, np.ndarray],
) -> tuple[str, list[str], np.ndarray]:
    if len(lidar_360_data) != 20:
        raise ValueError(
            "--diagnostic requires exactly one 20-frame Mid360 processing unit; "
            f"received {len(lidar_360_data)} frames"
        )
    blocks, block_timestamps = _accumulate_lidar_360_blocks(lidar_360_data)
    if len(blocks) != 1:
        raise RuntimeError(f"Expected exactly one accumulated block, got {len(blocks)}")
    block_timestamp, data_with_ind = next(iter(blocks.items()))
    frame_timestamps = block_timestamps[block_timestamp]
    return block_timestamp, frame_timestamps, data_with_ind


def _diagnose_accumulated_block(
    block_timestamp: str,
    frame_timestamps: list[str],
    data_with_ind: np.ndarray,
    model: torch.nn.Module,
    eps: float,
) -> dict[str, Any]:
    """Observe DBSCAN/features/LSTM without feeding results into the baseline."""
    if data_with_ind.size == 0:
        labels = np.empty((0,), dtype=np.int64)
        data = np.empty((0, 3), dtype=np.float64)
        time_ind = np.empty((0,), dtype=np.float64)
    else:
        time_ind = data_with_ind[:, 0]
        data = data_with_ind[:, 1:]
        # DIAGNOSTIC REPLAY: min_samples and the selected eps are observational.
        labels = _dbscan_labels(data, eps=eps, min_samples=10)
    features, cluster_labels_array = extract_feature_set_predict(data, labels, time_ind)
    cluster_ids = cluster_labels_array.reshape(-1).astype(np.int64)
    if features.shape != (len(cluster_ids), 20, 9):
        raise RuntimeError(f"Expected features [M,20,9], got {features.shape}")

    if len(cluster_ids):
        with torch.no_grad():
            logits_tensor = model(torch.as_tensor(features, dtype=torch.float32))
            probabilities_tensor = torch.softmax(logits_tensor, dim=1)
            predictions_tensor = torch.argmax(logits_tensor, dim=1)
        logits = logits_tensor.cpu().numpy()
        probabilities = probabilities_tensor.cpu().numpy()
        predictions = predictions_tensor.cpu().numpy().astype(np.int64)
    else:
        # Keep the required rank even when DBSCAN forms no non-noise clusters.
        logits = np.empty((0, 2), dtype=np.float32)
        probabilities = np.empty((0, 2), dtype=np.float32)
        predictions = np.empty((0,), dtype=np.int64)
    if logits.shape != (len(cluster_ids), 2):
        raise RuntimeError(f"Expected logits [M,2], got {logits.shape}")
    if probabilities.shape != (len(cluster_ids), 2):
        raise RuntimeError(f"Expected probabilities [M,2], got {probabilities.shape}")

    centers = np.empty((len(cluster_ids), 3), dtype=np.float64)
    num_points = np.empty((len(cluster_ids),), dtype=np.int64)
    per_frame_counts = np.empty((len(cluster_ids), 20), dtype=np.int64)
    records = []
    for cluster_index, cluster_id in enumerate(cluster_ids):
        member_mask = labels == cluster_id
        cluster_points = data[member_mask]
        cluster_times = time_ind[member_mask]
        centers[cluster_index] = cluster_points.mean(axis=0)
        num_points[cluster_index] = cluster_points.shape[0]
        per_frame_counts[cluster_index] = np.asarray([
            np.count_nonzero(cluster_times == frame_index)
            for frame_index in range(1, 21)
        ])
        records.append({
            "cluster_id": int(cluster_id),
            "num_points": int(num_points[cluster_index]),
            "center_xyz": centers[cluster_index].tolist(),
            "per_frame_point_count": per_frame_counts[cluster_index].tolist(),
            "feature_20x9": features[cluster_index].tolist(),
            "lstm_logits": logits[cluster_index].tolist(),
            "lstm_probability": probabilities[cluster_index].tolist(),
            "predicted_class": int(predictions[cluster_index]),
            "gt_oracle_status": "PENDING_STATUS_CHECK",
            "oracle_min_center_distance_m": None,
            "oracle_min_point_distance_m": None,
            "oracle_hit_1m": None,
            "oracle_hit_2m": None,
            "oracle_hit_5m": None,
        })
    p_uav = probabilities[:, 1] if len(cluster_ids) else np.empty((0,), dtype=np.float32)
    boundary_index = int(np.argmin(np.abs(p_uav - 0.5))) if len(p_uav) else None
    noise_points = int(np.count_nonzero(labels == -1))
    return {
        "block_timestamp": block_timestamp,
        "frame_timestamps": frame_timestamps,
        "dbscan_eps": eps,
        "dbscan_min_samples": 10,
        "features": features,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "cluster_ids": cluster_ids,
        "centers": centers,
        "num_points": num_points,
        "per_frame_counts": per_frame_counts,
        "records": records,
        "summary": {
            "dbscan_clusters": int(len(cluster_ids)),
            "noise_points": noise_points,
            "largest_cluster_points": 0 if not len(num_points) else int(num_points.max()),
            "median_cluster_points": 0.0 if not len(num_points) else float(np.median(num_points)),
            "max_P_uav": None if not len(p_uav) else float(p_uav.max()),
            "mean_P_uav": None if not len(p_uav) else float(p_uav.mean()),
            "positive_clusters": int(np.count_nonzero(predictions == 1)),
            "closest_to_decision_boundary_cluster": None if boundary_index is None else {
                "cluster_id": int(cluster_ids[boundary_index]),
                "P_uav": float(p_uav[boundary_index]),
            },
        },
    }


def build_pre_lstm_cluster_diagnostics(
    lidar_360_data: dict[str, np.ndarray],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Replay the unchanged eps=2 Mid360 diagnostic path for observation only."""
    block_timestamp, frame_timestamps, data_with_ind = _prepare_single_diagnostic_block(
        lidar_360_data
    )
    model = load_original_checkpoint(checkpoint_path)
    return _diagnose_accumulated_block(
        block_timestamp, frame_timestamps, data_with_ind, model, eps=2
    )


def build_eps_comparison_diagnostics(
    lidar_360_data: dict[str, np.ndarray], checkpoint_path: Path,
) -> dict[str, dict[str, Any]]:
    """Run eps=2 baseline replay and eps=1 diagnostic on one shared point cloud."""
    block_timestamp, frame_timestamps, data_with_ind = _prepare_single_diagnostic_block(
        lidar_360_data
    )
    model = load_original_checkpoint(checkpoint_path)
    return {
        "eps2": _diagnose_accumulated_block(
            block_timestamp, frame_timestamps, data_with_ind, model, eps=2
        ),
        "eps1": _diagnose_accumulated_block(
            block_timestamp, frame_timestamps, data_with_ind, model, eps=1
        ),
    }


def save_pre_lstm_cluster_diagnostics(
    diagnostic: dict[str, Any], output_dir: Path, calibration_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_status, oracle_detail = _gt_oracle_status(calibration_path)
    for record in diagnostic["records"]:
        record["gt_oracle_status"] = oracle_status
    np.savez_compressed(
        output_dir / "cluster_features.npz",
        features=diagnostic["features"],
        logits=diagnostic["logits"],
        probabilities=diagnostic["probabilities"],
        predictions=diagnostic["predictions"],
        cluster_ids=diagnostic["cluster_ids"],
        centers=diagnostic["centers"],
        num_points=diagnostic["num_points"],
        per_frame_counts=diagnostic["per_frame_counts"],
    )
    json_payload = {
        "diagnostic_only": True,
        "affects_baseline_output": False,
        "block_timestamp": diagnostic["block_timestamp"],
        "frame_timestamps": diagnostic["frame_timestamps"],
        "dbscan": {
            "eps": diagnostic["dbscan_eps"],
            "min_samples": diagnostic["dbscan_min_samples"],
        },
        "shapes": {
            "features": list(diagnostic["features"].shape),
            "logits": list(diagnostic["logits"].shape),
            "probabilities": list(diagnostic["probabilities"].shape),
        },
        "gt_oracle_status": oracle_status,
        "gt_oracle_detail": oracle_detail,
        "summary": diagnostic["summary"],
        "clusters": diagnostic["records"],
    }
    write_json(json_payload, output_dir / "cluster_diagnostics.json")

    print(f"DBSCAN clusters: {len(diagnostic['cluster_ids'])}")
    print("cluster | points | center_xyz | P(bg) | P(uav) | pred")
    for index, cluster_id in enumerate(diagnostic["cluster_ids"]):
        center = ",".join(f"{value:.3f}" for value in diagnostic["centers"][index])
        probability = diagnostic["probabilities"][index]
        print(
            f"{int(cluster_id)} | {int(diagnostic['num_points'][index])} | "
            f"[{center}] | {probability[0]:.3f} | {probability[1]:.3f} | "
            f"{int(diagnostic['predictions'][index])}"
        )
    summary = diagnostic["summary"]
    print(f"max_P_uav={summary['max_P_uav']}")
    print(f"mean_P_uav={summary['mean_P_uav']}")
    print(f"positive_clusters={summary['positive_clusters']}")
    print(
        "closest_to_decision_boundary_cluster="
        f"{summary['closest_to_decision_boundary_cluster']}"
    )
    print(f"gt_oracle_status={oracle_status}")
    return json_payload


def save_eps_comparison_diagnostics(
    comparison: dict[str, dict[str, Any]], diagnostics_dir: Path,
) -> dict[str, Any]:
    """Save the eps=1 sidecar and compact eps=2/eps=1 comparison."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    eps1 = comparison["eps1"]
    np.savez_compressed(
        diagnostics_dir / "eps1_cluster_features.npz",
        features=eps1["features"],
        logits=eps1["logits"],
        probabilities=eps1["probabilities"],
        predictions=eps1["predictions"],
        cluster_ids=eps1["cluster_ids"],
        centers=eps1["centers"],
        num_points=eps1["num_points"],
        per_frame_counts=eps1["per_frame_counts"],
    )
    rows = []
    for name in ("eps2", "eps1"):
        diagnostic = comparison[name]
        summary = diagnostic["summary"]
        rows.append({
            "eps": diagnostic["dbscan_eps"],
            "min_samples": diagnostic["dbscan_min_samples"],
            "num_clusters": summary["dbscan_clusters"],
            "noise_points": summary["noise_points"],
            "largest_cluster_points": summary["largest_cluster_points"],
            "median_cluster_points": summary["median_cluster_points"],
            "positive_clusters": summary["positive_clusters"],
            "max_P_uav": summary["max_P_uav"],
            "mean_P_uav": summary["mean_P_uav"],
        })
    payload = {
        "diagnostic_only": True,
        "affects_baseline_output": False,
        "gt_used": False,
        "shared_accumulated_point_cloud": True,
        "baseline": "eps=2,min_samples=10",
        "control": "eps=1,min_samples=10",
        "comparison": rows,
        "eps1_shapes": {
            "features": list(eps1["features"].shape),
            "logits": list(eps1["logits"].shape),
            "probabilities": list(eps1["probabilities"].shape),
        },
    }
    write_json(payload, diagnostics_dir / "eps_comparison.json")

    print("eps | num_clusters | noise_points | largest_cluster_points | "
          "median_cluster_points | positive_clusters | max_P_uav | mean_P_uav")
    for row in rows:
        print(
            f"{row['eps']} | {row['num_clusters']} | {row['noise_points']} | "
            f"{row['largest_cluster_points']} | {row['median_cluster_points']:.1f} | "
            f"{row['positive_clusters']} | {row['max_P_uav']} | {row['mean_P_uav']}"
        )
    print("eps=1 clusters:")
    print("cluster | points | center_xyz | P(bg) | P(uav) | pred")
    for index, cluster_id in enumerate(eps1["cluster_ids"]):
        center = ",".join(f"{value:.3f}" for value in eps1["centers"][index])
        probability = eps1["probabilities"][index]
        print(
            f"{int(cluster_id)} | {int(eps1['num_points'][index])} | "
            f"[{center}] | {probability[0]:.3f} | {probability[1]:.3f} | "
            f"{int(eps1['predictions'][index])}"
        )
    return payload


def checkpoint_audit(source_repo: Path, checkpoint: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_repo": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
        "source_repo_path": str(source_repo.resolve()),
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_load_success": False,
    }
    try:
        result["source_worktree_head"] = subprocess.check_output(
            ["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True
        ).strip()
        result["source_commit_matches"] = result["source_worktree_head"] == SOURCE_COMMIT
        result["source_worktree_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(source_repo), "status", "--porcelain"], text=True
        ).strip())
        result["checkpoint_commit_blob"] = subprocess.check_output(
            [
                "git", "-C", str(source_repo), "rev-parse",
                f"{SOURCE_COMMIT}:point_cloud_processing/tracker/lstm_model.pth",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        result["source_git_error"] = repr(exc)
        result["source_commit_matches"] = False
    if checkpoint.is_file():
        result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        try:
            result["checkpoint_worktree_blob"] = subprocess.check_output(
                ["git", "-C", str(source_repo), "hash-object", str(checkpoint)], text=True
            ).strip()
            result["checkpoint_matches_commit_blob"] = (
                result["checkpoint_worktree_blob"] == result.get("checkpoint_commit_blob")
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            result["checkpoint_blob_check_error"] = repr(exc)
        try:
            load_original_checkpoint(checkpoint)
            result["checkpoint_load_success"] = True
        except Exception as exc:  # pragma: no cover - real checkpoint reporting
            result["checkpoint_load_error"] = repr(exc)
    else:
        result["checkpoint_status"] = "MISSING_ORIGINAL_CHECKPOINT"
    return result


def dependency_audit() -> dict[str, Any]:
    result = {}
    try:
        import sklearn
        result["scikit_learn"] = {"available": True, "version": sklearn.__version__}
    except ImportError:
        result["scikit_learn"] = {
            "available": False,
            "required_for_full_run": True,
            "install_hint": "python -m pip install scikit-learn==1.3.2",
        }
    return result


def print_dry_run(
    manifest_paths: list[Path], index: dict[tuple[str, str], list[SensorFrame]],
    index_audit: dict[str, Any], schemas: dict[str, Any], checkpoint: dict[str, Any],
    dependencies: dict[str, Any], output_dir: Path,
    processing_units: list[ProcessingUnit], unit_audit: dict[str, Any],
) -> None:
    print("mode: dry-run (no DBSCAN/LSTM inference)")
    print("detected_manifests:")
    for path in manifest_paths:
        print(f"  - {path}")
    print("sequences:", sorted({key[0] for key in index}))
    print(
        "timestamp_source: NPY filename stem; selection: every original sensor frame "
        "inside each manifest GT time range"
    )
    for key in sorted(index):
        name = f"{key[0]}/{key[1]}"
        counts = index_audit["by_sequence_sensor"][name]
        example = schemas[name]["example"]
        print(
            f"{name}: source_frames={counts['source_directory_frames']} "
            f"selected_unique={counts['unique_sensor_frames']} "
            f"cross_split_refs={counts['frames_referenced_by_both_train_and_val']}"
        )
        print(
            f"  unique_by_split={counts['unique_sensor_frames_by_split']} "
            f"multi_range_frames={counts['frames_selected_by_multiple_temporal_ranges']}"
        )
        print(
            f"  example timestamp={example['timestamp']} path={example['path']} "
            f"shape={tuple(example['shape'])} dtype={example['dtype']}"
        )
        print(
            f"  schema dtypes={schemas[name]['dtype_counts']} "
            f"columns={schemas[name]['column_counts']} rows={schemas[name]['row_count']}"
        )
        if schemas[name]["extra_columns_detected"]:
            print("  WARNING extra columns detected; baseline will pass XYZ columns 0:3 only")
    print("original_lstm_checkpoint:")
    print(f"  path={checkpoint['checkpoint_path']}")
    print(f"  exists={checkpoint['checkpoint_exists']}")
    print(f"  load_success={checkpoint['checkpoint_load_success']}")
    print(f"  matches_commit_blob={checkpoint.get('checkpoint_matches_commit_blob')}")
    print(f"  sha256={checkpoint.get('checkpoint_sha256')}")
    print("source:")
    print(f"  repo={checkpoint['source_repo']} commit={checkpoint['source_commit']}")
    print(f"  worktree_head_matches={checkpoint.get('source_commit_matches')}")
    print(f"  worktree_dirty={checkpoint.get('source_worktree_dirty')}")
    print("dependencies:", json.dumps(dependencies, ensure_ascii=False))
    source_counts = index_audit["by_sequence_sensor"]
    mavic2_groups = [
        group for group in unit_audit["source_groups"]
        if group["sequence_id"] == "Mavic2"
    ]
    if mavic2_groups:
        print("mavic2_temporal_blocks:")
        for group in mavic2_groups:
            selected = group["selected_frames"]
            print(
                f"  {group['split']}/block{group['temporal_block']:02d}: "
                f"range=[{group['t_start']:.6f},{group['t_end']:.6f}] "
                f"source_mid360={source_counts['Mavic2/lidar_360']['source_directory_frames']} "
                f"selected_mid360={selected['lidar_360']} "
                f"source_avia={source_counts['Mavic2/livox_avia']['source_directory_frames']} "
                f"selected_avia={selected['livox_avia']} "
                f"chunks={group['generated_chunks']}"
            )
    print("processing_unit_policy:")
    print(f"  strategy={unit_audit['strategy']}")
    print(f"  max_mid360_frames_per_unit={unit_audit['max_mid360_frames_per_unit']}")
    print(f"  processing_units={len(processing_units)}")
    if len(processing_units) <= 5:
        print(f"  selected_units={[unit.qualified_name for unit in processing_units]}")
    print(
        "  multi_unit_reference_frames_assigned_once="
        f"{unit_audit['multi_unit_reference_frames_assigned_once']}"
    )
    per_sequence_units = Counter(unit.sequence_id for unit in processing_units)
    print(f"  units_by_sequence={dict(sorted(per_sequence_units.items()))}")
    largest = {
        sensor: max(
            (len(unit.sensor_frames[sensor]) for unit in processing_units), default=0
        )
        for sensor in SENSORS
    }
    print(f"  largest_unit_frames={largest}")
    print(f"  avia_processed_point_upper_bound_per_unit={largest['livox_avia'] * 100}")
    print("planned_output_directory:", output_dir.resolve())


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Multi-Modal-UAV candidate baseline audit", "",
        f"- Source: `{audit['source']['source_repo']}`",
        f"- Commit: `{audit['source']['source_commit']}`",
        f"- Checkpoint: `{audit['source']['checkpoint_path']}`",
        f"- Seed: `{audit['seed']}`", "",
        "本报告只确认 source-faithful preprocessing/candidate 流程；不包含 GT proposal recall 或准确率结论。",
        "", "## Sequences", "",
    ]
    for sequence, units in audit["sequences"].items():
        lines.extend([f"### {sequence}", ""])
        for unit_name, item in units.items():
            lines.extend([
                f"#### {unit_name}", "",
                f"- Avia: `{json.dumps(item['avia'], ensure_ascii=False)}`",
                f"- Mid360: `{json.dumps(item['mid360'], ensure_ascii=False)}`",
                f"- Fusion: `{json.dumps(item['fusion'], ensure_ascii=False)}`",
                f"- Candidate: `{json.dumps(item['candidate'], ensure_ascii=False)}`", "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_full(
    processing_units: list[ProcessingUnit], output_dir: Path,
    checkpoint_path: Path, source_audit: dict[str, Any], index_audit: dict[str, Any],
    schemas: dict[str, Any], unit_audit: dict[str, Any], seed: int,
    diagnostic: bool = False,
    lidar_calibration_path: Path = PROJECT_ROOT / "calibration/lidar360_frame_resolution.json",
) -> None:
    if not source_audit["checkpoint_exists"] or not source_audit["checkpoint_load_success"]:
        raise RuntimeError("MISSING_ORIGINAL_CHECKPOINT or checkpoint load failure; Mid360 run stopped")
    dependencies = dependency_audit()
    if not dependencies["scikit_learn"]["available"]:
        raise RuntimeError("scikit-learn is missing; full DBSCAN path cannot run")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if diagnostic:
        if len(processing_units) != 1:
            raise ValueError(
                "--diagnostic supports exactly one processing unit; select it with --unit"
            )
        mid360_count = len(processing_units[0].sensor_frames["lidar_360"])
        if mid360_count != 20:
            raise ValueError(
                "--diagnostic supports exactly 20 Mid360 frames; "
                f"selected unit contains {mid360_count}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    # REPRODUCIBILITY CHANGE:
    # does not modify FPS algorithm
    np.random.seed(seed)
    audit: dict[str, Any] = {
        "source": source_audit, "seed": seed, "test_read": False,
        "adapter": index_audit, "processing_units": unit_audit,
        "schemas": schemas, "sequences": {},
        "warnings": [
            "Each adapter processing unit runs the unchanged source preprocessing independently.",
            "No coordinate correction is applied even if the two LiDAR frames may differ.",
        ],
    }
    for unit in processing_units:
        print(f"Processing {unit.sequence_id}/{unit.name}", flush=True)
        sequence_dir = output_dir / unit.sequence_id / unit.name
        avia_raw = load_xyz_frames(unit.sensor_frames["livox_avia"])
        lidar_raw = load_xyz_frames(unit.sensor_frames["lidar_360"])
        avia_audit = process_lidar_livox(
            avia_raw, sequence_dir / "livox_avia_processed", max_pts=100
        )
        mid_audit = process_lidar_360(
            lidar_raw, sequence_dir / "lidar_360_processed", checkpoint_path
        )
        diagnostic_audit = None
        if diagnostic:
            # DIAGNOSTIC SIDECAR:
            # Replay the exact pre-LSTM path only for observation. The replayed
            # labels/predictions are never used by fusion or candidate generation.
            diagnostic_comparison = build_eps_comparison_diagnostics(
                lidar_raw, checkpoint_path
            )
            diagnostic_data = diagnostic_comparison["eps2"]
            if diagnostic_data["summary"]["dbscan_clusters"] != mid_audit["dbscan_clusters"]:
                raise RuntimeError(
                    "Diagnostic replay cluster count differs from baseline processing: "
                    f"{diagnostic_data['summary']['dbscan_clusters']} vs "
                    f"{mid_audit['dbscan_clusters']}"
                )
            if diagnostic_data["summary"]["positive_clusters"] != mid_audit["lstm_positive_clusters"]:
                raise RuntimeError(
                    "Diagnostic replay prediction count differs from baseline processing: "
                    f"{diagnostic_data['summary']['positive_clusters']} vs "
                    f"{mid_audit['lstm_positive_clusters']}"
                )
            diagnostic_audit = save_pre_lstm_cluster_diagnostics(
                diagnostic_data,
                sequence_dir / "diagnostics/pre_lstm_clusters",
                lidar_calibration_path,
            )
            diagnostic_audit["eps_comparison"] = save_eps_comparison_diagnostics(
                diagnostic_comparison, sequence_dir / "diagnostics"
            )
        avia_processed = read_lidar_files(sequence_dir / "livox_avia_processed")
        lidar_processed = read_lidar_files(sequence_dir / "lidar_360_processed")
        fusion_audit = process_fusion(
            avia_processed, lidar_processed, sequence_dir / "lidar_fusion"
        )
        fusion_data = read_lidar_files(sequence_dir / "lidar_fusion")
        candidate_audit = write_candidate_sidecar(
            fusion_data, sequence_dir / "baseline_candidates"
        )
        audit["sequences"].setdefault(unit.sequence_id, {})[unit.name] = {
            "split": unit.split,
            "temporal_block": unit.temporal_block,
            "chunk_index": unit.chunk_index,
            "time_start": unit.time_start,
            "time_end": unit.time_end,
            "avia": avia_audit, "mid360": mid_audit,
            "fusion": fusion_audit, "candidate": candidate_audit,
            "pre_lstm_diagnostic": diagnostic_audit,
        }
    write_json(audit, output_dir / "baseline_audit.json")
    write_summary(audit, output_dir / "baseline_summary.md")
    print(output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "run"), default="dry-run")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/official/v1"))
    parser.add_argument("--splits", nargs="+", choices=ALLOWED_SPLITS, default=["train", "val"])
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument(
        "--unit", default=None,
        help="Run one exact generated unit, e.g. Mavic2/train_block00_chunk000.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diagnostic", action="store_true",
        help=(
            "Save a read-only DBSCAN-to-LSTM diagnostic for one exact 20-frame unit. "
            "This does not change baseline candidates."
        ),
    )
    parser.add_argument(
        "--lidar-calibration", type=Path,
        default=PROJECT_ROOT / "calibration/lidar360_frame_resolution.json",
        help="Credibility metadata used only to decide whether GT oracle audit is allowed.",
    )
    parser.add_argument(
        "--max-frames-per-unit", type=int, default=200,
        help=(
            "Mid360 frame cap defining shared time chunks; must be >=20 and "
            "divisible by 20. Avia is selected by the same time boundaries."
        ),
    )
    parser.add_argument(
        "--source-repo", type=Path,
        default=Path("/home/jasoncui/projects/open_source/Multi-Modal-UAV"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/mmuav_candidate_baseline_f11b573",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diagnostic and args.mode != "run":
        raise ValueError("--diagnostic must be used together with --mode run")
    checkpoint = args.checkpoint or (
        args.source_repo / "point_cloud_processing/tracker/lstm_model.pth"
    )
    rows, manifest_paths = read_manifests(args.manifest_dir, args.splits)
    index, index_audit, time_ranges = build_unique_sensor_frame_index(
        rows, args.dataset_root, args.sequence
    )
    processing_units, unit_audit = build_processing_units(
        index, time_ranges, args.max_frames_per_unit
    )
    processing_units = select_processing_units(processing_units, args.unit)
    unit_audit["selected_processing_unit_count"] = len(processing_units)
    unit_audit["selected_processing_units"] = [
        unit.qualified_name for unit in processing_units
    ]
    schemas = inspect_schemas(index)
    source_audit = checkpoint_audit(args.source_repo, checkpoint)
    dependencies = dependency_audit()
    if args.mode == "dry-run":
        print_dry_run(
            manifest_paths, index, index_audit, schemas,
            source_audit, dependencies, args.output_dir,
            processing_units, unit_audit,
        )
        return
    run_full(
        processing_units, args.output_dir, checkpoint, source_audit,
        index_audit, schemas, unit_audit, args.seed,
        diagnostic=args.diagnostic,
        lidar_calibration_path=args.lidar_calibration,
    )


if __name__ == "__main__":
    main()
