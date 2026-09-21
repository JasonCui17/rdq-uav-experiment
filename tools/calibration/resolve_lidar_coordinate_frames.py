#!/usr/bin/env python3
"""Independently fit lidar_360->GT and livox_avia->GT coordinate transforms.

Only train data determine each transform. Validation and same-sequence shuffled
validation are evaluation-only. Test manifests are never opened.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.utils.io import write_json  # noqa: E402


STATUS_OUT_OF_RANGE = "OUT_OF_RANGE"
STATUS_NO_VALID_CLOUD = "NO_VALID_CLOUD"
STATUS_ELIGIBLE = "ELIGIBLE"


@dataclass(frozen=True)
class SensorSpec:
    name: str
    directory: str
    min_range_m: float
    max_range_m: float
    distance_bins_m: tuple[float, ...]
    observability: dict[str, Any]


def proper_axis_rotations() -> list[np.ndarray]:
    """Return all 24 orientation-preserving signed-axis permutations."""
    rotations: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return rotations


def timestamp_index(directory: Path) -> tuple[np.ndarray, list[Path]]:
    entries: list[tuple[float, Path]] = []
    for path in directory.glob("*.npy"):
        try:
            entries.append((float(path.stem), path))
        except ValueError:
            continue
    entries.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in entries], dtype=np.float64), [item[1] for item in entries]


def nearest_timestamp(query: float, timestamps: np.ndarray) -> int | None:
    if not len(timestamps):
        return None
    insertion = int(np.searchsorted(timestamps, query))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    return min(candidates, key=lambda index: abs(float(timestamps[index]) - query))


def load_xyz(path: Path) -> np.ndarray:
    points = np.asarray(np.load(path, allow_pickle=False))
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected (N,>=3) point cloud, got {points.shape}: {path}")
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0, axis=1)
    return xyz[valid]


def read_manifest(path: Path) -> list[dict[str, str]]:
    if path.stem not in {"train", "val"}:
        raise ValueError(f"Only train/val manifests are permitted, got: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_from_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])], dtype=np.float64)


def build_records(
    rows: list[dict[str, str]], dataset_root: Path, sensor: SensorSpec, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexes: dict[str, tuple[np.ndarray, list[Path]]] = {}
    records: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    sequences = sorted({row["sequence_id"] for row in rows})
    for sequence in sequences:
        indexes[sequence] = timestamp_index(dataset_root / sequence / sensor.directory)

    if sensor.observability.get("enabled", False):
        raise ValueError(
            f"{sensor.name}: angular observability was enabled without an independently "
            "resolved installed sensor frame. Disable it or implement a verified mask."
        )

    for row in tqdm(rows, desc=f"load {split} {sensor.name}", unit="GT"):
        target = target_from_row(row)
        target_range = float(np.linalg.norm(target))
        base = {
            "sample_id": row["sample_id"], "sequence_id": row["sequence_id"],
            "gt_time": float(row["gt_time"]), "target_range_m": target_range,
        }
        if not (sensor.min_range_m <= target_range <= sensor.max_range_m):
            statuses.append({**base, "status": STATUS_OUT_OF_RANGE})
            continue

        timestamps, paths = indexes[row["sequence_id"]]
        nearest = nearest_timestamp(base["gt_time"], timestamps)
        if nearest is None:
            statuses.append({**base, "status": STATUS_NO_VALID_CLOUD, "reason": "NO_CLOUD_FILE"})
            continue
        cloud_time = float(timestamps[nearest])
        points = load_xyz(paths[nearest])
        if not len(points):
            statuses.append({
                **base, "status": STATUS_NO_VALID_CLOUD, "reason": "EMPTY_AFTER_FILTER",
                "cloud_time": cloud_time, "abs_dt_s": abs(cloud_time - base["gt_time"]),
            })
            continue
        record = {
            **base, "status": STATUS_ELIGIBLE, "points": points,
            "cloud_path": str(paths[nearest]), "cloud_time": cloud_time,
            "abs_dt_s": abs(cloud_time - base["gt_time"]), "target": target,
        }
        records.append(record)
        statuses.append({key: value for key, value in record.items() if key not in {"points", "target"}})

    counts = {
        "total": len(statuses),
        "in_range": sum(item["status"] != STATUS_OUT_OF_RANGE for item in statuses),
        "out_of_range": sum(item["status"] == STATUS_OUT_OF_RANGE for item in statuses),
        "valid_cloud": len(records),
        "no_valid_cloud": sum(item["status"] == STATUS_NO_VALID_CLOUD for item in statuses),
    }
    gaps = np.asarray([record["abs_dt_s"] for record in records], dtype=np.float64)
    counts["eligible"] = len(records)
    counts["by_sequence"] = {}
    for sequence in sequences:
        sequence_statuses = [item for item in statuses if item["sequence_id"] == sequence]
        counts["by_sequence"][sequence] = {
            "total": len(sequence_statuses),
            "in_range": sum(item["status"] != STATUS_OUT_OF_RANGE for item in sequence_statuses),
            "out_of_range": sum(item["status"] == STATUS_OUT_OF_RANGE for item in sequence_statuses),
            "valid_cloud": sum(item["status"] == STATUS_ELIGIBLE for item in sequence_statuses),
            "no_valid_cloud": sum(item["status"] == STATUS_NO_VALID_CLOUD for item in sequence_statuses),
        }
    counts["timestamp_gap_ms"] = {
        "mean": float(gaps.mean() * 1000.0) if len(gaps) else math.nan,
        "median": float(np.median(gaps) * 1000.0) if len(gaps) else math.nan,
        "p95": float(np.quantile(gaps, 0.95) * 1000.0) if len(gaps) else math.nan,
        "max": float(gaps.max() * 1000.0) if len(gaps) else math.nan,
    }
    return records, {"counts": counts, "statuses": statuses}


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def paired_targets(records: list[dict[str, Any]], shuffled: bool) -> list[dict[str, Any]]:
    if not shuffled:
        return records
    output: list[dict[str, Any] | None] = [None] * len(records)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["sequence_id"]].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: records[index]["gt_time"])
        shift = max(1, len(indices) // 2) if len(indices) > 1 else 0
        for local_index, record_index in enumerate(indices):
            output[record_index] = records[indices[(local_index + shift) % len(indices)]]
    return [item for item in output if item is not None]


def nearest_distances(
    records: list[dict[str, Any]], rotation: np.ndarray, translation: np.ndarray,
    targets: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    if targets is None:
        targets = records
    distances = np.empty(len(records), dtype=np.float64)
    for index, (record, target_record) in enumerate(zip(records, targets)):
        transformed = transform_points(record["points"], rotation, translation)
        delta = transformed - target_record["target"]
        distances[index] = math.sqrt(float(np.einsum("ij,ij->i", delta, delta).min()))
    return distances


def nearest_correspondences(
    records: list[dict[str, Any]], rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected, targets, distances = [], [], []
    for record in records:
        transformed = transform_points(record["points"], rotation, translation)
        delta = transformed - record["target"]
        squared = np.einsum("ij,ij->i", delta, delta)
        index = int(np.argmin(squared))
        selected.append(np.asarray(record["points"][index], dtype=np.float64))
        targets.append(record["target"])
        distances.append(math.sqrt(float(squared[index])))
    return np.asarray(selected), np.asarray(targets), np.asarray(distances)


def robust_score(distances: np.ndarray, clip_m: float) -> float:
    return float(np.mean(np.minimum(distances, clip_m)))


def fit_translation(
    records: list[dict[str, Any]], rotation: np.ndarray, bound_m: float,
    iterations: int, clip_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(iterations):
        selected, targets, _ = nearest_correspondences(records, rotation, translation)
        proposed = np.median(targets - selected @ rotation.T, axis=0)
        proposed = np.clip(proposed, -bound_m, bound_m)
        if np.linalg.norm(proposed - translation) < 1e-5:
            translation = proposed
            break
        translation = proposed
    distances = nearest_distances(records, rotation, translation)
    return translation, distances, robust_score(distances, clip_m)


def fit_axis(
    records: list[dict[str, Any]], sensor_name: str, bound_m: float,
    iterations: int, clip_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidates = []
    for rotation in tqdm(proper_axis_rotations(), desc=f"fit 24 axes {sensor_name}", unit="axis"):
        translation, distances, score = fit_translation(
            records, rotation, bound_m, iterations, clip_m
        )
        candidates.append((score, float(np.median(distances)), rotation, translation))
    score, median, rotation, translation = min(candidates, key=lambda item: (item[0], item[1]))
    return rotation, translation, {
        "train_robust_score": score, "train_median_m": median, "candidates": 24,
    }


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, target_mean - rotation @ source_mean


def refine_rigid(
    records: list[dict[str, Any]], initial_rotation: np.ndarray,
    initial_translation: np.ndarray, bound_m: float, iterations: int, clip_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rotation, translation = initial_rotation.copy(), initial_translation.copy()
    distances = nearest_distances(records, rotation, translation)
    score = robust_score(distances, clip_m)
    initial_score = score
    accepted = 0
    for _ in range(iterations):
        selected, targets, distances = nearest_correspondences(records, rotation, translation)
        cutoff = min(2.0, float(np.quantile(distances, 0.5)))
        keep = distances <= cutoff
        if int(keep.sum()) < 12:
            break
        proposed_rotation, proposed_translation = kabsch(selected[keep], targets[keep])
        if np.any(np.abs(proposed_translation) > bound_m):
            break
        proposed_distances = nearest_distances(records, proposed_rotation, proposed_translation)
        proposed_score = robust_score(proposed_distances, clip_m)
        if proposed_score >= score - 1e-6:
            break
        rotation, translation, distances, score = (
            proposed_rotation, proposed_translation, proposed_distances, proposed_score
        )
        accepted += 1
    improvement = 1.0 - score / initial_score if initial_score > 0 else 0.0
    return rotation, translation, {
        "accepted_iterations": accepted, "initial_train_robust_score": initial_score,
        "train_robust_score": score, "relative_robust_improvement": improvement,
        "train_median_m": float(np.median(distances)),
    }


def distance_metrics(distances: np.ndarray) -> dict[str, Any]:
    if not len(distances):
        return {
            "eligible_frames": 0, "nearest_3d_mean_m": math.nan,
            "nearest_3d_median_m": math.nan, "nearest_3d_p75_m": math.nan,
            "nearest_3d_p90_m": math.nan, "hit_within_0.5m": math.nan,
            "hit_within_1m": math.nan, "hit_within_2m": math.nan,
        }
    return {
        "eligible_frames": len(distances),
        "nearest_3d_mean_m": float(distances.mean()),
        "nearest_3d_median_m": float(np.median(distances)),
        "nearest_3d_p75_m": float(np.quantile(distances, 0.75)),
        "nearest_3d_p90_m": float(np.quantile(distances, 0.90)),
        **{f"hit_within_{radius:g}m": float(np.mean(distances <= radius))
           for radius in (0.5, 1.0, 2.0)},
    }


def evaluate(
    records: list[dict[str, Any]], rotation: np.ndarray, translation: np.ndarray,
    bins: tuple[float, ...], shuffled: bool = False,
) -> dict[str, Any]:
    targets = paired_targets(records, shuffled)
    distances = nearest_distances(records, rotation, translation, targets)
    result = distance_metrics(distances)
    by_sequence = {}
    for sequence in sorted({record["sequence_id"] for record in records}):
        indices = [i for i, record in enumerate(records) if record["sequence_id"] == sequence]
        by_sequence[sequence] = distance_metrics(distances[indices])
    by_range = {}
    target_ranges = np.asarray([target["target_range_m"] for target in targets])
    for bin_index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        upper_match = target_ranges <= upper if bin_index == len(bins) - 2 else target_ranges < upper
        include = (target_ranges >= lower) & upper_match
        by_range[f"{lower:g}-{upper:g}m"] = distance_metrics(distances[include])
    result["by_sequence"] = by_sequence
    result["by_gt_range"] = by_range
    result["shuffle"] = shuffled
    result["hit_denominator"] = "eligible_frames_only"
    return result


def transform_payload(sensor: str, rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    payload = {
        "convention": "p_gt = R_gt_from_sensor @ p_sensor + t_gt_from_sensor",
        "rotation_gt_from_sensor": rotation.tolist(),
        "translation_gt_from_sensor_m": translation.tolist(),
        "rotation_euler_xyz_deg": Rotation.from_matrix(rotation).as_euler("xyz", degrees=True).tolist(),
    }
    payload[f"rotation_gt_from_{sensor}"] = rotation.tolist()
    payload[f"translation_gt_from_{sensor}_m"] = translation.tolist()
    return payload


def resolve_sensor(
    sensor: SensorSpec, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]],
    dataset_root: Path, fit_config: dict[str, Any], validation_config: dict[str, Any],
) -> dict[str, Any]:
    train, train_audit = build_records(train_rows, dataset_root, sensor, "train")
    val, val_audit = build_records(val_rows, dataset_root, sensor, "val")
    if len(train) < 24 or len(val) < 1:
        raise RuntimeError(f"Insufficient eligible frames for {sensor.name}: train={len(train)}, val={len(val)}")

    bound = float(fit_config["translation_bound_m"])
    translation_iterations = int(fit_config["translation_iterations"])
    rigid_iterations = int(fit_config["rigid_iterations"])
    clip_m = float(fit_config["robust_distance_clip_m"])
    min_continuous = float(fit_config["min_continuous_improvement"])

    identity_r, identity_t = np.eye(3), np.zeros(3)
    axis_r, axis_t, axis_fit = fit_axis(
        train, sensor.name, bound, translation_iterations, clip_m
    )
    rigid_r, rigid_t, rigid_fit = refine_rigid(
        train, axis_r, axis_t, bound, rigid_iterations, clip_m
    )
    selected_name = (
        "continuous_rigid" if rigid_fit["relative_robust_improvement"] >= min_continuous
        else "axis_translation"
    )
    candidates = {
        "identity": (identity_r, identity_t, {"fit": "none"}),
        "axis_translation": (axis_r, axis_t, axis_fit),
        "continuous_rigid": (rigid_r, rigid_t, rigid_fit),
    }
    results = {}
    for name, (rotation, translation, fit) in candidates.items():
        results[name] = {
            "transform": transform_payload(sensor.name, rotation, translation), "fit": fit,
            "train": evaluate(train, rotation, translation, sensor.distance_bins_m),
            "val": evaluate(val, rotation, translation, sensor.distance_bins_m),
            "val_same_sequence_shuffle": evaluate(
                val, rotation, translation, sensor.distance_bins_m, shuffled=True
            ),
        }

    identity_median = results["identity"]["val"]["nearest_3d_median_m"]
    selected_val = results[selected_name]["val"]
    selected_shuffle = results[selected_name]["val_same_sequence_shuffle"]
    val_gain = 1.0 - selected_val["nearest_3d_median_m"] / identity_median
    shuffle_median = selected_shuffle["nearest_3d_median_m"]
    normal_vs_shuffle_gain = 1.0 - selected_val["nearest_3d_median_m"] / shuffle_median
    credible = (
        val_gain >= float(validation_config["min_identity_median_gain"])
        and normal_vs_shuffle_gain >= float(validation_config["min_normal_vs_shuffle_median_gain"])
    )
    return {
        "sensor": sensor.name,
        "scope": {"test_read": False, "train": train_audit["counts"], "val": val_audit["counts"]},
        "eligibility": {
            "min_range_m": sensor.min_range_m, "max_range_m": sensor.max_range_m,
            "distance_bins_m": list(sensor.distance_bins_m),
            "observability": sensor.observability,
            "out_of_range_excluded_from_miss_hit_and_fit": True,
            "no_valid_cloud_excluded_from_fit": True,
        },
        "fit_independence": {
            "radar_transform_loaded": False,
            "shared_transform_between_lidars": False,
            "train_only_fit": True,
        },
        "selection": {
            "selected_transform": selected_name,
            "continuous_min_relative_improvement": min_continuous,
            "continuous_actual_relative_improvement": rigid_fit["relative_robust_improvement"],
            "reason": (
                "continuous improvement met threshold" if selected_name == "continuous_rigid"
                else "continuous improvement was small; retain discrete coordinate convention"
            ),
        },
        "credibility": {
            "val_median_gain_over_identity": val_gain,
            "val_normal_vs_shuffle_median_gain": normal_vs_shuffle_gain,
            "credible": credible,
            "criteria": validation_config,
        },
        "results": results,
        "status_records": {"train": train_audit["statuses"], "val": val_audit["statuses"]},
    }


def load_sensor_specs(config: dict[str, Any]) -> list[SensorSpec]:
    specs = []
    for name in ("lidar_360", "livox_avia"):
        item = config["sensors"][name]
        specs.append(SensorSpec(
            name=name, directory=str(item["directory"]),
            min_range_m=float(item["min_range_m"]), max_range_m=float(item["max_range_m"]),
            distance_bins_m=tuple(float(value) for value in item["distance_bins_m"]),
            observability=dict(item["observability"]),
        ))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/calibration/lidar_frame_resolution.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--calibration-dir", type=Path, default=PROJECT_ROOT / "calibration")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    manifest_dir = Path(config["manifest_dir"])
    # Intentionally name only train/val; no directory scan and no test access.
    train_rows = read_manifest(manifest_dir / "train.csv")
    val_rows = read_manifest(manifest_dir / "val.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"stage4_lidar_frame_resolution_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    args.calibration_dir.mkdir(parents=True, exist_ok=True)

    summary = {"test_read": False, "config": str(args.config.resolve()), "sensors": {}}
    for sensor in load_sensor_specs(config):
        print(f"\n=== Resolve {sensor.name} independently ===", flush=True)
        payload = resolve_sensor(
            sensor, train_rows, val_rows, dataset_root,
            config["fit"], config["validation"],
        )
        output_name = "lidar360_frame_resolution.json" if sensor.name == "lidar_360" else "livox_avia_frame_resolution.json"
        write_json(payload, args.calibration_dir / output_name)
        write_json(payload, run_dir / output_name)
        summary["sensors"][sensor.name] = {
            "calibration_file": str((args.calibration_dir / output_name).resolve()),
            "selected_transform": payload["selection"]["selected_transform"],
            "credibility": payload["credibility"],
            "scope": payload["scope"],
        }
    write_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
