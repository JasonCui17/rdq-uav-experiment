#!/usr/bin/env python3
"""Fit the smallest untested Radar->GT frame hypotheses on train, verify on val."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402


def proper_axis_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return rotations


def load_split(
    manifest: Path, dataset_root: Path, trajectories: dict[str, PositionTrajectory]
) -> list[dict[str, Any]]:
    split = manifest.stem
    if split not in {"train", "val"}:
        raise ValueError("Radar frame resolution is restricted to train/val")
    with manifest.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    records = []
    for row in tqdm(source, desc=f"load {split}", unit="frame"):
        path = Path(row["radar_path"])
        if not path.is_absolute():
            path = dataset_root / path
        points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected (N,>=3), got {points.shape}: {path}")
        points = points[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        sequence = row["sequence_id"]
        target, valid = trajectories[sequence].evaluate(float(row["radar_time"]))
        if not bool(valid) or not len(points):
            continue
        records.append(
            {
                "sample_id": row["sample_id"], "split": split, "sequence_id": sequence,
                "points": points, "target": np.asarray(target, dtype=np.float64),
                "bbox": np.asarray([float(row[f"official_bbox_{k}"]) for k in ("x1", "y1", "x2", "y2")]),
            }
        )
    return records


def nearest_correspondences(
    records: list[dict[str, Any]], rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected, targets, distances = [], [], []
    for record in records:
        transformed = transform_points(record["points"], rotation, translation)
        delta = transformed - record["target"]
        index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        selected.append(record["points"][index])
        targets.append(record["target"])
        distances.append(float(np.linalg.norm(delta[index])))
    return np.asarray(selected), np.asarray(targets), np.asarray(distances)


def robust_score(distances: np.ndarray) -> float:
    return float(np.mean(np.minimum(distances, 5.0)))


def fit_translation_icp(
    records: list[dict[str, Any]], rotation: np.ndarray, iterations: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(iterations):
        selected, targets, _ = nearest_correspondences(records, rotation, translation)
        proposed = np.median(targets - selected @ rotation.T, axis=0)
        proposed = np.clip(proposed, -2.0, 2.0)
        if np.linalg.norm(proposed - translation) < 1e-5:
            translation = proposed
            break
        translation = proposed
    return translation, nearest_correspondences(records, rotation, translation)[2]


def fit_axis_translation(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidates = []
    for rotation in tqdm(proper_axis_rotations(), desc="fit 24 Radar axis hypotheses", unit="axis"):
        translation, distances = fit_translation_icp(records, rotation)
        candidates.append((robust_score(distances), float(np.median(distances)), rotation, translation))
    best = min(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[3], {"train_robust_score": best[0], "train_median_m": best[1], "candidates": 24}


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def refine_rigid_icp(
    records: list[dict[str, Any]], initial_rotation: np.ndarray, initial_translation: np.ndarray,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rotation, translation = initial_rotation.copy(), initial_translation.copy()
    _, _, current_distances = nearest_correspondences(records, rotation, translation)
    current_score = robust_score(current_distances)
    accepted = 0
    for _ in range(iterations):
        selected, targets, distances = nearest_correspondences(records, rotation, translation)
        cutoff = min(2.0, float(np.quantile(distances, 0.5)))
        keep = distances <= cutoff
        if keep.sum() < 12:
            break
        proposed_rotation, proposed_translation = kabsch(selected[keep], targets[keep])
        if np.any(np.abs(proposed_translation) > 2.0):
            break
        proposed_distances = nearest_correspondences(
            records, proposed_rotation, proposed_translation
        )[2]
        proposed_score = robust_score(proposed_distances)
        if proposed_score >= current_score - 1e-6:
            break
        rotation, translation = proposed_rotation, proposed_translation
        current_distances, current_score = proposed_distances, proposed_score
        accepted += 1
    return rotation, translation, {
        "accepted_iterations": accepted, "train_robust_score": current_score,
        "train_median_m": float(np.median(current_distances)),
    }


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def half_cycle(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["sequence_id"]].append(index)
    result: list[dict[str, Any] | None] = [None] * len(records)
    for indices in grouped.values():
        shift = max(1, len(indices) // 2) if len(indices) > 1 else 0
        for local, index in enumerate(indices):
            result[index] = records[indices[(local + shift) % len(indices)]]
    return [record for record in result if record is not None]


def evaluate(
    records: list[dict[str, Any]], rotation: np.ndarray, translation: np.ndarray,
    camera: OmniRadtanCamera, camera_rotation: np.ndarray, camera_translation: np.ndarray,
    shuffled: bool = False,
) -> dict[str, Any]:
    targets = half_cycle(records) if shuffled else records
    distances_3d, distances_px = [], []
    coverage = {8: 0, 16: 0, 32: 0, 64: 0}
    for record, target_record in zip(records, targets):
        points_gt = transform_points(record["points"], rotation, translation)
        distances_3d.append(float(np.linalg.norm(points_gt - target_record["target"], axis=1).min()))
        pixels, valid = camera.project(
            transform_points(points_gt, camera_rotation, camera_translation), require_in_image=True
        )
        pixels = pixels[valid]
        if len(pixels):
            pixel_distances = np.linalg.norm(pixels - bbox_center(target_record["bbox"]), axis=1)
            nearest = float(pixel_distances.min())
        else:
            nearest = math.nan
        distances_px.append(nearest)
        for radius in coverage:
            coverage[radius] += int(math.isfinite(nearest) and nearest <= radius)
    d3 = np.asarray(distances_3d)
    dpx = np.asarray(distances_px)
    finite = np.isfinite(dpx)
    return {
        "frames": len(records), "nearest_3d_mean_m": float(d3.mean()),
        "nearest_3d_median_m": float(np.median(d3)),
        **{f"hit_within_{r:g}m": float((d3 <= r).mean()) for r in (0.5, 1.0, 2.0)},
        "nearest_image_mean_px": float(dpx[finite].mean()) if finite.any() else math.nan,
        "nearest_image_median_px": float(np.median(dpx[finite])) if finite.any() else math.nan,
        **{f"image_coverage_{r}px": coverage[r] / len(records) for r in coverage},
        "no_valid_image_projection_rate": float((~finite).mean()),
    }


def transform_payload(rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any]:
    return {
        "rotation_gt_from_radar": rotation.tolist(),
        "translation_gt_from_radar_m": translation.tolist(),
        "rotation_euler_xyz_deg": Rotation.from_matrix(rotation).as_euler("xyz", degrees=True).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "manifests_oracle_left_fixed256_bbox")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/official/v1"))
    parser.add_argument("--camera-config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml")
    parser.add_argument("--camera-calibration", type=Path, default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--calibration-output", type=Path, default=PROJECT_ROOT / "calibration/radar_frame_resolution.json")
    args = parser.parse_args()

    config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    fitted_camera = json.loads(args.camera_calibration.read_text(encoding="utf-8"))
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    camera_rotation = np.asarray(fitted_camera["cameras"]["left"]["rotation_camera_from_gt"])
    camera_translation = np.asarray(fitted_camera["cameras"]["left"]["translation_camera_from_gt_m"])
    sequence_ids = ["Mavic2", "Mavic3", "Avata", "M300", "Pham4"]
    trajectories = {
        sequence: PositionTrajectory.from_directory(args.dataset_root / sequence / "ground_truth")
        for sequence in sequence_ids
    }
    train = load_split(args.manifest_dir / "train.csv", args.dataset_root, trajectories)
    val = load_split(args.manifest_dir / "val.csv", args.dataset_root, trajectories)
    transforms: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {
        "identity": (np.eye(3), np.zeros(3), {"fit": "none"})
    }
    axis_r, axis_t, axis_fit = fit_axis_translation(train)
    transforms["global_axis_translation"] = (axis_r, axis_t, axis_fit)
    rigid_r, rigid_t, rigid_fit = refine_rigid_icp(train, axis_r, axis_t)
    transforms["global_rigid_icp"] = (rigid_r, rigid_t, rigid_fit)

    results: dict[str, Any] = {}
    for name, (rotation, translation, fit_info) in transforms.items():
        results[name] = {
            "transform": transform_payload(rotation, translation), "fit": fit_info,
            "train": evaluate(train, rotation, translation, camera, camera_rotation, camera_translation),
            "val": evaluate(val, rotation, translation, camera, camera_rotation, camera_translation),
            "val_same_sequence_shuffle": evaluate(
                val, rotation, translation, camera, camera_rotation, camera_translation, shuffled=True
            ),
            "val_by_sequence": {
                sequence: evaluate(
                    [record for record in val if record["sequence_id"] == sequence],
                    rotation, translation, camera, camera_rotation, camera_translation,
                )
                for sequence in sequence_ids
            },
        }

    identity_val = results["identity"]["val"]
    global_candidates = ["global_axis_translation", "global_rigid_icp"]
    best_global = min(global_candidates, key=lambda name: results[name]["val"]["nearest_3d_median_m"])
    best_val = results[best_global]["val"]
    median_gain = 1.0 - best_val["nearest_3d_median_m"] / identity_val["nearest_3d_median_m"]
    image_gain = best_val["image_coverage_32px"] - identity_val["image_coverage_32px"]
    sequence_global_checks = {}
    for sequence in sequence_ids:
        identity_sequence = results["identity"]["val_by_sequence"][sequence]
        fitted_sequence = results[best_global]["val_by_sequence"][sequence]
        sequence_gain = 1.0 - (
            fitted_sequence["nearest_3d_median_m"]
            / identity_sequence["nearest_3d_median_m"]
        )
        sequence_global_checks[sequence] = {
            "median_3d_gain": sequence_gain,
            "median_3d_below_2m_or_gain_30pct": bool(
                fitted_sequence["nearest_3d_median_m"] <= 2.0 or sequence_gain >= 0.30
            ),
        }
    global_success = (
        median_gain >= 0.30
        and image_gain >= 0.05
        and all(
            check["median_3d_below_2m_or_gain_30pct"]
            for check in sequence_global_checks.values()
        )
    )

    per_sequence = {}
    if not global_success:
        for sequence in sequence_ids:
            train_sequence = [record for record in train if record["sequence_id"] == sequence]
            val_sequence = [record for record in val if record["sequence_id"] == sequence]
            rotation, translation, fit_info = fit_axis_translation(train_sequence)
            rotation, translation, rigid_info = refine_rigid_icp(
                train_sequence, rotation, translation
            )
            per_sequence[sequence] = {
                "transform": transform_payload(rotation, translation),
                "fit": {"axis": fit_info, "rigid": rigid_info},
                "train": evaluate(train_sequence, rotation, translation, camera, camera_rotation, camera_translation),
                "val": evaluate(val_sequence, rotation, translation, camera, camera_rotation, camera_translation),
                "val_same_sequence_shuffle": evaluate(
                    val_sequence, rotation, translation, camera, camera_rotation, camera_translation, shuffled=True
                ),
            }

    payload = {
        "scope": {"train_frames": len(train), "val_frames": len(val), "test_read": False},
        "already_completed_not_repeated": [
            "GT-to-left-camera 24 proper signed-axis bootstrap",
            "GT-to-left-camera continuous SE(3) and clock-offset fit",
            "held-out official bbox projection evaluation",
            "Radar identity-frame nearest-point vs half-sequence-shift audit",
        ],
        "new_tests": ["global Radar axis/sign permutation + bounded translation",
                      "global continuous rigid ICP refinement",
                      "per-sequence transforms only if global fixed transform fails"],
        "success_rule": "val median 3D nearest-distance gain >=30% and image Coverage@32 gain >=5pp",
        "best_global": best_global, "global_median_3d_gain": median_gain,
        "global_image_coverage32_gain": image_gain, "global_success": global_success,
        "global_sequence_consistency": sequence_global_checks,
        "global_results": results, "per_sequence_results": per_sequence,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"stage4_radar_frame_resolution_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(payload, output_dir / "report.json")
    write_json(payload, args.calibration_output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
