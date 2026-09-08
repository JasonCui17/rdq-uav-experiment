#!/usr/bin/env python3
"""Stage 4.11: audit confirmed Radar target-candidate rules on train/val.

The released MMAUD V1 ``radar_enhance_pcl`` arrays contain XYZ only.  This
tool therefore never invents Power, Doppler, SNR, cluster, or track fields.
The GT-range rule is optional and is reported as an oracle, not as a
deployable input rule.
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
from typing import Any, Callable

import numpy as np
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402

RADII_PX = (8, 16, 32, 64)


def load_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    if expected_split not in {"train", "val"}:
        raise ValueError("Stage 4.11 permits train and val only")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"Unexpected or empty split: {path}")
    return rows


def load_xyz(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    array = np.asarray(np.load(path, allow_pickle=False))
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"Released V1 Radar must be exactly (N,3) XYZ; got {array.shape}: {path}"
        )
    points = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], {
        "released_columns": int(points.shape[1]),
        "input_points": int(len(points)),
        "nonfinite_points": int((~finite).sum()),
    }


def candidate_rules(
    max_range_m: float,
    gt_range_gate_m: float | None,
) -> dict[str, Callable[[np.ndarray, np.ndarray | None], np.ndarray]]:
    """Return only rules supported by fields in released V1 XYZ arrays."""
    rules: dict[str, Callable[[np.ndarray, np.ndarray | None], np.ndarray]] = {
        "released_xyz_all": lambda points, _target_radar: np.ones(len(points), dtype=bool),
        "finite_range_le_50m": lambda points, _target_radar: (
            np.linalg.norm(points, axis=1) <= max_range_m
        ),
    }
    if gt_range_gate_m is not None:
        def gt_range_gate(points: np.ndarray, target_radar: np.ndarray | None) -> np.ndarray:
            if target_radar is None:
                return np.zeros(len(points), dtype=bool)
            expected_range = float(np.linalg.norm(target_radar))
            return np.abs(np.linalg.norm(points, axis=1) - expected_range) <= gt_range_gate_m

        rules["oracle_gt_range_gate"] = gt_range_gate
    return rules


def deterministic_shuffle_indices(rows: list[dict[str, str]]) -> list[int]:
    """Half-cycle permutation within each sequence and split."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["sequence_id"]].append(index)
    result = list(range(len(rows)))
    for indices in groups.values():
        offset = max(1, len(indices) // 2) if len(indices) > 1 else 0
        for local_index, index in enumerate(indices):
            result[index] = indices[(local_index + offset) % len(indices)]
    return result


def bbox_center(row: dict[str, str]) -> np.ndarray:
    x1, y1, x2, y2 = (
        float(row[f"official_bbox_{key}"]) for key in ("x1", "y1", "x2", "y2")
    )
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def score_candidates(
    candidates_radar: np.ndarray,
    target_gt: np.ndarray,
    target_pixel: np.ndarray,
    rotation_gt_from_radar: np.ndarray,
    translation_gt_from_radar: np.ndarray,
    camera: OmniRadtanCamera,
    rotation_camera_from_gt: np.ndarray,
    translation_camera_from_gt: np.ndarray,
) -> dict[str, Any]:
    candidate_count = len(candidates_radar)
    if candidate_count:
        candidates_gt = transform_points(
            candidates_radar, rotation_gt_from_radar, translation_gt_from_radar
        )
        nearest_3d = float(np.linalg.norm(candidates_gt - target_gt[None, :], axis=1).min())
        pixels, valid = camera.project(
            transform_points(
                candidates_gt, rotation_camera_from_gt, translation_camera_from_gt
            ),
            require_in_image=True,
        )
        pixels = pixels[valid]
    else:
        nearest_3d = math.nan
        pixels = np.empty((0, 2), dtype=np.float64)

    if len(pixels):
        distances = np.linalg.norm(pixels - target_pixel[None, :], axis=1)
        nearest_2d = float(distances.min())
    else:
        distances = np.empty(0, dtype=np.float64)
        nearest_2d = math.nan

    result: dict[str, Any] = {
        "candidate_count": candidate_count,
        "candidate_nonempty": int(candidate_count > 0),
        "valid_projected_candidate_count": int(len(pixels)),
        "nearest_candidate_gt_3d_m": nearest_3d,
        "nearest_candidate_gt_2d_px": nearest_2d,
    }
    for radius in RADII_PX:
        result[f"coverage_{radius}px"] = int(bool(len(distances)) and np.any(distances <= radius))
    return result


def summarize(rows: list[dict[str, Any]], split: str, candidate: str, pairing: str) -> dict[str, Any]:
    d3 = np.asarray([row["nearest_candidate_gt_3d_m"] for row in rows], dtype=np.float64)
    d2 = np.asarray([row["nearest_candidate_gt_2d_px"] for row in rows], dtype=np.float64)
    finite3, finite2 = np.isfinite(d3), np.isfinite(d2)
    output: dict[str, Any] = {
        "split": split,
        "candidate_rule": candidate,
        "pairing": pairing,
        "frames": len(rows),
        "candidate_nonempty_rate": float(np.mean([row["candidate_nonempty"] for row in rows])),
        "candidate_points_mean": float(np.mean([row["candidate_count"] for row in rows])),
        "candidate_points_median": float(np.median([row["candidate_count"] for row in rows])),
        "valid_projected_points_mean": float(
            np.mean([row["valid_projected_candidate_count"] for row in rows])
        ),
        "nearest_3d_mean_m": float(d3[finite3].mean()) if finite3.any() else math.nan,
        "nearest_3d_median_m": float(np.median(d3[finite3])) if finite3.any() else math.nan,
        "nearest_2d_mean_px": float(d2[finite2].mean()) if finite2.any() else math.nan,
        "nearest_2d_median_px": float(np.median(d2[finite2])) if finite2.any() else math.nan,
    }
    for radius in RADII_PX:
        output[f"coverage_{radius}px"] = float(
            np.mean([row[f"coverage_{radius}px"] for row in rows])
        )
    return output


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


def load_transform(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = payload["global_results"]["global_rigid_icp"]["transform"]
    rotation = np.asarray(transform["rotation_gt_from_radar"], dtype=np.float64)
    translation = np.asarray(transform["translation_gt_from_radar_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Invalid Radar->GT transform in {path}")
    return rotation, translation, transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=PROJECT_ROOT / "manifests_oracle_left_fixed256_bbox",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/v1"))
    parser.add_argument(
        "--radar-transform", type=Path,
        default=PROJECT_ROOT / "calibration/radar_frame_resolution.json",
    )
    parser.add_argument(
        "--camera-config", type=Path,
        default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml",
    )
    parser.add_argument(
        "--camera-calibration", type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument("--max-range-m", type=float, default=50.0)
    parser.add_argument("--with-gt-range-oracle", action="store_true")
    parser.add_argument("--gt-range-gate-m", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    radar_rotation, radar_translation, transform_payload = load_transform(args.radar_transform)
    camera_config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    fitted_camera = json.loads(args.camera_calibration.read_text(encoding="utf-8"))
    camera = OmniRadtanCamera.from_config(camera_config["cameras"]["left"])
    camera_fit = fitted_camera["cameras"]["left"]
    camera_rotation = np.asarray(camera_fit["rotation_camera_from_gt"], dtype=np.float64)
    camera_translation = np.asarray(camera_fit["translation_camera_from_gt_m"], dtype=np.float64)

    sequence_ids = ["Mavic2", "Mavic3", "Avata", "M300", "Pham4"]
    trajectories = {
        sequence: PositionTrajectory.from_directory(args.dataset_root / sequence / "ground_truth")
        for sequence in sequence_ids
    }
    rules = candidate_rules(
        args.max_range_m,
        args.gt_range_gate_m if args.with_gt_range_oracle else None,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"stage4_target_association_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    frame_results: list[dict[str, Any]] = []

    for split in ("train", "val"):
        rows = load_rows(args.manifest_dir / f"{split}.csv", split)
        shuffle_indices = deterministic_shuffle_indices(rows)
        started = time.perf_counter()
        nonempty = 0
        running_3d: list[float] = []
        bar = tqdm(
            enumerate(rows), total=len(rows), desc=split, unit="frame",
            bar_format="{desc} | {n_fmt}/{total_fmt} | {postfix} | {rate_fmt} | ETA {remaining}",
        )
        for index, row in bar:
            radar_path = Path(row["radar_path"])
            if not radar_path.is_absolute():
                radar_path = args.dataset_root / radar_path
            points, input_info = load_xyz(radar_path)
            target_gt, target_valid = trajectories[row["sequence_id"]].evaluate(float(row["radar_time"]))
            if not bool(target_valid):
                raise ValueError(f"GT trajectory unavailable at Radar time: {row['sample_id']}")
            target_gt = np.asarray(target_gt, dtype=np.float64)
            # Convert current GT target into the released Radar frame. This is
            # used only by the explicitly named oracle range rule.
            target_radar = (target_gt - radar_translation) @ radar_rotation

            shuffled_row = rows[shuffle_indices[index]]
            shuffled_gt, shuffled_valid = trajectories[shuffled_row["sequence_id"]].evaluate(
                float(shuffled_row["radar_time"])
            )
            if not bool(shuffled_valid):
                raise ValueError(f"Shuffled GT unavailable: {shuffled_row['sample_id']}")

            for rule_name, rule in rules.items():
                mask = rule(points, target_radar)
                candidates = points[mask]
                common = {
                    "sample_id": row["sample_id"], "split": split,
                    "sequence_id": row["sequence_id"], "candidate_rule": rule_name,
                    "rule_is_oracle": int(rule_name.startswith("oracle_")),
                    "radar_time": float(row["radar_time"]),
                    "image_radar_gap_ms": abs(float(row["image_time"]) - float(row["radar_time"])) * 1000.0,
                    **input_info,
                }
                real_score = score_candidates(
                    candidates, target_gt, bbox_center(row), radar_rotation, radar_translation,
                    camera, camera_rotation, camera_translation,
                )
                frame_results.append({
                    **common, "pairing": "real", "target_sample_id": row["sample_id"],
                    **real_score,
                })
                frame_results.append({
                    **common, "pairing": "shuffle_same_sequence",
                    "target_sample_id": shuffled_row["sample_id"],
                    **score_candidates(
                        candidates, np.asarray(shuffled_gt), bbox_center(shuffled_row),
                        radar_rotation, radar_translation, camera, camera_rotation,
                        camera_translation,
                    ),
                })
                if rule_name == "released_xyz_all":
                    nonempty += real_score["candidate_nonempty"]
                    if math.isfinite(real_score["nearest_candidate_gt_3d_m"]):
                        running_3d.append(real_score["nearest_candidate_gt_3d_m"])
            processed = index + 1
            bar.set_postfix_str(
                f"raw_nonempty={nonempty/processed:.3f} | "
                f"median3d={np.median(running_3d) if running_3d else math.nan:.2f}m | "
                f"samples/s={processed/max(time.perf_counter()-started, 1e-9):.1f}",
                refresh=False,
            )
        bar.close()

    summary_rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for rule_name in rules:
            for pairing in ("real", "shuffle_same_sequence"):
                selected = [
                    row for row in frame_results
                    if row["split"] == split and row["candidate_rule"] == rule_name
                    and row["pairing"] == pairing
                ]
                summary_rows.append(summarize(selected, split, rule_name, pairing))

    write_csv(frame_results, output_dir / "association_predictions.csv")
    write_csv(summary_rows, output_dir / "association_summary.csv")
    report = {
        "stage": "4.11 Radar Target Association Audit",
        "scope": {"splits_read": ["train", "val"], "test_read": False},
        "radar_input_schema": {
            "released_v1_shape": "(N,3)", "available_fields": ["x", "y", "z"],
            "derived_confirmed_field": "range=sqrt(x^2+y^2+z^2)",
            "unavailable_not_invented": [
                "Power", "Doppler", "Alpha", "Beta", "SNR", "cluster_id",
                "track_id", "VeloX", "VeloY", "VeloZ", "Class", "Confidence",
            ],
        },
        "candidate_rules": {
            "released_xyz_all": "all finite released enhanced-cloud XYZ points",
            "finite_range_le_50m": (
                f"finite XYZ with derived Euclidean range <= {args.max_range_m:g} m; "
                "the cutoff is an existing project preprocessing threshold, not an official label"
            ),
            **({
                "oracle_gt_range_gate": (
                    f"abs(point_range - GT-derived sensor range) <= {args.gt_range_gate_m:g} m; "
                    "GT-dependent oracle, never a deployable candidate rule"
                )
            } if args.with_gt_range_oracle else {}),
        },
        "transform": {
            "source": str(args.radar_transform.resolve()),
            "selected_entry": "global_results.global_rigid_icp.transform",
            **transform_payload,
        },
        "null_control": "deterministic same-sequence half-cycle target permutation",
        "summary": summary_rows,
        "outputs": {
            "predictions": "association_predictions.csv",
            "summary": "association_summary.csv",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
