#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.spatiotemporal import CenterObservation, fit_spatiotemporal_calibration  # noqa: E402
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


def load_visible(path: Path) -> list[CenterObservation]:
    observations = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["visible"].strip().lower() not in {"1", "true", "yes"}:
                continue
            if row.get("calibration_split", "fit") != "fit" or not row["u"] or not row["v"]:
                continue
            observations.append(
                CenterObservation(
                    row["sequence_id"], row["camera"], float(row["image_time"]),
                    float(row["u"]), float(row["v"]), float(row.get("confidence", 1.0))
                )
            )
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap extrinsics from 24 axis-aligned rotations")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "calibration/initial_extrinsics.json")
    parser.add_argument(
        "--max-observations-per-camera",
        type=int,
        default=300,
        help="Uniformly subsample only for the 24-way bootstrap; final fitting uses all rows",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    observations = load_visible(args.annotations)
    sequence_ids = {observation.sequence_id for observation in observations}
    trajectories = {
        name: PositionTrajectory.from_directory(Path(config["dataset_root"]) / name / "ground_truth")
        for name in sequence_ids
    }
    selected = {}
    offsets = []
    observed_camera_names = sorted({observation.camera for observation in observations})
    for camera_name in observed_camera_names:
        if camera_name not in config["cameras"]:
            raise ValueError(f"Camera {camera_name!r} is absent from the calibration config")
        camera_config = config["cameras"][camera_name]
        camera_observations = [obs for obs in observations if obs.camera == camera_name]
        if len(camera_observations) > args.max_observations_per_camera:
            selection = np.linspace(
                0, len(camera_observations) - 1, args.max_observations_per_camera, dtype=int
            )
            camera_observations = [camera_observations[index] for index in selection]
        print(camera_name, "bootstrap_observations", len(camera_observations))
        camera = OmniRadtanCamera.from_config(camera_config)
        candidates = []
        for rotation in proper_axis_rotations():
            solution = fit_spatiotemporal_calibration(
                camera_observations,
                trajectories,
                {camera_name: camera},
                {camera_name: rotation},
                {camera_name: np.zeros(3)},
                max_time_offset_s=float(config["time"]["max_abs_offset_s"]),
            )
            errors = np.linalg.norm(solution.residuals_px.reshape(-1, 2), axis=1)
            candidates.append((float(np.median(errors)), solution))
        score, best = min(candidates, key=lambda item: item[0])
        selected[camera_name] = {
            "rotation_camera_from_gt": best.rotation_camera_from_gt[camera_name].tolist(),
            "translation_camera_from_gt_m": best.translation_camera_from_gt[camera_name].tolist(),
            "bootstrap_median_error_px": score,
        }
        offsets.append(best.time_offset_s)
        print(camera_name, "median_px", score, "offset_s", best.time_offset_s)
    payload = {
        "time_offset_s": float(np.median(offsets)),
        "cameras": selected,
        "warning": "Bootstrap only; run fit_spatiotemporal.py and validate on held-out clicks.",
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
