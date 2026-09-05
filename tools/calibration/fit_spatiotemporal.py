#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.spatiotemporal import (  # noqa: E402
    CenterObservation,
    fit_spatiotemporal_calibration,
    reprojection_errors,
)
from rdq_uav.utils.io import write_json  # noqa: E402


def load_observations(path: Path) -> tuple[list[CenterObservation], list[CenterObservation]]:
    groups = {"fit": [], "validation": []}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["visible"].strip().lower() not in {"1", "true", "yes"}:
                continue
            if not row["u"].strip() or not row["v"].strip():
                continue
            split = row.get("calibration_split", "fit")
            if split not in groups:
                raise ValueError(f"Unknown calibration_split={split}")
            groups[split].append(
                CenterObservation(
                    sequence_id=row["sequence_id"],
                    camera=row["camera"],
                    image_time=float(row["image_time"]),
                    u=float(row["u"]),
                    v=float(row["v"]),
                    weight=float(row.get("confidence", 1.0)),
                )
            )
    return groups["fit"], groups["validation"]


def error_summary(errors: np.ndarray) -> dict:
    valid = np.isfinite(errors).all(axis=1)
    norms = np.linalg.norm(errors[valid], axis=1)
    if not len(norms):
        return {"count": 0}
    return {
        "count": int(len(norms)),
        "mean_px": float(norms.mean()),
        "median_px": float(np.median(norms)),
        "p95_px": float(np.percentile(norms, 95)),
        "max_px": float(norms.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit GT-to-fisheye extrinsics and time offset")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument("--initial", type=Path, required=True, help="JSON with initial camera transforms")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "calibration/fitted_calibration.json")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    initial = json.loads(args.initial.read_text(encoding="utf-8"))
    all_cameras = {
        name: OmniRadtanCamera.from_config(camera_config)
        for name, camera_config in config["cameras"].items()
    }
    fit_observations, validation_observations = load_observations(args.annotations)
    observed_cameras = {obs.camera for obs in fit_observations + validation_observations}
    cameras = {name: camera for name, camera in all_cameras.items() if name in observed_cameras}
    if not cameras:
        raise ValueError("No usable camera observations were loaded")
    unknown_cameras = observed_cameras - set(all_cameras)
    if unknown_cameras:
        raise ValueError(f"Annotations reference cameras absent from config: {unknown_cameras}")
    sequence_ids = {obs.sequence_id for obs in fit_observations + validation_observations}
    trajectories = {
        sequence_id: PositionTrajectory.from_directory(
            Path(config["dataset_root"]) / sequence_id / "ground_truth"
        )
        for sequence_id in sequence_ids
    }
    rotations = {
        name: np.asarray(initial["cameras"][name]["rotation_camera_from_gt"], dtype=np.float64)
        for name in cameras
    }
    translations = {
        name: np.asarray(initial["cameras"][name]["translation_camera_from_gt_m"], dtype=np.float64)
        for name in cameras
    }
    solution = fit_spatiotemporal_calibration(
        fit_observations,
        trajectories,
        cameras,
        rotations,
        translations,
        initial_time_offset_s=float(initial.get("time_offset_s", 0.0)),
        max_time_offset_s=float(config["time"]["max_abs_offset_s"]),
    )
    payload = solution.summary()
    payload["fit"] = error_summary(
        reprojection_errors(
            fit_observations,
            trajectories,
            cameras,
            solution.rotation_camera_from_gt,
            solution.translation_camera_from_gt,
            solution.time_offset_s,
        )
    )
    payload["validation"] = error_summary(
        reprojection_errors(
            validation_observations,
            trajectories,
            cameras,
            solution.rotation_camera_from_gt,
            solution.translation_camera_from_gt,
            solution.time_offset_s,
        )
    )
    payload["source_config"] = str(args.config.resolve())
    payload["source_annotations"] = str(args.annotations.resolve())
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
