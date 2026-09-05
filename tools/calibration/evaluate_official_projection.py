#!/usr/bin/env python3
"""Evaluate fitted GT-to-left-fisheye calibration on official 2D boxes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.spatiotemporal import CenterObservation, reprojection_errors  # noqa: E402
from rdq_uav.calibration.roi import clamped_square_roi, sphere_projection_extent  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402


def summarize(
    rows: list[dict[str, str]], errors: np.ndarray, roi_contains_bbox: np.ndarray
) -> dict:
    valid = np.isfinite(errors).all(axis=1)
    norms = np.linalg.norm(errors[valid], axis=1)
    if not len(norms):
        return {"count": 0, "invalid_projection_count": int((~valid).sum())}
    valid_rows = [row for row, keep in zip(rows, valid) if keep]
    half_width = np.asarray([float(row["bbox_width_px"]) / 2.0 for row in valid_rows])
    half_height = np.asarray([float(row["bbox_height_px"]) / 2.0 for row in valid_rows])
    valid_errors = errors[valid]
    inside = (np.abs(valid_errors[:, 0]) <= half_width) & (
        np.abs(valid_errors[:, 1]) <= half_height
    )
    return {
        "count": int(len(norms)),
        "invalid_projection_count": int((~valid).sum()),
        "mean_px": float(norms.mean()),
        "median_px": float(np.median(norms)),
        "p95_px": float(np.percentile(norms, 95)),
        "max_px": float(norms.max()),
        "inside_bbox_rate": float(inside.mean()),
        "within_5px_rate": float((norms <= 5.0).mean()),
        "within_10px_rate": float((norms <= 10.0).mean()),
        "official_bbox_fully_inside_oracle_roi_rate": float(roi_contains_bbox.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official MMAUD 2D reprojection")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_projection_evaluation.json",
    )
    parser.add_argument("--max-manifest-dt", type=float, default=0.12)
    parser.add_argument("--shared-radius-m", type=float, default=1.0)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--min-side-px", type=int, default=128)
    parser.add_argument("--max-side-px", type=int, default=512)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fitted = json.loads(args.calibration.read_text(encoding="utf-8"))
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    rotation = np.asarray(
        fitted["cameras"]["left"]["rotation_camera_from_gt"], dtype=np.float64
    )
    translation = np.asarray(
        fitted["cameras"]["left"]["translation_camera_from_gt_m"], dtype=np.float64
    )
    offset = float(fitted["time_offset_s"])

    with args.mapping.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in source_rows
        if row["match_status"] == "exact"
        and row["experiment_split"] in {"train", "val", "test"}
        and abs(float(row["dt_to_manifest_s"])) <= args.max_manifest_dt
    ]
    sequence_ids = sorted({row["class_name"] for row in rows})
    trajectories = {
        sequence_id: PositionTrajectory.from_directory(
            Path(config["dataset_root"]) / sequence_id / "ground_truth"
        )
        for sequence_id in sequence_ids
    }
    observations = [
        CenterObservation(
            sequence_id=row["class_name"],
            camera="left",
            image_time=float(row["image_time"]),
            u=float(row["u"]),
            v=float(row["v"]),
        )
        for row in rows
    ]
    errors = reprojection_errors(
        observations,
        trajectories,
        {"left": camera},
        {"left": rotation},
        {"left": translation},
        offset,
    )
    roi_contains_bbox = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        point_gt, valid_time = trajectories[row["class_name"]].evaluate(
            float(row["image_time"]) + offset
        )
        if not bool(valid_time):
            continue
        try:
            center_uv, extent_wh = sphere_projection_extent(
                point_gt, args.shared_radius_m, camera, rotation, translation
            )
        except ValueError:
            continue
        if not bool(camera.in_image(center_uv)):
            continue
        x1, y1, x2, y2 = clamped_square_roi(
            center_uv,
            extent_wh,
            camera.width,
            camera.height,
            args.context_scale,
            args.min_side_px,
            args.max_side_px,
        )
        u, v = float(row["u"]), float(row["v"])
        half_width = float(row["bbox_width_px"]) / 2.0
        half_height = float(row["bbox_height_px"]) / 2.0
        roi_contains_bbox[index] = (
            x1 <= u - half_width
            and x2 >= u + half_width
            and y1 <= v - half_height
            and y2 >= v + half_height
        )

    payload = {
        "time_offset_s": offset,
        "selection": {
            "exact_pixel_match": True,
            "max_abs_dt_to_manifest_s": args.max_manifest_dt,
            "count": len(rows),
        },
        "oracle_roi": {
            "shared_radius_m": args.shared_radius_m,
            "context_scale": args.context_scale,
            "min_side_px": args.min_side_px,
            "max_side_px": args.max_side_px,
            "class_dependent_roi_size": False,
        },
        "overall": summarize(rows, errors, roi_contains_bbox),
        "by_experiment_split": {},
        "by_class": {},
        "source_mapping": str(args.mapping.resolve()),
        "source_calibration": str(args.calibration.resolve()),
    }
    for field, destination in (
        ("experiment_split", payload["by_experiment_split"]),
        ("class_name", payload["by_class"]),
    ):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[row[field]].append(index)
        for name, indices_list in sorted(grouped.items()):
            indices = np.asarray(indices_list, dtype=int)
            destination[name] = summarize(
                [rows[index] for index in indices], errors[indices], roi_contains_bbox[indices]
            )

    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
