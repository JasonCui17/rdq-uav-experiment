#!/usr/bin/env python3
"""Project time-compensated GT centers and build left-camera Oracle ROI manifests."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402
from rdq_uav.calibration.roi import (  # noqa: E402
    centered_square_roi,
    clamped_square_roi,
    sphere_projection_extent,
)
from rdq_uav.data.manifest import compute_radar_stats  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe left Oracle ROI manifests")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manifests_oracle_left")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
    )
    parser.add_argument("--shared-radius-m", type=float, default=1.0)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--min-side-px", type=int, default=128)
    parser.add_argument("--max-side-px", type=int, default=512)
    parser.add_argument(
        "--roi-mode", choices=("physical_shared", "fixed_pixel"), default="physical_shared"
    )
    parser.add_argument("--fixed-side-px", type=int, default=256)
    parser.add_argument(
        "--boundary-policy",
        choices=("shift", "drop"),
        default="shift",
        help="drop keeps the target exactly centered; shift retains edge samples",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}; pass --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fitted = json.loads(args.calibration.read_text(encoding="utf-8"))
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    camera_fit = fitted["cameras"]["left"]
    rotation = np.asarray(camera_fit["rotation_camera_from_gt"], dtype=np.float64)
    translation = np.asarray(camera_fit["translation_camera_from_gt_m"], dtype=np.float64)
    offset = float(fitted["time_offset_s"])
    trajectories = {
        class_name: PositionTrajectory.from_directory(
            Path(config["dataset_root"]) / class_name / "ground_truth"
        )
        for class_name in ("Mavic2", "Mavic3", "Avata", "M300", "Pham4")
    }

    summary: dict[str, object] = {
        "method": "GT-projected Oracle ROI; not an end-to-end detector",
        "camera": "left",
        "shared_radius_m": args.shared_radius_m,
        "roi_mode": args.roi_mode,
        "fixed_side_px": args.fixed_side_px if args.roi_mode == "fixed_pixel" else None,
        "boundary_policy": args.boundary_policy,
        "context_scale": args.context_scale,
        "min_side_px": args.min_side_px,
        "max_side_px": args.max_side_px,
        "time_offset_s": offset,
        "class_dependent_roi_size": False,
        "splits": {},
    }
    all_kept_rows = []
    for split in ("train", "val", "test"):
        with (args.source_dir / f"{split}.csv").open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        kept = []
        dropped = Counter()
        sides = []
        by_class = defaultdict(lambda: {"source": 0, "kept": 0})
        for row in rows:
            class_name = row["class_name"]
            by_class[class_name]["source"] += 1
            point_gt, valid_time = trajectories[class_name].evaluate(
                float(row["image_time"]) + offset
            )
            if not bool(valid_time):
                dropped["outside_gt_time_range"] += 1
                continue
            center_uv, center_valid = camera.project(
                transform_points(point_gt, rotation, translation), require_in_image=True
            )
            if not bool(center_valid):
                dropped["center_outside_left_image"] += 1
                continue
            if args.roi_mode == "fixed_pixel":
                extent = np.asarray([args.fixed_side_px, args.fixed_side_px], dtype=float)
                side = args.fixed_side_px
            else:
                try:
                    projected_center, extent = sphere_projection_extent(
                        point_gt,
                        args.shared_radius_m,
                        camera,
                        rotation,
                        translation,
                    )
                except ValueError:
                    dropped["invalid_sphere_projection"] += 1
                    continue
                if np.linalg.norm(projected_center - center_uv) > 1e-6:
                    raise RuntimeError("Inconsistent center projection")
                side = int(
                    np.ceil(
                        np.clip(
                            float(np.max(extent)) * args.context_scale,
                            args.min_side_px,
                            args.max_side_px,
                        )
                    )
                )
            if args.boundary_policy == "drop":
                roi = centered_square_roi(
                    center_uv, side, camera.width, camera.height
                )
                if roi is None:
                    dropped["centered_roi_crosses_image_boundary"] += 1
                    continue
                x1, y1, x2, y2 = roi
            else:
                x1, y1, x2, y2 = clamped_square_roi(
                    center_uv,
                    extent,
                    camera.width,
                    camera.height,
                    args.context_scale if args.roi_mode == "physical_shared" else 1.0,
                    args.min_side_px if args.roi_mode == "physical_shared" else side,
                    args.max_side_px if args.roi_mode == "physical_shared" else side,
                )
            enriched = dict(row)
            enriched.update(
                {
                    "roi_camera": "left",
                    "roi_center_u": float(center_uv[0]),
                    "roi_center_v": float(center_uv[1]),
                    "roi_x1": x1,
                    "roi_y1": y1,
                    "roi_x2": x2,
                    "roi_y2": y2,
                    "roi_side_px": x2 - x1,
                    "roi_shared_radius_m": args.shared_radius_m,
                    "roi_context_scale": args.context_scale,
                    "roi_oracle": "1",
                    "roi_mode": args.roi_mode,
                    "roi_boundary_policy": args.boundary_policy,
                }
            )
            kept.append(enriched)
            sides.append(x2 - x1)
            by_class[class_name]["kept"] += 1
        if not kept:
            raise RuntimeError(f"No valid rows for {split}")
        with (args.output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kept[0]))
            writer.writeheader()
            writer.writerows(kept)
        all_kept_rows.extend(kept)
        side_array = np.asarray(sides)
        summary["splits"][split] = {
            "source_count": len(rows),
            "kept_count": len(kept),
            "dropped": dict(dropped),
            "by_class": dict(by_class),
            "roi_side_px_quantiles": {
                "min": int(side_array.min()),
                "p50": float(np.quantile(side_array, 0.5)),
                "p95": float(np.quantile(side_array, 0.95)),
                "max": int(side_array.max()),
            },
        }
    source_radar_stats = json.loads(
        (args.source_dir / "radar_stats.json").read_text(encoding="utf-8")
    )
    radar_stats = compute_radar_stats(
        all_kept_rows,
        Path(config["dataset_root"]),
        max_range_m=float(source_radar_stats["max_range_m"]),
    )
    write_json(radar_stats, args.output_dir / "radar_stats.json")
    summary["radar_stats"] = radar_stats
    for filename in ("class_mapping.json",):
        source = args.source_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)
    write_json(summary, args.output_dir / "oracle_roi_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote Oracle ROI manifests to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
