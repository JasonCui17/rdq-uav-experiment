#!/usr/bin/env python3
"""Select the earliest 20-frame Mavic2 unit with strong official bbox coverage."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.run_mmuav_candidate_baseline import (  # noqa: E402
    build_processing_units,
    build_unique_sensor_frame_index,
    read_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/official/v1"),
    )
    parser.add_argument(
        "--official-2d-mapping", type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/mmuav_bbox_unit_selection/unit_selection_audit.json",
    )
    parser.add_argument("--max-bbox-time-gap-ms", type=float, default=150.0)
    return parser.parse_args()


def image_timestamp_index(directory: Path) -> tuple[np.ndarray, list[str]]:
    entries = sorted(
        (float(path.stem), path.name) for path in directory.glob("*.png")
    )
    return np.asarray([item[0] for item in entries]), [item[1] for item in entries]


def nearest_image_name(times: np.ndarray, names: list[str], query: float) -> str:
    insertion = int(np.searchsorted(times, query))
    candidates = [max(0, insertion - 1), min(len(times) - 1, insertion)]
    index = min(candidates, key=lambda item: (abs(times[item] - query), times[item]))
    return names[index]


def official_mavic2_images(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        entries = sorted(
            (float(row["image_time"]), Path(row["image_path"]).name)
            for row in csv.DictReader(handle)
            if row.get("class_name") == "Mavic2" and row.get("match_status") == "exact"
        )
    return np.asarray([item[0] for item in entries]), [item[1] for item in entries]


def main() -> None:
    args = parse_args()
    rows, manifests = read_manifests(args.manifest_dir, ["train"])
    index, index_audit, ranges = build_unique_sensor_frame_index(
        rows, args.dataset_root, ["Mavic2"]
    )
    units, unit_audit = build_processing_units(index, ranges, 20)
    image_timestamp_index(args.dataset_root / "Mavic2/image")  # validates raw image source
    official_times, official_names = official_mavic2_images(args.official_2d_mapping)
    records = []
    for unit in units:
        mid360 = unit.sensor_frames["lidar_360"]
        gaps_ms = []
        for frame in mid360:
            image_name = nearest_image_name(
                official_times, official_names, float(frame.timestamp)
            )
            image_time = float(Path(image_name).stem)
            gap_ms = abs(image_time - float(frame.timestamp)) * 1000
            if gap_ms <= args.max_bbox_time_gap_ms:
                gaps_ms.append(gap_ms)
        records.append({
            "unit_name": unit.qualified_name,
            "t_start": unit.time_start,
            "t_end": unit.time_end,
            "mid360_frames": len(mid360),
            "official_bbox_matches": len(gaps_ms),
            "bbox_coverage_ratio": len(gaps_ms) / len(mid360) if mid360 else 0.0,
            "matched_bbox_gap_ms_median": None if not gaps_ms else float(np.median(gaps_ms)),
            "matched_bbox_gap_ms_max": None if not gaps_ms else float(np.max(gaps_ms)),
        })
    eligible = [
        record for record in records
        if record["mid360_frames"] == 20 and record["official_bbox_matches"] >= 15
    ]
    if not eligible:
        raise RuntimeError("No exact 20-frame unit has at least 15 official bbox matches")
    # Prefer complete coverage; use earliest time only as a deterministic tie-break.
    selected = min(
        eligible,
        key=lambda record: (-record["bbox_coverage_ratio"], record["t_start"], record["unit_name"]),
    )
    payload = {
        "mode": "dry_selector_no_dbscan",
        "selection_inputs": "timestamps and official bbox availability only",
        "max_bbox_time_gap_ms": args.max_bbox_time_gap_ms,
        "gt_coordinates_used": False,
        "manifests": [str(path.resolve()) for path in manifests],
        "processing_policy": unit_audit["strategy"],
        "source_index_policy": index_audit["selection_policy"],
        "units": records,
        "selected_unit": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for record in records:
        print(
            f"{record['unit_name']} | [{record['t_start']:.6f},{record['t_end']:.6f}] | "
            f"mid360={record['mid360_frames']} | bbox={record['official_bbox_matches']} | "
            f"coverage={record['bbox_coverage_ratio']:.3f}"
        )
    print(f"selected_unit = {selected['unit_name']}")
    print(f"unit_selection_audit = {args.output.resolve()}")


if __name__ == "__main__":
    main()
