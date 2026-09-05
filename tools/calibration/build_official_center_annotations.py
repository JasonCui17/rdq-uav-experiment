#!/usr/bin/env python3
"""Convert exact official 2D mappings into leakage-safe calibration observations."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build left-fisheye center observations")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_center_annotations.csv",
    )
    parser.add_argument("--max-manifest-dt", type=float, default=0.12)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force intentionally")
    with args.mapping.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    eligible: dict[str, list[dict[str, str]]] = defaultdict(list)
    excluded = Counter()
    for row in rows:
        if row["match_status"] != "exact":
            excluded["not_exact"] += 1
            continue
        if not row["experiment_split"]:
            excluded["no_experiment_split"] += 1
            continue
        if abs(float(row["dt_to_manifest_s"])) > args.max_manifest_dt:
            excluded["manifest_dt_too_large"] += 1
            continue
        if row["experiment_split"] != "train":
            excluded[f'heldout_{row["experiment_split"]}'] += 1
            continue
        eligible[row["class_name"]].append(row)

    output_rows = []
    for class_name, class_rows in sorted(eligible.items()):
        class_rows.sort(key=lambda row: float(row["image_time"]))
        for index, row in enumerate(class_rows):
            calibration_split = "validation" if index % args.validation_every == 0 else "fit"
            output_rows.append(
                {
                    "sample_id": row["nearest_sample_id"],
                    "sequence_id": class_name,
                    "class_name": class_name,
                    "image_path": row["image_path"],
                    "image_time": row["image_time"],
                    "camera": "left",
                    "u": row["u"],
                    "v": row["v"],
                    "visible": "1",
                    "confidence": "1.0",
                    "source": "MMAUD_2D_exact_pixel_match",
                    "calibration_split": calibration_split,
                    "bbox_width_px": row["bbox_width_px"],
                    "bbox_height_px": row["bbox_height_px"],
                    "dt_to_manifest_s": row["dt_to_manifest_s"],
                }
            )

    if not output_rows:
        raise RuntimeError(f"No eligible training observations; excluded={dict(excluded)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} observations to {args.output.resolve()}")
    print("calibration_split", dict(Counter(row["calibration_split"] for row in output_rows)))
    print("class", dict(Counter(row["class_name"] for row in output_rows)))
    print("excluded", dict(excluded))


if __name__ == "__main__":
    main()
