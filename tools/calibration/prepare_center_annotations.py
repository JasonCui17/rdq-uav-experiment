#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def uniformly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frames for UAV center annotation")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/train.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "calibration/center_annotations.csv")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force intentionally")

    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[row["class_name"]].append(row)

    output_rows = []
    for class_name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["image_time"]))
        selected = uniformly_spaced(rows, args.per_class)
        for sample_index, row in enumerate(selected):
            annotation_split = "validation" if sample_index % 5 == 0 else "fit"
            for camera in ("left", "right"):
                output_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "sequence_id": row["sequence_id"],
                        "class_name": class_name,
                        "image_path": row["image_path"],
                        "image_time": row["image_time"],
                        "camera": camera,
                        "u": "",
                        "v": "",
                        "visible": "",
                        "confidence": "1.0",
                        "source": "manual_or_official_2d",
                        "calibration_split": annotation_split,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} camera rows to {args.output.resolve()}")
    print("Fill u/v in each 1280x960 camera half; visible=1 for usable centers, 0 otherwise.")


if __name__ == "__main__":
    main()
