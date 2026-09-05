#!/usr/bin/env python3
"""Attach exact MMAUD 2D boxes to the fixed-ROI manifests.

The output contains the intersection of samples available to all bbox
counterfactual modes, so Full/Erase/Foreground-only comparisons cannot differ
because of sample selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.data.manifest import compute_radar_stats  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402


def load_exact_left_boxes(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    boxes: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["camera"] != "left" or row["match_status"] != "exact":
                continue
            key = (row["class_name"], Path(row["image_path"]).name)
            if key in boxes:
                raise ValueError(f"Duplicate exact official bbox mapping: {key}")
            boxes[key] = row
    if not boxes:
        raise ValueError(f"No exact left-camera boxes found in {path}")
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a shared fixed-ROI subset with exact official 2D boxes"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "manifests_oracle_left_fixed256",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "manifests_oracle_left_fixed256_bbox",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/v1"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}; pass --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official = load_exact_left_boxes(args.mapping)
    summary: dict[str, object] = {
        "method": "exact official MMAUD 2D bbox matched to fixed 256 px Oracle ROI",
        "source_dir": str(args.source_dir.resolve()),
        "official_mapping": str(args.mapping.resolve()),
        "splits": {},
    }
    train_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        with (args.source_dir / f"{split}.csv").open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        kept: list[dict[str, object]] = []
        dropped = Counter()
        by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"source": 0, "kept": 0})
        widths: list[float] = []
        heights: list[float] = []
        for row in rows:
            class_name = row["class_name"]
            by_class[class_name]["source"] += 1
            match = official.get((class_name, Path(row["image_path"]).name))
            if match is None:
                dropped["no_exact_official_left_bbox"] += 1
                continue
            center_x = float(match["u"])
            center_y = float(match["v"])
            width = float(match["bbox_width_px"])
            height = float(match["bbox_height_px"])
            bx1 = max(0.0, center_x - width / 2.0)
            by1 = max(0.0, center_y - height / 2.0)
            bx2 = min(1280.0, center_x + width / 2.0)
            by2 = min(960.0, center_y + height / 2.0)
            if bx1 >= bx2 or by1 >= by2:
                dropped["invalid_official_bbox"] += 1
                continue
            rx1, ry1, rx2, ry2 = [float(row[key]) for key in (
                "roi_x1", "roi_y1", "roi_x2", "roi_y2"
            )]
            if not (rx1 <= bx1 and ry1 <= by1 and bx2 <= rx2 and by2 <= ry2):
                dropped["official_bbox_not_fully_inside_roi"] += 1
                continue
            enriched: dict[str, object] = dict(row)
            enriched.update(
                {
                    "official_bbox_x1": bx1,
                    "official_bbox_y1": by1,
                    "official_bbox_x2": bx2,
                    "official_bbox_y2": by2,
                    "official_bbox_width_px": bx2 - bx1,
                    "official_bbox_height_px": by2 - by1,
                    "official_bbox_source": "MMAUD_2D_exact_pixel_match",
                    "official_2d_split": match["official_split"],
                }
            )
            kept.append(enriched)
            widths.append(bx2 - bx1)
            heights.append(by2 - by1)
            by_class[class_name]["kept"] += 1
        if not kept:
            raise RuntimeError(f"No matched rows for {split}")
        with (args.output_dir / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kept[0]))
            writer.writeheader()
            writer.writerows(kept)
        if split == "train":
            train_rows = kept
        summary["splits"][split] = {
            "source_count": len(rows),
            "kept_count": len(kept),
            "coverage": len(kept) / len(rows),
            "dropped": dict(dropped),
            "by_class": dict(by_class),
            "bbox_width_px": {
                "min": min(widths),
                "median": sorted(widths)[len(widths) // 2],
                "max": max(widths),
            },
            "bbox_height_px": {
                "min": min(heights),
                "median": sorted(heights)[len(heights) // 2],
                "max": max(heights),
            },
        }

    source_stats = json.loads(
        (args.source_dir / "radar_stats.json").read_text(encoding="utf-8")
    )
    radar_stats = compute_radar_stats(
        train_rows,
        args.dataset_root,
        max_range_m=float(source_stats["max_range_m"]),
    )
    write_json(radar_stats, args.output_dir / "radar_stats.json")
    summary["radar_stats"] = radar_stats
    for filename in ("class_mapping.json",):
        source = args.source_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)
    write_json(summary, args.output_dir / "official_bbox_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote official bbox manifests to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
