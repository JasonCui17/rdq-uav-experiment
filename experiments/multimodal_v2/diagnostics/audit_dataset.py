#!/usr/bin/env python3
"""Read-only audit of the V2 query, observation and supervision contracts."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import yaml

from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from rdq_uav.multimodal_v2.data import (
    V2_SPLIT_COUNTS, build_split_dataset, load_3d_target, load_sequence_splits, load_yolo_box,
)
from rdq_uav.multimodal_v2.radar_data import load_released_xyz


COUNT_KEYS = (
    "queries", "valid_gt3d", "valid_radar_samples", "valid_image_samples",
    "valid_bbox_samples", "supervised_images", "b0_supervised_samples",
    "b1_supervised_samples", "both_modalities_missing_samples",
)


def audit_sequence(dataset, records):
    counts = Counter()
    supervised_images = set()

    @lru_cache(maxsize=64)
    def event_points(path):
        # Same released-point reader and float32 conversion as Dataset.__getitem__.
        return torch.from_numpy(load_released_xyz(path)[0].astype(np.float32))

    for record in records:
        counts["queries"] += 1
        target = load_3d_target(dataset.target_3d_index.get(record.query_time))
        match = dataset.image_index.match(record.sequence_id, record.query_time)
        label = (dataset.root / record.sequence_id / dataset.label_directory /
                 f"{match.path.stem}.txt") if match.path is not None else None
        box = load_yolo_box(label, dataset.camera_wh) if label is not None else None
        points = [event_points(event.file_path) for event in
                  dataset.select_radar_events(record.sequence_id, record.query_time)]
        radar_valid = any(len(part) for part in points)
        image_valid = match.valid
        bbox_valid = box is not None

        counts["valid_gt3d"] += int(target.valid)
        counts["valid_radar_samples"] += int(radar_valid)
        counts["valid_image_samples"] += int(image_valid)
        counts["valid_bbox_samples"] += int(bbox_valid)
        counts["both_modalities_missing_samples"] += int(not radar_valid and not image_valid)
        counts["b1_supervised_samples"] += int(image_valid and bbox_valid)
        if image_valid and bbox_valid:
            supervised_images.add(str(match.path))

        # CandidateLoss.labels takes the minimum point-to-GT distance per voxel.
        # A positive voxel exists exactly when some input point is within 1 m.
        if target.valid and radar_valid:
            counts["b0_supervised_samples"] += int(any(
                bool((torch.linalg.vector_norm(part - target.xyz_m, dim=1) <= 1.0).any())
                for part in points if len(part)
            ))

    counts["supervised_images"] = len(supervised_images)
    return {key: counts[key] for key in COUNT_KEYS}


def audit_config(config, root):
    groups = {}
    split_path = resolve_project_path(config["data"]["split_file"], root)
    split_sequences = load_sequence_splits(split_path)
    for split in V2_SPLIT_COUNTS:
        dataset = build_split_dataset(config, root, split)
        by_sequence = {}
        for sequence in split_sequences[split]:
            records = [record for record in dataset.query_records if record.sequence_id == sequence]
            by_sequence[sequence] = audit_sequence(dataset, records)
        totals = {key: sum(row[key] for row in by_sequence.values()) for key in COUNT_KEYS}
        groups[split] = {"totals": totals, "sequences": by_sequence}
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve_project_path(args.config, ROOT)
    config = apply_runtime_path_overrides(yaml.safe_load(config_path.read_text()))
    if args.data_root is not None:
        config["data"]["root"] = str(resolve_project_path(args.data_root, ROOT))
    result = {
        "config": str(config_path), "data_root": str(config["data"]["root"]),
        "split_file": str(config["data"]["split_file"]),
        "b0_positive_rule": "valid 3D GT and at least one input point within 1.0 m",
        "groups": audit_config(config, ROOT),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
