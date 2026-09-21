#!/usr/bin/env python3
"""Generate P4 LiDAR-V2 -> left-image geometry measurements.

This is an audit-only oracle: GT is used to score whether the frozen LiDAR
geometry contract can support the later local HCI. It never feeds GT into the
model. Test splits are deliberately refused.

Coordinate contract:
    released Avia/Mid360 XYZ == LiDAR-V2/GT reference frame
    p_camera = R_camera_from_gt @ p_lidar + t_camera_from_gt

The same-sequence shuffled control keeps one LiDAR query fixed and replaces only
its GT/image target with a deterministic half-cycle target from the same
sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402
from rdq_uav.lidar_v2.data import LiDARUAVDataset  # noqa: E402
from rdq_uav.lidar_v2.geometry import HierarchyBuilder  # noqa: E402
from rdq_uav.multimodal_v1.data import LeftImageIndex  # noqa: E402
from rdq_uav.multimodal_v1.geometry_audit import (  # noqa: E402
    feature_neighborhood_hit,
    nearest_projected_distance_px,
)


@dataclass(frozen=True)
class AuditTarget:
    dataset_index: int
    sequence_id: str
    query_uid: object
    query_time: float
    target_xyz: np.ndarray
    target_pixel: np.ndarray
    image_time: float
    image_gap_s: float


def dino_eval_resize_scale(
    width: int,
    height: int,
    *,
    short_edge: int,
    max_size: int,
) -> tuple[float, float, int, int]:
    """Detectron2 ResizeShortestEdge geometry for deterministic DINO evaluation."""
    if min(width, height, short_edge, max_size) <= 0:
        raise ValueError("image dimensions and resize limits must be positive")
    scale = float(short_edge) / float(min(width, height))
    if float(max(width, height)) * scale > float(max_size):
        scale = float(max_size) / float(max(width, height))
    new_height = int(float(height) * scale + 0.5)
    new_width = int(float(width) * scale + 0.5)
    return new_width / width, new_height / height, new_width, new_height


def load_geometry(
    camera_config: Path,
    camera_calibration: Path,
) -> tuple[OmniRadtanCamera, np.ndarray, np.ndarray, float]:
    config = yaml.safe_load(camera_config.read_text(encoding="utf-8"))
    fitted = json.loads(camera_calibration.read_text(encoding="utf-8"))
    if fitted.get("time_convention") != "gt_query_time = image_time + time_offset_s":
        raise ValueError("unexpected camera calibration time convention")
    left = fitted["cameras"]["left"]
    camera = OmniRadtanCamera.from_config(config["cameras"]["left"])
    rotation = np.asarray(left["rotation_camera_from_gt"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(
        left["translation_camera_from_gt_m"], dtype=np.float64
    ).reshape(3)
    return camera, rotation, translation, float(fitted["time_offset_s"])


def project_reference(
    xyz: np.ndarray,
    camera: OmniRadtanCamera,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, bool]:
    point = np.asarray(xyz, dtype=np.float64).reshape(1, 3)
    pixel, valid = camera.project(
        transform_points(point, rotation, translation),
        require_in_image=True,
    )
    return pixel[0], bool(valid[0])


def record_target_xyz(record: dict[str, Any]) -> np.ndarray:
    if "target_path" in record:
        value = np.load(record["target_path"], allow_pickle=False)
    elif "target_xyz" in record:
        value = record["target_xyz"]
    else:
        raise KeyError("LiDAR dataset record has no target source")
    xyz = np.asarray(value, dtype=np.float64).reshape(-1)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"invalid target XYZ: {xyz}")
    return xyz


def collect_targets(
    dataset: LiDARUAVDataset,
    image_index: LeftImageIndex,
    camera: OmniRadtanCamera,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[AuditTarget]:
    targets: list[AuditTarget] = []
    for dataset_index, record in enumerate(dataset.records):
        match = image_index.match(record["sequence_id"], record["query_time"])
        if not match.valid or match.image_time is None or match.gap_s is None:
            continue
        target_xyz = record_target_xyz(record)
        target_pixel, valid = project_reference(
            target_xyz, camera, rotation, translation
        )
        if not valid:
            continue
        targets.append(
            AuditTarget(
                dataset_index=dataset_index,
                sequence_id=record["sequence_id"],
                query_uid=record["query_uid"],
                query_time=float(record["query_time"]),
                target_xyz=target_xyz,
                target_pixel=target_pixel,
                image_time=float(match.image_time),
                image_gap_s=float(match.gap_s),
            )
        )
    return targets


def deterministic_half_cycle(
    targets: list[AuditTarget],
) -> dict[int, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target.sequence_id].append(index)
    mapping: dict[int, int] = {}
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        indices.sort(key=lambda index: (targets[index].query_time, str(targets[index].query_uid)))
        offset = max(1, len(indices) // 2)
        for local, source_index in enumerate(indices):
            mapping[source_index] = indices[(local + offset) % len(indices)]
    return mapping


def project_level(
    centers: torch.Tensor,
    camera: OmniRadtanCamera,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = centers.detach().cpu().numpy().astype(np.float64, copy=False)
    if not len(xyz):
        return xyz, np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=bool)
    pixels, valid = camera.project(
        transform_points(xyz, rotation, translation),
        require_in_image=True,
    )
    return xyz, pixels, valid


def measure_target(
    projected_levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    target: AuditTarget,
    *,
    image_scale_xy: tuple[float, float],
) -> dict[str, Any]:
    l0_xyz, l0_pixels, l0_valid = projected_levels[0]
    nearest_3d = (
        float(np.linalg.norm(l0_xyz - target.target_xyz[None, :], axis=1).min())
        if len(l0_xyz)
        else None
    )
    nearest_px = nearest_projected_distance_px(
        l0_pixels, l0_valid, target.target_pixel
    )
    output: dict[str, Any] = {
        "nearest_3d_distance_m": nearest_3d,
        "image_reprojection_distance_px": nearest_px,
        "nearest_center_distance_px": nearest_px,
        # The 102-sequence LiDAR split has no per-query official 2D bbox contract.
        # Keep bbox distance unavailable rather than fabricate a box from GT.
        "nearest_bbox_distance_px": None,
    }
    for (stride, level) in zip((4, 8, 16), projected_levels, strict=True):
        _, pixels, valid = level
        output[f"feature_neighborhood_hit_stride_{stride}"] = feature_neighborhood_hit(
            pixels,
            valid,
            target.target_pixel,
            stride=stride,
            image_scale_xy=image_scale_xy,
        )
        output[f"valid_projected_token_count_stride_{stride}"] = int(valid.sum())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/official/train"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "outputs/mmuav_paper_reproduction/splits/splits.json",
    )
    parser.add_argument("--split", required=True, help="train_sub or validation_sub; test is refused")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--image-directory", default="Image")
    parser.add_argument("--max-image-gap-s", type=float, default=0.04)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument(
        "--dino-short-edge",
        type=int,
        default=800,
        help="detrex DINO deterministic eval ResizeShortestEdge",
    )
    parser.add_argument("--dino-max-size", type=int, default=1333)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "test" in args.split.lower():
        raise ValueError("P4 must not read a held-out test split")
    if args.max_events <= 0:
        raise ValueError("max-events must be positive")
    if args.max_image_gap_s < 0:
        raise ValueError("max-image-gap-s must be non-negative")

    camera, rotation, translation, time_offset_s = load_geometry(
        args.camera_config, args.camera_calibration
    )
    sx, sy, resized_width, resized_height = dino_eval_resize_scale(
        camera.width,
        camera.height,
        short_edge=args.dino_short_edge,
        max_size=args.dino_max_size,
    )

    dataset = LiDARUAVDataset(
        args.root,
        args.split_file,
        args.split,
        max_events=args.max_events,
    )
    image_index = LeftImageIndex(
        args.root,
        time_offset_s=time_offset_s,
        image_directory=args.image_directory,
        max_abs_gap_s=args.max_image_gap_s,
    )
    targets = collect_targets(dataset, image_index, camera, rotation, translation)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        targets = targets[: args.max_samples]
    shuffle = deterministic_half_cycle(targets)
    source_indices = sorted(shuffle)
    if not source_indices:
        raise RuntimeError("no auditable same-sequence target pairs were found")

    builder = HierarchyBuilder(scales=(0.5, 1.0, 2.0))
    records: list[dict[str, Any]] = []
    for ordinal, source_index in enumerate(source_indices, start=1):
        source = targets[source_index]
        shuffled = targets[shuffle[source_index]]
        sample = dataset[source.dataset_index]
        points = sample["points"]
        point_batch = torch.zeros(len(points), dtype=torch.long)
        hierarchy = builder(points, point_batch)
        projected_levels = [
            project_level(level.centers, camera, rotation, translation)
            for level in hierarchy.levels
        ]
        common = {
            "split": args.split,
            "source_sequence_id": source.sequence_id,
            "source_query_uid": source.query_uid,
            "source_query_time": source.query_time,
            "source_image_time": source.image_time,
            "source_image_query_gap_s": source.image_gap_s,
            "source_point_count": int(len(points)),
            "source_event_count": int(sample["event_count"]),
            "coordinate_contract": "released_lidar_xyz_equals_gt_reference_frame",
            "camera_transform": "p_camera=R_camera_from_gt@p_lidar+t_camera_from_gt",
            "time_convention": "query_time=image_time+time_offset_s",
            "time_offset_s": time_offset_s,
            "dino_eval_resize": {
                "source_width": camera.width,
                "source_height": camera.height,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "scale_x": sx,
                "scale_y": sy,
                "short_edge": args.dino_short_edge,
                "max_size": args.dino_max_size,
            },
        }
        records.append(
            {
                **common,
                "pairing": "real",
                "target_sequence_id": source.sequence_id,
                "target_query_uid": source.query_uid,
                "target_query_time": source.query_time,
                **measure_target(
                    projected_levels,
                    source,
                    image_scale_xy=(sx, sy),
                ),
            }
        )
        records.append(
            {
                **common,
                "pairing": "same_sequence_shuffled",
                "target_sequence_id": shuffled.sequence_id,
                "target_query_uid": shuffled.query_uid,
                "target_query_time": shuffled.query_time,
                **measure_target(
                    projected_levels,
                    shuffled,
                    image_scale_xy=(sx, sy),
                ),
            }
        )
        if ordinal % 500 == 0:
            print(f"processed {ordinal}/{len(source_indices)} LiDAR queries", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    metadata = {
        "split": args.split,
        "dataset_queries": len(dataset),
        "auditable_targets": len(targets),
        "paired_queries": len(source_indices),
        "measurement_records": len(records),
        "test_read": False,
        "coordinate_contract": "released_lidar_xyz_equals_gt_reference_frame",
        "camera_calibration": str(args.camera_calibration.resolve()),
        "camera_config": str(args.camera_config.resolve()),
        "time_offset_s": time_offset_s,
        "max_image_gap_s": args.max_image_gap_s,
        "dino_eval_resize": {
            "source": [camera.width, camera.height],
            "resized": [resized_width, resized_height],
            "scale_xy": [sx, sy],
            "short_edge": args.dino_short_edge,
            "max_size": args.dino_max_size,
        },
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
