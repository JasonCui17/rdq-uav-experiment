#!/usr/bin/env python3
"""Weak-oracle geometry audit for eps=2 and eps=1 positives in one fixed unit."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration import OmniRadtanCamera  # noqa: E402
from rdq_uav.calibration.omni import transform_points  # noqa: E402
from tools.visualize_mmuav_positive_clusters import (  # noqa: E402
    load_existing_diagnostic,
    replay_membership,
)
from tools.visualize_nearest_radar_projection import left_image  # noqa: E402

AUDIT_STATUS = "PROVISIONAL_UNVERIFIED_TRANSFORM"
GEOMETRY_CONFIDENCE = "LOW"
MAX_OFFICIAL_BBOX_GAP_MS = 150.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-inspection", action="store_true")
    parser.add_argument(
        "--diagnostic-dir", type=Path,
        default=PROJECT_ROOT / (
            "outputs/mmuav_bbox_covered_mavic2_block00_chunk004/Mavic2/"
            "train_block00_chunk004/diagnostics"
        ),
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/jasoncui/datasets/MMAUD/official/v1"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "manifests/train.csv"
    )
    parser.add_argument(
        "--official-2d-mapping", type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--lidar-calibration", type=Path,
        default=PROJECT_ROOT / "calibration/lidar360_frame_resolution.json",
    )
    parser.add_argument(
        "--camera-config", type=Path,
        default=PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml",
    )
    parser.add_argument(
        "--camera-calibration", type=Path,
        default=PROJECT_ROOT / "calibration/official_left_fitted_calibration.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def timestamp_index(directory: Path, suffix: str) -> tuple[np.ndarray, list[Path]]:
    entries = []
    for path in directory.glob(f"*{suffix}"):
        try:
            entries.append((float(path.stem), path))
        except ValueError:
            continue
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise FileNotFoundError(f"No timestamped {suffix} files in {directory}")
    return np.asarray([entry[0] for entry in entries]), [entry[1] for entry in entries]


def nearest_path(times: np.ndarray, paths: list[Path], query: float) -> tuple[float, Path]:
    insertion = int(np.searchsorted(times, query))
    candidates = [max(0, insertion - 1), min(len(times) - 1, insertion)]
    index = min(candidates, key=lambda item: (abs(times[item] - query), times[item]))
    return float(times[index]), paths[index]


def read_train_rows(path: Path) -> list[dict[str, str]]:
    if path.name != "train.csv":
        raise ValueError("Weak audit accepts train.csv only")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row for row in rows
        if row.get("split") == "train"
        and row.get("sequence_id") == "Mavic2"
        and int(row.get("temporal_block", -1)) == 0
    ]
    if not rows:
        raise ValueError("No Mavic2/train/block00 rows found")
    return rows


def read_official_bbox_records(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("class_name") != "Mavic2" or row.get("match_status") != "exact":
                continue
            result.append({
                "image_time": float(row["image_time"]),
                "image_path": row["image_path"],
                "x1": float(row["bbox_x_center_norm"]) * 1280
                - float(row["bbox_width_norm"]) * 1280 / 2,
                "y1": float(row["bbox_y_center_norm"]) * 960
                - float(row["bbox_height_norm"]) * 960 / 2,
                "x2": float(row["bbox_x_center_norm"]) * 1280
                + float(row["bbox_width_norm"]) * 1280 / 2,
                "y2": float(row["bbox_y_center_norm"]) * 960
                + float(row["bbox_height_norm"]) * 960 / 2,
            })
    return sorted(result, key=lambda item: item["image_time"])


def nearest_official_bbox(
    records: list[dict[str, Any]], query: float,
) -> tuple[dict[str, Any] | None, float]:
    times = np.asarray([record["image_time"] for record in records])
    insertion = int(np.searchsorted(times, query))
    candidates = [max(0, insertion - 1), min(len(times) - 1, insertion)]
    index = min(candidates, key=lambda item: (abs(times[item] - query), times[item]))
    gap_ms = abs(float(times[index]) - query) * 1000
    return (records[index] if gap_ms <= MAX_OFFICIAL_BBOX_GAP_MS else None), gap_ms


def load_provisional_transforms(
    lidar_calibration: Path, camera_calibration: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    lidar_payload = json.loads(lidar_calibration.read_text(encoding="utf-8"))
    selected_name = lidar_payload["selection"]["selected_transform"]
    selected = lidar_payload["results"][selected_name]["transform"]
    rotation_gt_from_lidar = np.asarray(
        selected["rotation_gt_from_sensor"], dtype=np.float64
    )
    translation_gt_from_lidar = np.asarray(
        selected["translation_gt_from_sensor_m"], dtype=np.float64
    )
    camera_payload = json.loads(camera_calibration.read_text(encoding="utf-8"))
    camera_fit = camera_payload["cameras"]["left"]
    rotation_camera_from_gt = np.asarray(
        camera_fit["rotation_camera_from_gt"], dtype=np.float64
    )
    translation_camera_from_gt = np.asarray(
        camera_fit["translation_camera_from_gt_m"], dtype=np.float64
    )
    for value, shape in (
        (rotation_gt_from_lidar, (3, 3)),
        (translation_gt_from_lidar, (3,)),
        (rotation_camera_from_gt, (3, 3)),
        (translation_camera_from_gt, (3,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid provisional transform shape/value: {value.shape}")
    return (
        rotation_gt_from_lidar,
        translation_gt_from_lidar,
        rotation_camera_from_gt,
        translation_camera_from_gt,
        {
            "gt_audit_status": AUDIT_STATUS,
            "geometry_confidence": GEOMETRY_CONFIDENCE,
            "lidar_calibration": str(lidar_calibration.resolve()),
            "lidar_selected_transform": selected_name,
            "lidar_transform_credible": bool(
                lidar_payload.get("credibility", {}).get("credible", False)
            ),
            "camera_calibration": str(camera_calibration.resolve()),
        },
    )


def load_gt_xyz(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
    if value.size < 3:
        raise ValueError(f"Invalid GT XYZ: {path} {value.shape}")
    return value[:3]


def render_overlay(
    image_path: Path, camera: OmniRadtanCamera, bbox: dict[str, float] | None,
    pixel: np.ndarray, valid: bool, eps: int, cluster_id: int, frame_index: int,
    probability: float, output_path: Path,
) -> None:
    image = left_image(image_path, camera.width, camera.height)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if bbox is not None:
        draw.rectangle((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]), outline=(0, 255, 0), width=4)
    if valid:
        u, v = pixel
        draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=(255, 0, 0), width=4)
        draw.line((u - 12, v, u + 12, v), fill=(255, 0, 0), width=3)
        draw.line((u, v - 12, u, v + 12), fill=(255, 0, 0), width=3)
    lines = [
        f"eps={eps} cluster={cluster_id} frame={frame_index} P_uav={probability:.4f}",
        f"{AUDIT_STATUS} | confidence={GEOMETRY_CONFIDENCE}",
        f"projection={'VALID' if valid else 'INVALID'}",
        "GT bbox=AVAILABLE" if bbox is not None else "GT bbox=OFFICIAL BBOX UNAVAILABLE",
    ]
    for line_index, line in enumerate(lines):
        y = 10 + line_index * 20
        draw.rectangle((8, y - 2, 720, y + 16), fill=(0, 0, 0))
        draw.text((12, y), line, fill=(255, 255, 0), font=font)
    image.save(output_path)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    columns = 4
    thumb_size = (320, 240)
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "black")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail(thumb_size)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * thumb_size[1]
        sheet.paste(tile, (x, y))
    sheet.save(output_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_cluster(
    rows: list[dict[str, Any]], probability: float, *, eps: int | None = None,
    transform_credible: bool | None = None,
) -> dict[str, Any]:
    center_distances = np.asarray([row["distance_center_to_gt_m"] for row in rows])
    point_distances = np.asarray([row["min_point_to_gt_m"] for row in rows])
    valid_rows = [row for row in rows if row["projection_valid"]]
    official_bbox_rows = [row for row in rows if row["official_bbox_available"]]
    bbox_rows = [row for row in official_bbox_rows if row["projection_valid"]]
    if transform_credible is False:
        verdict = "INCONCLUSIVE_UNVERIFIED_TRANSFORM"
    elif not bbox_rows:
        verdict = "INCONCLUSIVE"
    else:
        near16 = sum(row["bbox_center_error_px"] <= 16 for row in bbox_rows)
        near32 = sum(row["bbox_center_error_px"] <= 32 for row in bbox_rows)
        if near16 > len(bbox_rows) / 2:
            verdict = "LIKELY_TARGET_ALIGNED"
        elif len(bbox_rows) >= 5 and near32 <= len(bbox_rows) * 0.2:
            verdict = "LIKELY_BACKGROUND"
        else:
            verdict = "INCONCLUSIVE"
    return {
        "eps": eps,
        "cluster_id": int(rows[0]["cluster_id"]),
        "P_uav": probability,
        "total_points": int(sum(row.get("cluster_point_count", 0) for row in rows)),
        "active_frames": len(rows),
        "mean_center_gt_distance_m": float(center_distances.mean()),
        "median_center_gt_distance_m": float(np.median(center_distances)),
        "min_center_gt_distance_m": float(center_distances.min()),
        "min_point_gt_distance_m": float(point_distances.min()),
        **{f"hit_center_{limit}m": int(np.count_nonzero(center_distances <= limit)) for limit in (1, 2, 5)},
        **{f"hit_point_{limit}m": int(np.count_nonzero(point_distances <= limit)) for limit in (1, 2, 5)},
        "projected_valid_frames": len(valid_rows),
        "official_bbox_frames": len(official_bbox_rows),
        "inside_gt_bbox_frames": None if not bbox_rows else int(sum(row["inside_gt_bbox"] for row in bbox_rows)),
        "within_8px_frames": None if not bbox_rows else int(sum(row["bbox_center_error_px"] <= 8 for row in bbox_rows)),
        "within_16px_frames": None if not bbox_rows else int(sum(row["bbox_center_error_px"] <= 16 for row in bbox_rows)),
        "within_32px_frames": None if not bbox_rows else int(sum(row["bbox_center_error_px"] <= 32 for row in bbox_rows)),
        "median_bbox_center_error_px": None if not bbox_rows else float(
            np.median([row["bbox_center_error_px"] for row in bbox_rows])
        ),
        "median_GT_distance": float(np.median(center_distances)),
        "min_GT_distance": float(center_distances.min()),
        "weak_bbox_overlap": bool(any(
            bool(row["inside_gt_bbox"]) or row["bbox_center_error_px"] <= 32
            for row in bbox_rows
        )),
        "verdict": verdict,
        "geometry_confidence": GEOMETRY_CONFIDENCE,
        "gt_audit_status": AUDIT_STATUS,
    }


def build_timestamp_inspection(
    frame_timestamps: list[str], gt_times: np.ndarray, gt_paths: list[Path],
    image_times: np.ndarray, image_paths: list[Path], bbox_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inspection = []
    for frame_index, timestamp_text in enumerate(frame_timestamps, start=1):
        timestamp = float(timestamp_text)
        gt_time, _ = nearest_path(gt_times, gt_paths, timestamp)
        official_bbox, official_gap_ms = nearest_official_bbox(bbox_records, timestamp)
        if official_bbox is not None:
            image_time = float(official_bbox["image_time"])
            image_path = Path(official_bbox["image_path"])
        else:
            image_time, image_path = nearest_path(image_times, image_paths, timestamp)
        inspection.append({
            "frame_index": frame_index,
            "lidar_time": timestamp,
            "gt_gap_ms": abs(gt_time - timestamp) * 1000,
            "image_time": image_time,
            "image_gap_ms": abs(image_time - timestamp) * 1000,
            "image_path": str(image_path),
            "official_bbox_available": official_bbox is not None,
            "official_bbox_gap_ms": official_gap_ms,
        })
    return inspection


def audit_eps_positive_clusters(
    *, eps: int, arrays: np.lib.npyio.NpzFile, frame_timestamps: list[str],
    inspection: list[dict[str, Any]], sequence_root: Path,
    gt_times: np.ndarray, gt_paths: list[Path], bbox_records: list[dict[str, Any]],
    camera: OmniRadtanCamera, R_gl: np.ndarray, t_gl: np.ndarray,
    R_cg: np.ndarray, t_cg: np.ndarray, transform_credible: bool,
    output_dir: Path,
) -> list[dict[str, Any]]:
    positive_ids = arrays["cluster_ids"][arrays["predictions"] == 1].astype(int)
    points, point_frame_indices, labels = replay_membership(
        sequence_root / "lidar_360", frame_timestamps, arrays, eps=eps
    )
    summaries = []
    for cluster_id in positive_ids:
        probability_index = int(np.flatnonzero(arrays["cluster_ids"] == cluster_id)[0])
        probability = float(arrays["probabilities"][probability_index, 1])
        cluster_rows: list[dict[str, Any]] = []
        overlay_paths: list[Path] = []
        overlay_dir = output_dir / f"eps{eps}_positive_cluster_{cluster_id}_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for inspected in inspection:
            frame_index = inspected["frame_index"]
            member = (labels == cluster_id) & (point_frame_indices == frame_index)
            frame_points = points[member]
            if not len(frame_points):
                continue
            center_lidar = frame_points.mean(axis=0)
            points_common = transform_points(frame_points, R_gl, t_gl)
            center_common = transform_points(center_lidar.reshape(1, 3), R_gl, t_gl)[0]
            gt_time, gt_path = nearest_path(gt_times, gt_paths, float(inspected["lidar_time"]))
            gt_xyz = load_gt_xyz(gt_path)
            center_distance = float(np.linalg.norm(center_common - gt_xyz))
            min_point_distance = float(np.linalg.norm(points_common - gt_xyz, axis=1).min())
            center_camera = transform_points(center_common.reshape(1, 3), R_cg, t_cg)
            pixels, valid_array = camera.project(center_camera, require_in_image=True)
            pixel = pixels[0]
            valid = bool(valid_array[0])
            image_path = Path(inspected["image_path"])
            bbox, bbox_gap_ms = nearest_official_bbox(bbox_records, float(inspected["lidar_time"]))
            bbox_center_error = None
            inside_bbox = None
            if valid and bbox is not None:
                bbox_center = np.asarray([
                    (bbox["x1"] + bbox["x2"]) / 2,
                    (bbox["y1"] + bbox["y2"]) / 2,
                ])
                bbox_center_error = float(np.linalg.norm(pixel - bbox_center))
                inside_bbox = bool(
                    bbox["x1"] <= pixel[0] <= bbox["x2"]
                    and bbox["y1"] <= pixel[1] <= bbox["y2"]
                )
            row = {
                "gt_audit_status": AUDIT_STATUS,
                "geometry_confidence": GEOMETRY_CONFIDENCE,
                "eps": eps,
                "cluster_id": int(cluster_id),
                "frame_idx": frame_index,
                "timestamp": inspected["lidar_time"],
                "cluster_point_count": int(len(frame_points)),
                "cluster_center_lidar_x": float(center_lidar[0]),
                "cluster_center_lidar_y": float(center_lidar[1]),
                "cluster_center_lidar_z": float(center_lidar[2]),
                "cluster_center_common_x": float(center_common[0]),
                "cluster_center_common_y": float(center_common[1]),
                "cluster_center_common_z": float(center_common[2]),
                "gt_time": gt_time,
                "gt_gap_ms": abs(gt_time - float(inspected["lidar_time"])) * 1000,
                "gt_x": float(gt_xyz[0]), "gt_y": float(gt_xyz[1]), "gt_z": float(gt_xyz[2]),
                "distance_center_to_gt_m": center_distance,
                "min_point_to_gt_m": min_point_distance,
                "image_time": inspected["image_time"],
                "image_gap_ms": inspected["image_gap_ms"],
                "image_path": str(image_path),
                "projected_u": None if not valid else float(pixel[0]),
                "projected_v": None if not valid else float(pixel[1]),
                "projection_valid": valid,
                "official_bbox_available": bbox is not None,
                "official_bbox_gap_ms": bbox_gap_ms,
                "bbox_x1": None if bbox is None else bbox["x1"],
                "bbox_y1": None if bbox is None else bbox["y1"],
                "bbox_x2": None if bbox is None else bbox["x2"],
                "bbox_y2": None if bbox is None else bbox["y2"],
                "bbox_center_error_px": bbox_center_error,
                "inside_gt_bbox": inside_bbox,
            }
            cluster_rows.append(row)
            overlay_path = overlay_dir / f"frame_{frame_index:02d}_{image_path.stem}.png"
            render_overlay(
                image_path, camera, bbox, pixel, valid, eps, int(cluster_id),
                frame_index, probability, overlay_path,
            )
            overlay_paths.append(overlay_path)
        write_csv(output_dir / f"eps{eps}_positive_cluster_{cluster_id}_gt_audit.csv", cluster_rows)
        make_contact_sheet(
            overlay_paths, output_dir / f"eps{eps}_cluster_{cluster_id}_contact_sheet.png"
        )
        summaries.append(summarize_cluster(
            cluster_rows, probability, eps=eps, transform_credible=transform_credible
        ))
    del labels, point_frame_indices, points
    gc.collect()
    return summaries


def print_summary_table(title: str, summaries: list[dict[str, Any]]) -> None:
    print(title)
    print("cluster | P_uav | active | valid_proj | bbox_frames | inside | <=16px | median_px_error | verdict")
    for item in summaries:
        inside = "N/A" if item["inside_gt_bbox_frames"] is None else str(item["inside_gt_bbox_frames"])
        near16 = "N/A" if item["within_16px_frames"] is None else str(item["within_16px_frames"])
        median_px = item["median_bbox_center_error_px"]
        median_px_text = "N/A" if median_px is None else f"{median_px:.2f}"
        print(
            f"{item['cluster_id']} | {item['P_uav']:.4f} | {item['active_frames']}/20 | "
            f"{item['projected_valid_frames']} | {item['official_bbox_frames']} | {inside} | "
            f"{near16} | {median_px_text} | {item['verdict']}"
        )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.diagnostic_dir / "geometry_both_eps_audit"
    eps1_arrays, frame_timestamps = load_existing_diagnostic(args.diagnostic_dir)
    eps2_arrays = np.load(
        args.diagnostic_dir / "pre_lstm_clusters/cluster_features.npz", allow_pickle=False
    )
    for eps, arrays in ((2, eps2_arrays), (1, eps1_arrays)):
        for name in ("cluster_ids", "predictions", "probabilities", "num_points"):
            if name not in arrays.files:
                raise KeyError(f"eps={eps} diagnostic missing {name}")

    sequence_root = args.dataset_root / "Mavic2"
    gt_times, gt_paths = timestamp_index(sequence_root / "ground_truth", ".npy")
    image_times, image_paths = timestamp_index(sequence_root / "image", ".png")
    bbox_records = read_official_bbox_records(args.official_2d_mapping)
    inspection = build_timestamp_inspection(
        frame_timestamps, gt_times, gt_paths, image_times, image_paths, bbox_records
    )
    print(
        f"dry timestamp inspection: unit=Mavic2/train_block00_chunk004 frames=20 "
        f"range=[{frame_timestamps[0]},{frame_timestamps[-1]}] "
        f"median_gt_gap_ms={np.median([row['gt_gap_ms'] for row in inspection]):.3f} "
        f"median_image_gap_ms={np.median([row['image_gap_ms'] for row in inspection]):.3f} "
        f"official_bbox_matches={sum(row['official_bbox_available'] for row in inspection)}/20"
    )
    if args.dry_inspection:
        return

    with args.camera_config.open(encoding="utf-8") as handle:
        camera = OmniRadtanCamera.from_config(yaml.safe_load(handle)["cameras"]["left"])
    R_gl, t_gl, R_cg, t_cg, provenance = load_provisional_transforms(
        args.lidar_calibration, args.camera_calibration
    )
    transform_credible = bool(provenance["lidar_transform_credible"])
    if transform_credible:
        raise RuntimeError("This audit is scoped to the known credible=false provisional transform")
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        frame_timestamps=frame_timestamps, inspection=inspection,
        sequence_root=sequence_root, gt_times=gt_times, gt_paths=gt_paths,
        bbox_records=bbox_records, camera=camera, R_gl=R_gl, t_gl=t_gl,
        R_cg=R_cg, t_cg=t_cg, transform_credible=transform_credible,
        output_dir=output_dir,
    )
    eps2_summaries = audit_eps_positive_clusters(eps=2, arrays=eps2_arrays, **common)
    eps1_summaries = audit_eps_positive_clusters(eps=1, arrays=eps1_arrays, **common)

    base_payload = {
        "scope": "Mavic2/train_block00_chunk004 only",
        "gt_used_after_dbscan_lstm_only": True,
        "provisional_transform_used": True,
        "transform": provenance,
        "official_bbox_matches": int(sum(row["official_bbox_available"] for row in inspection)),
        "weak_bbox_overlap_definition": (
            "at least one valid projected center is inside the official bbox or <=32px from bbox center"
        ),
        "weak_bbox_overlap_is_not_target_hit": True,
    }
    for eps, summaries in ((2, eps2_summaries), (1, eps1_summaries)):
        payload = {
            **base_payload,
            "eps": eps,
            "positive_count": len(summaries),
            "whether_any_positive_has_weak_bbox_overlap": any(
                item["weak_bbox_overlap"] for item in summaries
            ),
            "clusters": summaries,
        }
        (output_dir / f"geometry_eps{eps}_positive_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    print_summary_table("SOURCE eps=2", eps2_summaries)
    print_summary_table("TRAIN-CONSISTENT eps=1", eps1_summaries)
    print(f"eps=2 positive count: {len(eps2_summaries)}")
    print(f"eps=1 positive count: {len(eps1_summaries)}")
    print(
        "whether_any_eps2_positive_has_weak_bbox_overlap: "
        f"{any(item['weak_bbox_overlap'] for item in eps2_summaries)}"
    )
    print(
        "whether_any_eps1_positive_has_weak_bbox_overlap: "
        f"{any(item['weak_bbox_overlap'] for item in eps1_summaries)}"
    )


if __name__ == "__main__":
    main()
