"""Deterministic merged-LiDAR preprocessing primitives for read-only audits."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LidarFrameEvent:
    sequence_id: str
    timestamp: float
    sensor_id: int
    sensor_name: str
    file_path: Path


def merge_frame_streams(streams: Iterable[Iterable[LidarFrameEvent]]) -> list[LidarFrameEvent]:
    events = [event for stream in streams for event in stream]
    return sorted(events, key=lambda event: (
        event.timestamp, event.sensor_id, event.sensor_name, str(event.file_path)
    ))


def select_last_history(
    merged_stream: list[LidarFrameEvent], prediction_timestamp: float, count: int = 20,
) -> list[LidarFrameEvent]:
    if count <= 0:
        return []
    times = np.fromiter((event.timestamp for event in merged_stream), dtype=np.float64)
    end = int(np.searchsorted(times, prediction_timestamp, side="right"))
    return merged_stream[max(0, end - count):end]


def load_released_xyz(path: Path) -> tuple[np.ndarray, int, int]:
    """Use existing MMUAV XYZ semantics and report raw/invalid row counts."""
    raw = np.asarray(np.load(path, allow_pickle=False))
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"Expected [N,>=3] point cloud at {path}, got {raw.shape}")
    xyz = np.asarray(raw[:, :3], dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    return xyz[valid], int(len(xyz)), int(np.count_nonzero(~valid))


def normalize_delta_t(delta_t: np.ndarray, window_duration: float) -> np.ndarray:
    delta_t = np.asarray(delta_t, dtype=np.float64)
    denominator = max(float(window_duration), np.finfo(np.float64).eps)
    result = delta_t / denominator
    return np.clip(result, -1.0, 0.0)


def concatenate_frames(
    selected: list[LidarFrameEvent], prediction_timestamp: float,
    loaded: list[tuple[np.ndarray, int, int]],
) -> dict[str, np.ndarray | int | float]:
    if len(selected) != len(loaded):
        raise ValueError("selected and loaded lengths differ")
    oldest = selected[0].timestamp if selected else prediction_timestamp
    duration = max(0.0, prediction_timestamp - oldest)
    xyz_parts, sensor_parts, dt_parts, frame_parts = [], [], [], []
    raw_total = invalid_total = 0
    per_frame_valid = []
    for frame_index, (event, (xyz, raw_rows, invalid_rows)) in enumerate(zip(selected, loaded)):
        if event.timestamp > prediction_timestamp:
            raise ValueError("Future frame reached concatenate_frames")
        raw_total += raw_rows
        invalid_total += invalid_rows
        per_frame_valid.append(len(xyz))
        if not len(xyz):
            continue
        xyz_parts.append(xyz)
        sensor_parts.append(np.full(len(xyz), event.sensor_id, dtype=np.int8))
        dt_parts.append(np.full(len(xyz), event.timestamp - prediction_timestamp, dtype=np.float64))
        frame_parts.append(np.full(len(xyz), frame_index, dtype=np.int16))
    xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3), dtype=np.float64)
    sensor = np.concatenate(sensor_parts) if sensor_parts else np.empty(0, dtype=np.int8)
    delta_t = np.concatenate(dt_parts) if dt_parts else np.empty(0, dtype=np.float64)
    frame_index = np.concatenate(frame_parts) if frame_parts else np.empty(0, dtype=np.int16)
    if len(xyz) != sum(per_frame_valid):
        raise AssertionError("Merged point conservation failed")
    return {
        "xyz": xyz, "sensor_id": sensor, "delta_t": delta_t,
        "delta_t_norm": normalize_delta_t(delta_t, duration),
        "frame_index": frame_index, "raw_total": raw_total,
        "invalid_total": invalid_total, "per_frame_valid": np.asarray(per_frame_valid),
        "window_duration": duration,
    }


_NEIGHBOR_CELL_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def isolated_point_keep_mask(points: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Exact zero-neighbor filter using hashes plus Euclidean verification."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {points.shape}")
    count = len(points)
    if count < 2:
        return np.zeros(count, dtype=bool)
    # Any two points in one such cell are at most radius apart.
    safe_cells = np.floor(points / (radius / np.sqrt(3.0))).astype(np.int64)
    _, safe_inverse, safe_counts = np.unique(
        safe_cells, axis=0, return_inverse=True, return_counts=True
    )
    keep = safe_counts[safe_inverse] > 1
    unresolved = np.flatnonzero(~keep)
    if not len(unresolved):
        return keep
    query_cells = np.floor(points / radius).astype(np.int64)
    order = np.lexsort((query_cells[:, 2], query_cells[:, 1], query_cells[:, 0]))
    sorted_cells = query_cells[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(sorted_cells, axis=0), axis=1)), count]
    cell_map = {
        tuple(sorted_cells[start]): order[start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    }
    radius_sq = radius * radius
    for point_index in unresolved:
        cell = query_cells[point_index]
        for offset in _NEIGHBOR_CELL_OFFSETS:
            candidates = cell_map.get(tuple(cell + offset))
            if candidates is None:
                continue
            candidates = candidates[candidates != point_index]
            if len(candidates) and np.any(
                np.einsum("ij,ij->i", points[candidates] - points[point_index],
                          points[candidates] - points[point_index]) <= radius_sq
            ):
                keep[point_index] = True
                break
    return keep


def support_to_gt(points: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float | int]:
    if not len(points):
        return {"d_min": np.nan, "n_0p5m": 0, "n_1m": 0, "n_2m": 0}
    distances = np.linalg.norm(np.asarray(points) - np.asarray(gt_xyz)[None, :], axis=1)
    return {
        "d_min": float(np.min(distances)),
        "n_0p5m": int(np.count_nonzero(distances <= 0.5)),
        "n_1m": int(np.count_nonzero(distances <= 1.0)),
        "n_2m": int(np.count_nonzero(distances <= 2.0)),
    }


def voxelize_level(
    points: np.ndarray, sensor_id: np.ndarray, delta_t: np.ndarray,
    frame_index: np.ndarray, voxel_size: float, origin: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    origin = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    coords_per_point = np.floor((points - origin) / voxel_size).astype(np.int64)
    if not len(points):
        return {
            "coords": np.empty((0, 3), dtype=np.int64), "centers": np.empty((0, 3)),
            "inverse": np.empty(0, dtype=np.int64), "counts": np.empty(0, dtype=np.int64),
            "point_order": np.empty(0, dtype=np.int64), "point_offsets": np.zeros(1, dtype=np.int64),
            "relative_xyz": np.empty((0, 3)), "relative_mean": np.empty((0, 3)),
            "relative_max_abs": np.empty((0, 3)), "avia_count": np.empty(0, dtype=np.int64),
            "mid360_count": np.empty(0, dtype=np.int64), "avia_fraction": np.empty(0),
            "mid360_fraction": np.empty(0), "delta_t_mean": np.empty(0),
            "delta_t_min": np.empty(0), "delta_t_max": np.empty(0),
            "frame_count": np.empty(0, dtype=np.int64),
        }
    coords, inverse, counts = np.unique(
        coords_per_point, axis=0, return_inverse=True, return_counts=True
    )
    centers = origin + (coords.astype(np.float64) + 0.5) * voxel_size
    relative = points - centers[inverse]
    number = len(coords)
    relative_mean = np.column_stack([
        np.bincount(inverse, weights=relative[:, axis], minlength=number) / counts
        for axis in range(3)
    ])
    relative_max_abs = np.zeros((number, 3), dtype=np.float64)
    for axis in range(3):
        np.maximum.at(relative_max_abs[:, axis], inverse, np.abs(relative[:, axis]))
    avia_count = np.bincount(inverse, weights=(sensor_id == 0), minlength=number).astype(np.int64)
    mid_count = counts - avia_count
    dt_sum = np.bincount(inverse, weights=delta_t, minlength=number)
    dt_min = np.full(number, np.inf)
    dt_max = np.full(number, -np.inf)
    np.minimum.at(dt_min, inverse, delta_t)
    np.maximum.at(dt_max, inverse, delta_t)
    pairs = np.unique(np.column_stack((inverse, frame_index)), axis=0)
    frame_count = np.bincount(pairs[:, 0], minlength=number)
    point_order = np.argsort(inverse, kind="stable")
    point_offsets = np.r_[0, np.cumsum(counts)]
    return {
        "coords": coords, "centers": centers, "inverse": inverse, "counts": counts,
        "point_order": point_order, "point_offsets": point_offsets,
        "relative_xyz": relative, "relative_mean": relative_mean,
        "relative_max_abs": relative_max_abs, "avia_count": avia_count,
        "mid360_count": mid_count, "avia_fraction": avia_count / counts,
        "mid360_fraction": mid_count / counts, "delta_t_mean": dt_sum / counts,
        "delta_t_min": dt_min, "delta_t_max": dt_max, "frame_count": frame_count,
    }


def build_parent(child_coords: np.ndarray) -> dict[str, np.ndarray | int]:
    child_coords = np.asarray(child_coords, dtype=np.int64)
    if not len(child_coords):
        return {
            "coords": np.empty((0, 3), dtype=np.int64), "child_parent": np.empty(0, dtype=np.int64),
            "child_count": np.empty(0, dtype=np.int64), "occupancy_mask": np.empty((0, 8), dtype=bool),
            "child_indices": np.empty(0, dtype=np.int64), "child_offsets": np.zeros(1, dtype=np.int64),
            "error_count": 0,
        }
    parent_per_child = np.floor_divide(child_coords, 2)
    parents, child_parent, child_count = np.unique(
        parent_per_child, axis=0, return_inverse=True, return_counts=True
    )
    local = child_coords - 2 * parents[child_parent]
    invalid_local = np.any((local < 0) | (local > 1), axis=1)
    slots = local[:, 0] * 4 + local[:, 1] * 2 + local[:, 2]
    mask = np.zeros((len(parents), 8), dtype=bool)
    mask[child_parent, slots] = True
    mask_errors = np.count_nonzero(mask.sum(axis=1) != child_count)
    order = np.argsort(child_parent, kind="stable")
    return {
        "coords": parents, "child_parent": child_parent, "child_count": child_count,
        "occupancy_mask": mask, "child_indices": order,
        "child_offsets": np.r_[0, np.cumsum(child_count)],
        "error_count": int(np.count_nonzero(invalid_local) + mask_errors),
    }


def fixed_stat_embedding(level: dict[str, np.ndarray], output_dim: int = 64) -> np.ndarray:
    count = level["counts"]
    features = np.column_stack((
        level["relative_mean"], level["relative_max_abs"], np.log1p(count),
        level["avia_fraction"], level["mid360_fraction"], level["delta_t_mean"],
        level["delta_t_min"], level["delta_t_max"],
    )) if len(count) else np.empty((0, 12), dtype=np.float64)
    rng = np.random.default_rng(20260916)
    weight = rng.standard_normal((12, output_dim), dtype=np.float64) / np.sqrt(12.0)
    return (features @ weight).astype(np.float32)


def packed_offsets(token_counts: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(list(token_counts), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    batch_index = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    return offsets, batch_index


def interpolate_position(
    timestamps: np.ndarray, positions: np.ndarray, query: float,
) -> tuple[np.ndarray, bool]:
    if query < timestamps[0] or query > timestamps[-1]:
        return np.full(3, np.nan), False
    result = np.asarray([np.interp(query, timestamps, positions[:, axis]) for axis in range(3)])
    return result, bool(np.isfinite(result).all())
