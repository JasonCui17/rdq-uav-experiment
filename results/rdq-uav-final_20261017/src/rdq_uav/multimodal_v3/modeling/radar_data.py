"""Radar frame reading used by the standalone V2 dataset."""
from __future__ import annotations
from dataclasses import dataclass
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
    return sorted(events, key=lambda event: (event.timestamp, event.sensor_id, event.sensor_name, str(event.file_path)))

def load_released_xyz(path: Path) -> tuple[np.ndarray, int, int]:
    raw = np.asarray(np.load(path, allow_pickle=False))
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"Expected [N,>=3] point cloud at {path}, got {raw.shape}")
    xyz = np.asarray(raw[:, :3], dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    return xyz[valid], int(len(xyz)), int(np.count_nonzero(~valid))
