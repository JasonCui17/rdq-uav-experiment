from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class PositionTrajectory:
    timestamps: np.ndarray
    positions: np.ndarray

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.timestamps.ndim != 1 or self.positions.shape != (len(self.timestamps), 3):
            raise ValueError("Trajectory must have timestamps (N,) and positions (N,3)")
        if len(self.timestamps) < 2 or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Trajectory timestamps must be strictly increasing")
        if not np.isfinite(self.positions).all():
            raise ValueError("Trajectory contains non-finite positions")
        relative_time = self.timestamps - self.timestamps[0]
        self._interpolator = PchipInterpolator(relative_time, self.positions, axis=0, extrapolate=False)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "PositionTrajectory":
        files = sorted(Path(directory).glob("*.npy"), key=lambda path: float(path.stem))
        if len(files) < 2:
            raise ValueError(f"Need at least two GT files in {directory}")
        timestamps = np.asarray([float(path.stem) for path in files], dtype=np.float64)
        positions = np.stack(
            [np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(3) for path in files]
        )
        return cls(timestamps, positions)

    def evaluate(self, query_timestamps: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_timestamps, dtype=np.float64)
        positions = np.asarray(self._interpolator(query - self.timestamps[0]), dtype=np.float64)
        valid = (query >= self.timestamps[0]) & (query <= self.timestamps[-1])
        valid &= np.isfinite(positions).all(axis=-1)
        return positions, valid

    def speed(self, query_timestamps: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_timestamps, dtype=np.float64)
        velocity = np.asarray(
            self._interpolator.derivative()(query - self.timestamps[0]), dtype=np.float64
        )
        valid = (query >= self.timestamps[0]) & (query <= self.timestamps[-1])
        valid &= np.isfinite(velocity).all(axis=-1)
        return velocity, valid
