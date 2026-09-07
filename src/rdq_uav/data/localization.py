from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rdq_uav.data.dataset import MMAUDDataset


def compute_position_stats(manifest_path: str | Path) -> dict[str, Any]:
    """Compute XYZ normalization from one explicitly supplied training manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    positions = np.asarray(
        [[float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(positions).all():
        raise ValueError("Position statistics contain non-finite GT coordinates")
    std = positions.std(axis=0, ddof=0)
    if np.any(std <= 0):
        raise ValueError(f"Position standard deviation must be positive, got {std.tolist()}")
    return {
        "mean": positions.mean(axis=0).tolist(),
        "std": std.tolist(),
        "count": int(len(positions)),
        "computed_from_split": "train",
        "manifest_path": str(manifest_path),
        "standard_deviation": "population_ddof_0",
    }


def compute_bbox_stats(
    manifest_path: str | Path, panorama_size: list[int]
) -> dict[str, Any]:
    """Compute normalized bbox size statistics from the full training manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if len(panorama_size) != 2:
        raise ValueError("panorama_size must be [height, width]")
    panorama_height, panorama_width = (int(value) for value in panorama_size)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    widths = np.asarray(
        [
            (float(row["official_bbox_x2"]) - float(row["official_bbox_x1"]))
            / panorama_width
            for row in rows
        ],
        dtype=np.float64,
    )
    heights = np.asarray(
        [
            (float(row["official_bbox_y2"]) - float(row["official_bbox_y1"]))
            / panorama_height
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(widths).all() or not np.isfinite(heights).all():
        raise ValueError("BBox statistics contain non-finite sizes")
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("BBox width/height must be positive")
    return {
        "width_mean": float(widths.mean()),
        "width_median": float(np.median(widths)),
        "width_std": float(widths.std(ddof=0)),
        "height_mean": float(heights.mean()),
        "height_median": float(np.median(heights)),
        "height_std": float(heights.std(ddof=0)),
        "num_train_samples": int(len(rows)),
        "computed_from_split": "train",
        "manifest_path": str(manifest_path),
        "bbox_format": "normalized_cxcywh_on_full_stitched_panorama",
        "panorama_size": [panorama_height, panorama_width],
        "standard_deviation": "population_ddof_0",
    }


class MMAUDLocalizationDataset(MMAUDDataset):
    """Train/val single-UAV localization dataset on fixed full dual-fisheye input."""

    def __init__(
        self,
        *args: Any,
        position_stats: dict[str, Any],
        panorama_size: list[int],
        **kwargs: Any,
    ) -> None:
        if str(kwargs.get("image_mode", "dual_full")) != "dual_full":
            raise ValueError(
                "Localization requires image_mode=dual_full; GT-derived oracle ROI modes leak bbox location"
            )
        if str(kwargs.get("bbox_mode", "full")) != "full":
            raise ValueError("Localization requires bbox_mode=full")
        super().__init__(*args, **kwargs)
        if len(panorama_size) != 2:
            raise ValueError("panorama_size must be [height, width]")
        self.panorama_height = int(panorama_size[0])
        self.panorama_width = int(panorama_size[1])
        if self.panorama_height <= 0 or self.panorama_width <= 0:
            raise ValueError("panorama dimensions must be positive")
        self.position_mean = torch.tensor(position_stats["mean"], dtype=torch.float32)
        self.position_std = torch.tensor(position_stats["std"], dtype=torch.float32)
        if self.position_mean.shape != (3,) or self.position_std.shape != (3,):
            raise ValueError("Position normalization mean/std must each have 3 values")
        if not bool(torch.all(self.position_std > 0)):
            raise ValueError("Position normalization std must be positive")

    def _normalized_bbox(self, row: dict[str, str]) -> torch.Tensor:
        keys = ("official_bbox_x1", "official_bbox_y1", "official_bbox_x2", "official_bbox_y2")
        if any(not row.get(key, "") for key in keys):
            raise ValueError(f"Missing official bbox columns for {row.get('sample_id', 'unknown')}")
        x1, y1, x2, y2 = (float(row[key]) for key in keys)
        if not (0.0 <= x1 < x2 <= self.panorama_width):
            raise ValueError(f"Invalid bbox x coordinates: {(x1, x2)}")
        if not (0.0 <= y1 < y2 <= self.panorama_height):
            raise ValueError(f"Invalid bbox y coordinates: {(y1, y2)}")
        return torch.tensor(
            [
                (x1 + x2) / (2.0 * self.panorama_width),
                (y1 + y2) / (2.0 * self.panorama_height),
                (x2 - x1) / self.panorama_width,
                (y2 - y1) / self.panorama_height,
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        position = sample["position"]
        assert isinstance(position, torch.Tensor)
        sample["bbox"] = self._normalized_bbox(self.rows[index])
        sample["position_normalized"] = (position - self.position_mean) / self.position_std
        return sample
