from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rdq_uav.data.transforms import DualFisheyeTransform
from rdq_uav.utils.seed import seed_worker


class RadarProcessor:
    def __init__(
        self,
        max_points: int,
        max_range_m: float,
        mean: list[float],
        std: list[float],
        training: bool,
        deterministic_eval_sampling: bool,
    ) -> None:
        self.max_points = int(max_points)
        self.max_range_m = float(max_range_m)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.training = training
        self.deterministic_eval_sampling = deterministic_eval_sampling
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("Radar normalization mean/std must each have 3 values")
        if np.any(self.std <= 0):
            raise ValueError("Radar normalization std must be positive")

    def _rng(self, sample_id: str) -> np.random.Generator:
        if self.training or not self.deterministic_eval_sampling:
            return np.random.default_rng(np.random.randint(0, 2**32 - 1))
        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def __call__(self, array: np.ndarray, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if array.size == 0:
            array = np.empty((0, 3), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"Expected radar (N,3), got {array.shape}")
        array = np.asarray(array, dtype=np.float32)
        valid = np.isfinite(array).all(axis=1)
        array = array[valid]
        array = array[np.linalg.norm(array, axis=1) <= self.max_range_m]
        if len(array) > self.max_points:
            indices = self._rng(sample_id).choice(len(array), self.max_points, replace=False)
            array = array[indices]
        count = len(array)
        output = np.zeros((self.max_points, 3), dtype=np.float32)
        mask = np.zeros(self.max_points, dtype=bool)
        if count:
            output[:count] = (array - self.mean) / self.std
            mask[:count] = True
        return torch.from_numpy(output), torch.from_numpy(mask)


class MMAUDDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        radar_stats_path: str | Path,
        image_size: list[int],
        image_mode: str,
        center_mask_fraction: float,
        bbox_mode: str,
        bbox_context_scale: float,
        max_radar_points: int,
        max_radar_range_m: float,
        training: bool,
        color_jitter: float = 0.0,
        deterministic_eval_sampling: bool = True,
        radar_mode: str = "normal",
        radar_shift_seconds: float = 0.0,
        image_decode_retries: int = 3,
        limit_per_class: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = Path(root).expanduser().resolve()
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if limit_per_class is not None:
            kept: dict[int, int] = defaultdict(int)
            limited = []
            for row in rows:
                label = int(row["class_id"])
                if kept[label] < limit_per_class:
                    limited.append(row)
                    kept[label] += 1
            rows = limited
        if not rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_path}")
        self.rows = rows
        self.image_decode_retries = max(1, int(image_decode_retries))
        stats = json.loads(Path(radar_stats_path).read_text(encoding="utf-8"))
        if abs(float(stats["max_range_m"]) - float(max_radar_range_m)) > 1e-9:
            raise ValueError("Configured radar max_range_m differs from normalization stats")
        self.image_transform = DualFisheyeTransform(
            image_size=image_size,
            training=training,
            color_jitter=color_jitter,
            image_mode=image_mode,
            center_mask_fraction=center_mask_fraction,
            bbox_mode=bbox_mode,
            bbox_context_scale=bbox_context_scale,
        )
        self.radar_processor = RadarProcessor(
            max_points=max_radar_points,
            max_range_m=max_radar_range_m,
            mean=stats["mean"],
            std=stats["std"],
            training=training,
            deterministic_eval_sampling=deterministic_eval_sampling,
        )
        if radar_mode not in {"normal", "zero", "shuffle_same_class", "shift"}:
            raise ValueError(f"Unsupported radar_mode: {radar_mode}")
        if radar_mode == "shift" and radar_shift_seconds == 0:
            raise ValueError("radar_shift_seconds must be non-zero for radar_mode=shift")
        self.radar_mode = radar_mode
        self.radar_indices = list(range(len(rows)))
        if radar_mode == "shuffle_same_class":
            by_class: dict[int, list[int]] = defaultdict(list)
            for index, row in enumerate(rows):
                by_class[int(row["class_id"])].append(index)
            for indices in by_class.values():
                if len(indices) < 2:
                    raise ValueError("Need at least two samples per class for radar shuffle")
                # A half-sequence cyclic shift avoids accidental near-time pairs
                # while preserving the class label.
                offset = max(1, len(indices) // 2)
                shifted = indices[offset:] + indices[:offset]
                for target, source in zip(indices, shifted):
                    self.radar_indices[target] = source
        elif radar_mode == "shift":
            by_class = defaultdict(list)
            for index, row in enumerate(rows):
                by_class[int(row["class_id"])].append(index)
            for indices in by_class.values():
                times = np.asarray([float(rows[index]["gt_time"]) for index in indices])
                for target in indices:
                    desired = float(rows[target]["gt_time"]) + radar_shift_seconds
                    position = int(np.searchsorted(times, desired))
                    candidates = []
                    if position < len(indices):
                        candidates.append(position)
                    if position > 0:
                        candidates.append(position - 1)
                    nearest = min(candidates, key=lambda item: abs(times[item] - desired))
                    self.radar_indices[target] = indices[nearest]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def labels(self) -> list[int]:
        return [int(row["class_id"]) for row in self.rows]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        last_image_error: Exception | None = None
        for attempt in range(self.image_decode_retries):
            try:
                with Image.open(image_path) as image:
                    # Force complete decoding while the file is open. Reopening
                    # on retry handles transient concurrent I/O/Pillow failures.
                    image.load()
                    image_tensor = self.image_transform(image, row)
                break
            except (OSError, ValueError) as exc:
                last_image_error = exc
                if attempt + 1 < self.image_decode_retries:
                    time.sleep(0.05 * (attempt + 1))
        else:
            raise OSError(
                f"Failed to decode image after {self.image_decode_retries} attempts: "
                f"sample_id={row['sample_id']} path={image_path}"
            ) from last_image_error

        radar_row = self.rows[self.radar_indices[index]]
        radar_path = self.root / radar_row["radar_path"]
        radar_array = np.load(radar_path, allow_pickle=False)
        radar, radar_mask = self.radar_processor(radar_array, row["sample_id"])
        if self.radar_mode == "zero":
            radar.zero_()
            radar_mask.zero_()

        gt_position = torch.tensor(
            [float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])],
            dtype=torch.float32,
        )
        return {
            "image": image_tensor,
            "radar": radar,
            "radar_mask": radar_mask,
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "position": gt_position,
            "distance": torch.tensor(float(row["distance_m"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "class_name": row["class_name"],
            "sequence_id": row["sequence_id"],
            "temporal_block": torch.tensor(int(row["temporal_block"]), dtype=torch.long),
            "gt_time": torch.tensor(float(row["gt_time"]), dtype=torch.float64),
            "bbox_area_px": torch.tensor(
                float(row.get("official_bbox_width_px", 0.0))
                * float(row.get("official_bbox_height_px", 0.0)),
                dtype=torch.float32,
            ),
        }


def build_dataloader(
    dataset: MMAUDDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
    sampler: torch.utils.data.Sampler[int] | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
