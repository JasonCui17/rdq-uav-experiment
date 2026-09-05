from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from rdq_uav.data.dataset import MMAUDDataset, build_dataloader


def make_dataset(
    config: dict[str, Any],
    split: str,
    radar_mode: str = "normal",
    radar_shift_seconds: float = 0.0,
    limit_per_class: int | None = None,
) -> MMAUDDataset:
    data_cfg = config["data"]
    manifest_dir = Path(data_cfg["manifest_dir"])
    return MMAUDDataset(
        manifest_path=manifest_dir / f"{split}.csv",
        root=data_cfg["root"],
        radar_stats_path=manifest_dir / "radar_stats.json",
        image_size=data_cfg["image_size"],
        image_mode=str(data_cfg.get("image_mode", "dual_full")),
        center_mask_fraction=float(data_cfg.get("center_mask_fraction", 0.0)),
        bbox_mode=str(data_cfg.get("bbox_mode", "full")),
        bbox_context_scale=float(data_cfg.get("bbox_context_scale", 1.0)),
        max_radar_points=int(data_cfg["radar"]["max_points"]),
        max_radar_range_m=float(data_cfg["radar"]["max_range_m"]),
        training=split == "train",
        color_jitter=float(data_cfg["train_color_jitter"]) if split == "train" else 0.0,
        deterministic_eval_sampling=bool(
            data_cfg["radar"]["deterministic_eval_sampling"]
        ),
        radar_mode=radar_mode,
        radar_shift_seconds=radar_shift_seconds,
        image_decode_retries=int(data_cfg.get("image_decode_retries", 3)),
        limit_per_class=limit_per_class,
    )


def make_loader(
    config: dict[str, Any],
    dataset: MMAUDDataset,
    split: str,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    data_cfg = config["data"]
    sampler = None
    shuffle = split == "train"
    if split == "train" and bool(config["train"]["class_balanced_sampler"]):
        counts = Counter(dataset.labels)
        weights = torch.tensor([1.0 / counts[label] for label in dataset.labels])
        generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
        shuffle = False
    num_workers = (
        int(data_cfg["num_workers"])
        if split == "train"
        else int(data_cfg.get("eval_num_workers", 0))
    )
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(data_cfg["pin_memory"]),
        persistent_workers=bool(data_cfg["persistent_workers"]),
        seed=int(config["experiment"]["seed"]),
        sampler=sampler,
    )
