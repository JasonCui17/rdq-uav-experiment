from __future__ import annotations

from pathlib import Path
from typing import Any

from rdq_uav.data.localization import MMAUDLocalizationDataset
from rdq_uav.experiment import make_loader


def make_localization_dataset(
    config: dict[str, Any],
    split: str,
    position_stats: dict[str, Any],
    limit_samples: int | None = None,
    radar_mode: str = "normal",
    radar_shift_seconds: float = 0.0,
) -> MMAUDLocalizationDataset:
    if split not in {"train", "val"}:
        raise ValueError("Stage-4 localization code is restricted to train/val")
    data_cfg = config["data"]
    manifest_dir = Path(data_cfg["manifest_dir"])
    dataset = MMAUDLocalizationDataset(
        manifest_path=manifest_dir / f"{split}.csv",
        root=data_cfg["root"],
        radar_stats_path=manifest_dir / "radar_stats.json",
        image_size=data_cfg["image_size"],
        image_mode=str(data_cfg["image_mode"]),
        center_mask_fraction=float(data_cfg.get("center_mask_fraction", 0.0)),
        bbox_mode=str(data_cfg["bbox_mode"]),
        bbox_context_scale=float(data_cfg.get("bbox_context_scale", 1.0)),
        max_radar_points=int(data_cfg["radar"]["max_points"]),
        max_radar_range_m=float(data_cfg["radar"]["max_range_m"]),
        training=split == "train",
        color_jitter=float(data_cfg["train_color_jitter"]) if split == "train" else 0.0,
        deterministic_eval_sampling=bool(data_cfg["radar"]["deterministic_eval_sampling"]),
        radar_mode=radar_mode,
        radar_shift_seconds=radar_shift_seconds,
        image_decode_retries=int(data_cfg.get("image_decode_retries", 3)),
        position_stats=position_stats,
        panorama_size=data_cfg["panorama_size"],
        expected_panorama_size=data_cfg["panorama_size"],
    )
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("limit_samples must be positive")
        dataset.rows = dataset.rows[:limit_samples]
        dataset.radar_indices = list(range(len(dataset.rows)))
    return dataset


__all__ = ["make_localization_dataset", "make_loader"]
