#!/usr/bin/env python3
"""Independent PyTorch Lightning entry point for frozen Multimodal V1 E5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.plugins.precision import MixedPrecision
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/multimodal_v1/e5_annotated20_lightning.yaml"
# When this file is executed directly, Python adds ``tools/`` rather than the
# repository root to sys.path.  Both are needed: ``src`` contains the package,
# while the unchanged reference E5 implementation intentionally remains in
# ``tools/train_multimodal_v1_full.py`` and is reused as the equivalence seam.
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path


def resolve(value: str | Path) -> Path:
    return resolve_project_path(value, ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--devices", default=None)
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    return parser.parse_args()


def build_datasets(cfg: dict[str, Any], train_limit: int | None, val_limit: int | None):
    from rdq_uav.lidar_v2.data import LiDARUAVDataset
    from rdq_uav.multimodal_v1.data import LeftImageIndex
    from rdq_uav.multimodal_v1.vision.ssod_data import load_label_manifest
    from tools.train_multimodal_v1_full import E5QueryDataset

    data = cfg["data"]
    root, split_file = resolve(data["root"]), resolve(data["split_file"])
    train_lidar = LiDARUAVDataset(root, split_file, data["train_split"], max_events=int(data["max_events"]))
    val_lidar = LiDARUAVDataset(root, split_file, data["val_split"], max_events=int(data["max_events"]))
    train_sequences = {str(record["sequence_id"]) for record in train_lidar.records}
    val_sequences = {str(record["sequence_id"]) for record in val_lidar.records}
    overlap = train_sequences & val_sequences
    if overlap:
        raise RuntimeError(
            f"sequence-level train/validation leakage detected ({len(overlap)} sequences)"
        )
    geometry = json.loads(resolve(data["geometry_calibration"]).read_text())
    image_index = LeftImageIndex(
        root,
        time_offset_s=float(geometry["time_offset_s"]),
        max_abs_gap_s=float(data["max_image_gap_s"]),
    )
    manifest = load_label_manifest(resolve(data["annotation_manifest"]), require_boxes=False)
    camera_cfg = yaml.safe_load(resolve(data["camera_config"]).read_text())
    camera_wh = tuple(map(int, camera_cfg["cameras"]["left"]["resolution"]))
    image_args = {
        "camera_wh": camera_wh,
        "short_edge": int(data["dino_short_edge"]),
        "max_size": int(data["dino_max_size"]),
    }
    train_dataset = E5QueryDataset(
        train_lidar, image_index, manifest,
        train_sequences,
        data["geometry_calibration"], **image_args,
    )
    val_dataset = E5QueryDataset(
        val_lidar, image_index, manifest,
        val_sequences,
        data["geometry_calibration"], **image_args,
    )
    if train_limit is not None:
        train_dataset.indices = train_dataset.indices[:train_limit]
        train_dataset.matches = train_dataset.matches[:train_limit]
    if val_limit is not None:
        val_dataset.indices = val_dataset.indices[:val_limit]
        val_dataset.matches = val_dataset.matches[:val_limit]
    return train_lidar, train_dataset, val_dataset


def parse_devices(raw: str | None):
    if raw is None:
        return 1
    if raw.isdigit():
        return int(raw)
    return [int(item) for item in raw.split(",")]


def build_precision_plugin(
    precision: str,
    accelerator: str,
    lightning_cfg: dict[str, Any],
) -> str | MixedPrecision:
    """Build the explicit FP16 scaler used by the verified 8 GB GPU path.

    PyTorch's default initial scale (65536) overflowed the first six optimizer
    attempts of the real E5 gate. Lightning advanced ``global_step`` and the
    scheduler for those skipped attempts. Starting from the observed stable
    scale avoids silently losing the beginning of the warmup schedule. Native
    Lightning resume restores the checkpointed scaler over this initial value.
    """

    if precision != "16-mixed" or accelerator == "cpu":
        return precision
    initial_scale = float(lightning_cfg.get("amp_initial_scale", 1024.0))
    if initial_scale <= 0:
        raise ValueError("lightning.amp_initial_scale must be positive")
    scaler = torch.cuda.amp.GradScaler(init_scale=initial_scale)
    return MixedPrecision(precision="16-mixed", device="cuda", scaler=scaler)


def main() -> None:
    args = parse_args()
    cfg = apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    lightning_cfg = cfg["lightning"]
    accelerator = args.accelerator or lightning_cfg["accelerator"]
    precision = args.precision or lightning_cfg["precision"]
    precision_setting = build_precision_plugin(precision, accelerator, lightning_cfg)
    precision_plugins = [precision_setting] if isinstance(precision_setting, MixedPrecision) else None
    trainer_precision = None if precision_plugins else precision_setting
    devices = parse_devices(args.devices) if args.devices is not None else lightning_cfg["devices"]
    output = resolve(args.output or cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    native_last = output / "checkpoints" / "last.ckpt"
    ckpt_path = None
    if args.resume:
        ckpt_path = native_last if args.resume == "auto" else resolve(args.resume)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Lightning resume checkpoint not found: {ckpt_path}")
    elif native_last.exists() and not args.fast_dev_run:
        raise FileExistsError(f"{native_last} exists; pass --resume auto instead of overwriting")

    seed = int(cfg["experiment"]["seed"])
    L.seed_everything(seed, workers=True)
    train_lidar, train_dataset, val_dataset = build_datasets(cfg, args.train_limit, args.val_limit)

    from rdq_uav.multimodal_v1.lightning_system import (
        LexicographicBestCheckpoint,
        MultimodalV1DataModule,
        MultimodalV1LightningModule,
        PeakMemoryMonitor,
        StopAfterEpoch,
        validate_lightning_config,
    )
    from tools.train_multimodal_v1_full import build_runtime, collate_e5, collate_e5_train

    validate_lightning_config(cfg)
    pin_memory = accelerator != "cpu" and torch.cuda.is_available()
    datamodule = MultimodalV1DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_collate=collate_e5_train,
        val_collate=collate_e5,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
        seed=seed,
        pin_memory=pin_memory,
        gate_order=args.max_updates is not None or args.fast_dev_run,
    )
    runtime = build_runtime(cfg, train_lidar, torch.device("cpu"))
    module = MultimodalV1LightningModule(runtime, cfg)
    if args.legacy_checkpoint is not None:
        module.load_legacy_weights(resolve(args.legacy_checkpoint))

    checkpoint_dir = output / "checkpoints"
    callbacks: list[Any] = [PeakMemoryMonitor()]
    if args.stop_after_epoch is not None:
        callbacks.append(StopAfterEpoch(args.stop_after_epoch))
    if not args.fast_dev_run:
        checkpoint_every_updates = int(lightning_cfg.get("checkpoint_every_updates", 250))
        if checkpoint_every_updates <= 0:
            raise ValueError("lightning.checkpoint_every_updates must be positive")
        callbacks.extend([
            # A full epoch is roughly two hours on the target RTX 3070. Keep a
            # rolling intra-epoch recovery point and also save immediately if
            # Lightning catches an exception.
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_last=True,
                save_top_k=0,
                every_n_train_steps=checkpoint_every_updates,
                save_on_exception=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ])
        if args.max_updates is None:
            callbacks.append(LexicographicBestCheckpoint(checkpoint_dir / "best.ckpt"))
    early = lightning_cfg["early_stopping"]
    if bool(early["enabled"]) and not args.fast_dev_run and args.max_updates is None:
        callbacks.append(EarlyStopping(
            monitor="val/final_3d_success_1m",
            mode="max",
            patience=int(early["patience"]),
            min_delta=float(early["min_delta"]),
            check_finite=True,
        ))
    loggers: list[Any] | bool = False if args.fast_dev_run else [
        CSVLogger(save_dir=output / "logs", name="csv"),
        TensorBoardLogger(save_dir=output / "logs", name="tensorboard"),
    ]
    max_steps = int(args.max_updates) if args.max_updates is not None else -1
    trainer = L.Trainer(
        default_root_dir=output,
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(cfg["training"]["epochs"]),
        max_steps=max_steps,
        precision=trainer_precision,
        plugins=precision_plugins,
        accumulate_grad_batches=int(cfg["training"]["accumulate"]),
        gradient_clip_val=float(cfg["training"]["grad_clip_norm"]),
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=int(cfg["validation"]["every_epochs"]),
        deterministic=lightning_cfg["deterministic"],
        benchmark=False,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=int(cfg["logging"]["log_every_updates"]),
        num_sanity_val_steps=int(lightning_cfg["num_sanity_val_steps"]),
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=not args.fast_dev_run,
        limit_val_batches=0 if args.max_updates is not None else 1.0,
    )
    cfg["runtime"] = {
        "framework": "lightning",
        "lightning_version": L.__version__,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "train_queries": len(train_dataset),
        "val_queries": len(val_dataset),
        "max_updates": args.max_updates,
    }
    if not args.fast_dev_run:
        (output / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (output / "effective_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
