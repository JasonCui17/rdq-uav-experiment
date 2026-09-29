#!/usr/bin/env python3
"""Manual training entry for candidate-level Multimodal V2 B2/B3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
from torch.utils.data import Subset
import yaml

from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from rdq_uav.multimodal_v2.data import build_datasets
from rdq_uav.multimodal_v2.training import (
    MultimodalV2DataModule, MultimodalV2LightningModule, build_runtime,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/multimodal_v2/configs/b2_radar_reads_vision.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--e5-visual-checkpoint", type=Path)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "auto"))
    parser.add_argument("--devices")
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulate", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto")
    return parser.parse_args()


def resolve(path):
    return resolve_project_path(path, ROOT)


def main():
    args = parse_args()
    cfg = apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    if args.e5_visual_checkpoint:
        cfg["initialization"]["e5_visual_checkpoint"] = str(args.e5_visual_checkpoint)
    for argument, section, key in (
        (args.num_workers, "data", "num_workers"), (args.batch_size, "training", "batch_size"),
        (args.accumulate, "training", "accumulate"), (args.epochs, "training", "epochs"),
    ):
        if argument is not None:
            cfg[section][key] = argument
    if cfg["experiment"]["stage"] not in {"B2", "B3"}:
        raise ValueError("training entry accepts B2/B3 only; B0/B1 are evaluation gates")
    accelerator = args.accelerator or cfg["lightning"]["accelerator"]
    precision = args.precision or cfg["lightning"]["precision"]
    devices = int(args.devices) if args.devices and args.devices.isdigit() else (args.devices or cfg["lightning"]["devices"])
    output = resolve(args.output or cfg["experiment"]["output_dir"])
    last = output / "checkpoints" / "last.ckpt"
    ckpt = None
    if args.resume:
        ckpt = last if args.resume == "auto" else resolve(args.resume)
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt}")
    elif last.exists() and not args.fast_dev_run:
        raise FileExistsError(f"{last} exists; use --resume auto or a new output")
    L.seed_everything(int(cfg["experiment"]["seed"]), workers=True)
    train_lidar, train_data, val_data = build_datasets(cfg, ROOT)
    if args.train_limit is not None: train_data = Subset(train_data, range(min(args.train_limit, len(train_data))))
    if args.val_limit is not None: val_data = Subset(val_data, range(min(args.val_limit, len(val_data))))
    runtime = build_runtime(cfg, ROOT, torch.device("cpu"))
    module = MultimodalV2LightningModule(runtime, cfg)
    data = MultimodalV2DataModule(
        train_data, val_data, batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
        seed=int(cfg["experiment"]["seed"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    callbacks = [] if args.fast_dev_run else [
        ModelCheckpoint(dirpath=output / "checkpoints", save_last=True, save_top_k=0,
                        every_n_train_steps=int(cfg["lightning"]["checkpoint_every_updates"]),
                        save_on_exception=True),
        ModelCheckpoint(dirpath=output / "checkpoints", filename="best", save_top_k=1,
                        monitor="val/success_1m", mode="max", every_n_epochs=1),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = False if args.fast_dev_run else [
        CSVLogger(output / "logs", name="csv"), TensorBoardLogger(output / "logs", name="tensorboard")
    ]
    trainer = L.Trainer(
        default_root_dir=output, accelerator=accelerator, devices=devices,
        precision=precision, max_epochs=int(cfg["training"]["epochs"]),
        max_steps=-1 if args.max_updates is None else int(args.max_updates),
        accumulate_grad_batches=int(cfg["training"]["accumulate"]),
        gradient_clip_val=float(cfg["training"]["grad_clip_norm"]),
        callbacks=callbacks, logger=logger,
        log_every_n_steps=int(cfg["logging"]["every_updates"]),
        num_sanity_val_steps=int(cfg["lightning"]["num_sanity_val_steps"]),
        deterministic=cfg["lightning"]["deterministic"], fast_dev_run=args.fast_dev_run,
        limit_val_batches=0 if args.max_updates is not None else 1.0,
    )
    cfg["runtime"] = {
        "framework": "lightning", "train_queries": len(train_data), "val_queries": len(val_data),
        "precision": precision, "accelerator": accelerator, "devices": devices,
        "max_updates": args.max_updates, "old_pre_stage_hci": False,
    }
    if not args.fast_dev_run:
        (output / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (output / "effective_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    trainer.fit(module, datamodule=data, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
