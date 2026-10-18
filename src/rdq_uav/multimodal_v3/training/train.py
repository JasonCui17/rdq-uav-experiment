#!/usr/bin/env python3
"""Train the standalone radar, vision, or multimodal V2 stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
from torch.utils.data import Subset
import yaml

from rdq_uav.multimodal_v3.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from rdq_uav.multimodal_v3.modeling.data import build_datasets
from rdq_uav.multimodal_v3.modeling.training import (
    MultimodalV2DataModule, MultimodalV2LightningModule, build_runtime,
)
from rdq_uav.multimodal_v3.modeling.monitoring import StageProgressBar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/final/radar.yaml"))
    parser.add_argument("--output", type=Path)
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
    parser.add_argument("--train-indices", type=int, nargs="+")
    parser.add_argument("--val-indices", type=int, nargs="+")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--val-check-interval", type=float, help="Validation frequency within each epoch, e.g. 0.25")
    parser.add_argument("--ap-every-val-batches", type=int, help="Refresh provisional COCO AP every N validation batches")

    parser.add_argument("--early-stopping-patience", type=int)

    parser.add_argument("--early-stopping-baseline", type=float)
    return parser.parse_args()


def resolve(path):
    return resolve_project_path(path, ROOT)


def main(*, runtime_builder=None, train_dataset_transform=None):
    args = parse_args()
    cfg = apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    for argument, section, key in (
        (args.num_workers, "data", "num_workers"), (args.batch_size, "training", "batch_size"),
        (args.accumulate, "training", "accumulate"), (args.epochs, "training", "epochs"),
    ):
        if argument is not None:
            cfg[section][key] = argument
    stage = str(cfg["experiment"]["stage"])
    if args.val_check_interval is not None:
        if not 0 < args.val_check_interval <= 1:
            raise ValueError("val-check-interval must be in (0,1]")
        cfg["lightning"]["val_check_interval"] = args.val_check_interval
    if args.ap_every_val_batches is not None:
        if args.ap_every_val_batches < 1:
            raise ValueError("ap-every-val-batches must be positive")
        cfg["logging"]["ap_every_val_batches"] = args.ap_every_val_batches
    if stage not in {"B0", "B1", "B2", "B3", "B4"}:
        raise ValueError(f"unknown training stage {stage}")
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
    train_data, val_data = build_datasets(cfg, ROOT)
    if train_dataset_transform is not None:
        train_data = train_dataset_transform(train_data, cfg)
    if args.train_indices is not None:
        if args.train_limit is not None or any(i < 0 or i >= len(train_data) for i in args.train_indices):
            raise ValueError("--train-indices must be in range and cannot be combined with --train-limit")
        train_data = Subset(train_data, args.train_indices)
    if args.val_indices is not None:
        if args.val_limit is not None or any(i < 0 or i >= len(val_data) for i in args.val_indices):
            raise ValueError("--val-indices must be in range and cannot be combined with --val-limit")
        val_data = Subset(val_data, args.val_indices)
    if args.train_limit is not None: train_data = Subset(train_data, range(min(args.train_limit, len(train_data))))
    if args.val_limit is not None: val_data = Subset(val_data, range(min(args.val_limit, len(val_data))))
    builder = build_runtime if runtime_builder is None else runtime_builder
    runtime = builder(cfg, ROOT, torch.device("cpu"), initialize_from_baselines=ckpt is None)
    module = MultimodalV2LightningModule(runtime, cfg)
    data = MultimodalV2DataModule(
        train_data, val_data, batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
        seed=int(cfg["experiment"]["seed"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_cfg = cfg.get("checkpoint", {})
    monitor = checkpoint_cfg.get("monitor", "val/rmse_3d_m" if stage == "B0" else ("val/AP" if stage == "B1" else "val/success_1m"))
    monitor_mode = checkpoint_cfg.get("mode", "min" if stage == "B0" else "max")
    filename = checkpoint_cfg.get("filename", "best_rmse" if stage == "B0" else "best")
    if monitor_mode not in {"min", "max"}:
        raise ValueError("checkpoint.mode must be min or max")
    callbacks = [] if args.fast_dev_run else [
        ModelCheckpoint(dirpath=output / "checkpoints", save_last=True, save_top_k=0,
                        every_n_epochs=1,
                        save_on_exception=True),
        ModelCheckpoint(dirpath=output / "checkpoints", filename=filename, save_top_k=1,
                        monitor=monitor, mode=monitor_mode, every_n_epochs=1),
        LearningRateMonitor(logging_interval="step"),
    ]

    if args.early_stopping_patience is not None and not args.fast_dev_run:
        from lightning.pytorch.callbacks import EarlyStopping

        if args.early_stopping_patience < 1:
            raise ValueError("early-stopping-patience must be positive")
        if args.max_updates is not None:
            raise ValueError("early stopping requires validation; remove --max-updates")

        early_stop = EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=args.early_stopping_patience,
            min_delta=0.0,
            check_on_train_epoch_end=False,
            verbose=True,
        )

        # 首次添加早停时，从恢复 checkpoint 中读取已有最佳成绩。
        # 如果 checkpoint 已有 EarlyStopping 状态，Lightning 会恢复该状态。
        if ckpt is not None:
            previous = torch.load(ckpt, map_location="cpu", weights_only=False)
            scores = [
                state["best_model_score"]
                for key, state in previous.get("callbacks", {}).items()
                if "ModelCheckpoint" in str(key)
                and monitor in str(key)
                and state.get("best_model_score") is not None
            ]
            if scores:
                early_stop.best_score = torch.tensor(
                    (min if monitor_mode == "min" else max)(float(score) for score in scores)
                )

        if args.early_stopping_baseline is not None:
            import math
            if not math.isfinite(args.early_stopping_baseline):
                raise ValueError("early-stopping-baseline must be finite")
            early_stop.best_score = torch.tensor(args.early_stopping_baseline)
        callbacks.append(early_stop)

    if cfg.get("efficiency", {}).get("enabled", False):
        if stage != "B0":
            raise ValueError("efficiency pilot currently measures B0 only")
        from rdq_uav.multimodal_v3.modeling.efficiency import EfficiencyRecorder
        callbacks.append(EfficiencyRecorder(output / "efficiency"))
    callbacks.append(StageProgressBar(refresh_rate=1))
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
        val_check_interval=float(cfg["lightning"].get("val_check_interval", 1.0)),
    )
    cfg["runtime"] = {
        "framework": "lightning", "train_queries": len(train_data), "val_queries": len(val_data),
        "precision": precision, "accelerator": accelerator, "devices": devices,
        "max_updates": args.max_updates,
        "radar_architecture": runtime.initialization.get("radar_architecture"),
    }
    if not args.fast_dev_run:
        (output / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (output / "effective_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    trainer.fit(module, datamodule=data, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
