#!/usr/bin/env python3
"""Evaluate one Lightning Multimodal V1 checkpoint in verified FP32 mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def _devices(value: str):
    return int(value) if value.isdigit() else [int(item) for item in value.split(",")]


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config, ROOT)
    checkpoint_path = resolve_project_path(args.checkpoint, ROOT)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = apply_runtime_path_overrides(yaml.safe_load(config_path.read_text()))
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    cfg["validation"]["precision"] = "fp32"

    from tools.train_multimodal_v1_lightning import build_datasets
    from tools.train_multimodal_v1_full import build_runtime, collate_e5, collate_e5_train
    from rdq_uav.multimodal_v1.lightning_system import (
        MultimodalV1DataModule,
        MultimodalV1LightningModule,
    )

    L.seed_everything(int(cfg["experiment"]["seed"]), workers=True)
    train_lidar, train_dataset, val_dataset = build_datasets(
        cfg, train_limit=1, val_limit=args.val_limit
    )
    data = cfg["data"]
    datamodule = MultimodalV1DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_collate=collate_e5_train,
        val_collate=collate_e5,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(data["num_workers"]),
        prefetch_factor=int(data.get("prefetch_factor", 2)),
        seed=int(cfg["experiment"]["seed"]),
        pin_memory=args.accelerator == "gpu" and torch.cuda.is_available(),
    )
    runtime = build_runtime(cfg, train_lidar, torch.device("cpu"))
    module = MultimodalV1LightningModule(runtime, cfg)
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=_devices(args.devices),
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        deterministic=cfg["lightning"]["deterministic"],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )
    results = trainer.validate(
        module, datamodule=datamodule, ckpt_path=checkpoint_path, verbose=False
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one validation result, got {len(results)}")
    metrics = {}
    for key, value in results[0].items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(f"validation metric {key!r} is not scalar: {value.shape}")
            value = value.detach().cpu().item()
        if not isinstance(value, (int, float)):
            raise TypeError(f"validation metric {key!r} is not numeric: {type(value).__name__}")
        metrics[key] = float(value)
    output = resolve_project_path(args.output, ROOT)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "validation_queries": len(val_dataset),
        "precision": "32-true",
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    lines = [
        "# Multimodal V1 evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Validation queries: {len(val_dataset)}",
        "- Precision: FP32",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value:.8g} |" for key, value in sorted(metrics.items()))
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
