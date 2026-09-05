#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.engine.trainer import run_epoch  # noqa: E402
from rdq_uav.experiment import make_dataset, make_loader  # noqa: E402
from rdq_uav.models import build_model, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import atomic_torch_save, write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled MMAUD fusion baselines")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--limit-per-class", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    train_dataset = make_dataset(config, "train", limit_per_class=args.limit_per_class)
    val_dataset = make_dataset(config, "val", limit_per_class=args.limit_per_class)
    train_loader = make_loader(
        config, train_dataset, "train", int(config["train"]["batch_size"])
    )
    val_loader = make_loader(
        config, val_dataset, "val", int(config["evaluation"]["batch_size"])
    )
    resume = config["train"].get("resume")
    model = build_model(
        config["model"], load_backbone_pretrained=not bool(resume)
    ).to(device)
    parameter_groups = build_parameter_groups(
        model,
        backbone_lr=float(config["train"]["backbone_lr"]),
        new_modules_lr=float(config["train"]["new_modules_lr"]),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(config["train"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["train"]["epochs"])
    )
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(config["train"]["label_smoothing"])
    )
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment']['name']}_{config['model']['variant']}_s{seed}_{timestamp}"
    run_dir = Path(config["experiment"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"run_dir={run_dir}")

    start_epoch = 0
    best_value = float("-inf")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint["best_value"])

    history = []
    auxiliary_cfg = config["model"]["auxiliary_position"]
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        print(f"epoch={epoch + 1}/{config['train']['epochs']}")
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=int(config["model"]["num_classes"]),
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp_enabled,
            grad_clip_norm=float(config["train"]["grad_clip_norm"]),
            auxiliary_position_weight=float(auxiliary_cfg["loss_weight"])
            if bool(auxiliary_cfg["enabled"])
            else 0.0,
            log_interval=int(config["train"]["log_interval"]),
        )
        val_metrics, predictions = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=int(config["model"]["num_classes"]),
            criterion=criterion,
            optimizer=None,
            scaler=None,
            amp=amp_enabled,
            auxiliary_position_weight=float(auxiliary_cfg["loss_weight"])
            if bool(auxiliary_cfg["enabled"])
            else 0.0,
            log_interval=0,
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items() if isinstance(value, (int, float))},
            **{f"val_{key}": value for key, value in val_metrics.items() if isinstance(value, (int, float))},
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_new": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        write_json(val_metrics, run_dir / "val_metrics_latest.json")
        metric_name = str(config["train"]["checkpoint_metric"])
        current_value = float(val_metrics[metric_name])
        is_best = current_value > best_value
        if is_best:
            best_value = current_value
        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_value": best_value,
            "config": config,
        }
        atomic_torch_save(checkpoint_payload, run_dir / "last.pt")
        if is_best:
            atomic_torch_save(checkpoint_payload, run_dir / "best.pt")
            write_json(predictions, run_dir / "val_predictions_best.json")
        print(
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
