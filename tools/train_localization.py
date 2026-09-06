#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import LocalizationLoss, run_localization_epoch  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import atomic_torch_save, write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


def write_prediction_csv(rows: list[dict], path: Path) -> None:
    serialized = []
    for row in rows:
        serialized.append(
            {
                **row,
                "pred_bbox": str(row["pred_bbox"]),
                "gt_bbox": str(row["gt_bbox"]),
                "pred_xyz": str(row["pred_xyz"]),
                "gt_xyz": str(row["gt_xyz"]),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0]))
        writer.writeheader()
        writer.writerows(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal single-UAV 2D/3D localizer")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if config.get("task", {}).get("name") != "single_uav_joint_2d_3d_localization":
        raise ValueError("Localization trainer requires the Stage-4 localization task config")

    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(config["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")

    train_dataset = make_localization_dataset(
        config, "train", position_stats, limit_samples=args.limit_samples
    )
    val_dataset = make_localization_dataset(
        config, "val", position_stats, limit_samples=args.limit_samples
    )
    train_loader = make_loader(config, train_dataset, "train", int(config["train"]["batch_size"]))
    val_loader = make_loader(config, val_dataset, "val", int(config["evaluation"]["batch_size"]))

    resume = config["train"].get("resume")
    model = build_localizer(config["model"], load_backbone_pretrained=not bool(resume)).to(device)
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
    criterion = LocalizationLoss(config["loss"])
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment']['name']}_{config['model']['variant']}_s{seed}_{timestamp}"
    run_dir = Path(config["experiment"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(position_stats, run_dir / "position_stats.json")
    print(f"device={device} run_dir={run_dir}")
    print(f"position_stats={position_stats}")

    start_epoch = 0
    mode = str(config["train"].get("checkpoint_mode", "min"))
    if mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be min or max")
    best_value = float("inf") if mode == "min" else float("-inf")
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint["best_value"])

    position_mean = torch.tensor(position_stats["mean"], dtype=torch.float32)
    position_std = torch.tensor(position_stats["std"], dtype=torch.float32)
    image_height, image_width = (int(x) for x in config["data"]["image_size"])
    history: list[dict] = []
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        print(f"epoch={epoch + 1}/{config['train']['epochs']}")
        common = dict(
            model=model,
            device=device,
            criterion=criterion,
            position_mean=position_mean,
            position_std=position_std,
            processed_height=image_height,
            processed_stitched_width=2 * image_width,
            amp=amp_enabled,
        )
        train_metrics, _ = run_localization_epoch(
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=float(config["train"]["grad_clip_norm"]),
            log_interval=int(config["train"]["log_interval"]),
            **common,
        )
        val_metrics, predictions = run_localization_epoch(
            loader=val_loader,
            optimizer=None,
            scaler=None,
            grad_clip_norm=None,
            log_interval=0,
            **common,
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))},
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_new": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        write_json(val_metrics, run_dir / "val_metrics_latest.json")
        current = float(val_metrics[str(config["train"]["checkpoint_metric"])])
        is_best = current < best_value if mode == "min" else current > best_value
        if is_best:
            best_value = current
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_value": best_value,
            "position_stats": position_stats,
            "config": config,
        }
        atomic_torch_save(payload, run_dir / "last.pt")
        if is_best:
            atomic_torch_save(payload, run_dir / "best.pt")
            write_json(predictions, run_dir / "val_predictions_best.json")
            write_prediction_csv(predictions, run_dir / "val_predictions_best.csv")
            write_json(val_metrics, run_dir / "val_metrics_best.json")
        print(
            f"train total={train_metrics['total_loss']:.4f} "
            f"bbox_l1={train_metrics['bbox_l1_loss']:.4f} "
            f"giou={train_metrics['giou_loss']:.4f} xyz={train_metrics['position_loss']:.4f} | "
            f"val total={val_metrics['total_loss']:.4f} mean_iou={val_metrics['mean_iou']:.4f} "
            f"e3d={val_metrics['position_error_mean_m']:.4f}m"
        )


if __name__ == "__main__":
    main()
