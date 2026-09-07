#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_bbox_stats, compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import LocalizationLoss, LocalizationMetrics  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate_all(
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    criterion: LocalizationLoss,
    position_stats: dict[str, Any],
    image_size: list[int],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    height, view_width = (int(value) for value in image_size)
    meter = LocalizationMetrics(height, 2 * view_width)
    mean = torch.tensor(position_stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(position_stats["std"], dtype=torch.float32, device=device)
    totals: dict[str, float] = {}
    count = 0
    for raw_batch in batches:
        batch = move(raw_batch, device)
        outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
        losses = criterion(
            outputs["box"], batch["bbox"], outputs["position"], batch["position_normalized"]
        )
        pred_position_m = outputs["position"] * std + mean
        meter.update(outputs["box"], batch["bbox"], pred_position_m, batch["position"])
        batch_size = int(batch["bbox"].shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
    result = meter.compute()
    result.update({key: value / count for key, value in totals.items()})
    return result


def history_row(step: int, elapsed_seconds: float, values: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "total_loss": values["total_loss"],
        "bbox_l1_loss": values["bbox_l1_loss"],
        "bbox_center_l1_loss": values["bbox_center_l1_loss"],
        "bbox_size_l1_loss": values["bbox_size_l1_loss"],
        "giou_loss": values["giou_loss"],
        "position_loss": values["position_loss"],
        "mean_iou": values["mean_iou"],
        "recall_iou_0.5": values["recall_iou_0.5"],
        "center_error_px": values["center_error_px_mean"],
        "width_abs_error": values["width_abs_error_mean"],
        "height_abs_error": values["height_abs_error_mean"],
        "position_error_mean_m": values["position_error_mean_m"],
        "pred_width_mean": values["pred_width_mean"],
        "gt_width_mean": values["gt_width_mean"],
        "pred_height_mean": values["pred_height_mean"],
        "gt_height_mean": values["gt_height_mean"],
    }


def write_history(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_row(row: dict[str, float | int], total_steps: int) -> None:
    print(
        f"sanity_step={row['step']}/{total_steps} total={row['total_loss']:.6f} "
        f"center_l1={row['bbox_center_l1_loss']:.6f} size_l1={row['bbox_size_l1_loss']:.6f} "
        f"giou={row['giou_loss']:.6f} xyz={row['position_loss']:.6f} "
        f"mean_iou={row['mean_iou']:.4f} recall50={row['recall_iou_0.5']:.4f} "
        f"center_px={row['center_error_px']:.3f} width_err={row['width_abs_error']:.6f} "
        f"height_err={row['height_abs_error']:.6f} e3d={row['position_error_mean_m']:.3f}m",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-protocol Stage-4 bbox longer-overfit sanity check")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-interval", type=int, default=50)
    parser.add_argument(
        "--parameterization",
        choices=("sigmoid_cxcywh", "sigmoid_center_log_size"),
        default="sigmoid_cxcywh",
    )
    parser.add_argument("--output-prefix", default="stage4_bbox_sanity")
    args = parser.parse_args()
    if args.samples != 20:
        raise ValueError("This controlled comparison is fixed to exactly 20 samples")
    if not 500 <= args.steps <= 1000:
        raise ValueError("Longer-overfit must use 500-1000 optimizer steps")
    if args.history_interval <= 0 or args.steps % args.history_interval != 0:
        raise ValueError("history_interval must be positive and divide steps exactly")

    config = load_config(args.config)
    if config["model"]["variant"] != "rdq":
        raise ValueError("BBox sanity run is fixed to the existing RDQ variant")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(config["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    bbox_stats = compute_bbox_stats(
        manifest_dir / "train.csv", config["data"]["panorama_size"]
    )
    config["model"]["bbox_parameterization"] = args.parameterization
    config["model"]["bbox_reference_wh"] = (
        [bbox_stats["width_median"], bbox_stats["height_median"]]
        if args.parameterization == "sigmoid_center_log_size"
        else None
    )
    dataset = make_localization_dataset(config, "train", position_stats, limit_samples=args.samples)
    # The val loader mode gives deterministic ordering while retaining exactly
    # the same already-materialized train samples and transforms.
    loader = make_loader(config, dataset, "val", args.batch_size)
    cached_batches = list(loader)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["experiment"]["output_dir"]) / f"{args.output_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(position_stats, run_dir / "position_stats.json")
    write_json(bbox_stats, run_dir / "bbox_stats.json")

    model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
    criterion = LocalizationLoss(config["loss"])
    first = move(cached_batches[0], device)
    diagnostic = model(first["image"], first["radar"], first["radar_mask"], return_attention=True)
    diagnostic_losses = criterion(
        diagnostic["box"], first["bbox"], diagnostic["position"], first["position_normalized"]
    )
    diagnostic_losses["total_loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
    if not gradient_finite:
        raise RuntimeError("Initial backward produced a missing or non-finite gradient")
    model.zero_grad(set_to_none=True)

    parameter_groups = build_parameter_groups(
        model,
        backbone_lr=float(config["train"]["backbone_lr"]),
        new_modules_lr=float(config["train"]["new_modules_lr"]),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(config["train"]["weight_decay"])
    )
    iterator = itertools.cycle(cached_batches)
    started = time.perf_counter()
    initial = evaluate_all(model, cached_batches, criterion, position_stats, config["data"]["image_size"], device)
    history = [history_row(0, 0.0, initial)]
    print(f"device={device} run_dir={run_dir} samples={len(dataset)} seed={seed}")
    print_row(history[-1], args.steps)

    for step in range(1, args.steps + 1):
        model.train()
        batch = move(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
        losses = criterion(
            outputs["box"], batch["bbox"], outputs["position"], batch["position_normalized"]
        )
        if not bool(torch.isfinite(losses["total_loss"])):
            raise RuntimeError(f"Non-finite loss at step {step}")
        losses["total_loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
        optimizer.step()

        if step % args.history_interval == 0:
            values = evaluate_all(
                model, cached_batches, criterion, position_stats, config["data"]["image_size"], device
            )
            row = history_row(step, time.perf_counter() - started, values)
            history.append(row)
            write_history(history, run_dir / "bbox_sanity_history.csv")
            print_row(row, args.steps)

    final = evaluate_all(model, cached_batches, criterion, position_stats, config["data"]["image_size"], device)
    comparison_keys = (
        "bbox_center_l1_loss",
        "bbox_size_l1_loss",
        "giou_loss",
        "mean_iou",
        "recall_iou_0.5",
        "center_error_px_mean",
        "width_abs_error_mean",
        "height_abs_error_mean",
        "position_error_mean_m",
    )
    best_mean_iou_row = max(history, key=lambda row: float(row["mean_iou"]))
    best_recall_row = max(history, key=lambda row: float(row["recall_iou_0.5"]))
    finite_history = all(
        math.isfinite(float(value))
        for row in history
        for key, value in row.items()
        if key not in {"step"}
    )
    report = {
        "protocol": {
            "samples": args.samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "history_interval": args.history_interval,
            "seed": seed,
            "variant": config["model"]["variant"],
            "bbox_parameterization": args.parameterization,
            "bbox_reference_wh": config["model"]["bbox_reference_wh"],
            "optimizer": "AdamW",
            "backbone_lr": config["train"]["backbone_lr"],
            "new_modules_lr": config["train"]["new_modules_lr"],
            "loss_weights": config["loss"],
            "image_mode": config["data"]["image_mode"],
            "augmentation": {"train_color_jitter": config["data"]["train_color_jitter"]},
            "test_split_accessed": False,
        },
        "input_shapes": {
            "image": list(first["image"].shape),
            "radar": list(first["radar"].shape),
            "radar_mask": list(first["radar_mask"].shape),
            "bbox": list(first["bbox"].shape),
            "position": list(first["position"].shape),
        },
        "gradient_finite": gradient_finite and finite_history,
        "initial": initial,
        "final": final,
        "comparison_table": [
            {"metric": key, "initial": float(initial[key]), "final": float(final[key])}
            for key in comparison_keys
        ],
        "best_observed": {
            "mean_iou": float(best_mean_iou_row["mean_iou"]),
            "mean_iou_step": int(best_mean_iou_row["step"]),
            "recall_iou_0.5": float(best_mean_iou_row["recall_iou_0.5"]),
            "center_error_px": float(best_mean_iou_row["center_error_px"]),
            "width_abs_error": float(best_mean_iou_row["width_abs_error"]),
            "height_abs_error": float(best_mean_iou_row["height_abs_error"]),
            "position_error_mean_m": float(best_mean_iou_row["position_error_mean_m"]),
            "max_recall_iou_0.5": float(best_recall_row["recall_iou_0.5"]),
            "max_recall_iou_0.5_step": int(best_recall_row["step"]),
        },
        "delta": {
            key: float(final[key]) - float(initial[key])
            for key in comparison_keys
        },
        "stage4_sanity_pass": bool(
            float(final["mean_iou"]) >= 0.5
            and float(final["recall_iou_0.5"]) > 0.0
            and float(final["width_abs_error_mean"]) < float(initial["width_abs_error_mean"])
            and float(final["height_abs_error_mean"]) < float(initial["height_abs_error_mean"])
        ),
        "position_stats": position_stats,
        "bbox_stats": bbox_stats,
    }
    write_history(history, run_dir / "bbox_sanity_history.csv")
    write_json(report, run_dir / "bbox_sanity_report.json")
    print(f"report={run_dir / 'bbox_sanity_report.json'}")
    print(f"stage4_sanity_pass={report['stage4_sanity_pass']}")


if __name__ == "__main__":
    main()
