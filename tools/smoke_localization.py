#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import LocalizationLoss, LocalizationMetrics  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def average_losses(
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    criterion: LocalizationLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for raw in batches:
        batch = move(raw, device)
        output = model(batch["image"], batch["radar"], batch["radar_mask"])
        losses = criterion(
            output["box"], batch["bbox"], output["position"], batch["position_normalized"]
        )
        size = int(batch["bbox"].shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        count += size
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def localization_metrics(
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    position_stats: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    meter = LocalizationMetrics(288, 768)
    mean = torch.tensor(position_stats["mean"], device=device)
    std = torch.tensor(position_stats["std"], device=device)
    for raw in batches:
        batch = move(raw, device)
        output = model(batch["image"], batch["radar"], batch["radar_mask"])
        pred_position_m = output["position"] * std + mean
        meter.update(output["box"], batch["bbox"], pred_position_m, batch["position"])
    return meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-4 dataset/forward/backward/overfit smoke test")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not 5 <= args.samples <= 20:
        raise ValueError("Smoke overfit sample count must be between 5 and 20")
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(config["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    dataset = make_localization_dataset(config, "train", position_stats, args.samples)
    loader = make_loader(config, dataset, "val", args.batch_size)
    cached_batches = list(loader)
    first = cached_batches[0]

    model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
    criterion = LocalizationLoss(config["loss"])
    batch = move(first, device)
    output = model(batch["image"], batch["radar"], batch["radar_mask"], return_attention=True)
    losses = criterion(
        output["box"], batch["bbox"], output["position"], batch["position_normalized"]
    )
    losses["total_loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    gradient_finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)

    metric_sanity = LocalizationMetrics(288, 768)
    metric_sanity.update(first["bbox"], first["bbox"], first["position"], first["position"])
    metric_sanity_result = metric_sanity.compute()
    metric_sanity_ok = (
        abs(float(metric_sanity_result["mean_iou"]) - 1.0) < 1e-6
        and float(metric_sanity_result["position_error_mean_m"]) == 0.0
    )

    print(f"device={device} samples={len(dataset)}")
    print(f"image shape={tuple(first['image'].shape)}")
    print(f"radar shape={tuple(first['radar'].shape)} mask={tuple(first['radar_mask'].shape)}")
    print(f"bbox shape={tuple(first['bbox'].shape)} position shape={tuple(first['position'].shape)}")
    print(
        f"pred box shape={tuple(output['box'].shape)} range="
        f"[{float(output['box'].min()):.6f}, {float(output['box'].max()):.6f}]"
    )
    print(f"pred xyz(normalized) shape={tuple(output['position'].shape)}")
    print("one_batch_losses", {k: float(v.detach()) for k, v in losses.items()})
    print(f"gradient_finite={gradient_finite} metric_sanity={metric_sanity_ok}")
    if not gradient_finite or not metric_sanity_ok:
        raise RuntimeError("Forward/backward or metric sanity check failed")

    model.zero_grad(set_to_none=True)
    groups = build_parameter_groups(
        model,
        backbone_lr=float(config["train"]["backbone_lr"]),
        new_modules_lr=float(config["train"]["new_modules_lr"]),
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))
    initial = average_losses(model, cached_batches, criterion, device)
    initial_metrics = localization_metrics(model, cached_batches, position_stats, device)
    model.train()
    iterator = itertools.cycle(cached_batches)
    for step in range(1, args.steps + 1):
        train_batch = move(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            train_batch["image"], train_batch["radar"], train_batch["radar_mask"]
        )
        step_losses = criterion(
            prediction["box"],
            train_batch["bbox"],
            prediction["position"],
            train_batch["position_normalized"],
        )
        step_losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
        optimizer.step()
        if step % 20 == 0 or step == args.steps:
            print(
                f"overfit_step={step}/{args.steps} total={float(step_losses['total_loss']):.6f} "
                f"bbox_l1={float(step_losses['bbox_l1_loss']):.6f} "
                f"center_l1={float(step_losses['bbox_center_l1_loss']):.6f} "
                f"size_l1={float(step_losses['bbox_size_l1_loss']):.6f} "
                f"giou={float(step_losses['giou_loss']):.6f} "
                f"xyz={float(step_losses['position_loss']):.6f}",
                flush=True,
            )
    final = average_losses(model, cached_batches, criterion, device)
    final_metrics = localization_metrics(model, cached_batches, position_stats, device)
    overfit_ok = (
        final["bbox_l1_loss"] < 0.5 * initial["bbox_l1_loss"]
        and final["position_loss"] < 0.5 * initial["position_loss"]
    )
    report = {
        "samples": len(dataset),
        "steps": args.steps,
        "device": str(device),
        "input_shapes": {
            "image": list(first["image"].shape),
            "radar": list(first["radar"].shape),
            "radar_mask": list(first["radar_mask"].shape),
            "bbox": list(first["bbox"].shape),
            "position": list(first["position"].shape),
        },
        "prediction_shapes": {"box": list(output["box"].shape), "position": list(output["position"].shape)},
        "pred_box_range": [float(output["box"].min()), float(output["box"].max())],
        "one_batch_losses": {k: float(v.detach()) for k, v in losses.items()},
        "gradient_finite": gradient_finite,
        "metric_sanity": metric_sanity_ok,
        "overfit_initial": initial,
        "overfit_final": final,
        "overfit_initial_metrics": initial_metrics,
        "overfit_final_metrics": final_metrics,
        "overfit_success": overfit_ok,
        "position_stats": position_stats,
        "test_split_accessed": False,
    }
    run_dir = Path(config["experiment"]["output_dir"]) / (
        "stage4_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(position_stats, run_dir / "position_stats.json")
    write_json(report, run_dir / "smoke_report.json")
    print(f"initial={initial}")
    print(f"final={final}")
    print(f"initial_metrics={initial_metrics}")
    print(f"final_metrics={final_metrics}")
    print(f"overfit_success={overfit_ok} report={run_dir / 'smoke_report.json'}")
    if not overfit_ok:
        raise RuntimeError("20-sample overfit did not reduce both bbox and xyz losses by at least 50%")


if __name__ == "__main__":
    main()
