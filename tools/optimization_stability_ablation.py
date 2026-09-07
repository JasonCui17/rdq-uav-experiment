#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_bbox_stats, compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import LocalizationLoss, LocalizationMetrics  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


VARIANTS = (
    {"name": "baseline", "freeze_bn": False, "new_modules_lr": 1e-3},
    {"name": "freeze_bn", "freeze_bn": True, "new_modules_lr": 1e-3},
    {"name": "low_lr", "freeze_bn": False, "new_modules_lr": 1e-4},
)
TAIL_STEPS = {400, 450, 500, 550, 600}


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def batch_norm_modules(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]


def set_training_mode(model: nn.Module, freeze_bn: bool) -> None:
    model.train()
    if freeze_bn:
        for module in batch_norm_modules(model):
            module.eval()


def bn_statistics(model: nn.Module) -> dict[str, float | int]:
    modules = batch_norm_modules(model)
    if not modules:
        return {
            "bn_mean_abs_running_mean": 0.0,
            "bn_mean_running_var": 0.0,
            "bn_min_running_var": 0.0,
            "bn_max_running_var": 0.0,
            "bn_num_batches_tracked": 0,
            "bn_num_layers": 0,
        }
    running_means = torch.cat([module.running_mean.detach().cpu().float() for module in modules])
    running_vars = torch.cat([module.running_var.detach().cpu().float() for module in modules])
    tracked = [int(module.num_batches_tracked) for module in modules]
    tracked_value: int | float = tracked[0] if len(set(tracked)) == 1 else float(sum(tracked) / len(tracked))
    return {
        "bn_mean_abs_running_mean": float(running_means.abs().mean()),
        "bn_mean_running_var": float(running_vars.mean()),
        "bn_min_running_var": float(running_vars.min()),
        "bn_max_running_var": float(running_vars.max()),
        "bn_num_batches_tracked": tracked_value,
        "bn_num_layers": len(modules),
    }


def head_gradient_norm(module: nn.Module) -> float:
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


@torch.no_grad()
def evaluate_all(
    model: nn.Module,
    batches: list[dict[str, Any]],
    criterion: LocalizationLoss,
    position_stats: dict[str, Any],
    image_size: list[int],
    device: torch.device,
    *,
    training_mode: bool,
    freeze_bn: bool,
) -> dict[str, float | int]:
    if training_mode:
        set_training_mode(model, freeze_bn)
    else:
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


def train_mode_diagnostic_without_side_effects(
    model: nn.Module,
    batches: list[dict[str, Any]],
    criterion: LocalizationLoss,
    position_stats: dict[str, Any],
    image_size: list[int],
    device: torch.device,
    freeze_bn: bool,
) -> dict[str, float | int]:
    modules = batch_norm_modules(model)
    buffers = [
        (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for module in modules
    ]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return evaluate_all(
            model,
            batches,
            criterion,
            position_stats,
            image_size,
            device,
            training_mode=True,
            freeze_bn=freeze_bn,
        )
    finally:
        for module, (running_mean, running_var, tracked) in zip(modules, buffers):
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            module.num_batches_tracked.copy_(tracked)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.eval()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def history_row(
    variant: dict[str, Any],
    step: int,
    elapsed: float,
    eval_values: dict[str, float | int],
    train_values: dict[str, float | int],
    box_grad: float,
    position_grad: float,
    bn_stats: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "variant": variant["name"],
        "freeze_bn": variant["freeze_bn"],
        "new_modules_lr": variant["new_modules_lr"],
        "step": step,
        "elapsed_seconds": elapsed,
        "mean_iou": eval_values["mean_iou"],
        "median_iou": eval_values["median_iou"],
        "recall_iou_0.5": eval_values["recall_iou_0.5"],
        "center_error_px": eval_values["center_error_px_mean"],
        "width_error": eval_values["width_abs_error_mean"],
        "height_error": eval_values["height_abs_error_mean"],
        "xyz_error_m": eval_values["position_error_mean_m"],
        "bbox_l1": eval_values["bbox_regression_loss"],
        "position_loss": eval_values["position_loss"],
        "box_head_grad_norm": box_grad,
        "position_head_grad_norm": position_grad,
        "train_mode_mean_iou": train_values["mean_iou"],
        "eval_mode_mean_iou": eval_values["mean_iou"],
        "train_eval_iou_gap": float(train_values["mean_iou"]) - float(eval_values["mean_iou"]),
        **bn_stats,
    }


def print_row(row: dict[str, Any], steps: int) -> None:
    print(
        f"{row['variant']} step={row['step']}/{steps} eval_iou={row['mean_iou']:.4f} "
        f"train_iou={row['train_mode_mean_iou']:.4f} gap={row['train_eval_iou_gap']:+.4f} "
        f"r50={row['recall_iou_0.5']:.2f} center={row['center_error_px']:.2f}px "
        f"w={row['width_error']:.6f} h={row['height_error']:.6f} xyz={row['xyz_error_m']:.3f}m "
        f"box_grad={row['box_head_grad_norm']:.3g} bn_var={row['bn_mean_running_var']:.4g} "
        f"bn_n={row['bn_num_batches_tracked']}",
        flush=True,
    )


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=lambda row: float(row["mean_iou"]))
    final = rows[-1]
    tail = [row for row in rows if int(row["step"]) in TAIL_STEPS]
    if len(tail) != len(TAIL_STEPS):
        raise RuntimeError("Tail summary does not contain steps 400,450,500,550,600")
    tail_ious = torch.tensor([float(row["mean_iou"]) for row in tail], dtype=torch.float64)
    tail_gaps = torch.tensor([float(row["train_eval_iou_gap"]) for row in tail], dtype=torch.float64)
    return {
        "name": rows[0]["variant"],
        "freeze_bn": rows[0]["freeze_bn"],
        "new_modules_lr": rows[0]["new_modules_lr"],
        "best_step": int(best["step"]),
        "best_mean_iou": float(best["mean_iou"]),
        "best_recall_iou_0.5": float(best["recall_iou_0.5"]),
        "best_center_error_px": float(best["center_error_px"]),
        "best_width_error": float(best["width_error"]),
        "best_height_error": float(best["height_error"]),
        "best_xyz_error_m": float(best["xyz_error_m"]),
        "tail_mean_iou": float(tail_ious.mean()),
        "tail_std_iou_sample": float(tail_ious.std(unbiased=True)),
        "tail_min_iou": float(tail_ious.min()),
        "tail_mean_abs_train_eval_gap": float(tail_gaps.abs().mean()),
        "tail_signed_train_eval_gap": float(tail_gaps.mean()),
        "final_iou": float(final["mean_iou"]),
        "final_recall_iou_0.5": float(final["recall_iou_0.5"]),
        "final_train_mode_iou": float(final["train_mode_mean_iou"]),
        "final_train_eval_iou_gap": float(final["train_eval_iou_gap"]),
        "final_bn_mean_abs_running_mean": float(final["bn_mean_abs_running_mean"]),
        "final_bn_mean_running_var": float(final["bn_mean_running_var"]),
        "final_bn_min_running_var": float(final["bn_min_running_var"]),
        "final_bn_max_running_var": float(final["bn_max_running_var"]),
        "final_bn_num_batches_tracked": final["bn_num_batches_tracked"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4.3 tiny-batch optimization stability")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()
    if (args.samples, args.steps, args.batch_size, args.eval_interval) != (20, 600, 2, 50):
        raise ValueError("Stage 4.3 is fixed to samples=20, steps=600, batch=2, interval=50")

    base = load_config(args.config)
    if base["model"]["variant"] != "rdq":
        raise ValueError("Stage 4.3 requires RDQ")
    if base["model"].get("bbox_parameterization") != "sigmoid_cxcywh":
        raise ValueError("Stage 4.3 requires sigmoid_cxcywh")
    base["loss"]["bbox_regression"] = "l1"
    base["loss"]["giou_weight"] = 0.0
    if float(base["loss"]["bbox_l1_weight"]) != 5.0 or float(base["loss"]["position_weight"]) != 1.0:
        raise ValueError("Stage 4.3 requires bbox weight 5 and position weight 1")
    seed = int(base["experiment"]["seed"])
    if seed != 42:
        raise ValueError("Stage 4.3 requires seed 42")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(base["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    bbox_stats = compute_bbox_stats(manifest_dir / "train.csv", base["data"]["panorama_size"])
    dataset = make_localization_dataset(base, "train", position_stats, limit_samples=args.samples)
    cached_batches = list(make_loader(base, dataset, "val", args.batch_size))
    if sum(int(batch["bbox"].shape[0]) for batch in cached_batches) != 20:
        raise RuntimeError("Expected exactly 20 cached samples")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base["experiment"]["output_dir"]) / f"stage4_optimization_stability_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(position_stats, run_dir / "position_stats.json")
    write_json(bbox_stats, run_dir / "bbox_stats.json")
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    print(f"device={device} run_dir={run_dir} samples=20 seed=42", flush=True)

    for variant in VARIANTS:
        config = copy.deepcopy(base)
        config["train"]["new_modules_lr"] = variant["new_modules_lr"]
        (run_dir / f"config_{variant['name']}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        seed_everything(seed)
        model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
        criterion = LocalizationLoss(config["loss"])
        groups = build_parameter_groups(
            model,
            backbone_lr=float(config["train"]["backbone_lr"]),
            new_modules_lr=float(config["train"]["new_modules_lr"]),
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))
        iterator = itertools.cycle(cached_batches)
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()

        for step in range(1, args.steps + 1):
            set_training_mode(model, bool(variant["freeze_bn"]))
            batch = move(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
            losses = criterion(
                outputs["box"], batch["bbox"], outputs["position"], batch["position_normalized"]
            )
            if not bool(torch.isfinite(losses["total_loss"])):
                raise RuntimeError(f"{variant['name']}: non-finite loss at step {step}")
            losses["total_loss"].backward()
            box_grad = head_gradient_norm(model.box_head)
            position_grad = head_gradient_norm(model.position_head)
            if not math.isfinite(box_grad) or not math.isfinite(position_grad):
                raise RuntimeError(f"{variant['name']}: non-finite gradient at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            optimizer.step()

            if step % args.eval_interval == 0:
                stats_before_diagnostics = bn_statistics(model)
                eval_values = evaluate_all(
                    model, cached_batches, criterion, position_stats, config["data"]["image_size"],
                    device, training_mode=False, freeze_bn=bool(variant["freeze_bn"]),
                )
                train_values = train_mode_diagnostic_without_side_effects(
                    model, cached_batches, criterion, position_stats, config["data"]["image_size"],
                    device, bool(variant["freeze_bn"]),
                )
                if bn_statistics(model) != stats_before_diagnostics:
                    raise RuntimeError("Train-mode diagnostic changed BatchNorm buffers")
                row = history_row(
                    variant, step, time.perf_counter() - started, eval_values, train_values,
                    box_grad, position_grad, stats_before_diagnostics,
                )
                rows.append(row)
                all_rows.append(row)
                print_row(row, args.steps)
                write_csv(all_rows, run_dir / "optimization_stability_history.csv")

        summary = summarize_variant(rows)
        summaries.append(summary)
        write_json(
            {
                "protocol": {
                    "samples": 20, "steps": 600, "batch_size": 2, "eval_interval": 50,
                    "seed": 42, "bbox_loss": "l1", "giou_weight": 0.0,
                    "bbox_parameterization": "sigmoid_cxcywh", "test_split_accessed": False,
                },
                "runs": summaries,
            },
            run_dir / "optimization_stability_report.json",
        )

    write_csv(summaries, run_dir / "optimization_stability_comparison.csv")
    print(f"completed={run_dir}", flush=True)


if __name__ == "__main__":
    main()
