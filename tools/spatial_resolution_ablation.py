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


VARIANTS = (("stride16", 3), ("stride8", 2))
TAIL_STEPS = {400, 450, 500, 550, 600}


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def head_gradient_norm(module: nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


@torch.no_grad()
def evaluate_all(
    model: nn.Module,
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
        output = model(batch["image"], batch["radar"], batch["radar_mask"])
        losses = criterion(
            output["box"], batch["bbox"], output["position"], batch["position_normalized"]
        )
        pred_position = output["position"] * std + mean
        meter.update(output["box"], batch["bbox"], pred_position, batch["position"])
        batch_size = int(batch["bbox"].shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
    result = meter.compute()
    result.update({key: value / count for key, value in totals.items()})
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_geometry(
    model: nn.Module,
    batch: dict[str, Any],
    image_size: list[int],
    device: torch.device,
    bbox_stats: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        moved = move(batch, device)
        output = model(moved["image"], moved["radar"], moved["radar_mask"])
    grid = output["visual_grid"]
    if not isinstance(grid, tuple) or len(grid) != 2:
        raise RuntimeError(f"Unexpected visual_grid: {grid}")
    feature_h, feature_w = int(grid[0]), int(grid[1])
    input_h, input_w = (int(value) for value in image_size)
    views = int(batch["image"].shape[1])
    stride_h, stride_w = input_h / feature_h, input_w / feature_w
    stitched_feature_width = views * feature_w
    return {
        "feature_height": feature_h,
        "feature_width_per_view": feature_w,
        "views": views,
        "visual_token_count": views * feature_h * feature_w,
        "effective_stride_h": stride_h,
        "effective_stride_w": stride_w,
        "median_bbox_width_cells": bbox_stats["width_median"] * stitched_feature_width,
        "median_bbox_height_cells": bbox_stats["height_median"] * feature_h,
        "mean_bbox_width_cells": bbox_stats["width_mean"] * stitched_feature_width,
        "mean_bbox_height_cells": bbox_stats["height_mean"] * feature_h,
    }


def make_row(
    variant: str,
    out_index: int,
    step: int,
    elapsed: float,
    values: dict[str, float | int],
    box_grad: float,
    peak_memory_mb: float,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "out_index": out_index,
        "step": step,
        "elapsed_seconds": elapsed,
        "center_error_px_mean": values["center_error_px_mean"],
        "center_error_px_median": values["center_error_px_median"],
        "center_error_lt_2px": values["center_error_lt_2px"],
        "center_error_lt_4px": values["center_error_lt_4px"],
        "mean_iou": values["mean_iou"],
        "median_iou": values["median_iou"],
        "recall_iou_0.5": values["recall_iou_0.5"],
        "position_error_mean_m": values["position_error_mean_m"],
        "bbox_l1": values["bbox_regression_loss"],
        "position_loss": values["position_loss"],
        "box_head_grad_norm": box_grad,
        "peak_gpu_memory_mb": peak_memory_mb,
        "visual_token_count": geometry["visual_token_count"],
    }


def summarize(rows: list[dict[str, Any]], runtime: float, geometry: dict[str, Any]) -> dict[str, Any]:
    best_center = min(rows, key=lambda row: float(row["center_error_px_mean"]))
    best_iou = max(rows, key=lambda row: float(row["mean_iou"]))
    tail = [row for row in rows if int(row["step"]) in TAIL_STEPS]
    if len(tail) != 5:
        raise RuntimeError("Missing fixed tail checkpoints")

    def mean_std(key: str) -> tuple[float, float]:
        values = torch.tensor([float(row[key]) for row in tail], dtype=torch.float64)
        return float(values.mean()), float(values.std(unbiased=True))

    tail_center_mean, tail_center_std = mean_std("center_error_px_mean")
    tail_iou_mean, tail_iou_std = mean_std("mean_iou")
    return {
        "variant": rows[0]["variant"],
        **geometry,
        "best_center_step": int(best_center["step"]),
        "best_center_error_px_mean": float(best_center["center_error_px_mean"]),
        "best_center_error_px_median": float(best_center["center_error_px_median"]),
        "best_center_lt_2px": float(best_center["center_error_lt_2px"]),
        "best_center_lt_4px": float(best_center["center_error_lt_4px"]),
        "best_center_checkpoint_mean_iou": float(best_center["mean_iou"]),
        "best_center_checkpoint_recall_iou_0.5": float(best_center["recall_iou_0.5"]),
        "best_center_checkpoint_3d_error_m": float(best_center["position_error_mean_m"]),
        "best_iou_step": int(best_iou["step"]),
        "best_mean_iou": float(best_iou["mean_iou"]),
        "best_iou_recall_iou_0.5": float(best_iou["recall_iou_0.5"]),
        "tail_center_error_mean_px": tail_center_mean,
        "tail_center_error_std_px_sample": tail_center_std,
        "tail_mean_iou": tail_iou_mean,
        "tail_iou_std_sample": tail_iou_std,
        "runtime_seconds": runtime,
        "peak_gpu_memory_mb": max(float(row["peak_gpu_memory_mb"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4.5 spatial-resolution ablation")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()
    if (args.samples, args.steps, args.batch_size, args.eval_interval) != (20, 600, 2, 50):
        raise ValueError("Stage 4.5 is fixed to 20 samples, 600 steps, batch2, eval every50")

    base = load_config(args.config)
    if base["model"]["variant"] != "rdq" or base["model"].get("bbox_parameterization") != "sigmoid_cxcywh":
        raise ValueError("Stage 4.5 requires RDQ with sigmoid_cxcywh")
    base["loss"]["bbox_regression"] = "l1"
    base["loss"]["giou_weight"] = 0.0
    base["train"]["new_modules_lr"] = 1e-4
    base["train"]["backbone_lr"] = 1e-4
    if int(base["experiment"]["seed"]) != 42:
        raise ValueError("Stage 4.5 requires seed42")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(base["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    bbox_stats = compute_bbox_stats(manifest_dir / "train.csv", base["data"]["panorama_size"])
    dataset = make_localization_dataset(base, "train", position_stats, limit_samples=20)
    cached_batches = list(make_loader(base, dataset, "val", 2))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base["experiment"]["output_dir"]) / f"stage4_spatial_resolution_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(position_stats, run_dir / "position_stats.json")
    write_json(bbox_stats, run_dir / "bbox_stats.json")
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    print(f"device={device} run_dir={run_dir}", flush=True)

    # Resolve both geometries by real forward passes before either training run.
    resolved_geometry: dict[str, dict[str, Any]] = {}
    for preview_name, preview_index in VARIANTS:
        preview_config = copy.deepcopy(base)
        preview_config["model"]["backbone"]["out_index"] = preview_index
        seed_everything(42)
        preview_model = build_localizer(
            preview_config["model"], load_backbone_pretrained=False
        ).to(device)
        geometry = feature_geometry(
            preview_model,
            cached_batches[0],
            preview_config["data"]["image_size"],
            device,
            bbox_stats,
        )
        resolved_geometry[preview_name] = geometry
        print(
            f"PRETRAIN_GEOMETRY {preview_name}: "
            f"feature={geometry['feature_height']}x{geometry['feature_width_per_view']} "
            f"tokens={geometry['visual_token_count']} stride="
            f"{geometry['effective_stride_h']:.1f}x{geometry['effective_stride_w']:.1f} "
            f"median_cells={geometry['median_bbox_width_cells']:.3f}x"
            f"{geometry['median_bbox_height_cells']:.3f} mean_cells="
            f"{geometry['mean_bbox_width_cells']:.3f}x{geometry['mean_bbox_height_cells']:.3f}",
            flush=True,
        )
        del preview_model

    for name, out_index in VARIANTS:
        config = copy.deepcopy(base)
        config["model"]["backbone"]["out_index"] = out_index
        (run_dir / f"config_{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        seed_everything(42)
        model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
        criterion = LocalizationLoss(config["loss"])
        geometry = resolved_geometry[name]
        groups = build_parameter_groups(model, 1e-4, 1e-4)
        optimizer = torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))
        iterator = itertools.cycle(cached_batches)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        variant_rows: list[dict[str, Any]] = []
        for step in range(1, 601):
            model.train()
            batch = move(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"], batch["radar"], batch["radar_mask"])
            losses = criterion(
                output["box"], batch["bbox"], output["position"], batch["position_normalized"]
            )
            if not bool(torch.isfinite(losses["total_loss"])):
                raise RuntimeError(f"{name}: non-finite loss at step {step}")
            losses["total_loss"].backward()
            box_grad = head_gradient_norm(model.box_head)
            if not math.isfinite(box_grad):
                raise RuntimeError(f"{name}: non-finite gradient at step {step}")
            nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            optimizer.step()
            if step % 50 == 0:
                values = evaluate_all(
                    model, cached_batches, criterion, position_stats, config["data"]["image_size"], device
                )
                peak_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
                )
                row = make_row(
                    name, out_index, step, time.perf_counter() - started, values, box_grad, peak_mb, geometry
                )
                variant_rows.append(row)
                all_rows.append(row)
                print(
                    f"{name} step={step}/600 center={row['center_error_px_mean']:.2f}px "
                    f"p2={row['center_error_lt_2px']:.2f} p4={row['center_error_lt_4px']:.2f} "
                    f"iou={row['mean_iou']:.3f} r50={row['recall_iou_0.5']:.2f} "
                    f"xyz={row['position_error_mean_m']:.3f}m",
                    flush=True,
                )
                write_csv(all_rows, run_dir / "spatial_resolution_history.csv")
        runtime = time.perf_counter() - started
        summaries.append(summarize(variant_rows, runtime, geometry))
        write_json(
            {"protocol": {"test_split_accessed": False}, "runs": summaries},
            run_dir / "spatial_resolution_report.json",
        )

    write_csv(summaries, run_dir / "spatial_resolution_comparison.csv")
    baseline, highres = summaries
    center_improvement = 1.0 - highres["tail_center_error_mean_px"] / baseline["tail_center_error_mean_px"]
    iou_improvement = highres["tail_mean_iou"] - baseline["tail_mean_iou"]
    if center_improvement >= 0.30 or iou_improvement >= 0.10:
        status = "supported"
    elif center_improvement < 0.10 and abs(iou_improvement) < 0.03:
        status = "rejected"
    else:
        status = "inconclusive"
    report = {
        "hypothesis": "stride16 visual resolution is the primary tiny-UAV 2D localization bottleneck",
        "controlled_variable": "ResNet18 out_index only: 3 versus 2",
        "fixed_variables": {
            "model": "RDQ", "bbox_loss": "L1-only", "giou_weight": 0.0,
            "bbox_parameterization": "sigmoid_cxcywh", "new_modules_lr": 1e-4,
            "backbone_lr": 1e-4, "seed": 42, "samples": 20, "batch_size": 2,
            "steps": 600, "eval_interval": 50, "bn": "normal",
        },
        "runs": summaries,
        "tail_center_error_improvement": center_improvement,
        "tail_mean_iou_improvement": iou_improvement,
        "status": status,
        "test_split_accessed": False,
        "three_d_interpretation_limit": "Stage4.4 found ~44.7% time/background Gain3D; 3D is not a primary decision metric",
    }
    write_json(report, run_dir / "spatial_resolution_report.json")
    print(
        f"decision center_improvement={center_improvement:.4f} "
        f"tail_iou_delta={iou_improvement:.4f} H2={status}",
        flush=True,
    )
    print(f"completed={run_dir}", flush=True)


if __name__ == "__main__":
    main()
