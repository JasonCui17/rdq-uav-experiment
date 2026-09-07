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
from torchvision.ops import generalized_box_iou_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_bbox_stats, compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import (  # noqa: E402
    LocalizationLoss,
    LocalizationMetrics,
    aligned_iou,
    cxcywh_to_xyxy,
)
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


VARIANTS = (
    ("smooth_l1_giou", "smooth_l1", 2.0),
    ("l1_giou", "l1", 2.0),
    ("l1_only", "l1", 0.0),
)


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def head_gradient_norm(module: torch.nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


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


def diagnostic_gradient_norms(
    model: torch.nn.Module,
    raw_batch: dict[str, Any],
    criterion: LocalizationLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    model.zero_grad(set_to_none=True)
    batch = move(raw_batch, device)
    outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
    losses = criterion(
        outputs["box"], batch["bbox"], outputs["position"], batch["position_normalized"]
    )
    losses["total_loss"].backward()
    box_norm = head_gradient_norm(model.box_head)
    position_norm = head_gradient_norm(model.position_head)
    model.zero_grad(set_to_none=True)
    if not math.isfinite(box_norm) or not math.isfinite(position_norm):
        raise RuntimeError("Initial head gradient norm is not finite")
    return box_norm, position_norm


def make_history_row(
    name: str,
    regression: str,
    giou_weight: float,
    step: int,
    elapsed: float,
    values: dict[str, float | int],
    box_grad_norm: float,
    position_grad_norm: float,
) -> dict[str, Any]:
    return {
        "loss_variant": name,
        "bbox_regression": regression,
        "giou_weight": giou_weight,
        "step": step,
        "elapsed_seconds": elapsed,
        "total_loss": values["total_loss"],
        "bbox_regression_loss": values["bbox_regression_loss"],
        "bbox_center_loss": values["bbox_center_loss"],
        "bbox_size_loss": values["bbox_size_loss"],
        "giou_loss": values["giou_loss"],
        "position_loss": values["position_loss"],
        "mean_iou": values["mean_iou"],
        "median_iou": values["median_iou"],
        "recall_iou_0.5": values["recall_iou_0.5"],
        "center_error_px_mean": values["center_error_px_mean"],
        "width_abs_error_mean": values["width_abs_error_mean"],
        "height_abs_error_mean": values["height_abs_error_mean"],
        "position_error_mean_m": values["position_error_mean_m"],
        "box_head_grad_norm": box_grad_norm,
        "position_head_grad_norm": position_grad_norm,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_row(row: dict[str, Any], steps: int) -> None:
    print(
        f"{row['loss_variant']} step={row['step']}/{steps} total={row['total_loss']:.6f} "
        f"reg={row['bbox_regression_loss']:.6f} center={row['bbox_center_loss']:.6f} "
        f"size={row['bbox_size_loss']:.6f} giou={row['giou_loss']:.6f} "
        f"xyz={row['position_loss']:.6f} iou={row['mean_iou']:.4f} "
        f"recall50={row['recall_iou_0.5']:.3f} center_px={row['center_error_px_mean']:.2f} "
        f"werr={row['width_abs_error_mean']:.6f} herr={row['height_abs_error_mean']:.6f} "
        f"e3d={row['position_error_mean_m']:.3f}m "
        f"grad_box={row['box_head_grad_norm']:.4g} grad_xyz={row['position_head_grad_norm']:.4g}",
        flush=True,
    )


def synthetic_sensitivity(bbox_stats: dict[str, Any], canvas: list[int]) -> list[dict[str, Any]]:
    height, width = int(canvas[0]), int(canvas[1])
    gt = torch.tensor(
        [[0.5, 0.5, bbox_stats["width_median"], bbox_stats["height_median"]]],
        dtype=torch.float32,
    )

    def row(case: str, pred: torch.Tensor, offset_px: float | None) -> dict[str, Any]:
        pred_xyxy = cxcywh_to_xyxy(pred)
        gt_xyxy = cxcywh_to_xyxy(gt)
        return {
            "case": case,
            "horizontal_center_offset_px": offset_px,
            "l1": float(torch.nn.functional.l1_loss(pred, gt)),
            "giou_loss": float(generalized_box_iou_loss(pred_xyxy, gt_xyxy)),
            "iou": float(aligned_iou(pred_xyxy, gt_xyxy)[0]),
        }

    rows = [row("perfect", gt.clone(), 0.0)]
    size_larger = gt.clone()
    size_larger[:, 2:] *= 1.10
    rows.append(row("width_height_10pct_larger", size_larger, 0.0))
    for pixels in (1, 2, 4, 8):
        shifted = gt.clone()
        shifted[:, 0] += pixels / width
        rows.append(row(f"center_x_plus_{pixels}px", shifted, float(pixels)))
    return rows


def select_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "step",
        "total_loss",
        "bbox_regression_loss",
        "bbox_center_loss",
        "bbox_size_loss",
        "giou_loss",
        "position_loss",
        "mean_iou",
        "median_iou",
        "recall_iou_0.5",
        "center_error_px_mean",
        "width_abs_error_mean",
        "height_abs_error_mean",
        "position_error_mean_m",
        "box_head_grad_norm",
        "position_head_grad_norm",
    )
    return {key: row[key] for key in keys}


def write_markdown(report: dict[str, Any], output_dir: Path, path: Path) -> None:
    lines = [
        "# Stage 4.2：BBox Loss 配对消融",
        "",
        "本实验仅改变 bbox regression loss 与 GIoU 权重。模型、数据、参数化、优化器、学习率、样本顺序和 XYZ 分支保持一致；未访问 test。",
        "",
        "## 配对结果",
        "",
        "| Loss | Initial IoU | Best IoU | Best step | Best Recall@0.5 | Center error (px) | Width error | Height error | XYZ error (m) | Final IoU | Final Recall@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        initial, best, final = run["initial"], run["best"], run["final"]
        lines.append(
            f"| {run['name']} | {initial['mean_iou']:.6f} | {best['mean_iou']:.6f} | "
            f"{best['step']} | {best['recall_iou_0.5']:.3f} | "
            f"{best['center_error_px_mean']:.3f} | {best['width_abs_error_mean']:.6f} | "
            f"{best['height_abs_error_mean']:.6f} | {best['position_error_mean_m']:.3f} | "
            f"{final['mean_iou']:.6f} | {final['recall_iou_0.5']:.3f} |"
        )
    lines += [
        "",
        "## Tiny-box synthetic sensitivity",
        "",
        f"使用 train bbox median size，processed canvas 为 {report['protocol']['canvas_width']}×{report['protocol']['canvas_height']}。10% 尺寸偏差指宽高均放大 10%；中心偏差仅沿 x 轴。",
        "",
        "| Case | Offset (px) | L1 | GIoU loss | IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["synthetic_sensitivity"]:
        lines.append(
            f"| {item['case']} | {item['horizontal_center_offset_px']:.1f} | "
            f"{item['l1']:.8f} | {item['giou_loss']:.6f} | {item['iou']:.6f} |"
        )
    lines += [
        "",
        "## Head gradient norm（step 50–600）",
        "",
        "| Loss | Box-head mean | Box-head min | Box-head max | Position-head mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["gradient_summary"]:
        lines.append(
            f"| {item['loss_variant']} | {item['box_head_mean']:.6f} | "
            f"{item['box_head_min']:.6f} | {item['box_head_max']:.6f} | "
            f"{item['position_head_mean']:.6f} |"
        )
    lines += [
        "",
        "## 可复现信息",
        "",
        f"完整历史与机器可读报告位于 `{output_dir.relative_to(PROJECT_ROOT)}`。Best checkpoint criterion 为固定 20 样本全集上的 Mean IoU。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict paired Stage-4.2 bbox-loss ablation")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()
    if args.samples != 20 or args.steps != 600 or args.batch_size != 2:
        raise ValueError("Stage 4.2 protocol is fixed to samples=20, steps=600, batch_size=2")
    if args.eval_interval != 50:
        raise ValueError("Stage 4.2 protocol is fixed to eval_interval=50")

    base_config = load_config(args.config)
    if base_config["model"]["variant"] != "rdq":
        raise ValueError("Stage 4.2 is fixed to model.variant=rdq")
    if base_config["model"].get("bbox_parameterization", "sigmoid_cxcywh") != "sigmoid_cxcywh":
        raise ValueError("Stage 4.2 is fixed to sigmoid_cxcywh")
    if float(base_config["loss"]["bbox_l1_weight"]) != 5.0:
        raise ValueError("Stage 4.2 is fixed to bbox_weight=5.0")
    if float(base_config["loss"]["position_weight"]) != 1.0:
        raise ValueError("Stage 4.2 is fixed to position_weight=1.0")

    seed = int(base_config["experiment"]["seed"])
    if seed != 42:
        raise ValueError("Stage 4.2 is fixed to seed=42")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(base_config["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    bbox_stats = compute_bbox_stats(manifest_dir / "train.csv", base_config["data"]["panorama_size"])
    dataset = make_localization_dataset(base_config, "train", position_stats, limit_samples=args.samples)
    loader = make_loader(base_config, dataset, "val", args.batch_size)
    cached_batches = list(loader)
    if sum(int(batch["bbox"].shape[0]) for batch in cached_batches) != 20:
        raise RuntimeError("The cached paired set does not contain exactly 20 samples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_config["experiment"]["output_dir"]) / f"stage4_bbox_loss_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(position_stats, run_dir / "position_stats.json")
    write_json(bbox_stats, run_dir / "bbox_stats.json")

    all_history: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    print(f"device={device} run_dir={run_dir} samples=20 seed=42", flush=True)
    for name, regression, giou_weight in VARIANTS:
        config = copy.deepcopy(base_config)
        config["loss"]["bbox_regression"] = regression
        config["loss"]["giou_weight"] = giou_weight
        (run_dir / f"config_{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        # Re-seeding here makes model initialization identical across all loss variants.
        seed_everything(seed)
        model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
        criterion = LocalizationLoss(config["loss"])
        initial_grad = diagnostic_gradient_norms(model, cached_batches[0], criterion, device)
        initial_values = evaluate_all(
            model, cached_batches, criterion, position_stats, config["data"]["image_size"], device
        )
        started = time.perf_counter()
        rows = [
            make_history_row(
                name, regression, giou_weight, 0, 0.0, initial_values, *initial_grad
            )
        ]
        print_row(rows[-1], args.steps)

        parameter_groups = build_parameter_groups(
            model,
            backbone_lr=float(config["train"]["backbone_lr"]),
            new_modules_lr=float(config["train"]["new_modules_lr"]),
        )
        optimizer = torch.optim.AdamW(
            parameter_groups, weight_decay=float(config["train"]["weight_decay"])
        )
        iterator = itertools.cycle(cached_batches)
        for step in range(1, args.steps + 1):
            model.train()
            batch = move(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
            losses = criterion(
                outputs["box"], batch["bbox"], outputs["position"], batch["position_normalized"]
            )
            if not bool(torch.isfinite(losses["total_loss"])):
                raise RuntimeError(f"{name}: non-finite loss at step {step}")
            losses["total_loss"].backward()
            box_grad = head_gradient_norm(model.box_head)
            position_grad = head_gradient_norm(model.position_head)
            if not math.isfinite(box_grad) or not math.isfinite(position_grad):
                raise RuntimeError(f"{name}: non-finite head gradient at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            optimizer.step()

            if step % args.eval_interval == 0:
                values = evaluate_all(
                    model,
                    cached_batches,
                    criterion,
                    position_stats,
                    config["data"]["image_size"],
                    device,
                )
                row = make_history_row(
                    name,
                    regression,
                    giou_weight,
                    step,
                    time.perf_counter() - started,
                    values,
                    box_grad,
                    position_grad,
                )
                rows.append(row)
                all_history.append(row)
                print_row(row, args.steps)
                write_csv(all_history, run_dir / "bbox_loss_history.csv")

        # Include step 0 once the variant is complete, preserving chronological order later.
        all_history.append(rows[0])
        all_history.sort(key=lambda item: (VARIANTS.index(next(v for v in VARIANTS if v[0] == item["loss_variant"])), item["step"]))
        write_csv(all_history, run_dir / "bbox_loss_history.csv")
        best = max(rows, key=lambda item: float(item["mean_iou"]))
        reports.append(
            {
                "name": name,
                "bbox_regression": regression,
                "giou_weight": giou_weight,
                "initial": select_summary(rows[0]),
                "best": select_summary(best),
                "final": select_summary(rows[-1]),
            }
        )
        partial = {
            "protocol": {"samples": 20, "steps": 600, "seed": 42, "test_split_accessed": False},
            "bbox_stats": bbox_stats,
            "position_stats": position_stats,
            "runs": reports,
        }
        write_json(partial, run_dir / "loss_ablation_report.json")

    comparison_rows: list[dict[str, Any]] = []
    for run in reports:
        comparison_rows.append(
            {
                "loss_variant": run["name"],
                "initial_mean_iou": run["initial"]["mean_iou"],
                "best_mean_iou": run["best"]["mean_iou"],
                "best_step": run["best"]["step"],
                "best_recall_iou_0.5": run["best"]["recall_iou_0.5"],
                "best_center_error_px": run["best"]["center_error_px_mean"],
                "best_width_error": run["best"]["width_abs_error_mean"],
                "best_height_error": run["best"]["height_abs_error_mean"],
                "best_xyz_error_m": run["best"]["position_error_mean_m"],
                "best_box_head_grad_norm": run["best"]["box_head_grad_norm"],
                "best_position_head_grad_norm": run["best"]["position_head_grad_norm"],
                "final_mean_iou": run["final"]["mean_iou"],
                "final_recall_iou_0.5": run["final"]["recall_iou_0.5"],
            }
        )
    write_csv(comparison_rows, run_dir / "loss_comparison.csv")

    height, view_width = (int(value) for value in base_config["data"]["image_size"])
    report = {
        "protocol": {
            "samples": 20,
            "steps": 600,
            "batch_size": 2,
            "eval_interval": 50,
            "seed": 42,
            "model_variant": "rdq",
            "bbox_parameterization": "sigmoid_cxcywh",
            "bbox_weight": 5.0,
            "position_weight": 1.0,
            "optimizer": "AdamW",
            "canvas_width": 2 * view_width,
            "canvas_height": height,
            "test_split_accessed": False,
        },
        "bbox_stats": bbox_stats,
        "position_stats": position_stats,
        "runs": reports,
        "synthetic_sensitivity": synthetic_sensitivity(bbox_stats, [height, 2 * view_width]),
        "gradient_summary": [
            {
                "loss_variant": name,
                "box_head_mean": sum(
                    float(row["box_head_grad_norm"])
                    for row in all_history
                    if row["loss_variant"] == name and int(row["step"]) > 0
                ) / (args.steps // args.eval_interval),
                "box_head_min": min(
                    float(row["box_head_grad_norm"])
                    for row in all_history
                    if row["loss_variant"] == name and int(row["step"]) > 0
                ),
                "box_head_max": max(
                    float(row["box_head_grad_norm"])
                    for row in all_history
                    if row["loss_variant"] == name and int(row["step"]) > 0
                ),
                "position_head_mean": sum(
                    float(row["position_head_grad_norm"])
                    for row in all_history
                    if row["loss_variant"] == name and int(row["step"]) > 0
                ) / (args.steps // args.eval_interval),
            }
            for name, _, _ in VARIANTS
        ],
    }
    write_json(report, run_dir / "loss_ablation_report.json")
    write_markdown(report, run_dir, PROJECT_ROOT / "docs/STAGE4_BBOX_LOSS_ABLATION.md")
    print(f"completed={run_dir}", flush=True)


if __name__ == "__main__":
    main()
