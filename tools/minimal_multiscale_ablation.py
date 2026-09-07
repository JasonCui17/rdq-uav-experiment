#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_bbox_stats, compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import LocalizationLoss  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402
from spatial_resolution_ablation import (  # noqa: E402
    evaluate_all,
    feature_geometry,
    head_gradient_norm,
    make_row,
    move,
    summarize,
    write_csv,
)


VARIANTS = (
    {"name": "stride8_only", "fusion": "none", "out_index": 2},
    {"name": "minimal_fpn", "fusion": "minimal_fpn", "out_index": 2},
)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_count_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4.6 minimal multi-scale fusion ablation")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()
    if (args.samples, args.steps, args.batch_size, args.eval_interval) != (20, 600, 2, 50):
        raise ValueError("Stage 4.6 is fixed to 20 samples, 600 steps, batch2, eval every50")

    base = load_config(args.config)
    if base["model"]["variant"] != "rdq" or base["model"].get("bbox_parameterization") != "sigmoid_cxcywh":
        raise ValueError("Stage 4.6 requires RDQ with sigmoid_cxcywh")
    base["loss"]["bbox_regression"] = "l1"
    base["loss"]["giou_weight"] = 0.0
    base["train"]["new_modules_lr"] = 1e-4
    base["train"]["backbone_lr"] = 1e-4
    if int(base["experiment"]["seed"]) != 42:
        raise ValueError("Stage 4.6 requires seed42")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(base["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    bbox_stats = compute_bbox_stats(manifest_dir / "train.csv", base["data"]["panorama_size"])
    dataset = make_localization_dataset(base, "train", position_stats, limit_samples=20)
    cached_batches = list(make_loader(base, dataset, "val", 2))
    if sum(int(batch["bbox"].shape[0]) for batch in cached_batches) != 20:
        raise RuntimeError("Expected exactly 20 cached samples")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base["experiment"]["output_dir"]) / f"stage4_minimal_multiscale_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(position_stats, run_dir / "position_stats.json")
    write_json(bbox_stats, run_dir / "bbox_stats.json")
    print(f"device={device} run_dir={run_dir}", flush=True)

    resolved: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        config = copy.deepcopy(base)
        config["model"]["backbone"]["out_index"] = variant["out_index"]
        config["model"]["backbone"]["fusion"] = variant["fusion"]
        seed_everything(42)
        preview = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
        geometry = feature_geometry(
            preview, cached_batches[0], config["data"]["image_size"], device, bbox_stats
        )
        resolved[variant["name"]] = {**geometry, **parameter_counts(preview)}
        print(
            f"PRETRAIN_GEOMETRY {variant['name']}: "
            f"feature={geometry['feature_height']}x{geometry['feature_width_per_view']} "
            f"tokens={geometry['visual_token_count']} "
            f"params={resolved[variant['name']]['parameter_count_total']} "
            f"trainable={resolved[variant['name']]['parameter_count_trainable']}",
            flush=True,
        )
        del preview

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant in VARIANTS:
        config = copy.deepcopy(base)
        config["model"]["backbone"]["out_index"] = variant["out_index"]
        config["model"]["backbone"]["fusion"] = variant["fusion"]
        (run_dir / f"config_{variant['name']}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        seed_everything(42)
        model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
        criterion = LocalizationLoss(config["loss"])
        optimizer = torch.optim.AdamW(
            build_parameter_groups(model, 1e-4, 1e-4),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        iterator = itertools.cycle(cached_batches)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for step in range(1, 601):
            model.train()
            batch = move(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"], batch["radar"], batch["radar_mask"])
            losses = criterion(
                output["box"], batch["bbox"], output["position"], batch["position_normalized"]
            )
            if not bool(torch.isfinite(losses["total_loss"])):
                raise RuntimeError(f"{variant['name']}: non-finite loss at step {step}")
            losses["total_loss"].backward()
            box_grad = head_gradient_norm(model.box_head)
            if not math.isfinite(box_grad):
                raise RuntimeError(f"{variant['name']}: non-finite gradient at step {step}")
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
                    variant["name"], variant["out_index"], step,
                    time.perf_counter() - started, values, box_grad, peak_mb,
                    resolved[variant["name"]],
                )
                row["fusion"] = variant["fusion"]
                row.update(parameter_counts(model))
                rows.append(row)
                all_rows.append(row)
                print(
                    f"{variant['name']} step={step}/600 center={row['center_error_px_mean']:.2f}px "
                    f"p2={row['center_error_lt_2px']:.2f} p4={row['center_error_lt_4px']:.2f} "
                    f"iou={row['mean_iou']:.3f} r50={row['recall_iou_0.5']:.2f} "
                    f"xyz={row['position_error_mean_m']:.3f}m",
                    flush=True,
                )
                write_csv(all_rows, run_dir / "multiscale_history.csv")
        summary = summarize(rows, time.perf_counter() - started, resolved[variant["name"]])
        summary["fusion"] = variant["fusion"]
        summaries.append(summary)

    write_csv(summaries, run_dir / "multiscale_comparison.csv")
    baseline, multiscale = summaries
    center_improvement = 1.0 - multiscale["tail_center_error_mean_px"] / baseline["tail_center_error_mean_px"]
    iou_improvement = multiscale["tail_mean_iou"] - baseline["tail_mean_iou"]
    if center_improvement >= 0.20 or iou_improvement >= 0.08:
        status = "supported"
    elif center_improvement < 0.10 and iou_improvement < 0.03:
        status = "rejected"
    else:
        status = "inconclusive"
    report = {
        "hypothesis": "adding stride16 semantics to stride8 improves tiny-UAV 2D localization",
        "controlled_variable": "stride8-only versus minimal additive stride8+stride16 fusion",
        "fixed_variables": {
            "model": "RDQ", "bbox_loss": "L1-only", "giou_weight": 0.0,
            "bbox_parameterization": "sigmoid_cxcywh", "backbone_lr": 1e-4,
            "new_modules_lr": 1e-4, "seed": 42, "samples": 20,
            "batch_size": 2, "steps": 600, "eval_interval": 50,
        },
        "runs": summaries,
        "tail_center_error_improvement": center_improvement,
        "tail_mean_iou_improvement": iou_improvement,
        "status": status,
        "test_split_accessed": False,
        "three_d_is_secondary": True,
    }
    write_json(report, run_dir / "report.json")
    print(
        f"decision center_improvement={center_improvement:.4f} "
        f"tail_iou_delta={iou_improvement:.4f} H3={status}", flush=True
    )
    print(f"completed={run_dir}", flush=True)


if __name__ == "__main__":
    main()
