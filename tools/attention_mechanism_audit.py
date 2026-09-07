#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
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
from rdq_uav.data.localization import compute_position_stats  # noqa: E402
from rdq_uav.engine.attention_audit import (  # noqa: E402
    attention_center_diagnostics,
    stitched_token_centers,
)
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


TAIL_STEPS = {400, 450, 500, 550, 600}
METHODS = (
    "fused_token",
    "mean_attention",
    "peak_attention",
    "best_fixed_head",
    "oracle_best_head",
    "grid_oracle",
)


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def center_summary(errors: torch.Tensor) -> dict[str, float]:
    errors = errors.detach().float().cpu()
    return {
        "center_error_px_mean": float(errors.mean()),
        "center_error_px_median": float(errors.quantile(0.5)),
        "center_error_lt_2px": float((errors < 2).float().mean()),
        "center_error_lt_4px": float((errors < 4).float().mean()),
        "center_error_lt_8px": float((errors < 8).float().mean()),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_attention(
    model: torch.nn.Module,
    cached_batches: list[dict[str, Any]],
    device: torch.device,
    image_size: list[int],
) -> tuple[dict[str, dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    image_h, view_w = (int(value) for value in image_size)
    stitched_w = 2 * view_w
    all_errors: dict[str, list[torch.Tensor]] = {name: [] for name in METHODS}
    all_entropy: list[torch.Tensor] = []
    all_normalized_entropy: list[torch.Tensor] = []
    all_mass1: list[torch.Tensor] = []
    all_mass2: list[torch.Tensor] = []
    per_head_errors: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []
    observed_shape: tuple[int, ...] | None = None
    observed_grid: tuple[int, int] | None = None
    token_centers: torch.Tensor | None = None
    cell_w = cell_h = 0.0

    # First pass collects every head, allowing one fixed best head to be chosen
    # over the complete diagnostic set rather than separately per mini-batch.
    batch_records: list[dict[str, Any]] = []
    for raw_batch in cached_batches:
        batch = move(raw_batch, device)
        output = model(
            batch["image"], batch["radar"], batch["radar_mask"], return_attention=True
        )
        attention = output["attention"]
        grid = output["visual_grid"]
        if not isinstance(attention, torch.Tensor):
            raise RuntimeError("RDQ did not return attention weights")
        if not isinstance(grid, tuple) or len(grid) != 2:
            raise RuntimeError(f"Unexpected visual grid: {grid}")
        shape = tuple(int(value) for value in attention.shape)
        if observed_shape is None:
            observed_shape = shape
            observed_grid = (int(grid[0]), int(grid[1]))
            views = int(batch["image"].shape[1])
            if views != 2:
                raise RuntimeError(f"Stage 4.7 requires dual view input, got {views}")
            token_centers = stitched_token_centers(
                observed_grid[0], observed_grid[1], views, image_h, view_w,
                device=device, dtype=attention.dtype,
            )
            cell_h = image_h / observed_grid[0]
            cell_w = view_w / observed_grid[1]
        elif shape[1:] != observed_shape[1:]:
            raise RuntimeError(f"Attention shape changed: {observed_shape} -> {shape}")
        assert token_centers is not None
        diagnostics = attention_center_diagnostics(
            attention,
            batch["bbox"][:, :2],
            token_centers,
            stitched_width=stitched_w,
            image_height=image_h,
            cell_width=cell_w,
            cell_height=cell_h,
        )
        fused_px = output["box"][:, :2] * output["box"].new_tensor([stitched_w, image_h])
        gt_px = batch["bbox"][:, :2] * batch["bbox"].new_tensor([stitched_w, image_h])
        fused_error = torch.linalg.vector_norm(fused_px - gt_px, dim=-1)
        batch_records.append(
            {
                "raw_batch": raw_batch,
                "fused_px": fused_px.detach(),
                "gt_px": gt_px.detach(),
                "diagnostics": diagnostics,
            }
        )
        all_errors["fused_token"].append(fused_error)
        all_errors["mean_attention"].append(diagnostics["mean_attention_error_px"])
        all_errors["peak_attention"].append(diagnostics["peak_attention_error_px"])
        all_errors["oracle_best_head"].append(diagnostics["best_head_error_px"])
        all_errors["grid_oracle"].append(diagnostics["grid_oracle_error_px"])
        per_head_errors.append(diagnostics["per_head_error_px"])
        all_entropy.append(diagnostics["attention_entropy"])
        all_normalized_entropy.append(diagnostics["attention_entropy_normalized"])
        all_mass1.append(diagnostics["gt_mass_1cell"])
        all_mass2.append(diagnostics["gt_mass_2cell"])

    head_error_matrix = torch.cat(per_head_errors, dim=0)
    fixed_head_means = head_error_matrix.mean(dim=0)
    best_fixed_head = int(fixed_head_means.argmin())
    all_errors["best_fixed_head"] = [head_error_matrix[:, best_fixed_head]]

    summaries = {
        name: center_summary(torch.cat(all_errors[name]))
        for name in METHODS
    }
    entropy = torch.cat(all_entropy).float().cpu()
    normalized_entropy = torch.cat(all_normalized_entropy).float().cpu()
    mass1 = torch.cat(all_mass1).float().cpu()
    mass2 = torch.cat(all_mass2).float().cpu()
    assert observed_shape is not None and observed_grid is not None and token_centers is not None

    # Uniform references use the exact per-sample neighborhoods, including seam effects.
    uniform_mass1: list[torch.Tensor] = []
    uniform_mass2: list[torch.Tensor] = []
    offset = 0
    for record in batch_records:
        gt_px = record["gt_px"]
        delta = (token_centers[None] - gt_px[:, None]).abs()
        uniform_mass1.append(
            ((delta[..., 0] <= cell_w) & (delta[..., 1] <= cell_h)).float().mean(dim=-1)
        )
        uniform_mass2.append(
            ((delta[..., 0] <= 2 * cell_w) & (delta[..., 1] <= 2 * cell_h)).float().mean(dim=-1)
        )
        raw_batch = record["raw_batch"]
        diag = record["diagnostics"]
        batch_count = int(gt_px.shape[0])
        for index in range(batch_count):
            sample_id = raw_batch.get("sample_id", [f"sample_{offset + index}"] * batch_count)[index]
            prediction_rows.append(
                {
                    "sample_id": str(sample_id),
                    "fused_center_x_px": float(record["fused_px"][index, 0]),
                    "fused_center_y_px": float(record["fused_px"][index, 1]),
                    "gt_center_x_px": float(gt_px[index, 0]),
                    "gt_center_y_px": float(gt_px[index, 1]),
                    "fused_error_px": float(
                        torch.linalg.vector_norm(record["fused_px"][index] - gt_px[index])
                    ),
                    "mean_attention_error_px": float(diag["mean_attention_error_px"][index]),
                    "peak_attention_error_px": float(diag["peak_attention_error_px"][index]),
                    "best_fixed_head": best_fixed_head,
                    "best_fixed_head_error_px": float(diag["per_head_error_px"][index, best_fixed_head]),
                    "oracle_best_head_index": int(diag["best_head_index"][index]),
                    "oracle_best_head_error_px": float(diag["best_head_error_px"][index]),
                    "grid_oracle_error_px": float(diag["grid_oracle_error_px"][index]),
                    "attention_entropy_normalized": float(diag["attention_entropy_normalized"][index]),
                    "gt_mass_1cell": float(diag["gt_mass_1cell"][index]),
                    "gt_mass_2cell": float(diag["gt_mass_2cell"][index]),
                }
            )
        offset += batch_count

    shared = {
        "attention_shape": list(observed_shape),
        "visual_grid_per_view": list(observed_grid),
        "visual_token_count": int(token_centers.shape[0]),
        "cell_width_px": cell_w,
        "cell_height_px": cell_h,
        "best_fixed_head_index": best_fixed_head,
        "per_head_center_error_px_mean": [float(value) for value in fixed_head_means.cpu()],
        "attention_entropy_mean": float(entropy.mean()),
        "attention_entropy_median": float(entropy.quantile(0.5)),
        "attention_entropy_normalized_mean": float(normalized_entropy.mean()),
        "attention_entropy_normalized_median": float(normalized_entropy.quantile(0.5)),
        "gt_mass_1cell_mean": float(mass1.mean()),
        "gt_mass_1cell_median": float(mass1.quantile(0.5)),
        "gt_mass_2cell_mean": float(mass2.mean()),
        "gt_mass_2cell_median": float(mass2.quantile(0.5)),
        "uniform_mass_1cell_mean": float(torch.cat(uniform_mass1).mean()),
        "uniform_mass_2cell_mean": float(torch.cat(uniform_mass2).mean()),
    }
    return summaries, shared, prediction_rows


def history_rows(
    step: int,
    elapsed: float,
    summaries: dict[str, dict[str, float]],
    shared: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        rows.append(
            {
                "step": step,
                "elapsed_seconds": elapsed,
                "method": method,
                **summaries[method],
                "best_fixed_head_index": shared["best_fixed_head_index"],
                "attention_entropy_normalized_mean": shared["attention_entropy_normalized_mean"],
                "gt_mass_1cell_mean": shared["gt_mass_1cell_mean"],
                "gt_mass_2cell_mean": shared["gt_mass_2cell_mean"],
                "box_head_grad_norm": 0.0,
            }
        )
    return rows


def summarize_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        best = min(selected, key=lambda row: float(row["center_error_px_mean"]))
        tail = [row for row in selected if int(row["step"]) in TAIL_STEPS]
        if len(tail) != 5:
            raise RuntimeError(f"Missing tail checkpoints for {method}")
        values = torch.tensor(
            [float(row["center_error_px_mean"]) for row in tail], dtype=torch.float64
        )
        result[method] = {
            "best_step": int(best["step"]),
            "best": {
                key: float(best[key])
                for key in (
                    "center_error_px_mean", "center_error_px_median",
                    "center_error_lt_2px", "center_error_lt_4px", "center_error_lt_8px",
                )
            },
            "tail_center_error_px_mean": float(values.mean()),
            "tail_center_error_px_std_sample": float(values.std(unbiased=True)),
            "tail_center_error_px_min": float(values.min()),
            "tail_center_error_lt_2px_mean": float(
                torch.tensor([float(row["center_error_lt_2px"]) for row in tail]).mean()
            ),
            "tail_center_error_lt_4px_mean": float(
                torch.tensor([float(row["center_error_lt_4px"]) for row in tail]).mean()
            ),
            "tail_center_error_lt_8px_mean": float(
                torch.tensor([float(row["center_error_lt_8px"]) for row in tail]).mean()
            ),
            "final": {
                key: float(selected[-1][key])
                for key in (
                    "center_error_px_mean", "center_error_px_median",
                    "center_error_lt_2px", "center_error_lt_4px", "center_error_lt_8px",
                )
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4.7 RDQ attention mechanism audit")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()
    if (args.samples, args.steps, args.batch_size, args.eval_interval) != (20, 600, 2, 50):
        raise ValueError("Stage 4.7 is fixed to 20 samples, 600 steps, batch2, eval every50")

    config = copy.deepcopy(load_config(args.config))
    if config["model"]["variant"] != "rdq":
        raise ValueError("Stage 4.7 requires RDQ")
    config["model"]["backbone"]["out_index"] = 2
    config["model"]["backbone"]["fusion"] = "none"
    config["model"]["bbox_parameterization"] = "sigmoid_cxcywh"
    config["train"]["backbone_lr"] = 1e-4
    config["train"]["new_modules_lr"] = 1e-4
    config["data"]["train_color_jitter"] = 0.0
    if int(config["experiment"]["seed"]) != 42:
        raise ValueError("Stage 4.7 requires seed42")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(config["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    dataset = make_localization_dataset(config, "train", position_stats, limit_samples=20)
    cached_batches = list(make_loader(config, dataset, "val", 2))
    if sum(int(batch["bbox"].shape[0]) for batch in cached_batches) != 20:
        raise RuntimeError("Fixed audit dataset does not contain exactly 20 samples")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["experiment"]["output_dir"]) / f"stage4_attention_audit_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(position_stats, run_dir / "position_stats.json")

    seed_everything(42)
    model = build_localizer(config["model"], load_backbone_pretrained=False).to(device)
    groups = build_parameter_groups(model, 1e-4, 1e-4)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))
    iterator = itertools.cycle(cached_batches)

    print(f"device={device} run_dir={run_dir}", flush=True)
    initial_summaries, initial_shared, initial_predictions = evaluate_attention(
        model, cached_batches, device, config["data"]["image_size"]
    )
    print(
        f"ATTENTION_SHAPE={initial_shared['attention_shape']} "
        f"GRID={initial_shared['visual_grid_per_view']} "
        f"TOKENS={initial_shared['visual_token_count']}",
        flush=True,
    )
    if initial_shared["attention_shape"] != [2, 8, 1, 3456]:
        raise RuntimeError(
            f"Expected real per-head [2,8,1,3456], got {initial_shared['attention_shape']}"
        )

    rows = history_rows(0, 0.0, initial_summaries, initial_shared)
    started = time.perf_counter()
    best_fused_error = float("inf")
    best_predictions = initial_predictions
    best_step = 0
    last_grad_norm = 0.0
    for step in range(1, 601):
        model.train()
        batch = move(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["image"], batch["radar"], batch["radar_mask"])
        # This stage intentionally trains only normalized center L1.  Size and
        # XYZ branches remain present but contribute no objective.
        center_loss = torch.nn.functional.l1_loss(output["box"][:, :2], batch["bbox"][:, :2])
        if not bool(torch.isfinite(center_loss)):
            raise RuntimeError(f"Non-finite center loss at step {step}")
        center_loss.backward()
        grads = [
            parameter.grad.detach().float().square().sum()
            for parameter in model.box_head.parameters()
            if parameter.grad is not None
        ]
        last_grad_norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
        if not math.isfinite(last_grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
        optimizer.step()

        if step % 50 == 0:
            summaries, shared, predictions = evaluate_attention(
                model, cached_batches, device, config["data"]["image_size"]
            )
            elapsed = time.perf_counter() - started
            new_rows = history_rows(step, elapsed, summaries, shared)
            for row in new_rows:
                row["box_head_grad_norm"] = last_grad_norm
            rows.extend(new_rows)
            fused_error = summaries["fused_token"]["center_error_px_mean"]
            if fused_error < best_fused_error:
                best_fused_error = fused_error
                best_step = step
                best_predictions = predictions
                torch.save({"step": step, "model": model.state_dict()}, run_dir / "best_fused_center.pt")
            write_csv(rows, run_dir / "attention_history.csv")
            write_csv(predictions, run_dir / f"predictions_step{step:04d}.csv")
            print(
                f"step={step}/600 fused={fused_error:.2f}px "
                f"mean_attn={summaries['mean_attention']['center_error_px_mean']:.2f}px "
                f"peak={summaries['peak_attention']['center_error_px_mean']:.2f}px "
                f"best_fixed_h{shared['best_fixed_head_index']}="
                f"{summaries['best_fixed_head']['center_error_px_mean']:.2f}px "
                f"entropy={shared['attention_entropy_normalized_mean']:.3f} "
                f"mass1={shared['gt_mass_1cell_mean']:.4f} mass2={shared['gt_mass_2cell_mean']:.4f}",
                flush=True,
            )

    write_csv(best_predictions, run_dir / "best_fused_checkpoint_predictions.csv")
    summary = summarize_history(rows)
    final_shared = shared
    fused_tail = summary["fused_token"]["tail_center_error_px_mean"]
    mean_tail = summary["mean_attention"]["tail_center_error_px_mean"]
    fixed_tail = summary["best_fixed_head"]["tail_center_error_px_mean"]
    oracle_tail = summary["oracle_best_head"]["tail_center_error_px_mean"]
    grid_error = summary["grid_oracle"]["tail_center_error_px_mean"]
    attention_vs_fused_gain = 1.0 - mean_tail / fused_tail
    fixed_vs_mean_gain = 1.0 - fixed_tail / mean_tail
    oracle_vs_mean_gain = 1.0 - oracle_tail / mean_tail

    # Predeclared, mechanism-oriented rules. The oracle-best-head result is
    # never treated as a deployable predictor by itself.
    fixed_head_is_spatially_useful = (
        fixed_tail <= 16.0
        and summary["best_fixed_head"]["tail_center_error_lt_8px_mean"] >= 0.50
    )
    oracle_head_is_spatially_useful = (
        oracle_tail <= 16.0
        and summary["oracle_best_head"]["tail_center_error_lt_8px_mean"] >= 0.50
    )
    if attention_vs_fused_gain >= 0.30:
        diagnosis = "single_token_bottleneck_supported"
    elif fixed_vs_mean_gain >= 0.30 and fixed_head_is_spatially_useful:
        diagnosis = "head_aggregation_issue"
    elif oracle_vs_mean_gain >= 0.30 and oracle_head_is_spatially_useful:
        diagnosis = "head_specialization_inconclusive"
    else:
        diagnosis = "radar_query_correspondence_problem"
    quantization_likely_limiting = grid_error >= 0.5 * min(fused_tail, mean_tail)
    report = {
        "hypothesis": "Radar-conditioned query attends near the UAV; a single fused token may discard spatial location",
        "controlled_operation": "read genuine per-head RDQ attention and decode token coordinates without changing the model",
        "fixed_variables": {
            "backbone": "ResNet18 stride8-only out_index=2",
            "fusion": "RDQ with unchanged radar encoder and CrossAttentionBlock",
            "samples": 20,
            "seed": 42,
            "batch_size": 2,
            "learning_rates": {"backbone": 1e-4, "new_modules": 1e-4},
            "optimizer_steps": 600,
            "training_objective": "normalized CXCY center L1 only",
        },
        "attention_contract": final_shared,
        "metrics": summary,
        "best_fused_checkpoint_step": best_step,
        "tail_attention_vs_fused_gain": attention_vs_fused_gain,
        "tail_best_fixed_head_vs_mean_gain": fixed_vs_mean_gain,
        "tail_oracle_best_head_vs_mean_gain": oracle_vs_mean_gain,
        "quantization_likely_limiting": quantization_likely_limiting,
        "status": "rejected" if diagnosis == "radar_query_correspondence_problem" else "supported",
        "attention_near_uav_status": (
            "rejected" if diagnosis == "radar_query_correspondence_problem" else "supported"
        ),
        "single_token_bottleneck_status": (
            "supported" if diagnosis == "single_token_bottleneck_supported" else "inconclusive"
        ),
        "primary_diagnosis": diagnosis,
        "test_split_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(report, run_dir / "attention_audit_report.json")
    print(
        f"decision={diagnosis} attention_vs_fused_gain={attention_vs_fused_gain:.4f} "
        f"fixed_head_vs_mean_gain={fixed_vs_mean_gain:.4f} "
        f"grid_quantization_likely_limiting={quantization_likely_limiting}",
        flush=True,
    )
    print(f"completed={run_dir}", flush=True)


if __name__ == "__main__":
    main()
