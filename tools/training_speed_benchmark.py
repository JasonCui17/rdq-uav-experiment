#!/usr/bin/env python3
"""Single-configuration training throughput benchmark for Stage 4.

Each invocation benchmarks exactly one (workers, batch size) pair and appends
the result. It never launches a formal experiment or reads validation/test.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_bbox_stats, compute_position_stats  # noqa: E402
from rdq_uav.engine.localization import LocalizationLoss  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.models import build_localizer, build_parameter_groups  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


RESULT_FIELDS = [
    "timestamp", "phase", "status", "config", "gpu", "cpu_cores", "train_samples",
    "batch_size", "workers", "pin_memory", "persistent_workers", "prefetch_factor",
    "amp", "warmup_batches", "measure_batches", "measured_samples",
    "data_time_ms", "host_to_device_time_ms", "forward_time_ms",
    "backward_optimizer_time_ms", "compute_time_ms", "total_batch_time_ms",
    "batches_per_s", "samples_per_s", "peak_allocated_gb", "peak_reserved_gb",
    "gpu_total_gb", "reserved_fraction", "estimated_epoch_seconds",
    "estimated_150_epoch_hours", "stop_reason", "optimizer_lrs",
]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_success_by_pair(rows: list[dict[str, str]], phase: str) -> dict[tuple[int, int], dict[str, str]]:
    latest: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if row["phase"] == phase and row["status"] == "ok":
            latest[(int(row["workers"]), int(row["batch_size"]))] = row
    return latest


def write_recommendation(results_path: Path, output_path: Path) -> None:
    rows = read_results(results_path)
    worker_rows = list(latest_success_by_pair(rows, "workers").values())
    batch_rows = list(latest_success_by_pair(rows, "batch").values())
    recommendation: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": {
            "workers": "smallest worker count within 5% of maximum samples/s",
            "batch_stop": "OOM, reserved memory >85%, or <5% samples/s gain after doubling",
            "warning": "Batch size is a throughput candidate only; validate quality separately.",
        },
        "tested": {
            "workers": sorted({int(row["workers"]) for row in worker_rows}),
            "batches": sorted({int(row["batch_size"]) for row in batch_rows}),
        },
    }
    best_workers = None
    if worker_rows:
        maximum = max(float(row["samples_per_s"]) for row in worker_rows)
        eligible = [row for row in worker_rows if float(row["samples_per_s"]) >= 0.95 * maximum]
        chosen = min(eligible, key=lambda row: int(row["workers"]))
        best_workers = int(chosen["workers"])
        recommendation["best_workers"] = best_workers
        recommendation["worker_samples_per_s"] = float(chosen["samples_per_s"])
        recommendation["workers_complete"] = {0, 2, 4, 8}.issubset(
            {int(row["workers"]) for row in worker_rows if int(row["batch_size"]) == 8}
        )
        # The selected workers-phase batch=8 run has identical runtime
        # controls to the batch-phase baseline, so reuse it rather than asking
        # the user to benchmark the same pair twice.
        if not any(
            int(row["workers"]) == best_workers and int(row["batch_size"]) == 8
            for row in batch_rows
        ):
            worker_baseline = next(
                (
                    row for row in worker_rows
                    if int(row["workers"]) == best_workers and int(row["batch_size"]) == 8
                ),
                None,
            )
            if worker_baseline is not None:
                batch_rows.append(worker_baseline)
    if batch_rows:
        candidates = sorted(
            (row for row in batch_rows if best_workers is None or int(row["workers"]) == best_workers),
            key=lambda row: int(row["batch_size"]),
        )
        if not candidates:
            candidates = sorted(batch_rows, key=lambda row: int(row["batch_size"]))
        allowed: list[dict[str, str]] = []
        previous = None
        for row in candidates:
            if float(row["reserved_fraction"]) > 0.85:
                break
            if previous is not None and int(row["batch_size"]) == 2 * int(previous["batch_size"]):
                gain = float(row["samples_per_s"]) / float(previous["samples_per_s"]) - 1.0
                if gain < 0.05:
                    break
            allowed.append(row)
            previous = row
        if allowed:
            chosen = max(allowed, key=lambda row: float(row["samples_per_s"]))
            recommendation.update(
                {
                    "best_throughput_batch": int(chosen["batch_size"]),
                    "best_throughput_samples_per_s": float(chosen["samples_per_s"]),
                    "estimated_epoch_seconds": float(chosen["estimated_epoch_seconds"]),
                    "estimated_150_epoch_hours": float(chosen["estimated_150_epoch_hours"]),
                    "quality_validation_candidates": [
                        int(row["batch_size"])
                        for row in sorted(allowed, key=lambda x: float(x["samples_per_s"]), reverse=True)[:2]
                    ],
                }
            )
    output_path.write_text(yaml.safe_dump(recommendation, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--phase", choices=("workers", "batch"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/runtime_tuning")
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size <= 0:
        raise ValueError("workers must be >=0 and batch-size must be positive")
    if args.warmup_batches < 0 or args.measure_batches <= 0:
        raise ValueError("warmup must be >=0 and measure-batches must be positive")
    if args.phase == "workers" and args.batch_size != 8:
        raise ValueError("The workers phase is controlled at batch-size=8")

    config = load_config(args.config, args.overrides)
    if config.get("task", {}).get("name") != "single_uav_joint_2d_3d_localization":
        raise ValueError("Benchmark requires a Stage-4 localization config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage-4 runtime benchmark")
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device("cuda")
    pin_memory = args.workers > 0
    persistent_workers = args.workers > 0
    prefetch_factor = 2 if args.workers > 0 else None
    config["data"]["num_workers"] = args.workers
    config["data"]["pin_memory"] = pin_memory
    config["data"]["persistent_workers"] = persistent_workers
    config["data"]["prefetch_factor"] = 2

    manifest_dir = Path(config["data"]["manifest_dir"])
    position_stats = compute_position_stats(manifest_dir / "train.csv")
    bbox_stats = compute_bbox_stats(manifest_dir / "train.csv", config["data"]["panorama_size"])
    if config["model"].get("bbox_parameterization") == "sigmoid_center_log_size":
        config["model"]["bbox_reference_wh"] = [bbox_stats["width_median"], bbox_stats["height_median"]]
    dataset = make_localization_dataset(config, "train", position_stats)
    loader = make_loader(config, dataset, "train", args.batch_size)
    model = build_localizer(config["model"]).to(device)
    parameter_groups = build_parameter_groups(
        model,
        backbone_lr=float(config["train"]["backbone_lr"]),
        new_modules_lr=float(config["train"]["new_modules_lr"]),
    )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(config["train"]["weight_decay"]))
    criterion = LocalizationLoss(config["loss"])
    amp_enabled = bool(config["train"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_bytes = torch.cuda.get_device_properties(device).total_memory
    group_lrs = {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
    group_details = {
        str(group.get("name", index)): {
            "lr": float(group["lr"]),
            "trainable_parameters": sum(
                parameter.numel() for parameter in group["params"]
                if parameter.requires_grad
            ),
        }
        for index, group in enumerate(optimizer.param_groups)
    }
    settings = {
        "GPU": gpu_name,
        "CPU cores": os.cpu_count(),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor if prefetch_factor is not None else "N/A",
        "AMP": amp_enabled,
        "optimizer_parameter_groups": group_details,
    }
    print(yaml.safe_dump(settings, sort_keys=False), flush=True)
    print(f"Python={platform.python_version()} PyTorch={torch.__version__} train_samples={len(dataset)}")

    position_mean = torch.tensor(position_stats["mean"], dtype=torch.float32, device=device)
    position_std = torch.tensor(position_stats["std"], dtype=torch.float32, device=device)
    totals = {key: 0.0 for key in ("data", "h2d", "forward", "backward", "total")}
    measured_samples = 0
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)

    def next_batch() -> tuple[dict[str, Any], float]:
        nonlocal iterator
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, time.perf_counter() - started

    try:
        model.train()
        total_iterations = args.warmup_batches + args.measure_batches
        progress = None
        for iteration in range(total_iterations):
            torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
            raw_batch, data_seconds = next_batch()
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()
            batch = move_batch(raw_batch, device)
            torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - h2d_start

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
                losses = criterion(outputs["box"], batch["bbox"], outputs["position"], batch["position_normalized"])
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            if scaler.is_enabled():
                scaler.scale(losses["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
                optimizer.step()
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - backward_start
            total_seconds = time.perf_counter() - batch_start

            if iteration >= args.warmup_batches:
                # Start tqdm only after warmup so its displayed rate/ETA uses
                # the same measurement window as the CSV batch/s result.
                if progress is None:
                    progress = tqdm(
                        total=args.measure_batches, desc="Benchmark", unit="batch",
                        bar_format="{desc} | {n_fmt}/{total_fmt} | {percentage:3.0f}% | "
                        "{postfix} | {rate_fmt} | ETA {remaining}",
                    )
                count = int(batch["bbox"].shape[0])
                measured_samples += count
                totals["data"] += data_seconds
                totals["h2d"] += h2d_seconds
                totals["forward"] += forward_seconds
                totals["backward"] += backward_seconds
                totals["total"] += total_seconds
                measured_batches = iteration - args.warmup_batches + 1
                gpu_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
                progress.set_postfix_str(
                    f"GPU_mem={gpu_memory:.2f}G | data_ms={1000*totals['data']/measured_batches:.1f} "
                    f"| compute_ms={1000*(totals['forward']+totals['backward'])/measured_batches:.1f} "
                    f"| samples/s={measured_samples/totals['total']:.1f}", refresh=False,
                )
                progress.update(1)
        assert progress is not None
        progress.close()
        measured_batches = args.measure_batches
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        samples_per_s = measured_samples / totals["total"]
        batches_per_s = measured_batches / totals["total"]
        reserved_fraction = peak_reserved / gpu_total_bytes
        stop_reason = "reserved_memory_above_85_percent" if reserved_fraction > 0.85 else ""
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "phase": args.phase,
            "status": "ok", "config": str(args.config.resolve()), "gpu": gpu_name,
            "cpu_cores": os.cpu_count(), "train_samples": len(dataset), "batch_size": args.batch_size,
            "workers": args.workers, "pin_memory": pin_memory,
            "persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor or 0,
            "amp": amp_enabled, "warmup_batches": args.warmup_batches,
            "measure_batches": measured_batches, "measured_samples": measured_samples,
            "data_time_ms": 1000 * totals["data"] / measured_batches,
            "host_to_device_time_ms": 1000 * totals["h2d"] / measured_batches,
            "forward_time_ms": 1000 * totals["forward"] / measured_batches,
            "backward_optimizer_time_ms": 1000 * totals["backward"] / measured_batches,
            "compute_time_ms": 1000 * (totals["forward"] + totals["backward"]) / measured_batches,
            "total_batch_time_ms": 1000 * totals["total"] / measured_batches,
            "batches_per_s": batches_per_s, "samples_per_s": samples_per_s,
            "peak_allocated_gb": peak_allocated / (1024**3),
            "peak_reserved_gb": peak_reserved / (1024**3),
            "gpu_total_gb": gpu_total_bytes / (1024**3), "reserved_fraction": reserved_fraction,
            "estimated_epoch_seconds": len(dataset) / samples_per_s,
            "estimated_150_epoch_hours": len(dataset) * 150 / samples_per_s / 3600,
            "stop_reason": stop_reason, "optimizer_lrs": str(group_lrs),
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "phase": args.phase,
            "status": "oom", "config": str(args.config.resolve()), "gpu": gpu_name,
            "cpu_cores": os.cpu_count(), "train_samples": len(dataset), "batch_size": args.batch_size,
            "workers": args.workers, "pin_memory": pin_memory,
            "persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor or 0,
            "amp": amp_enabled, "warmup_batches": args.warmup_batches,
            "measure_batches": 0, "measured_samples": 0,
            "gpu_total_gb": gpu_total_bytes / (1024**3), "stop_reason": "cuda_oom",
            "optimizer_lrs": str(group_lrs),
        }
        print("CUDA OOM: stop increasing batch size.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "benchmark_results.csv"
    recommendation_path = args.output_dir / "recommended_runtime_config.yaml"
    append_result(results_path, row)
    write_recommendation(results_path, recommendation_path)
    print(yaml.safe_dump(row, sort_keys=False), flush=True)
    print(f"results={results_path.resolve()}")
    print(f"recommendation={recommendation_path.resolve()}")


if __name__ == "__main__":
    main()
