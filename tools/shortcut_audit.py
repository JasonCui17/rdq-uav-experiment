#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.data.localization import compute_position_stats  # noqa: E402
from rdq_uav.localization_experiment import make_loader, make_localization_dataset  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


CANVAS_HEIGHT = 288
CANVAS_WIDTH = 768
LOW_FREQUENCY_SIZE = (36, 48)
BACKGROUND_EPOCHS = 100
BACKGROUND_BATCH_SIZE = 64
BACKGROUND_LR = 1e-3
TIME_GAP_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0, math.inf)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if path.name not in {"train.csv", "val.csv"}:
        raise ValueError("Shortcut audit is restricted to train/val manifests")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def targets(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(row["gt_time"]) for row in rows], dtype=np.float64)
    centers = np.asarray(
        [
            [
                (float(row["official_bbox_x1"]) + float(row["official_bbox_x2"])) / (2 * 2560),
                (float(row["official_bbox_y1"]) + float(row["official_bbox_y2"])) / (2 * 960),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    positions = np.asarray(
        [[float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])] for row in rows],
        dtype=np.float64,
    )
    return times, centers, positions


def error_metrics(
    pred_center: np.ndarray,
    gt_center: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
) -> dict[str, float]:
    scale = np.asarray([CANVAS_WIDTH, CANVAS_HEIGHT], dtype=np.float64)
    center_error = np.linalg.norm((pred_center - gt_center) * scale, axis=1)
    xyz_delta = pred_xyz - gt_xyz
    xyz_error = np.linalg.norm(xyz_delta, axis=1)
    return {
        "center_error_mean_px": float(center_error.mean()),
        "center_error_median_px": float(np.median(center_error)),
        "position_error_mean_m": float(xyz_error.mean()),
        "position_error_median_m": float(np.median(xyz_error)),
        "mae_x_m": float(np.abs(xyz_delta[:, 0]).mean()),
        "mae_y_m": float(np.abs(xyz_delta[:, 1]).mean()),
        "mae_z_m": float(np.abs(xyz_delta[:, 2]).mean()),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_predictions(
    train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any]]:
    _, train_centers, train_xyz = targets(train_rows)
    val_times, val_centers, val_xyz = targets(val_rows)
    global_center = train_centers.mean(axis=0)
    global_xyz = train_xyz.mean(axis=0)

    grouped: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for row in train_rows:
        _, center, xyz = targets([row])
        grouped[row["sequence_id"]].append((float(row["gt_time"]), np.concatenate([center[0], xyz[0]])))
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sequence, values in grouped.items():
        ordered = sorted(values, key=lambda value: value[0])
        series[sequence] = (
            np.asarray([item[0] for item in ordered], dtype=np.float64),
            np.stack([item[1] for item in ordered]),
        )

    predictions: dict[str, list[np.ndarray]] = {
        method: [] for method in ("global_mean", "sequence_mean", "nearest_train_time", "linear_interpolation")
    }
    prediction_rows: list[dict[str, Any]] = []
    nearest_gaps: list[float] = []
    nearest_errors_2d: list[float] = []
    nearest_errors_3d: list[float] = []
    for index, row in enumerate(val_rows):
        sequence = row["sequence_id"]
        if sequence not in series:
            raise RuntimeError(f"No training samples for validation sequence {sequence}")
        train_times, train_values = series[sequence]
        sequence_mean = train_values.mean(axis=0)
        time_value = val_times[index]
        insertion = int(np.searchsorted(train_times, time_value))
        candidates = [min(max(insertion, 0), len(train_times) - 1)]
        if insertion > 0:
            candidates.append(insertion - 1)
        nearest_index = min(candidates, key=lambda idx: abs(train_times[idx] - time_value))
        nearest = train_values[nearest_index]
        nearest_gap = abs(float(train_times[nearest_index] - time_value))

        if insertion <= 0:
            interpolated = train_values[0]
            bracketed = False
        elif insertion >= len(train_times):
            interpolated = train_values[-1]
            bracketed = False
        else:
            left_time, right_time = train_times[insertion - 1], train_times[insertion]
            alpha = (time_value - left_time) / (right_time - left_time)
            interpolated = train_values[insertion - 1] * (1.0 - alpha) + train_values[insertion] * alpha
            bracketed = True

        method_values = {
            "global_mean": np.concatenate([global_center, global_xyz]),
            "sequence_mean": sequence_mean,
            "nearest_train_time": nearest,
            "linear_interpolation": interpolated,
        }
        for method, prediction in method_values.items():
            predictions[method].append(prediction)
            center_delta_px = (prediction[:2] - val_centers[index]) * np.asarray([CANVAS_WIDTH, CANVAS_HEIGHT])
            e2d = float(np.linalg.norm(center_delta_px))
            e3d = float(np.linalg.norm(prediction[2:] - val_xyz[index]))
            prediction_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "sequence_id": sequence,
                    "temporal_block": row["temporal_block"],
                    "gt_time": row["gt_time"],
                    "method": method,
                    "pred_cx": prediction[0], "pred_cy": prediction[1],
                    "gt_cx": val_centers[index, 0], "gt_cy": val_centers[index, 1],
                    "pred_x": prediction[2], "pred_y": prediction[3], "pred_z": prediction[4],
                    "gt_x": val_xyz[index, 0], "gt_y": val_xyz[index, 1], "gt_z": val_xyz[index, 2],
                    "center_error_px": e2d,
                    "position_error_m": e3d,
                    "nearest_train_time_gap_s": nearest_gap if method in {"nearest_train_time", "linear_interpolation"} else "",
                    "interpolation_bracketed": bracketed if method == "linear_interpolation" else "",
                }
            )
            if method == "nearest_train_time":
                nearest_gaps.append(nearest_gap)
                nearest_errors_2d.append(e2d)
                nearest_errors_3d.append(e3d)

    metrics: dict[str, dict[str, float]] = {}
    for method, values in predictions.items():
        array = np.stack(values)
        metrics[method] = error_metrics(array[:, :2], val_centers, array[:, 2:], val_xyz)

    gaps = np.asarray(nearest_gaps, dtype=np.float64)
    gap_report: dict[str, Any] = {
        "mean_s": float(gaps.mean()),
        "median_s": float(np.median(gaps)),
        "p95_s": float(np.quantile(gaps, 0.95)),
        "max_s": float(gaps.max()),
        "bins": [],
    }
    e2 = np.asarray(nearest_errors_2d)
    e3 = np.asarray(nearest_errors_3d)
    for low, high in zip(TIME_GAP_EDGES[:-1], TIME_GAP_EDGES[1:]):
        mask = (gaps >= low) & (gaps < high)
        label = f"{low:g}-{high:g}s" if math.isfinite(high) else f">={low:g}s"
        gap_report["bins"].append(
            {
                "bin": label,
                "count": int(mask.sum()),
                "center_error_mean_px": float(e2[mask].mean()) if mask.any() else None,
                "position_error_mean_m": float(e3[mask].mean()) if mask.any() else None,
            }
        )
    return prediction_rows, metrics, gap_report


class BackgroundRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def make_frozen_encoder(device: torch.device) -> nn.Module:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import timm

    encoder = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg")
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


@torch.no_grad()
def extract_background_features(
    loader: DataLoader, encoder: nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    features: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    encoder.eval()
    for batch in loader:
        images = batch["image"].to(device)
        batch_size, views = images.shape[:2]
        low_frequency = F.interpolate(
            images.flatten(0, 1), size=LOW_FREQUENCY_SIZE, mode="area"
        )
        encoded = encoder(low_frequency).reshape(batch_size, views, -1).flatten(1)
        features.append(encoded.cpu())
        targets_all.append(torch.cat([batch["bbox"][:, :2], batch["position"]], dim=1))
        for index in range(batch_size):
            metadata.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "sequence_id": batch["sequence_id"][index],
                    "temporal_block": int(batch["temporal_block"][index]),
                    "gt_time": float(batch["gt_time"][index]),
                }
            )
    return torch.cat(features), torch.cat(targets_all), metadata


def background_audit(
    config: dict[str, Any], position_stats: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    train_dataset = make_localization_dataset(config, "train", position_stats)
    val_dataset = make_localization_dataset(config, "val", position_stats)
    train_loader = make_loader(config, train_dataset, "val", 64)
    val_loader = make_loader(config, val_dataset, "val", 64)
    encoder = make_frozen_encoder(device)
    train_features, train_targets, _ = extract_background_features(train_loader, encoder, device)
    val_features, val_targets, val_metadata = extract_background_features(val_loader, encoder, device)
    target_mean = train_targets.double().mean(dim=0).float()
    target_std = train_targets.double().std(dim=0, unbiased=False).float().clamp_min(1e-6)
    normalized_train_targets = (train_targets - target_mean) / target_std

    seed_everything(42)
    head = BackgroundRegressor(int(train_features.shape[1])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=BACKGROUND_LR, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(42)
    feature_loader = DataLoader(
        TensorDataset(train_features, normalized_train_targets),
        batch_size=BACKGROUND_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, BACKGROUND_EPOCHS + 1):
        head.train()
        total = 0.0
        count = 0
        for feature_batch, target_batch in feature_loader:
            feature_batch, target_batch = feature_batch.to(device), target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(feature_batch)
            loss = F.smooth_l1_loss(prediction, target_batch)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(feature_batch)
            count += len(feature_batch)
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "train_loss": total / count})
            print(f"background epoch={epoch}/{BACKGROUND_EPOCHS} train_loss={total/count:.6f}", flush=True)

    head.eval()
    with torch.no_grad():
        normalized_prediction = head(val_features.to(device)).cpu()
    prediction = normalized_prediction * target_std + target_mean
    pred_np, gt_np = prediction.numpy().astype(np.float64), val_targets.numpy().astype(np.float64)
    metrics = error_metrics(pred_np[:, :2], gt_np[:, :2], pred_np[:, 2:], gt_np[:, 2:])
    rows: list[dict[str, Any]] = []
    for metadata, pred, gt in zip(val_metadata, pred_np, gt_np):
        e2 = float(np.linalg.norm((pred[:2] - gt[:2]) * np.asarray([CANVAS_WIDTH, CANVAS_HEIGHT])))
        e3 = float(np.linalg.norm(pred[2:] - gt[2:]))
        rows.append(
            {
                **metadata,
                "pred_cx": pred[0], "pred_cy": pred[1], "gt_cx": gt[0], "gt_cy": gt[1],
                "pred_x": pred[2], "pred_y": pred[3], "pred_z": pred[4],
                "gt_x": gt[2], "gt_y": gt[3], "gt_z": gt[4],
                "center_error_px": e2, "position_error_m": e3,
            }
        )
    protocol = {
        "low_frequency_per_view": list(LOW_FREQUENCY_SIZE),
        "encoder": "frozen_timm_resnet18_a1_in1k",
        "pooling": "global_average_pool_per_view_then_concatenate",
        "head": "Linear(1024,128)-ReLU-Linear(128,5)",
        "epochs": BACKGROUND_EPOCHS,
        "batch_size": BACKGROUND_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": BACKGROUND_LR,
        "target_normalization": {"mean": target_mean.tolist(), "std": target_std.tolist()},
        "training_history": history,
        "uses_gt_bbox_for_input": False,
        "uses_radar": False,
    }
    return rows, metrics, protocol


def gain(method: dict[str, float], global_metrics: dict[str, float]) -> tuple[float, float]:
    return (
        1.0 - method["center_error_mean_px"] / global_metrics["center_error_mean_px"],
        1.0 - method["position_error_mean_m"] / global_metrics["position_error_mean_m"],
    )


def status(score: float, supported_threshold: float) -> str:
    if score >= supported_threshold:
        return "supported"
    if score >= 0.30:
        return "moderate"
    return "weak"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4.4 localization shortcut audit")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/localization/rdq.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["data"]["image_mode"] != "dual_full" or config["data"]["bbox_mode"] != "full":
        raise ValueError("Shortcut audit requires full fixed dual-view input")
    if list(config["data"]["image_size"]) != [288, 384]:
        raise ValueError("Shortcut audit requires the fixed 288x384 per-view canvas")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(config["data"]["manifest_dir"])
    train_path, val_path = manifest_dir / "train.csv", manifest_dir / "val.csv"
    train_rows, val_rows = read_manifest(train_path), read_manifest(val_path)
    position_stats = compute_position_stats(train_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["experiment"]["output_dir"]) / f"stage4_shortcut_audit_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"device={device} run_dir={run_dir} train={len(train_rows)} val={len(val_rows)}", flush=True)

    trajectory_rows, trajectory_metrics, gap_report = trajectory_predictions(train_rows, val_rows)
    write_csv(trajectory_rows, run_dir / "trajectory_predictions.csv")
    background_rows, background_metrics, background_protocol = background_audit(
        config, position_stats, device
    )
    write_csv(background_rows, run_dir / "background_predictions.csv")

    global_metrics = trajectory_metrics["global_mean"]
    summary_rows: list[dict[str, Any]] = []
    all_metrics = {**trajectory_metrics, "background_only": background_metrics}
    for method, metrics in all_metrics.items():
        gain_2d, gain_3d = gain(metrics, global_metrics)
        summary_rows.append({"method": method, **metrics, "gain_2d": gain_2d, "gain_3d": gain_3d})
    write_csv(summary_rows, run_dir / "shortcut_summary.csv")

    trajectory_candidates = [row for row in summary_rows if row["method"] in {"nearest_train_time", "linear_interpolation"}]
    trajectory_score = max(min(float(row["gain_2d"]), float(row["gain_3d"])) for row in trajectory_candidates)
    background_row = next(row for row in summary_rows if row["method"] == "background_only")
    background_score = min(float(background_row["gain_2d"]), float(background_row["gain_3d"]))
    report = {
        "research_question": "Can session/time/background predict 2D center and 3D XYZ in UAV-positive frames?",
        "protocol": {
            "train_manifest": str(train_path.resolve()),
            "val_manifest": str(val_path.resolve()),
            "test_split_accessed": False,
            "canvas": [CANVAS_HEIGHT, CANVAS_WIDTH],
            "trajectory_lookup_scope": "same sequence train samples only",
            "interpolation_outside_train_range": "clamp to nearest train endpoint",
            "trajectory_status_score": "max over nearest/interpolation of min(Gain2D, Gain3D)",
            "background_status_score": "min(Gain2D, Gain3D)",
        },
        "trajectory": {
            "hypothesis": "train-only session/time information predicts validation target center and XYZ",
            "controlled_variable": "trajectory estimator",
            "fixed_variables": "same train/val manifest and canvas",
            "metrics": trajectory_metrics,
            "nearest_time_gap": gap_report,
            "score": trajectory_score,
            "status": status(trajectory_score, 0.70),
        },
        "background": {
            "hypothesis": "fixed low-frequency full images predict validation target center and XYZ",
            "controlled_variable": "low-frequency RGB evidence without bbox or radar",
            "fixed_variables": "same train/val manifest and fixed dual-view transform",
            "metrics": background_metrics,
            "protocol": background_protocol,
            "score": background_score,
            "status": status(background_score, 0.50),
        },
        "summary": summary_rows,
    }
    write_json(report, run_dir / "shortcut_report.json")
    print(json.dumps({"summary": summary_rows, "trajectory_status": report["trajectory"]["status"], "background_status": report["background"]["status"]}, indent=2), flush=True)
    print(f"completed={run_dir}", flush=True)


if __name__ == "__main__":
    main()
