#!/usr/bin/env python3
"""Train and evaluate the public MMUAV 9D ordinary-LSTM classifier."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rdq_uav.mmuav.attention_lstm import AttentionLSTMClassifier  # noqa: E402
from rdq_uav.mmuav.public_lstm import (  # noqa: E402
    PublicLSTMClassifier,
    binary_metrics,
    public_augment_batch,
)


DEFAULT_DATASET = REPO_ROOT / "outputs/mmuav_paper_reproduction/datasets/public_9d"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/mmuav_paper_reproduction/classification/public_9d"
DEFAULT_ATTENTION_OUTPUT = (
    REPO_ROOT / "outputs/mmuav_paper_reproduction/classification/attention_9d"
)
DEFAULT_OFFICIAL = Path(
    "/home/jasoncui/projects/open_source/Multi-Modal-UAV/"
    "point_cloud_processing/tracker/lstm_model.pth"
)


@dataclass
class EarlyStoppingState:
    patience: int = 15
    min_delta: float = 0.0
    best_loss: float = float("inf")
    best_epoch: int = 0
    counter: int = 0

    def update(self, val_loss: float, epoch: int) -> tuple[bool, bool]:
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
        else:
            self.counter += 1
        return improved, self.counter >= self.patience


def create_model(name: str) -> nn.Module:
    if name == "public_lstm":
        return PublicLSTMClassifier()
    if name == "attention_lstm":
        return AttentionLSTMClassifier()
    raise ValueError(f"Unknown model: {name}")


def forward_model(
    model: nn.Module, inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(inputs)
    if isinstance(output, tuple):
        logits, attention = output
        return logits, attention
    return output, None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(path / "feature_train.npy", allow_pickle=False)
    y_train = np.load(path / "label_train.npy", allow_pickle=False).reshape(-1)
    x_val = np.load(path / "feature_val.npy", allow_pickle=False)
    y_val = np.load(path / "label_val.npy", allow_pickle=False).reshape(-1)
    for name, array in (("x_train", x_train), ("x_val", x_val)):
        if array.ndim != 3 or array.shape[1:] != (20, 9):
            raise ValueError(f"{name} must be [N,20,9], got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if len(x_train) != len(y_train) or len(x_val) != len(y_val):
        raise ValueError("Feature/label length mismatch")
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Train and validation datasets must be non-empty")
    if np.sum(y_train == 1) == 0 or np.sum(y_val == 1) == 0:
        raise RuntimeError("Positive labels are required in both train and validation data")
    return x_train, y_train, x_val, y_val


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
) -> tuple[float, dict[str, float | int], np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    total_loss = 0.0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    attention_batches: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, attention = forward_model(model, inputs)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            targets.append(labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            if attention is not None:
                attention_batches.append(attention.cpu().numpy())
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    attention_array = np.concatenate(attention_batches) if attention_batches else None
    return (
        total_loss / len(target), binary_metrics(target, prediction).as_dict(),
        target, prediction, attention_array,
    )


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint at {path}")
    return payload


def save_confusion(path: Path, metrics: dict[str, float | int], title: str) -> None:
    matrix = np.array([
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks((0, 1), labels=("background", "UAV"))
    axis.set_yticks((0, 1), labels=("background", "UAV"))
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_attention_statistics(
    output_dir: Path, attention: np.ndarray, checkpoint: Path,
) -> None:
    if attention.ndim != 2 or attention.shape[1] != 20:
        raise ValueError(f"Expected validation attention [N,20], got {attention.shape}")
    mean = attention.mean(axis=0)
    std = attention.std(axis=0)
    timesteps = [f"t{index:02d}" for index in range(1, 21)]
    payload = {
        "checkpoint": str(checkpoint),
        "num_validation_samples": int(attention.shape[0]),
        "mean_attention_weight_per_timestep": dict(zip(timesteps, mean.tolist())),
        "std_attention_weight_per_timestep": dict(zip(timesteps, std.tolist())),
        "mean_weight_sum": float(mean.sum()),
    }
    (output_dir / "attention_statistics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    positions = np.arange(1, 21)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.errorbar(positions, mean, yerr=std, marker="o", capsize=3)
    axis.axhline(1.0 / 20.0, color="gray", linestyle="--", label="uniform=0.05")
    axis.set_xticks(positions)
    axis.set_xlabel("LSTM timestep")
    axis.set_ylabel("Attention weight (mean ± std)")
    axis.set_title("M1 validation attention weights")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "attention_weights.png", dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_ATTENTION_OUTPUT if args.model == "attention_lstm" else DEFAULT_OUTPUT
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    x_train, y_train, x_val, y_val = load_dataset(args.dataset_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(
        f"device={device} train={len(y_train)} positive={int(y_train.sum())} "
        f"val={len(y_val)} positive={int(y_val.sum())}"
    )
    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    # The public script augments the complete tensor once before DataLoader creation.
    preaugment_rng = np.random.default_rng(args.seed)
    train_tensor = public_augment_batch(train_tensor, preaugment_rng)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_labels), batch_size=args.batch_size,
        shuffle=True, generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = create_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    batch_rng = np.random.default_rng(args.seed + 1)
    early_stopping = EarlyStoppingState(args.patience, args.min_delta)
    best_f1 = -1.0
    best_f1_epoch = 0
    history: list[dict[str, Any]] = []
    early_stopped = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_targets: list[np.ndarray] = []
        train_predictions: list[np.ndarray] = []
        for inputs, labels in train_loader:
            inputs = public_augment_batch(inputs, batch_rng).to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_model(model, inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            train_targets.append(labels.detach().cpu().numpy())
            train_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        train_target = np.concatenate(train_targets)
        train_prediction = np.concatenate(train_predictions)
        train_metrics = binary_metrics(train_target, train_prediction).as_dict()
        train_loss = running_loss / len(train_target)
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        improved, should_stop = early_stopping.update(val_loss, epoch)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "patience_counter": early_stopping.counter,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if improved:
            torch.save(model.state_dict(), args.output_dir / "best_val_loss.pth")
        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_f1_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "best_val_f1.pth")
        print(
            f"epoch={epoch:02d}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"patience_counter={early_stopping.counter}/{args.patience}"
        )
        if should_stop:
            early_stopped = True
            print("EARLY_STOPPING")
            print(f"best_epoch={early_stopping.best_epoch}")
            print(f"best_val_loss={early_stopping.best_loss:.8f}")
            print(f"stopped_epoch={epoch}")
            break
    torch.save(model.state_dict(), args.output_dir / "last.pth")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    comparisons: list[dict[str, Any]] = []
    if args.model == "attention_lstm":
        checkpoint_specs = [
            ("official_public_checkpoint", args.official_checkpoint, "public_lstm"),
            ("M0_retrained_best_loss", DEFAULT_OUTPUT / "best_val_loss.pth", "public_lstm"),
            ("M0_retrained_best_f1", DEFAULT_OUTPUT / "best_val_f1.pth", "public_lstm"),
            ("M1_attention_best_loss", args.output_dir / "best_val_loss.pth", "attention_lstm"),
            ("M1_attention_best_f1", args.output_dir / "best_val_f1.pth", "attention_lstm"),
        ]
        selected_name = "M1_attention_best_f1"
    else:
        checkpoint_specs = [
            ("official_public_checkpoint", args.official_checkpoint, "public_lstm"),
            ("M0_retrained_best_loss", args.output_dir / "best_val_loss.pth", "public_lstm"),
            ("M0_retrained_best_f1", args.output_dir / "best_val_f1.pth", "public_lstm"),
        ]
        selected_name = "M0_retrained_best_f1"
    metrics_by_name: dict[str, Any] = {}
    for name, checkpoint_path, architecture in checkpoint_specs:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Required comparison checkpoint is missing: {checkpoint_path}")
        candidate = create_model(architecture).to(device)
        candidate.load_state_dict(load_state_dict(checkpoint_path, device))
        loss, metrics, _, _, _ = evaluate(candidate, val_loader, criterion, device)
        item = {"model": name, "checkpoint": str(checkpoint_path), "val_loss": loss, **metrics}
        comparisons.append(item)
        metrics_by_name[name] = item
    with (args.output_dir / "checkpoint_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    if args.model == "attention_lstm":
        comparison_path = args.output_dir.parent / "comparison_9d_attention.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    selected = metrics_by_name[selected_name]
    save_confusion(
        args.output_dir / "confusion_matrix.png", selected,
        f"{args.model} validation",
    )
    if args.model == "attention_lstm":
        attention_checkpoint = args.output_dir / "best_val_f1.pth"
        attention_model = create_model("attention_lstm").to(device)
        attention_model.load_state_dict(load_state_dict(attention_checkpoint, device))
        _, _, _, _, attention = evaluate(attention_model, val_loader, criterion, device)
        if attention is None:
            raise RuntimeError("Attention model did not return attention weights")
        save_attention_statistics(args.output_dir, attention, attention_checkpoint)
    report = {
        "training_config": {
            "model": args.model, "seed": args.seed, "max_epochs": args.epochs,
            "actual_epochs": len(history), "batch_size": args.batch_size,
            "optimizer": "Adam", "learning_rate": args.learning_rate,
            "loss": "CrossEntropyLoss", "checkpoint_selection": ["val_loss", "val_f1"],
            "patience": args.patience, "min_delta": args.min_delta,
            "early_stopped": early_stopped,
            "best_val_loss_epoch": early_stopping.best_epoch,
            "best_val_f1_epoch": best_f1_epoch,
            "augmentation": "public double application: preaugment once + per-batch",
            "class_balancing": "none",
        },
        "dataset": {
            "path": str(args.dataset_dir.resolve()), "train_samples": len(y_train),
            "train_positive": int(y_train.sum()), "val_samples": len(y_val),
            "val_positive": int(y_val.sum()),
        },
        "checkpoint_metrics": metrics_by_name,
        "all_background_collapse": bool(selected["positive_predictions"] == 0),
    }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"results={args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("public_lstm", "attention_lstm"), default="public_lstm"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
