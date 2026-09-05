#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.engine.metrics import ClassificationMetrics  # noqa: E402
from rdq_uav.engine.sequence import aggregate_temporal_blocks  # noqa: E402
from rdq_uav.engine.trainer import run_epoch  # noqa: E402
from rdq_uav.experiment import make_dataset, make_loader  # noqa: E402
from rdq_uav.models import build_model  # noqa: E402
from rdq_uav.utils.io import write_json  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


def metrics_for_predictions(rows: list[dict], num_classes: int) -> dict:
    meter = ClassificationMetrics(num_classes)
    meter.update_predictions(
        torch.tensor([row["prediction"] for row in rows]),
        torch.tensor([row["target"] for row in rows]),
    )
    return meter.compute()


def add_grouped_metrics(metrics: dict, predictions: list[dict], num_classes: int) -> None:
    bins = (("0-10m", 0.0, 10.0), ("10-20m", 10.0, 20.0), ("20-30m", 20.0, 30.0), ("30m+", 30.0, float("inf")))
    metrics["distance_bins"] = {}
    for name, lower, upper in bins:
        rows = [row for row in predictions if lower <= row["distance_m"] < upper]
        if rows:
            metrics["distance_bins"][name] = metrics_for_predictions(rows, num_classes)
    per_sequence = {}
    for class_name in sorted({row["class_name"] for row in predictions}):
        rows = [row for row in predictions if row["class_name"] == class_name]
        per_sequence[class_name] = metrics_for_predictions(rows, num_classes)
    metrics["per_sequence"] = per_sequence
    metrics["sequence_accuracy_macro"] = sum(
        value["accuracy"] for value in per_sequence.values()
    ) / len(per_sequence)
    if all(row.get("probabilities") is not None for row in predictions):
        all_metrics, _ = aggregate_temporal_blocks(predictions, num_classes)
        metrics["temporal_block_soft_vote_all"] = all_metrics
        # Official bbox area is available only in the oracle-bbox manifests.
        # Do not silently call arbitrary zero-score frames "bbox keyframes" for
        # ordinary manifests that do not carry 2D annotations.
        if any(float(row.get("bbox_area_px", 0.0)) > 0.0 for row in predictions):
            keyframe_metrics, keyframe_rows = aggregate_temporal_blocks(
                predictions,
                num_classes,
                top_k=5,
                min_gap_seconds=1.5,
                score_key="bbox_area_px",
            )
            metrics["temporal_block_soft_vote_top5_bbox_area_gap1.5s"] = keyframe_metrics
            metrics["temporal_block_keyframes"] = keyframe_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and radar ablations")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--radar-mode",
        choices=("normal", "zero", "shuffle_same_class", "shift"),
        default="normal",
    )
    parser.add_argument("--radar-shift-seconds", type=float, default=0.0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.config is None:
        if args.overrides:
            raise ValueError("--set overrides require an explicit --config")
        config = raw_checkpoint["config"]
    else:
        config = load_config(args.config, args.overrides)
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_dataset(
        config,
        args.split,
        radar_mode=args.radar_mode,
        radar_shift_seconds=args.radar_shift_seconds,
    )
    loader = make_loader(
        config, dataset, args.split, int(config["evaluation"]["batch_size"])
    )
    # The checkpoint contains the full backbone; downloading ImageNet weights
    # here is redundant and makes otherwise offline evaluation fragile.
    model = build_model(config["model"], load_backbone_pretrained=False).to(device)
    model.load_state_dict(raw_checkpoint["model"])
    criterion = torch.nn.CrossEntropyLoss()
    metrics, predictions = run_epoch(
        model=model,
        loader=loader,
        device=device,
        num_classes=int(config["model"]["num_classes"]),
        criterion=criterion,
        optimizer=None,
        amp=bool(config["train"]["amp"]) and device.type == "cuda",
        log_interval=0,
    )
    add_grouped_metrics(metrics, predictions, int(config["model"]["num_classes"]))
    shift_suffix = f"_{args.radar_shift_seconds:g}s" if args.radar_mode == "shift" else ""
    output_stem = f"{args.split}_{args.radar_mode}{shift_suffix}"
    output_dir = args.checkpoint.resolve().parent
    write_json(metrics, output_dir / f"metrics_{output_stem}.json")
    if bool(config["evaluation"]["save_predictions"]):
        csv_predictions = []
        for row in predictions:
            flat = dict(row)
            for index, probability in enumerate(flat.pop("probabilities")):
                flat[f"probability_{index}"] = probability
            csv_predictions.append(flat)
        fields = list(csv_predictions[0])
        with (output_dir / f"predictions_{output_stem}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_predictions)
    print(metrics)


if __name__ == "__main__":
    main()
