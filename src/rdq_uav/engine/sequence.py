from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from rdq_uav.engine.metrics import ClassificationMetrics


def _select_with_time_gap(
    rows: list[dict[str, Any]],
    top_k: int,
    min_gap_seconds: float,
    score_key: str,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
    selected: list[dict[str, Any]] = []
    for row in ranked:
        timestamp = float(row["gt_time"])
        if all(abs(timestamp - float(other["gt_time"])) >= min_gap_seconds for other in selected):
            selected.append(row)
            if len(selected) == top_k:
                break
    return selected


def aggregate_temporal_blocks(
    rows: list[dict[str, Any]],
    num_classes: int,
    *,
    top_k: int | None = None,
    min_gap_seconds: float = 0.0,
    score_key: str = "bbox_area_px",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Soft-vote frame probabilities inside each sequence/temporal block."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sequence_id"]), int(row["temporal_block"]))].append(row)
    meter = ClassificationMetrics(num_classes)
    aggregated = []
    for (sequence_id, temporal_block), group in sorted(grouped.items()):
        targets = {int(row["target"]) for row in group}
        if len(targets) != 1:
            raise ValueError(f"Mixed labels in {(sequence_id, temporal_block)}: {targets}")
        selected = (
            group
            if top_k is None
            else _select_with_time_gap(group, top_k, min_gap_seconds, score_key)
        )
        if not selected:
            raise ValueError(f"No keyframes selected for {(sequence_id, temporal_block)}")
        probabilities = torch.tensor(
            [row["probabilities"] for row in selected], dtype=torch.float64
        ).mean(dim=0)
        prediction = int(probabilities.argmax())
        target = targets.pop()
        meter.update_predictions(torch.tensor([prediction]), torch.tensor([target]))
        aggregated.append(
            {
                "sequence_id": sequence_id,
                "temporal_block": temporal_block,
                "target": target,
                "prediction": prediction,
                "probabilities": probabilities.tolist(),
                "frames_available": len(group),
                "frames_selected": len(selected),
                "selected_sample_ids": [row["sample_id"] for row in selected],
            }
        )
    metrics = meter.compute()
    metrics["groups"] = len(aggregated)
    metrics["frames_selected"] = sum(row["frames_selected"] for row in aggregated)
    return metrics, aggregated
