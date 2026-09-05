from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from rdq_uav.engine.metrics import ClassificationMetrics


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    auxiliary_position_weight: float = 0.0,
    log_interval: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    metrics = ClassificationMetrics(num_classes)
    totals: defaultdict[str, float] = defaultdict(float)
    predictions: list[dict[str, Any]] = []
    sample_count = 0
    start = time.perf_counter()

    for step, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        batch_size = int(batch["label"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.cuda.amp.autocast(enabled=amp) if device.type == "cuda" else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast_context:
            outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
            classification_loss = criterion(outputs["logits"], batch["label"])
            position_loss = torch.zeros((), device=device)
            if outputs["position"] is not None and auxiliary_position_weight > 0:
                position_loss = nn.functional.smooth_l1_loss(
                    outputs["position"], batch["position"]
                )
            loss = classification_loss + auxiliary_position_weight * position_loss

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        metrics.update(outputs["logits"], batch["label"])
        totals["loss"] += float(loss.detach()) * batch_size
        totals["classification_loss"] += float(classification_loss.detach()) * batch_size
        totals["position_loss"] += float(position_loss.detach()) * batch_size
        sample_count += batch_size
        probabilities = outputs["logits"].softmax(dim=1).detach().cpu()
        predicted = probabilities.argmax(dim=1).tolist()
        confidence = probabilities.amax(dim=1).tolist()
        for index in range(batch_size):
            predictions.append(
                {
                    "sample_id": raw_batch["sample_id"][index],
                    "target": int(raw_batch["label"][index]),
                    "prediction": int(predicted[index]),
                    "confidence": float(confidence[index]),
                    "probabilities": probabilities[index].tolist(),
                    "distance_m": float(raw_batch["distance"][index]),
                    "class_name": raw_batch["class_name"][index],
                    "sequence_id": raw_batch["sequence_id"][index],
                    "temporal_block": int(raw_batch["temporal_block"][index]),
                    "gt_time": float(raw_batch["gt_time"][index]),
                    "bbox_area_px": float(raw_batch["bbox_area_px"][index]),
                }
            )
        if training and log_interval > 0 and step % log_interval == 0:
            print(
                f"step={step}/{len(loader)} samples={sample_count} "
                f"loss={totals['loss'] / sample_count:.4f}",
                flush=True,
            )

    result = metrics.compute()
    divisor = max(sample_count, 1)
    result.update(
        {
            "loss": totals["loss"] / divisor,
            "classification_loss": totals["classification_loss"] / divisor,
            "position_loss": totals["position_loss"] / divisor,
            "samples": sample_count,
            "seconds": time.perf_counter() - start,
        }
    )
    return result, predictions
