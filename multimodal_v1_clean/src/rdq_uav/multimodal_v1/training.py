"""Training-only contracts for the frozen Multimodal V1 E5 architecture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .vision.ssod import source_xyxy_to_normalized_cxcywh, weighted_loss_sum


def load_lidar_checkpoint_strict(detector: torch.nn.Module, checkpoint: str) -> dict[str, Any]:
    """Load the audited spatial checkpoint without accepting key drift."""

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise ValueError("LiDAR checkpoint must contain model_state")
    detector.load_state_dict(payload["model_state"], strict=True)
    return dict(payload)


def _select_batch(value: Any, indices: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return value.index_select(0, indices)
    if isinstance(value, Mapping):
        return {key: _select_batch(item, indices) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        selected = [_select_batch(item, indices) for item in value]
        return type(value)(selected)
    return value


def select_dino_batch(output: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    """Select DINO's batch dimension while preserving decoder-level lists."""

    selected = {
        "pred_logits": output["pred_logits"].index_select(0, indices),
        "pred_boxes": output["pred_boxes"].index_select(0, indices),
    }
    if "aux_outputs" in output:
        selected["aux_outputs"] = [
            {
                "pred_logits": level["pred_logits"].index_select(0, indices),
                "pred_boxes": level["pred_boxes"].index_select(0, indices),
            }
            for level in output["aux_outputs"]
        ]
    if "enc_outputs" in output:
        selected["enc_outputs"] = {
            "pred_logits": output["enc_outputs"]["pred_logits"].index_select(0, indices),
            "pred_boxes": output["enc_outputs"]["pred_boxes"].index_select(0, indices),
        }
    return selected


def supervised_dino_loss(
    detector: torch.nn.Module,
    output: Mapping[str, Any],
    *,
    gt_box_xyxy_source: torch.Tensor,
    gt_2d_valid: torch.Tensor,
    transforms: Sequence[Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Compute native DINO supervision only for samples with verified 2D GT.

    Missing annotations are removed from the criterion batch.  They are never
    represented as empty detection targets, which would incorrectly supervise
    every query as background.
    """

    indices = torch.nonzero(gt_2d_valid, as_tuple=False).flatten()
    if len(indices) == 0:
        zero = output["pred_logits"].sum() * 0.0
        return zero, {}, 0
    selected = select_dino_batch(output, indices)
    targets = []
    for index in indices.tolist():
        normalized = source_xyxy_to_normalized_cxcywh(
            gt_box_xyxy_source[index], transforms[index]
        ).reshape(1, 4)
        targets.append(
            {
                "labels": torch.zeros(1, dtype=torch.long, device=normalized.device),
                "boxes": normalized,
            }
        )
    raw = detector.criterion(selected, targets, None)
    weighted = {
        key: value * float(detector.criterion.weight_dict[key])
        for key, value in raw.items()
        if key in detector.criterion.weight_dict
    }
    return weighted_loss_sum(weighted), weighted, len(indices)


def combine_e5_losses(
    radar_loss: torch.Tensor,
    vision_loss: torch.Tensor,
    fusion_loss: torch.Tensor,
    *,
    lambda_r: float,
    lambda_v: float,
    lambda_f: float,
) -> torch.Tensor:
    return lambda_r * radar_loss + lambda_v * vision_loss + lambda_f * fusion_loss


def stage_for_epoch(stages: Sequence[Mapping[str, Any]], epoch: int) -> str:
    matches = [
        str(item["name"])
        for item in stages
        if int(item["start_epoch"]) <= epoch <= int(item["end_epoch"])
    ]
    if len(matches) != 1:
        raise ValueError(f"epoch {epoch} must belong to exactly one training stage; got {matches}")
    return matches[0]


def set_e5_trainable(
    *,
    stage: str,
    lidar_detector: torch.nn.Module,
    dino_detector: torch.nn.Module,
    new_modules: Sequence[torch.nn.Module],
) -> None:
    """Apply the frozen T1/T2/T3 policy without changing module structure."""

    if stage not in {"T1", "T2", "T3"}:
        raise ValueError(f"unknown E5 training stage {stage}")
    for parameter in lidar_detector.parameters():
        parameter.requires_grad_(stage in {"T2", "T3"})
    for parameter in dino_detector.parameters():
        parameter.requires_grad_(stage in {"T2", "T3"})
    # Swin stays frozen in T2. T3 opens exactly its last two stages and their
    # official output normalizations; patch embedding/stages 0-1 remain frozen.
    for parameter in dino_detector.backbone.parameters():
        parameter.requires_grad_(False)
    if stage == "T3":
        for index in (2, 3):
            for parameter in dino_detector.backbone.layers[index].parameters():
                parameter.requires_grad_(True)
            norm = getattr(dino_detector.backbone, f"norm{index}", None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad_(True)
    for module in new_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
