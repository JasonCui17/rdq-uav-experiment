from __future__ import annotations
from typing import Any, Mapping, Sequence
import torch
from .data import ViewTransform

def source_xyxy_to_normalized_cxcywh(box_xyxy_source: torch.Tensor, transform: ViewTransform) -> torch.Tensor:
    box = transform.source_boxes_to_view(torch.as_tensor(box_xyxy_source))
    width, height = transform.view_wh
    x1, y1, x2, y2 = box.unbind(-1)
    out = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
    return out / out.new_tensor((width, height, width, height))

def weighted_loss_sum(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = list(losses.values())
    if not values:
        raise ValueError("loss dictionary is empty")
    return torch.stack([value if value.ndim == 0 else value.sum() for value in values]).sum()

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


