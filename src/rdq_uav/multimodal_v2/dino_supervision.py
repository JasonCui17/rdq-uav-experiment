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


def build_dino_targets(gt_box_xyxy_source: torch.Tensor, gt_2d_valid: torch.Tensor,
                       transforms: Sequence[Any]) -> list[dict[str, torch.Tensor]]:
    """Build view-normalized targets in compact visual order, before CDN.

    Empty entries are only placeholders for CDN batch indexing. Unlabeled
    images are excluded from both regular and denoising detection losses.
    """
    if len(gt_box_xyxy_source) != len(gt_2d_valid) or len(transforms) != len(gt_2d_valid):
        raise ValueError("DINO boxes, validity, and transforms must have equal batch size")
    targets = []
    for index, valid in enumerate(gt_2d_valid):
        boxes = gt_box_xyxy_source.new_empty((0, 4), dtype=torch.float32)
        if bool(valid):
            boxes = source_xyxy_to_normalized_cxcywh(
                gt_box_xyxy_source[index].float(), transforms[index],
            ).reshape(1, 4)
        targets.append({"labels": torch.zeros(len(boxes), dtype=torch.long, device=boxes.device),
                        "boxes": boxes})
    return targets


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
    all_targets = build_dino_targets(gt_box_xyxy_source, gt_2d_valid, transforms)
    targets = [all_targets[index] for index in indices.tolist()]
    dn_meta = output.get("dn_meta")
    if dn_meta is not None:
        dn_meta = dict(dn_meta)
        dn_meta["output_known_lbs_bboxes"] = select_dino_batch(
            dn_meta["output_known_lbs_bboxes"], indices,
        )
    raw = detector.criterion(selected, targets, dn_meta)
    weighted = {
        key: value * float(detector.criterion.weight_dict[key])
        for key, value in raw.items()
        if key in detector.criterion.weight_dict
    }
    return weighted_loss_sum(weighted), weighted, len(indices)


