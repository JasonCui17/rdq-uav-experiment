from __future__ import annotations

import torch

from rdq_uav.lidar_v2 import CandidateSelector
from rdq_uav.multimodal_v1.candidate.builders import _nms_xyxy


def _reference_box_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float):
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    order = torch.argsort(scores, descending=True, stable=True)
    kept = []
    while len(order):
        index = order[0]
        kept.append(index)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1, yy1 = torch.maximum(x1[index], x1[rest]), torch.maximum(y1[index], y1[rest])
        xx2, yy2 = torch.minimum(x2[index], x2[rest]), torch.minimum(y2[index], y2[rest])
        intersection = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        union = areas[index] + areas[rest] - intersection
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        order = rest[iou <= threshold]
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long)


def _reference_radius_nms(output, selector):
    ids = torch.nonzero(output["batch_index"] == 0).flatten()
    order = ids[selector._order(output["logits"][ids].float(), output["source_token_id"][ids])]
    kept = []
    for index in order[: selector.pre]:
        if not kept or torch.all(
            torch.linalg.vector_norm(
                output["pred_xyz"][torch.stack(kept)] - output["pred_xyz"][index], dim=1
            )
            > selector.radius
        ):
            kept.append(index)
        if len(kept) >= selector.final:
            break
    return torch.stack(kept) if kept else order[:0]


def test_fused_box_nms_matches_stable_reference_with_ties() -> None:
    generator = torch.Generator().manual_seed(42)
    starts = torch.rand((100, 2), generator=generator) * 100
    sizes = torch.rand((100, 2), generator=generator) * 30
    boxes = torch.cat((starts, starts + sizes), dim=1)
    boxes[1] = boxes[0]
    scores = torch.rand(100, generator=generator)
    scores[:4] = 0.75
    expected = _reference_box_nms(boxes, scores, 0.7)
    actual = _nms_xyxy(boxes, scores, 0.7)
    assert torch.equal(actual, expected)


def test_vectorized_radius_nms_matches_legacy_greedy_order() -> None:
    generator = torch.Generator().manual_seed(7)
    count = 137
    output = {
        "logits": torch.randn(count, generator=generator),
        "pred_xyz": torch.randn((count, 3), generator=generator) * 4,
        "fine_features": torch.randn((count, 128), generator=generator),
        "source_token_id": torch.arange(count - 1, -1, -1),
        "batch_index": torch.zeros(count, dtype=torch.long),
        "aux_stats": {"num_samples": 1},
    }
    output["logits"][:5] = 1.0
    output["pred_xyz"][1] = output["pred_xyz"][0]
    selector = CandidateSelector(
        {"selector": {"raw_topk": 20, "pre_nms_topk": 100, "nms_radius_m": 1.0, "final_topk": 20}}
    )
    expected = _reference_radius_nms(output, selector)
    actual = selector(output)[0]["nms"]["source_token_id"]
    assert torch.equal(actual, output["source_token_id"][expected])
