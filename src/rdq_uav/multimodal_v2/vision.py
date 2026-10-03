"""One shared Swin-DINO execution producing visual candidates and V0/V1."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .candidate_builders import RGBCandidateBuilder
from .dino_adapter import DINOAdapter

from .contracts import CandidateBatch


class VisionCandidateModel(nn.Module):
    def __init__(self, detector: nn.Module, *, pre_topk: int = 50,
                 final_topk: int = 10, nms_iou: float = 0.7) -> None:
        super().__init__()
        self.dino = DINOAdapter(detector)
        self.builder = RGBCandidateBuilder(
            query_dim=256, feature_dim=128, pre_topk=pre_topk,
            final_topk=final_topk, nms_iou=nms_iou,
        )

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor,
                source_image_wh: torch.Tensor) -> tuple[dict[str, Any], CandidateBatch]:
        pyramid = self.dino.swin(images)
        raw = self.dino.forward_from_pyramid(
            pyramid, image_masks, allow_training_candidate_path=True,
        )
        candidates = self.builder(raw, source_image_wh)
        return raw, candidates

    @property
    def detector(self) -> nn.Module:
        """Non-registering access to the single detector owned by DINOAdapter."""
        return self.dino.detector
