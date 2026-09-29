"""One shared Swin-DINO execution producing visual candidates and V0/V1."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from rdq_uav.multimodal_v1.candidate.builders import RGBCandidateBuilder
from rdq_uav.multimodal_v1.vision.dino_adapter import DINOAdapter

from .contracts import CandidateBatch


class VisionCandidateModel(nn.Module):
    def __init__(self, detector: nn.Module, *, pre_topk: int = 100,
                 final_topk: int = 50, nms_iou: float = 0.7) -> None:
        super().__init__()
        self.detector = detector
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
        old = self.builder(raw, source_image_wh)
        candidates = CandidateBatch(
            old.feature, old.score, old.xyz, old.xyz_valid, old.box_xyxy_px,
            old.box_valid, old.batch_index, old.source_index, "V",
        )
        return raw, candidates

    def load_e5_weights(self, checkpoint: str) -> dict[str, Any]:
        """Load only DINO and RGB projection from an audited E5 checkpoint."""
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("state_dict", payload)
        dino_prefix = "network.dino.detector."
        dino_state = {key[len(dino_prefix):]: value for key, value in state.items() if key.startswith(dino_prefix)}
        rgb_prefix = "network.rgb_candidates.feature_proj."
        rgb_state = {key[len(rgb_prefix):]: value for key, value in state.items() if key.startswith(rgb_prefix)}
        if not dino_state or not rgb_state:
            raise RuntimeError("E5 checkpoint lacks DINO/RGB candidate weights")
        self.detector.load_state_dict(dino_state, strict=True)
        self.builder.feature_proj.load_state_dict(rgb_state, strict=True)
        return payload
