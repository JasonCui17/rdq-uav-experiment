"""Unmodified LiDAR V2 candidate producer for Multimodal V2."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from rdq_uav.lidar_v2.model import LiDARUAVDetector
from rdq_uav.lidar_v2.selector import CandidateSelector
from rdq_uav.multimodal_v1.candidate.builders import RadarCandidateBuilder

from .contracts import CandidateBatch


class LiDARCandidateModel(nn.Module):
    def __init__(self, detector: LiDARUAVDetector, selector: CandidateSelector) -> None:
        super().__init__()
        self.detector = detector
        self.builder = RadarCandidateBuilder(selector)

    def forward(self, batch: Mapping[str, Any]) -> tuple[dict[str, Any], CandidateBatch]:
        raw = self.detector(batch)
        old = self.builder(raw)
        candidates = CandidateBatch(
            old.feature, old.score, old.xyz, old.xyz_valid, old.box_xyxy_px,
            old.box_valid, old.batch_index, old.source_index, "R",
        )
        return raw, candidates


def load_lidar_weights(detector: LiDARUAVDetector, checkpoint: str,
                       *, prefix: str | None = None) -> dict[str, Any]:
    """Strictly load either a native LiDAR V2 or Lightning E5 checkpoint."""
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload.get("model_state", payload))
    if prefix is not None:
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not state:
        raise RuntimeError(f"no LiDAR weights found in {checkpoint}")
    detector.load_state_dict(state, strict=True)
    return payload
