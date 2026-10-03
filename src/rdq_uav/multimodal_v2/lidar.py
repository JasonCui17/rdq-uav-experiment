"""Unmodified LiDAR V2 candidate producer for Multimodal V2."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .radar_model import LiDARUAVDetector
from .radar_selector import CandidateSelector
from .candidate_builders import RadarCandidateBuilder

from .contracts import CandidateBatch


class LiDARCandidateModel(nn.Module):
    def __init__(self, detector: LiDARUAVDetector, selector: CandidateSelector) -> None:
        super().__init__()
        self.detector = detector
        self.builder = RadarCandidateBuilder(selector)

    def forward(self, batch: Mapping[str, Any]) -> tuple[dict[str, Any], CandidateBatch]:
        raw = self.detector(batch)
        candidates = self.builder(raw)
        return raw, candidates
