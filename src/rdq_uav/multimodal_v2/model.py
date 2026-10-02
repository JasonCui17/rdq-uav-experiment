"""Single Multimodal V2 forward chain with no pre-stage HCI."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .geometry import ProjectionContext, project_omni_radtan

from .contracts import CandidateBatch, CrossModalEvidence, MultimodalOutput, validate_batch
from .interaction import CandidateCrossAttention, _zero_evidence
from .lidar import LiDARCandidateModel
from .scoring import CandidateScoring
from .vision import VisionCandidateModel


class MultimodalV2(nn.Module):
    """Independent candidates -> geometric evidence -> constrained ranking."""

    def __init__(self, lidar: LiDARCandidateModel, vision: VisionCandidateModel,
                 interaction: CandidateCrossAttention, scoring: CandidateScoring,
                 *, interaction_enabled: bool = True,
                 vision_reads_radar: bool = False,
                 vision_scoring_enabled: bool = False) -> None:
        super().__init__()
        self.lidar = lidar
        self.vision = vision
        self.interaction = interaction
        self.scoring = scoring
        self.interaction_enabled = bool(interaction_enabled)
        self.vision_reads_radar = bool(vision_reads_radar)
        self.vision_scoring_enabled = bool(vision_scoring_enabled)

    @staticmethod
    def _filter_modality(candidate: CandidateBatch, available: torch.Tensor) -> CandidateBatch:
        if candidate.n == 0:
            return candidate
        return candidate.index_select(available[candidate.batch_index])

    def forward(self, batch: Mapping[str, Any], images: torch.Tensor,
                image_padding_mask: torch.Tensor, projection: ProjectionContext) -> MultimodalOutput:
        validate_batch(batch)
        lidar_raw, radar = self.lidar(batch)
        vision_raw, visual = self.vision(images, image_padding_mask, batch["image_source_wh"])
        radar = self._filter_modality(radar, batch["m_R"])
        visual = self._filter_modality(visual, batch["m_V"])
        projected, projection_valid = project_omni_radtan(
            radar.xyz_m.float(), radar.batch_index, projection
        )
        radar = radar.with_projection(projected, projection_valid)
        if self.interaction_enabled:
            radar_evidence, vision_evidence = self.interaction(
                radar, visual, vision_raw["pyramid"].features[:2], image_padding_mask,
                projection, batch["m_R"], batch["m_V"], enable_radar_to_vision=self.vision_reads_radar,
            )
        else:
            radar_evidence, vision_evidence = _zero_evidence(radar), _zero_evidence(visual)
            radar_evidence = CrossModalEvidence(
                radar_evidence.feature, radar_evidence.valid, radar_evidence.token_count,
                radar_evidence.gate_weight, projected,
            )
        return self.scoring(
            radar, visual, radar_evidence, vision_evidence,
            num_samples=int(batch["num_samples"]),
            enable_vision_scoring=self.vision_scoring_enabled,
            diagnostics={
                "lidar_raw": lidar_raw,
                "vision_raw": vision_raw,
                "radar_evidence": radar_evidence,
                "vision_evidence": vision_evidence,
                "old_pre_stage_hci_used": False,
            },
        )
