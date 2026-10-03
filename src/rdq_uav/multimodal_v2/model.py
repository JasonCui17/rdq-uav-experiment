"""Single Multimodal V2 forward chain with no pre-stage HCI."""

from __future__ import annotations

from typing import Any, Mapping
from dataclasses import replace, fields

import torch
from torch import nn

from .geometry import ProjectionContext, project_omni_radtan

from .contracts import CandidateBatch, CrossModalEvidence, MultimodalOutput, validate_batch
from .interaction import CandidateCrossAttention, _zero_evidence
from .lidar import LiDARCandidateModel
from .scoring import CandidateScoring
from .vision import VisionCandidateModel
from .data import radar_model_batch


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
    def _projection_subset(projection: ProjectionContext, indices: torch.Tensor) -> ProjectionContext:
        return ProjectionContext(**{field.name: getattr(projection, field.name)[indices]
                                    for field in fields(ProjectionContext)})

    @staticmethod
    def _merge_evidence(base: CrossModalEvidence, selected: torch.Tensor,
                        value: CrossModalEvidence) -> CrossModalEvidence:
        result = {}
        for field in fields(CrossModalEvidence):
            tensor = getattr(base, field.name).clone()
            tensor[selected] = getattr(value, field.name)
            result[field.name] = tensor
        return CrossModalEvidence(**result)

    def forward(self, batch: Mapping[str, Any], images: torch.Tensor | None,
                image_padding_mask: torch.Tensor | None, projection: ProjectionContext) -> MultimodalOutput:
        validate_batch(batch)
        device = batch["m_R"].device
        radar_ids, vision_ids = batch["radar_batch_index"], batch["vision_batch_index"]
        lidar_raw, vision_raw, lidar_batch = None, None, None
        radar, visual = CandidateBatch.empty("R", device), CandidateBatch.empty("V", device)
        if len(radar_ids):
            lidar_batch = radar_model_batch(batch)
            lidar_raw, radar = self.lidar(lidar_batch)
            radar = replace(radar, batch_index=radar_ids[radar.batch_index])
        if len(vision_ids):
            if images is None or image_padding_mask is None or len(images) != len(vision_ids):
                raise ValueError("valid visual samples require matching preprocessed images/masks")
            vision_raw, visual = self.vision(images, image_padding_mask, batch["image_source_wh"])
            visual = replace(visual, batch_index=vision_ids[visual.batch_index])
        elif images is not None or image_padding_mask is not None:
            raise ValueError("absent vision must not provide preprocessed images")

        projected, projection_valid = project_omni_radtan(radar.xyz_m.float(), radar.batch_index, projection)
        radar = radar.with_projection(projected, projection_valid)
        radar_evidence = replace(_zero_evidence(radar), projected_xy_px=projected)
        vision_evidence = _zero_evidence(visual)
        # Feature pyramids use compact visual indices. Only dual-modality
        # candidates enter interaction; no full-B feature padding is allocated.
        paired_r = torch.nonzero(batch["m_V"][radar.batch_index], as_tuple=False).flatten()
        paired_v = torch.nonzero(batch["m_R"][visual.batch_index], as_tuple=False).flatten()
        if self.interaction_enabled and len(paired_r):
            original_to_visual = torch.full((int(batch["num_samples"]),), -1, dtype=torch.long, device=device)
            original_to_visual[vision_ids] = torch.arange(len(vision_ids), device=device)
            local_r = radar.index_select(paired_r)
            local_v = visual.index_select(paired_v)
            local_r = replace(local_r, batch_index=original_to_visual[local_r.batch_index])
            local_v = replace(local_v, batch_index=original_to_visual[local_v.batch_index])
            er, ev = self.interaction(
                local_r, local_v, vision_raw["pyramid"].features[:2], image_padding_mask,
                self._projection_subset(projection, vision_ids), batch["m_R"][vision_ids],
                batch["m_V"][vision_ids], enable_radar_to_vision=self.vision_reads_radar,
            )
            radar_evidence = self._merge_evidence(radar_evidence, paired_r, er)
            vision_evidence = self._merge_evidence(vision_evidence, paired_v, ev)
        return self.scoring(
            radar, visual, radar_evidence, vision_evidence,
            num_samples=int(batch["num_samples"]),
            enable_vision_scoring=self.vision_scoring_enabled,
            diagnostics={
                "lidar_raw": lidar_raw, "vision_raw": vision_raw, "lidar_batch": lidar_batch,
                "radar_batch_index": radar_ids, "vision_batch_index": vision_ids,
                "radar_evidence": radar_evidence, "vision_evidence": vision_evidence,
            },
        )
