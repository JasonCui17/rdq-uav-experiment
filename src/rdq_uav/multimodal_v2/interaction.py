"""Candidate-level, geometry-constrained bidirectional cross-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from rdq_uav.multimodal_v1.contracts import InteractionContext
from rdq_uav.multimodal_v1.interaction.geometry_local import project_omni_radtan

from .contracts import CandidateBatch, CrossModalEvidence


def _zero_evidence(candidate: CandidateBatch) -> CrossModalEvidence:
    return CrossModalEvidence(
        candidate.feature.new_zeros((candidate.n, 128)),
        torch.zeros(candidate.n, dtype=torch.bool, device=candidate.feature.device),
        torch.zeros(candidate.n, dtype=torch.long, device=candidate.feature.device),
        candidate.score.new_zeros(candidate.n),
        candidate.xyz_m.new_zeros((candidate.n, 2)),
    )


class _EvidenceAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.gate = nn.Linear(256, 1)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                padding: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The caller removes all-masked rows; this avoids undefined softmax.
        n, length, _ = memory.shape
        additive = bias.to(query.dtype)[:, None, :].expand(n, 4, length).reshape(n * 4, 1, length)
        value, _ = self.attention(
            query[:, None], memory, memory, key_padding_mask=padding,
            attn_mask=additive, need_weights=False,
        )
        value = self.norm(value[:, 0])
        gate = torch.sigmoid(self.gate(torch.cat((query, value), dim=1)))[:, 0]
        # Evidence and gate remain separate. CandidateScoring applies the gate
        # exactly once to the bounded logit correction required by the model
        # contract.
        return value, gate


class CandidateCrossAttention(nn.Module):
    """Read local cross-modal evidence without modifying either candidate set."""

    def __init__(self, vision_dims: Sequence[int] = (96, 192),
                 visual_radius: int = 1, lidar_neighbors: int = 16,
                 box_margin_px: float = 16.0) -> None:
        super().__init__()
        if tuple(vision_dims) != (96, 192):
            raise ValueError("the frozen Swin-T contract uses V0/V1 channels 96/192")
        self.visual_proj = nn.ModuleList([nn.Linear(dim, 128) for dim in vision_dims])
        # A 2D box does not determine depth. Preserve every selected LiDAR
        # hypothesis as a separate token and expose its metric range instead
        # of collapsing box-contained returns into a synthetic 3D target.
        self.depth_embed = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, 128))
        self.radar_reads_vision = _EvidenceAttention()
        self.vision_reads_radar = _EvidenceAttention()
        self.visual_radius = int(visual_radius)
        self.lidar_neighbors = int(lidar_neighbors)
        self.box_margin_px = float(box_margin_px)

    def _visual_memory(self, radar: CandidateBatch, pyramid: Sequence[torch.Tensor],
                       image_masks: torch.Tensor, context: InteractionContext):
        pixels, projection_valid = project_omni_radtan(radar.xyz_m.float(), radar.batch_index, context.projection)
        side = 2 * self.visual_radius + 1
        per_level = side * side
        total = per_level * len(self.visual_proj)
        memory = radar.feature.new_zeros((radar.n, total, 128))
        padding = torch.ones((radar.n, total), dtype=torch.bool, device=radar.feature.device)
        bias = radar.score.new_zeros((radar.n, total), dtype=torch.float32)
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-self.visual_radius, self.visual_radius + 1, device=radar.feature.device),
            torch.arange(-self.visual_radius, self.visual_radius + 1, device=radar.feature.device),
            indexing="ij",
        ), dim=-1).reshape(-1, 2)[:, [1, 0]]
        scale = context.projection.image_scale_xy[radar.batch_index].to(pixels.dtype)
        for level, (feature, projection) in enumerate(zip(pyramid, self.visual_proj)):
            _, _, height, width = feature.shape
            padded_height, padded_width = image_masks.shape[-2:]
            cell_scale = pixels.new_tensor((width / padded_width, height / padded_height))[None]
            anchor = torch.floor(pixels * scale * cell_scale).long()
            cells = anchor[:, None] + offsets[None]
            x, y = cells[..., 0], cells[..., 1]
            in_bounds = projection_valid[:, None] & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            resized_mask = torch.nn.functional.interpolate(
                image_masks[:, None].float(), size=(height, width), mode="nearest"
            )[:, 0].bool()
            safe_x, safe_y = x.clamp(0, width - 1), y.clamp(0, height - 1)
            visible = in_bounds & ~resized_mask[radar.batch_index[:, None], safe_y, safe_x]
            raw = feature.permute(0, 2, 3, 1)[radar.batch_index[:, None], safe_y, safe_x]
            start = level * per_level
            memory[:, start:start + per_level] = projection(raw)
            padding[:, start:start + per_level] = ~visible
            distance = torch.linalg.vector_norm(offsets.float(), dim=1)
            bias[:, start:start + per_level] = -distance[None]
        return memory, padding, bias, pixels, projection_valid

    def read_visual_for_radar(self, radar: CandidateBatch, pyramid: Sequence[torch.Tensor],
                              image_masks: torch.Tensor, context: InteractionContext) -> CrossModalEvidence:
        if radar.n == 0:
            return _zero_evidence(radar)
        memory, padding, bias, pixels, projected = self._visual_memory(radar, pyramid, image_masks, context)
        active = projected & context.m_R[radar.batch_index] & context.m_V[radar.batch_index] & (~padding).any(1)
        feature = radar.feature.new_zeros((radar.n, 128)); gate = radar.score.new_zeros(radar.n)
        if bool(active.any()):
            values, weights = self.radar_reads_vision(
                radar.feature[active], memory[active], padding[active], bias[active]
            )
            feature[active], gate[active] = values, weights.to(gate.dtype)
        return CrossModalEvidence(feature, active, (~padding).sum(1), gate, pixels)

    def read_radar_for_vision(self, vision: CandidateBatch, radar: CandidateBatch,
                              context: InteractionContext) -> CrossModalEvidence:
        if vision.n == 0:
            return _zero_evidence(vision)
        radar_pixels, radar_valid = project_omni_radtan(radar.xyz_m.float(), radar.batch_index, context.projection)
        k = self.lidar_neighbors
        memory = vision.feature.new_zeros((vision.n, k, 128))
        padding = torch.ones((vision.n, k), dtype=torch.bool, device=vision.feature.device)
        bias = vision.score.new_zeros((vision.n, k), dtype=torch.float32)
        for vi in range(vision.n):
            batch = vision.batch_index[vi]
            ids = torch.nonzero((radar.batch_index == batch) & radar_valid, as_tuple=False).flatten()
            if not len(ids) or not bool(context.m_R[batch] & context.m_V[batch]):
                continue
            box = vision.box_xyxy_px[vi].float()
            points = radar_pixels[ids].float()
            center = (box[:2] + box[2:]) * 0.5
            # A candidate is only evidence if its calibrated projection lies in
            # the box or a bounded margin. It is never promoted directly to GT.
            dx = torch.maximum(
                torch.maximum(box[0] - points[:, 0], points[:, 0] - box[2]),
                torch.zeros_like(points[:, 0]),
            )
            dy = torch.maximum(
                torch.maximum(box[1] - points[:, 1], points[:, 1] - box[3]),
                torch.zeros_like(points[:, 1]),
            )
            near_box = torch.sqrt(dx.square() + dy.square()) <= self.box_margin_px
            ids = ids[near_box]; points = points[near_box]
            if not len(ids):
                continue
            distance = torch.linalg.vector_norm(points - center, dim=1)
            order = torch.argsort(distance, stable=True)[:k]
            chosen = ids[order]; count = len(chosen)
            metric_range = torch.linalg.vector_norm(radar.xyz_m[chosen].float(), dim=1)
            depth = torch.log1p(metric_range)[:, None].to(radar.feature.dtype)
            memory[vi, :count] = radar.feature[chosen] + self.depth_embed(depth)
            padding[vi, :count] = False
            scale = max(1.0, float(torch.linalg.vector_norm(box[2:] - box[:2])))
            bias[vi, :count] = -distance[order] / scale
        active = context.m_V[vision.batch_index] & (~padding).any(1)
        feature = vision.feature.new_zeros((vision.n, 128)); gate = vision.score.new_zeros(vision.n)
        if bool(active.any()):
            values, weights = self.vision_reads_radar(
                vision.feature[active], memory[active], padding[active], bias[active]
            )
            feature[active], gate[active] = values, weights.to(gate.dtype)
        return CrossModalEvidence(
            feature, active, (~padding).sum(1), gate,
            vision.box_xyxy_px[:, :2].add(vision.box_xyxy_px[:, 2:]).mul(0.5),
        )

    def forward(self, radar: CandidateBatch, vision: CandidateBatch,
                pyramid: Sequence[torch.Tensor], image_masks: torch.Tensor,
                context: InteractionContext, *, enable_radar_to_vision: bool = True):
        r = self.read_visual_for_radar(radar, pyramid, image_masks, context)
        v = self.read_radar_for_vision(vision, radar, context) if enable_radar_to_vision else _zero_evidence(vision)
        return r, v
