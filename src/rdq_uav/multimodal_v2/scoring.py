"""Deterministic geometry association and task-specific score correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import CandidateBatch, CrossModalEvidence, MultimodalOutput

from .geometry import validate_geometry_gate, geometry_gate_margins

HYP_RV, HYP_R, HYP_V = 0, 1, 2


@dataclass(frozen=True)
class AssociationRows:
    radar_index: torch.Tensor
    vision_index: torch.Tensor
    batch_index: torch.Tensor
    hypothesis_type: torch.Tensor
    association_valid: torch.Tensor
    d_box_px: torch.Tensor
    d_center_normalized: torch.Tensor
    per_query: tuple[dict[str, Any], ...]
    rv_pairs: tuple[dict[str, Any], ...]


def point_to_box_distance(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Return pairwise Euclidean distance from image points to boxes."""
    px, py = points[:, 0, None], points[:, 1, None]
    x1, y1, x2, y2 = [boxes[:, index][None] for index in range(4)]
    dx = torch.maximum(torch.maximum(x1 - px, px - x2), torch.zeros_like(px + x1))
    dy = torch.maximum(torch.maximum(y1 - py, py - y2), torch.zeros_like(py + y1))
    return torch.sqrt(dx.square() + dy.square())


def associate_geometry_greedy(
    radar: CandidateBatch,
    vision: CandidateBatch,
    *,
    num_samples: int,
    geometry_gate: dict | None = None,
) -> AssociationRows:
    """Stable, rejectable one-to-one association in calibrated image geometry."""
    geometry_gate = validate_geometry_gate(geometry_gate)
    device = radar.feature.device if radar.n else vision.feature.device
    floating_template = radar.score if radar.n else vision.score
    rows: list[tuple[int, int, int, int, bool, float, float]] = []
    per_query: list[dict[str, Any]] = []
    rv_pairs: list[dict[str, Any]] = []

    for batch_idx in range(num_samples):
        radar_ids = torch.nonzero(
            radar.batch_index == batch_idx, as_tuple=False
        ).flatten()
        vision_ids = torch.nonzero(
            vision.batch_index == batch_idx, as_tuple=False
        ).flatten()
        valid_radar = (
            radar.projection_valid[radar_ids]
            & torch.isfinite(radar.projected_xy_px[radar_ids]).all(1)
        )
        boxes = vision.box_xyxy_px[vision_ids].float()
        valid_vision = (
            vision.has_box[vision_ids]
            & torch.isfinite(boxes).all(1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )

        feasible_items: list[tuple[float, int, int, int, int, float, float]] = []
        if len(radar_ids) and len(vision_ids):
            points = radar.projected_xy_px[radar_ids].float()
            distance_to_box = point_to_box_distance(points, boxes)
            center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            diagonal = torch.linalg.vector_norm(
                boxes[:, 2:] - boxes[:, :2], dim=1
            ).clamp_min(1.0)
            center_distance = torch.linalg.vector_norm(
                points[:, None] - center[None], dim=2
            ) / diagonal[None]
            feasible = (
                valid_radar[:, None]
                & valid_vision[None]
                & (distance_to_box <= geometry_gate_margins(radar.xyz_m[radar_ids], geometry_gate)[:, None])
            )
            local_radar, local_vision = torch.nonzero(feasible, as_tuple=True)
            for local_r, local_v in zip(local_radar.tolist(), local_vision.tolist()):
                radar_idx = int(radar_ids[local_r])
                vision_idx = int(vision_ids[local_v])
                d_box = float(distance_to_box[local_r, local_v])
                d_center = float(center_distance[local_r, local_v])
                feasible_items.append(
                    (
                        d_box + 0.01 * d_center,
                        int(radar.source_index[radar_idx]),
                        int(vision.source_index[vision_idx]),
                        radar_idx,
                        vision_idx,
                        d_box,
                        d_center,
                    )
                )

        feasible_items.sort(key=lambda item: (item[0], item[1], item[2]))
        matched_radar: set[int] = set()
        matched_vision: set[int] = set()
        conflicts = 0
        for _, _, _, radar_idx, vision_idx, d_box, d_center in feasible_items:
            if radar_idx in matched_radar or vision_idx in matched_vision:
                conflicts += 1
                continue
            matched_radar.add(radar_idx)
            matched_vision.add(vision_idx)
            rows.append(
                (radar_idx, vision_idx, batch_idx, HYP_RV, True, d_box, d_center)
            )
            rv_pairs.append(
                {
                    "batch_index": batch_idx,
                    "radar_source_index": int(radar.source_index[radar_idx]),
                    "vision_source_index": int(vision.source_index[vision_idx]),
                    "d_box_px": d_box,
                    "d_center_normalized": d_center,
                }
            )

        unmatched_radar = (int(index) for index in radar_ids if int(index) not in matched_radar)
        for radar_idx in sorted(
            unmatched_radar, key=lambda index: int(radar.source_index[index])
        ):
            rows.append((radar_idx, -1, batch_idx, HYP_R, False, 0.0, 0.0))
        unmatched_vision = (int(index) for index in vision_ids if int(index) not in matched_vision)
        for vision_idx in sorted(
            unmatched_vision, key=lambda index: int(vision.source_index[index])
        ):
            rows.append((-1, vision_idx, batch_idx, HYP_V, False, 0.0, 0.0))

        per_query.append(
            {
                "batch_index": batch_idx,
                "radar_candidates": len(radar_ids),
                "vision_candidates": len(vision_ids),
                "feasible_pairs": len(feasible_items),
                "rv": len(matched_radar),
                "r": len(radar_ids) - len(matched_radar),
                "v": len(vision_ids) - len(matched_vision),
                "invalid_radar_projection": int((~valid_radar).sum()),
                "invalid_vision_box": int((~valid_vision).sum()),
                "association_conflicts": conflicts,
            }
        )

    def long(column: int) -> torch.Tensor:
        return torch.tensor(
            [row[column] for row in rows], device=device, dtype=torch.long
        )

    def boolean(column: int) -> torch.Tensor:
        return torch.tensor(
            [row[column] for row in rows], device=device, dtype=torch.bool
        )

    def floating(column: int) -> torch.Tensor:
        return floating_template.new_tensor([row[column] for row in rows])

    return AssociationRows(
        long(0), long(1), long(2), long(3), boolean(4), floating(5), floating(6),
        tuple(per_query), tuple(rv_pairs),
    )


class EvidenceScoreHead(nn.Module):
    """Independent zero-initialized bounded logit corrections for 3D and 2D."""

    def __init__(self, max_abs_delta_logit: float = 2.0) -> None:
        super().__init__()
        self.radar = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.vision = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
        )
        nn.init.zeros_(self.radar[-1].weight)
        nn.init.zeros_(self.radar[-1].bias)
        nn.init.zeros_(self.vision[-1].weight)
        nn.init.zeros_(self.vision[-1].bias)
        self.max_abs_delta_logit = float(max_abs_delta_logit)

    def forward(
        self,
        base: torch.Tensor,
        own: torch.Tensor,
        evidence: torch.Tensor,
        valid: torch.Tensor,
        gate: torch.Tensor,
        *,
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.radar if source == "R" else self.vision
        raw = head(torch.cat((own, evidence), dim=1))[:, 0].float()
        delta = (
            self.max_abs_delta_logit
            * torch.tanh(raw)
            * gate.float()
            * valid.float()
        )
        safe = base.float().clamp(1e-6, 1 - 1e-6)
        changed = torch.sigmoid(torch.logit(safe) + delta).to(base.dtype)
        # Keep exact forward identity when delta is zero while preserving the
        # derivative through the zero-initialized final layer.
        exact_identity_with_gradient = base + (changed - changed.detach())
        result = torch.where(delta == 0, exact_identity_with_gradient, changed)
        return result, delta.to(base.dtype)


class CandidateScoring(nn.Module):
    def __init__(self, geometry_gate: dict | None = None,
                 max_abs_delta_logit: float = 2.0) -> None:
        super().__init__()
        self.geometry_gate = validate_geometry_gate(geometry_gate)
        self.score_head = EvidenceScoreHead(max_abs_delta_logit)

    def forward(
        self,
        radar: CandidateBatch,
        vision: CandidateBatch,
        radar_evidence: CrossModalEvidence,
        vision_evidence: CrossModalEvidence,
        *,
        num_samples: int,
        enable_vision_scoring: bool = False,
        diagnostics: dict | None = None,
    ) -> MultimodalOutput:
        association = associate_geometry_greedy(
            radar, vision, num_samples=num_samples,
            geometry_gate=self.geometry_gate,
        )
        n = len(association.batch_index)
        index_device = association.batch_index.device
        has_radar = association.radar_index >= 0
        has_vision = association.vision_index >= 0

        # R and V can legitimately arrive with different dtypes under AMP
        # (LiDAR FP16 candidates and FP32 DINO object-query features). Keep
        # every buffer in the dtype of the value it stores.
        radar_feature = radar.feature.new_zeros((n, 128))
        vision_feature = vision.feature.new_zeros((n, 128))
        xyz = radar.xyz_m.new_zeros((n, 3))
        box = vision.box_xyxy_px.new_zeros((n, 4))
        base_3d = radar.score.new_zeros(n)
        base_2d = vision.score.new_zeros(n)
        radar_source = torch.full(
            (n,), -1, device=index_device, dtype=torch.long
        )
        vision_source = torch.full_like(radar_source, -1)
        evidence_3d = radar_evidence.feature.new_zeros((n, 128))
        evidence_2d = vision_evidence.feature.new_zeros((n, 128))
        valid_3d = radar_evidence.valid.new_zeros(n)
        valid_2d = vision_evidence.valid.new_zeros(n)
        gate_3d = radar_evidence.gate_weight.new_zeros(n)
        gate_2d = vision_evidence.gate_weight.new_zeros(n)
        count_3d = radar_evidence.token_count.new_zeros(n)
        count_2d = vision_evidence.token_count.new_zeros(n)

        if bool(has_radar.any()):
            rows = torch.nonzero(has_radar, as_tuple=False).flatten()
            indices = association.radar_index[rows]
            radar_feature[rows] = radar.feature[indices]
            xyz[rows] = radar.xyz_m[indices]
            base_3d[rows] = radar.score[indices]
            radar_source[rows] = radar.source_index[indices]
            evidence_3d[rows] = radar_evidence.feature[indices]
            valid_3d[rows] = radar_evidence.valid[indices]
            gate_3d[rows] = radar_evidence.gate_weight[indices]
            count_3d[rows] = radar_evidence.token_count[indices]
        if bool(has_vision.any()):
            rows = torch.nonzero(has_vision, as_tuple=False).flatten()
            indices = association.vision_index[rows]
            vision_feature[rows] = vision.feature[indices]
            box[rows] = vision.box_xyxy_px[indices]
            base_2d[rows] = vision.score[indices]
            vision_source[rows] = vision.source_index[indices]
            evidence_2d[rows] = vision_evidence.feature[indices]
            valid_2d[rows] = vision_evidence.valid[indices]
            gate_2d[rows] = vision_evidence.gate_weight[indices]
            count_2d[rows] = vision_evidence.token_count[indices]

        score_3d = base_3d.clone()
        delta_3d = base_3d.new_zeros(n)
        score_2d = base_2d.clone()
        delta_2d = base_2d.new_zeros(n)
        if bool(has_radar.any()):
            rows = torch.nonzero(has_radar, as_tuple=False).flatten()
            score_3d[rows], delta_3d[rows] = self.score_head(
                base_3d[rows], radar_feature[rows], evidence_3d[rows],
                valid_3d[rows], gate_3d[rows], source="R",
            )
        if enable_vision_scoring and bool(has_vision.any()):
            rows = torch.nonzero(has_vision, as_tuple=False).flatten()
            score_2d[rows], delta_2d[rows] = self.score_head(
                base_2d[rows], vision_feature[rows], evidence_2d[rows],
                valid_2d[rows], gate_2d[rows], source="V",
            )

        details = dict(diagnostics or {})
        details.update(
            association_per_query=association.per_query,
            association_rv_pairs=association.rv_pairs,
        )
        return MultimodalOutput(
            base_3d, score_3d, delta_3d,
            base_2d, score_2d, delta_2d,
            xyz, has_radar, box, has_vision,
            association.batch_index, association.hypothesis_type,
            radar_source, vision_source,
            valid_3d, gate_3d, count_3d,
            valid_2d, gate_2d, count_2d,
            association.association_valid,
            association.d_box_px,
            association.d_center_normalized,
            radar, vision, details,
        )
