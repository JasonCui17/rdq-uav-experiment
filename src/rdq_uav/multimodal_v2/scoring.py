"""Geometry association and conservative evidence-based score correction."""

from __future__ import annotations

import torch
from torch import nn

from rdq_uav.multimodal_v1.candidate.association import associate_candidates
from rdq_uav.multimodal_v1.candidate.candidate_set import CandidateSet

from .contracts import CandidateBatch, CrossModalEvidence, MultimodalOutput


def _legacy(candidate: CandidateBatch) -> CandidateSet:
    return CandidateSet(
        score=candidate.score, feature=candidate.feature, xyz=candidate.xyz_m,
        xyz_valid=candidate.has_xyz, box_xyxy_px=candidate.box_xyxy_px,
        box_valid=candidate.has_box, batch_index=candidate.batch_index,
        source="radar" if candidate.source == "R" else "rgb",
        source_index=candidate.source_index,
    )


def _lookup_evidence(hypothesis_batch: torch.Tensor, hypothesis_source: torch.Tensor,
                     candidate: CandidateBatch, evidence: CrossModalEvidence):
    feature = candidate.feature.new_zeros((len(hypothesis_source), 128))
    valid = torch.zeros(len(hypothesis_source), dtype=torch.bool, device=feature.device)
    count = torch.zeros(len(hypothesis_source), dtype=torch.long, device=feature.device)
    gate = candidate.score.new_zeros(len(hypothesis_source))
    for index in range(candidate.n):
        match = (hypothesis_batch == candidate.batch_index[index]) & (hypothesis_source == candidate.source_index[index])
        if bool(match.any()):
            feature[match] = evidence.feature[index]
            valid[match] = evidence.valid[index]
            count[match] = evidence.token_count[index]
            gate[match] = evidence.gate_weight[index]
    return feature, valid, count, gate


class EvidenceScoreHead(nn.Module):
    """Zero-initialized, bounded correction of candidate logits.

    At initialization the explicit zero branch returns the exact original
    probability tensor, which makes B1 bitwise equivalent to B0.
    """

    def __init__(self, max_abs_delta_logit: float = 2.0) -> None:
        super().__init__()
        self.radar = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        self.vision = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.radar[-1].weight); nn.init.zeros_(self.radar[-1].bias)
        nn.init.zeros_(self.vision[-1].weight); nn.init.zeros_(self.vision[-1].bias)
        self.max_abs_delta_logit = float(max_abs_delta_logit)

    def forward(self, base_score: torch.Tensor, own_feature: torch.Tensor,
                evidence: torch.Tensor, evidence_valid: torch.Tensor,
                gate_weight: torch.Tensor, *, source: str):
        head = self.radar if source == "R" else self.vision
        raw = head(torch.cat((own_feature, evidence), dim=1))[:, 0].float()
        delta = self.max_abs_delta_logit * torch.tanh(raw) * gate_weight.float() * evidence_valid.float()
        safe = base_score.float().clamp(1e-6, 1.0 - 1e-6)
        rescored = torch.sigmoid(torch.logit(safe) + delta).to(base_score.dtype)
        # Required exact identity at initialization and for absent evidence.
        exact_identity_with_gradient = base_score + (rescored - rescored.detach())
        score = torch.where(delta == 0, exact_identity_with_gradient, rescored)
        return score, delta.to(base_score.dtype)


class CandidateScoring(nn.Module):
    def __init__(self, geometry_gate_px: float = 16.0,
                 max_abs_delta_logit: float = 2.0) -> None:
        super().__init__()
        self.geometry_gate_px = float(geometry_gate_px)
        self.score_head = EvidenceScoreHead(max_abs_delta_logit)

    def forward(self, radar: CandidateBatch, vision: CandidateBatch,
                radar_evidence: CrossModalEvidence, vision_evidence: CrossModalEvidence,
                projected_radar_xy: torch.Tensor, *, enable_vision_scoring: bool = False,
                diagnostics: dict | None = None) -> MultimodalOutput:
        hypotheses = associate_candidates(
            _legacy(radar), _legacy(vision), geometry_gate_px=self.geometry_gate_px,
            projected_radar_xy=projected_radar_xy,
        )
        n = hypotheses.n
        if n == 0:
            empty = radar.score.new_empty(0)
            return MultimodalOutput(
                empty, empty, empty, radar.xyz_m.new_empty((0, 3)),
                torch.empty(0, dtype=torch.bool, device=empty.device),
                radar.box_xyxy_px.new_empty((0, 4)), torch.empty(0, dtype=torch.bool, device=empty.device),
                torch.empty(0, dtype=torch.long, device=empty.device),
                torch.empty(0, dtype=torch.long, device=empty.device),
                torch.empty(0, dtype=torch.long, device=empty.device),
                torch.empty(0, dtype=torch.long, device=empty.device),
                torch.empty(0, dtype=torch.long, device=empty.device), empty,
                radar, vision, diagnostics or {},
            )
        r_feature, r_valid, r_count, r_gate = _lookup_evidence(
            hypotheses.batch_index, hypotheses.radar_source_index, radar, radar_evidence
        )
        v_feature, v_valid, v_count, v_gate = _lookup_evidence(
            hypotheses.batch_index, hypotheses.vision_source_index, vision, vision_evidence
        )
        has_r, has_v = hypotheses.m_R, hypotheses.m_V
        base = torch.where(has_r, hypotheses.radar_score, hypotheses.vision_score)
        post = base.clone(); delta = base.new_zeros(n)
        if bool(has_r.any()):
            indices = torch.nonzero(has_r, as_tuple=False).flatten()
            values, changes = self.score_head(
                base[indices], hypotheses.radar_feature[indices], r_feature[indices],
                r_valid[indices], r_gate[indices], source="R",
            )
            post[indices], delta[indices] = values, changes
        if enable_vision_scoring:
            only_v = ~has_r & has_v
            if bool(only_v.any()):
                indices = torch.nonzero(only_v, as_tuple=False).flatten()
                values, changes = self.score_head(
                    base[indices], hypotheses.vision_feature[indices], v_feature[indices],
                    v_valid[indices], v_gate[indices], source="V",
                )
                post[indices], delta[indices] = values, changes
        # Radar XYZ is copied verbatim. V-only rows retain storage zeros but the
        # explicit has_xyz=False contract excludes them from 3D ranking/loss.
        return MultimodalOutput(
            base, post, delta, hypotheses.radar_xyz, hypotheses.radar_xyz_valid,
            hypotheses.vision_box_xyxy_px, hypotheses.vision_box_valid,
            hypotheses.batch_index, hypotheses.hypothesis_type,
            hypotheses.radar_source_index, hypotheses.vision_source_index,
            torch.where(has_r, r_count, v_count), torch.where(has_r, r_gate, v_gate),
            radar, vision, diagnostics or {},
        )
