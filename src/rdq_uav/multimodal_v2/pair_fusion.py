"""B4: independent feature alignment, Hungarian pairing, fresh joint logits.

The discrete assignment is detached. Gradients flow through the selected pair
features and the complete per-query similarity matrices, never through SciPy.
Original detector scores select Top-K only; they are not scoring inputs.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .contracts import CandidateBatch, MultimodalOutput
from .geometry import geometry_gate_margins, validate_geometry_gate
from .scoring import point_to_box_distance


def select_topk(candidates: CandidateBatch, num_samples: int, topk: int) -> CandidateBatch:
    selected = []
    for batch in range(num_samples):
        ids = torch.nonzero(candidates.batch_index == batch).flatten()
        ids = ids[torch.argsort(candidates.source_index[ids], stable=True)]
        ids = ids[torch.argsort(candidates.score[ids].float(), descending=True, stable=True)]
        selected.append(ids[:topk])
    return candidates.index_select(torch.cat(selected))


def solve_assignment(cost: torch.Tensor, feasible: torch.Tensor) -> list[tuple[int, int]]:
    """Maximum valid cardinality, then minimum total semantic/geometric cost."""
    nr, nv = cost.shape
    if nr == 0 or nv == 0 or not bool(feasible.any()):
        return []
    real = cost.detach().double().cpu().numpy()
    allowed = feasible.detach().cpu().numpy()
    if not np.isfinite(real[allowed]).all() or (real[allowed] < 0).any():
        raise ValueError("matching cost must be finite and nonnegative on feasible edges")
    dummy = (min(nr, nv) + 1) * (float(real[allowed].max()) + 1)
    augmented = np.full((nr, nv + nr), dummy, dtype=np.float64)
    augmented[:, :nv] = np.where(allowed, real, np.inf)
    rows, columns = linear_sum_assignment(augmented)
    return [(int(r), int(v)) for r, v in zip(rows, columns) if v < nv]


class CandidatePairFusion(nn.Module):
    def __init__(self, *, topk: int = 50, geometry_gate: dict | None = None,
                 geometry_weight: float = 1., similarity_weight: float = 1.,
                 center_weight: float = .01, reject_outside_gate: bool = False) -> None:
        super().__init__()
        if topk <= 0:
            raise ValueError("pair topk must be positive")
        weights = (geometry_weight, similarity_weight, center_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("matching weights must be finite, nonnegative and not all zero")
        self.topk = int(topk)
        self.geometry_gate = validate_geometry_gate(geometry_gate)
        self.geometry_weight, self.similarity_weight, self.center_weight = weights
        self.reject_outside_gate = bool(reject_outside_gate)
        def adapter():
            return nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Linear(128, 128), nn.LayerNorm(128))
        self.radar_adapter, self.vision_adapter = adapter(), adapter()
        # R, V, absolute difference, product; six geometry values and two masks.
        self.pair_encoder = nn.Sequential(nn.LayerNorm(520), nn.Linear(520, 128),
                                          nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.joint_head = nn.Linear(128, 1)

    def forward(self, radar: CandidateBatch, vision: CandidateBatch, *, num_samples: int,
                diagnostics: dict | None = None) -> MultimodalOutput:
        # All B4 trainable layers and their intervening activations stay FP32.
        # Frozen detector features may arrive in FP16/BF16, but are constants
        # for this stage. This avoids scaled-gradient FP16 cast overflow.
        with torch.autocast(device_type=radar.feature.device.type, enabled=False):
            return self._forward_fp32(
                replace(radar, feature=radar.feature.float()),
                replace(vision, feature=vision.feature.float()),
                num_samples=num_samples, diagnostics=diagnostics,
            )

    def _forward_fp32(self, radar: CandidateBatch, vision: CandidateBatch, *, num_samples: int,
                      diagnostics: dict | None = None) -> MultimodalOutput:
        radar = select_topk(radar, num_samples, self.topk)
        vision = select_topk(vision, num_samples, self.topk)
        hr = self.radar_adapter(radar.feature)
        hv = self.vision_adapter(vision.feature)
        # Matmul/normalization in FP32, even under AMP.
        with torch.autocast(device_type=hr.device.type, enabled=False):
            ur = F.normalize(hr.float(), dim=1, eps=1e-6)
            uv = F.normalize(hv.float(), dim=1, eps=1e-6)
        records, alignment, audits, pairs = [], [], [], []
        for batch in range(num_samples):
            rids = torch.nonzero(radar.batch_index == batch).flatten()
            vids = torch.nonzero(vision.batch_index == batch).flatten()
            rids = rids[torch.argsort(radar.source_index[rids], stable=True)]
            vids = vids[torch.argsort(vision.source_index[vids], stable=True)]
            with torch.autocast(device_type=hr.device.type, enabled=False):
                sim = (ur[rids] @ uv[vids].T).clamp(-1, 1)
            valid_r = (radar.projection_valid[rids]
                       & torch.isfinite(radar.projected_xy_px[rids]).all(1)
                       & torch.isfinite(radar.xyz_m[rids]).all(1))
            boxes = vision.box_xyxy_px[vids].float()
            points = torch.nan_to_num(radar.projected_xy_px[rids].float())
            dbox = point_to_box_distance(points, boxes)
            diagonal = (boxes[:, 2:] - boxes[:, :2]).norm(dim=1).clamp_min(1)
            center = (boxes[:, :2] + boxes[:, 2:]) * .5
            dcenter = (points[:, None] - center[None]).norm(dim=2) / diagonal[None]
            margin = geometry_gate_margins(radar.xyz_m[rids], self.geometry_gate)
            valid_edges = valid_r[:, None].expand(len(rids), len(vids))
            geometry = torch.log1p(dbox / margin[:, None])
            cost = (self.geometry_weight * geometry
                    + self.center_weight * torch.log1p(dcenter)
                    + self.similarity_weight * (1 - sim) * .5)
            feasible = valid_edges & torch.isfinite(cost)
            if self.reject_outside_gate:
                feasible = feasible & (dbox <= margin[:, None])
            selected = solve_assignment(cost, feasible)
            alignment.append(dict(radar_index=rids, vision_index=vids,
                                  similarity=sim, valid_edges=valid_edges))
            used_r, used_v = set(), set()
            for ri, vi in selected:
                r, v = int(rids[ri]), int(vids[vi])
                records.append((r, v, batch, float(dbox[ri, vi]), float(dcenter[ri, vi])))
                used_r.add(r); used_v.add(v)
                pairs.append(dict(batch_index=batch, radar_source_index=int(radar.source_index[r]),
                                  vision_source_index=int(vision.source_index[v]),
                                  d_box_px=float(dbox[ri, vi]), d_center_normalized=float(dcenter[ri, vi]),
                                  similarity=float(sim[ri, vi].detach()), cost=float(cost[ri, vi].detach())))
            records.extend((int(r), -1, batch, 0., 0.) for r in rids if int(r) not in used_r)
            records.extend((-1, int(v), batch, 0., 0.) for v in vids if int(v) not in used_v)
            audits.append(dict(batch_index=batch, association_method="semantic_geometry_hungarian",
                               radar_candidates=len(rids), vision_candidates=len(vids),
                               similarity_shape=list(sim.shape), feasible_pairs=int(feasible.sum()),
                               rv=len(selected), r=len(rids)-len(selected), v=len(vids)-len(selected),
                               reject_outside_gate=self.reject_outside_gate,
                               invalid_radar_projection=int((~valid_r).sum())))

        device = radar.feature.device
        n = len(records)
        rid = torch.tensor([x[0] for x in records], device=device, dtype=torch.long)
        vid = torch.tensor([x[1] for x in records], device=device, dtype=torch.long)
        batch_ids = torch.tensor([x[2] for x in records], device=device, dtype=torch.long)
        has_r, has_v = rid >= 0, vid >= 0
        paired = has_r & has_v
        rfeat, vfeat = hr.new_zeros(n, 128), hv.new_zeros(n, 128)
        xyz, box = radar.xyz_m.new_zeros(n, 3), vision.box_xyxy_px.new_zeros(n, 4)
        rscore, vscore = torch.zeros(n, device=device), torch.zeros(n, device=device)
        rsource, vsource = rid.new_full((n,), -1), vid.new_full((n,), -1)
        dbox = torch.tensor([x[3] for x in records], device=device, dtype=torch.float32)
        dcenter = torch.tensor([x[4] for x in records], device=device, dtype=torch.float32)
        geometry = torch.zeros(n, 6, device=device)
        if bool(has_r.any()):
            ids = rid[has_r]
            rfeat[has_r] = hr[ids]; xyz[has_r] = radar.xyz_m[ids]
            rscore[has_r] = radar.score[ids].float(); rsource[has_r] = radar.source_index[ids]
            geometry[has_r, 2] = torch.log1p(torch.nan_to_num(xyz[has_r].float().norm(dim=1)))
            geometry[has_r, 5] = radar.projection_valid[ids].float()
        if bool(has_v.any()):
            ids = vid[has_v]
            vfeat[has_v] = hv[ids]; box[has_v] = vision.box_xyxy_px[ids]
            vscore[has_v] = vision.score[ids].float(); vsource[has_v] = vision.source_index[ids]
            geometry[has_v, 3:5] = torch.log1p((box[has_v, 2:].float()-box[has_v, :2].float()).clamp_min(0))
        if bool(paired.any()):
            scale = geometry_gate_margins(xyz[paired], self.geometry_gate)
            geometry[paired, 0] = torch.log1p(dbox[paired] / scale)
            geometry[paired, 1] = torch.log1p(dcenter[paired])
        pair_sim = torch.zeros(n, device=device)
        pair_sim[paired] = (ur[rid[paired]] * uv[vid[paired]]).sum(1).clamp(-1, 1)
        interaction_mask = paired[:, None].to(rfeat.dtype)
        inputs = torch.cat((rfeat, vfeat, (rfeat-vfeat).abs()*interaction_mask,
                            rfeat*vfeat*interaction_mask, geometry.to(rfeat.dtype),
                            has_r[:, None].to(rfeat.dtype), has_v[:, None].to(rfeat.dtype)), dim=1)
        joint_logit = self.joint_head(self.pair_encoder(inputs))[:, 0].float()
        probability = torch.sigmoid(joint_logit)
        score3 = torch.where(has_r, probability, 0.)
        score2 = torch.where(has_v, probability, 0.)
        types = torch.where(paired, 0, torch.where(has_r, 1, 2))
        details = dict(diagnostics or {})
        details.update(scoring_mode="fresh_joint_pair", association_per_query=tuple(audits),
                       association_rv_pairs=tuple(pairs), pair_alignment=tuple(alignment),
                       original_scores_used_for="topk_and_before_diagnostics_only")
        return MultimodalOutput(
            rscore, score3, score3-rscore, vscore, score2, score2-vscore,
            xyz, has_r, box, has_v, batch_ids, types, rsource, vsource,
            paired, paired.float(), paired.long(), paired, paired.float(), paired.long(),
            paired, dbox, dcenter, radar, vision, details,
            joint_logit=joint_logit, pair_similarity=pair_sim,
        )
