"""GT-masked joint classification, feature-distance and Top1 ranking for B4."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import CandidateBatch, MultimodalOutput
from .loss import CandidateRankingLoss, MultimodalTargets, box_iou_aligned


class JointPairLoss(nn.Module):
    """One UAV GT per query; geometry assignment is never a pseudo GT label.

The positive embedding penalty is (1-cos), half squared unit-vector distance.
Hard negatives penalize cos above a margin. Two-background pairs supervise
joint confidence but not alignment: disparate backgrounds need no common metric.
"""
    def __init__(self, *, xyz_positive_m: float = 1., box_positive_iou: float = .5,
                 xyz_boundary_m: float = .05, iou_boundary: float = .02,
                 hard_negative_weight: float = 2., ordinary_negative_weight: float = 1.,
                 lambda_cls: float = 1., lambda_align: float = 1., lambda_rank: float = .5,
                 negative_similarity_margin: float = .2, ranking_margin: float = 1.) -> None:
        super().__init__()
        values = (xyz_positive_m, box_positive_iou, xyz_boundary_m, iou_boundary,
                  hard_negative_weight, ordinary_negative_weight, lambda_cls,
                  lambda_align, lambda_rank, ranking_margin)
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("pair loss thresholds and weights must be finite and nonnegative")
        if not 0 < box_positive_iou <= 1 or xyz_positive_m <= 0:
            raise ValueError("invalid positive thresholds")
        if xyz_boundary_m >= xyz_positive_m or iou_boundary >= min(box_positive_iou, 1-box_positive_iou):
            raise ValueError("boundary tolerance must lie inside the positive threshold range")
        if not math.isfinite(negative_similarity_margin) or not -1 <= negative_similarity_margin < 1:
            raise ValueError("negative cosine margin must be in [-1,1)")
        self.xyz_positive_m, self.box_positive_iou = xyz_positive_m, box_positive_iou
        self.xyz_boundary_m, self.iou_boundary = xyz_boundary_m, iou_boundary
        self.hard_weight, self.ordinary_weight = hard_negative_weight, ordinary_negative_weight
        self.lambda_cls, self.lambda_align, self.lambda_rank = lambda_cls, lambda_align, lambda_rank
        self.sim_margin, self.rank_margin = negative_similarity_margin, ranking_margin

    def candidate_status(self, candidates: CandidateBatch, targets: MultimodalTargets) -> torch.Tensor:
        # -1 unknown/boundary, 0 confidently incorrect, 1 confidently correct.
        result = torch.full((candidates.n,), -1, device=candidates.feature.device, dtype=torch.long)
        for batch in range(len(targets.has_xyz)):
            ids = torch.nonzero(candidates.batch_index == batch).flatten()
            if candidates.source == "R" and bool(targets.has_xyz[batch]):
                distance = (candidates.xyz_m[ids].float()-targets.xyz_m[batch].float()).norm(dim=1)
                positive = distance <= self.xyz_positive_m-self.xyz_boundary_m
                negative = distance > self.xyz_positive_m+self.xyz_boundary_m
                finite = torch.isfinite(distance)
            elif candidates.source == "V" and bool(targets.has_box[batch]):
                iou = box_iou_aligned(candidates.box_xyxy_px[ids].float(), targets.box_xyxy_px[batch].float())
                positive = iou >= self.box_positive_iou+self.iou_boundary
                negative = iou < self.box_positive_iou-self.iou_boundary
                finite = torch.isfinite(iou)
            else:
                continue
            result[ids[positive & finite]] = 1
            result[ids[negative & finite]] = 0
        return result

    @staticmethod
    def _pair_masks(r: torch.Tensor, v: torch.Tensor):
        known = (r >= 0) & (v >= 0)
        positive = known & (r == 1) & (v == 1)
        hard = known & ((r == 1) ^ (v == 1))
        ordinary = known & (r == 0) & (v == 0)
        return positive, hard, ordinary

    def forward(self, output: MultimodalOutput, targets: MultimodalTargets) -> dict:
        if output.joint_logit is None:
            raise ValueError("JointPairLoss requires B4 joint logits")
        logits = output.joint_logit.float()
        zero = logits.sum()*0
        rstatus = self.candidate_status(output.radar_candidates, targets)
        vstatus = self.candidate_status(output.vision_candidates, targets)
        # Reuse public coverage counters without applying the old focal losses.
        with torch.no_grad():
            legacy = CandidateRankingLoss(lambda_3d=0., lambda_2d=0.)(output, targets)
        stats = {k: v for k, v in legacy.items() if k.startswith("n_")}
        stats.update(n_pair_positive=0, n_pair_hard_negative=0, n_pair_ordinary_negative=0,
                     n_pair_ignored=0, n_alignment_positive=0, n_alignment_hard_negative=0,
                     n_joint_loss_queries=0, n_joint_rank_queries=0)
        labels = torch.full_like(output.batch_index, -1)
        hard_rows = torch.zeros_like(labels, dtype=torch.bool)
        # Map retained candidate indices via source IDs within each query.
        for batch in range(len(targets.has_xyz)):
            rows = torch.nonzero(output.batch_index == batch).flatten()
            rlookup = {int(output.radar_candidates.source_index[i]): int(rstatus[i])
                       for i in torch.nonzero(output.radar_candidates.batch_index == batch).flatten()}
            vlookup = {int(output.vision_candidates.source_index[i]): int(vstatus[i])
                       for i in torch.nonzero(output.vision_candidates.batch_index == batch).flatten()}
            for row in rows.tolist():
                r = rlookup.get(int(output.radar_source_index[row]), -1)
                v = vlookup.get(int(output.vision_source_index[row]), -1)
                if bool(output.has_xyz[row] & output.has_box[row]):
                    if r < 0 or v < 0:
                        stats["n_pair_ignored"] += 1
                    elif r == v == 1:
                        labels[row] = 1; stats["n_pair_positive"] += 1
                    elif r != v:
                        labels[row] = 0; hard_rows[row] = True
                        stats["n_pair_hard_negative"] += 1
                    else:
                        labels[row] = 0; stats["n_pair_ordinary_negative"] += 1
                else:
                    labels[row] = r if bool(output.has_xyz[row]) else v

        cls_losses, rank_losses, align_losses = [], [], []
        for batch in range(len(targets.has_xyz)):
            in_query = output.batch_index == batch
            positive = in_query & (labels == 1)
            hard = in_query & (labels == 0) & hard_rows
            ordinary = in_query & (labels == 0) & ~hard_rows
            terms = []
            if bool(positive.any()):
                terms.append(F.softplus(-logits[positive]).mean())
            if bool(hard.any()) and self.hard_weight > 0:
                terms.append(self.hard_weight*F.softplus(logits[hard]).mean())
            if bool(ordinary.any()) and self.ordinary_weight > 0:
                terms.append(self.ordinary_weight*F.softplus(logits[ordinary]).mean())
            if terms:
                cls_losses.append(torch.stack(terms).sum())
                stats["n_joint_loss_queries"] += 1
            negative = hard | ordinary
            if bool(positive.any() & negative.any()):
                rank_losses.append(F.softplus(self.rank_margin+logits[negative].max()-logits[positive].max()))
                stats["n_joint_rank_queries"] += 1

        # Supervise all candidate pairs, not only the selected assignment: a bad
        # early assignment cannot hide every true pair from the alignment loss.
        for item in output.diagnostics["pair_alignment"]:
            r = rstatus[item["radar_index"]][:, None]
            v = vstatus[item["vision_index"]][None]
            positive, hard, _ = self._pair_masks(r, v)
            positive &= item["valid_edges"]
            hard &= item["valid_edges"]
            sim = item["similarity"].float()
            terms = []
            if bool(positive.any()):
                terms.append((1-sim[positive]).mean())
                stats["n_alignment_positive"] += int(positive.sum())
            if bool(hard.any()) and self.hard_weight > 0:
                terms.append(self.hard_weight*F.relu(sim[hard]-self.sim_margin).mean())
                stats["n_alignment_hard_negative"] += int(hard.sum())
            if terms:
                align_losses.append(torch.stack(terms).sum())

        cls = torch.stack(cls_losses).mean() if cls_losses else zero
        align = torch.stack(align_losses).mean() if align_losses else zero
        rank = torch.stack(rank_losses).mean() if rank_losses else zero
        loss = self.lambda_cls*cls+self.lambda_align*align+self.lambda_rank*rank
        return dict(loss=loss, loss_joint_cls=cls, loss_align=align, loss_joint_rank=rank,
                    loss_rank_3d=zero, loss_rank_2d=zero,
                    has_trainable_loss=bool((cls_losses and self.lambda_cls > 0)
                                           or (align_losses and self.lambda_align > 0)
                                           or (rank_losses and self.lambda_rank > 0)),
                    joint_labels=labels, **stats)
