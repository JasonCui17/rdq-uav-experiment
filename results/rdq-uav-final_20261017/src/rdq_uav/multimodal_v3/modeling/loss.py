"""Task-separated, label-masked candidate ranking losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import MultimodalOutput


@dataclass(frozen=True)
class MultimodalTargets:
    xyz_m: torch.Tensor
    has_xyz: torch.Tensor
    box_xyxy_px: torch.Tensor
    has_box: torch.Tensor


def box_iou_aligned(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes[:, :2], target[:2])
    bottom_right = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    target_area = (target[2:] - target[:2]).clamp_min(0).prod()
    return intersection / (area + target_area - intersection).clamp_min(1e-8)


class CandidateRankingLoss(nn.Module):
    """Focal ranking loss normalized independently for every labeled query."""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        xyz_positive_m: float = 1.0,
        xyz_ignore_m: float = 2.0,
        box_positive_iou: float = 0.5,
        box_ignore_iou: float = 0.3,
        lambda_3d: float = 1.0,
        lambda_2d: float = 1.0,
        negative_only_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.alpha = float(focal_alpha)
        self.gamma = float(focal_gamma)
        self.xyz_positive_m = float(xyz_positive_m)
        self.xyz_ignore_m = float(xyz_ignore_m)
        self.box_positive_iou = float(box_positive_iou)
        self.box_ignore_iou = float(box_ignore_iou)
        self.lambda_3d = float(lambda_3d)
        self.lambda_2d = float(lambda_2d)
        self.negative_only_weight = float(negative_only_weight)

    def _terms(self, score: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        score = score.float().clamp(1e-6, 1 - 1e-6)
        target = positive.float()
        # Lightning's FP16 autocast rejects probability-form BCE. Convert the
        # already-produced probability to its equivalent logit and use the
        # autocast-safe logits formulation; sigmoid(logit(score)) preserves
        # the same focal probability and the original loss semantics.
        logits = torch.logit(score)
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        probability = torch.where(positive, score, 1 - score)
        alpha = torch.where(positive, self.alpha, 1 - self.alpha)
        return alpha * (1 - probability).pow(self.gamma) * cross_entropy

    def _query_loss(
        self,
        score: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor | None, str]:
        if bool(positive.any()):
            valid = positive | negative
            return self._terms(score[valid], positive[valid]).sum() / positive.sum(), "positive"
        if bool(negative.any()):
            loss = self._terms(score[negative], positive[negative]).sum() / negative.sum()
            return self.negative_only_weight * loss, "negative_only"
        return None, "ignored_only"

    def forward(
        self, output: MultimodalOutput, targets: MultimodalTargets
    ) -> dict[str, torch.Tensor | int | bool]:
        zero = (output.score_3d_after.sum() + output.score_2d_after.sum()) * 0
        losses_3d: list[torch.Tensor] = []
        losses_2d: list[torch.Tensor] = []
        stats = {
            "n_gt3d": 0,
            "n_with_3d_candidate": 0,
            "n_with_positive_3d": 0,
            "n_negative_only_3d": 0,
            "n_no_3d_candidate": 0,
            "n_3d_loss_queries": 0,
            "n_gt2d": 0,
            "n_with_2d_candidate": 0,
            "n_with_positive_2d": 0,
            "n_negative_only_2d": 0,
            "n_no_2d_candidate": 0,
            "n_2d_loss_queries": 0,
        }

        for batch_idx in range(len(targets.has_xyz)):
            sample_rows = output.batch_index == batch_idx
            if bool(targets.has_xyz[batch_idx]):
                stats["n_gt3d"] += 1
                candidates = sample_rows & output.has_xyz
                if not bool(candidates.any()):
                    stats["n_no_3d_candidate"] += 1
                else:
                    stats["n_with_3d_candidate"] += 1
                    distance = torch.linalg.vector_norm(
                        output.xyz_m[candidates].float() - targets.xyz_m[batch_idx].float(),
                        dim=1,
                    )
                    positive = distance <= self.xyz_positive_m
                    negative = distance > self.xyz_ignore_m
                    query_loss, kind = self._query_loss(
                        output.score_3d_after[candidates], positive, negative
                    )
                    if kind == "positive":
                        stats["n_with_positive_3d"] += 1
                    elif kind == "negative_only":
                        stats["n_negative_only_3d"] += 1
                    if query_loss is not None and self.lambda_3d > 0:
                        losses_3d.append(query_loss)
                        stats["n_3d_loss_queries"] += 1

            if bool(targets.has_box[batch_idx]):
                stats["n_gt2d"] += 1
                candidates = sample_rows & output.has_box
                if not bool(candidates.any()):
                    stats["n_no_2d_candidate"] += 1
                else:
                    stats["n_with_2d_candidate"] += 1
                    iou = box_iou_aligned(
                        output.box_xyxy_px[candidates].float(),
                        targets.box_xyxy_px[batch_idx].float(),
                    )
                    positive = iou >= self.box_positive_iou
                    negative = iou <= self.box_ignore_iou
                    query_loss, kind = self._query_loss(
                        output.score_2d_after[candidates], positive, negative
                    )
                    if kind == "positive":
                        stats["n_with_positive_2d"] += 1
                    elif kind == "negative_only":
                        stats["n_negative_only_2d"] += 1
                    if query_loss is not None and self.lambda_2d > 0:
                        losses_2d.append(query_loss)
                        stats["n_2d_loss_queries"] += 1

        loss_3d = torch.stack(losses_3d).mean() if losses_3d else zero
        loss_2d = torch.stack(losses_2d).mean() if losses_2d else zero
        return {
            "loss": self.lambda_3d * loss_3d + self.lambda_2d * loss_2d,
            "loss_rank_3d": loss_3d,
            "loss_rank_2d": loss_2d,
            "has_trainable_loss": bool(
                (self.lambda_3d > 0 and losses_3d)
                or (self.lambda_2d > 0 and losses_2d)
            ),
            **stats,
        }
