"""Masked ranking supervision for unchanged multimodal candidates."""

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
    lt = torch.maximum(boxes[:, :2], target[:2]); rb = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (rb - lt).clamp_min(0).prod(1)
    area_a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    area_b = (target[2:] - target[:2]).clamp_min(0).prod()
    return intersection / (area_a + area_b - intersection).clamp_min(1e-8)


class CandidateRankingLoss(nn.Module):
    """Per-query focal ranking loss with independent 3D and 2D masks."""

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 xyz_positive_m: float = 1.0, xyz_ignore_m: float = 2.0,
                 box_positive_iou: float = 0.5, box_ignore_iou: float = 0.3,
                 lambda_3d: float = 1.0, lambda_2d: float = 1.0) -> None:
        super().__init__()
        self.alpha, self.gamma = float(focal_alpha), float(focal_gamma)
        self.xyz_positive_m, self.xyz_ignore_m = float(xyz_positive_m), float(xyz_ignore_m)
        self.box_positive_iou, self.box_ignore_iou = float(box_positive_iou), float(box_ignore_iou)
        self.lambda_3d, self.lambda_2d = float(lambda_3d), float(lambda_2d)

    def _focal(self, score: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        score = score[valid].float().clamp(1e-6, 1 - 1e-6)
        target = positive[valid].float()
        if not len(score):
            return score.sum()
        ce = F.binary_cross_entropy(score, target, reduction="none")
        pt = torch.where(target.bool(), score, 1 - score)
        alpha = torch.where(target.bool(), self.alpha, 1 - self.alpha)
        return (alpha * (1 - pt).pow(self.gamma) * ce).sum() / positive.sum().clamp_min(1)

    def forward(self, output: MultimodalOutput, targets: MultimodalTargets) -> dict[str, torch.Tensor | int]:
        zero = output.score_after.sum() * 0
        losses_3d, losses_2d = [], []
        positive_3d = positive_2d = supervised_3d = supervised_2d = 0
        for batch in range(len(targets.has_xyz)):
            rows = output.batch_index == batch
            if bool(targets.has_xyz[batch]):
                candidates = rows & output.has_xyz
                distance = torch.linalg.vector_norm(output.xyz_m[candidates].float() - targets.xyz_m[batch].float(), dim=1)
                positive = distance <= self.xyz_positive_m
                if bool(positive.any()):
                    valid = positive | (distance > self.xyz_ignore_m)
                    losses_3d.append(self._focal(output.score_after[candidates], positive, valid))
                    positive_3d += int(positive.sum()); supervised_3d += 1
            if bool(targets.has_box[batch]):
                candidates = rows & output.has_box
                if bool(candidates.any()):
                    iou = box_iou_aligned(output.box_xyxy_px[candidates].float(), targets.box_xyxy_px[batch].float())
                    positive = iou >= self.box_positive_iou
                    if bool(positive.any()):
                        valid = positive | (iou < self.box_ignore_iou)
                        losses_2d.append(self._focal(output.score_after[candidates], positive, valid))
                        positive_2d += int(positive.sum()); supervised_2d += 1
        loss_3d = torch.stack(losses_3d).mean() if losses_3d else zero
        loss_2d = torch.stack(losses_2d).mean() if losses_2d else zero
        return {
            "loss": self.lambda_3d * loss_3d + self.lambda_2d * loss_2d,
            "loss_rank_3d": loss_3d, "loss_rank_2d": loss_2d,
            "num_supervised_3d": supervised_3d, "num_supervised_2d": supervised_2d,
            "num_positive_3d": positive_3d, "num_positive_2d": positive_2d,
        }
