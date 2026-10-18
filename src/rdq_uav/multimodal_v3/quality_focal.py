"""Single-class Quality Focal Loss adapter over the existing DINO criterion.

GFL (NeurIPS 2020), arXiv:2006.04388; beta=2 as in MMDetection's QFL.
This changes classification supervision, not matching or box regression.
"""
from types import MethodType

import torch
from torch.nn import functional as F


def _xyxy(boxes):
    return torch.cat((boxes[..., :2]-boxes[..., 2:]/2,
                      boxes[..., :2]+boxes[..., 2:]/2), dim=-1)


def _matched_quality(self, outputs, targets, indices):
    logits = outputs['pred_logits'].float()
    if logits.shape[-1] != 1 or self.num_classes != 1:
        raise ValueError('MMV3 quality focal adapter requires one UAV class')
    quality = torch.zeros_like(logits)
    matched = torch.zeros_like(logits, dtype=torch.bool)
    # IoU is a detached training target: classification must not move boxes to
    # make its target easier. Matching remains the original DINO assignment.
    with torch.no_grad():
        for batch, (source, target) in enumerate(indices):
            if not len(source):
                continue
            if torch.any(targets[batch]['labels'][target] != 0):
                raise ValueError('Expected UAV class zero')
            pred = _xyxy(outputs['pred_boxes'][batch, source].float())
            truth = _xyxy(targets[batch]['boxes'][target].float())
            intersection = (torch.minimum(pred[:, 2:], truth[:, 2:])-
                            torch.maximum(pred[:, :2], truth[:, :2])).clamp(min=0).prod(-1)
            union = (pred[:, 2:]-pred[:, :2]).clamp(min=0).prod(-1) + \
                    (truth[:, 2:]-truth[:, :2]).clamp(min=0).prod(-1) - intersection
            matched[batch, source, 0] = True
            quality[batch, source, 0] = (intersection/union.clamp(min=1e-12)).clamp(0, 1)
    return logits, quality, matched


def quality_focal_labels(self, outputs, targets, indices, num_boxes):
    logits, quality, _ = _matched_quality(self, outputs, targets, indices)
    scale = (quality-logits.sigmoid()).abs().square()
    loss = F.binary_cross_entropy_with_logits(logits, quality, reduction='none') * scale
    return {'loss_class': loss.sum()/num_boxes}


def enable_quality_focal(criterion):
    if criterion.num_classes != 1 or criterion.loss_class_type != 'focal_loss':
        raise ValueError('Expected the existing single-class focal DINO criterion')
    criterion.loss_labels = MethodType(quality_focal_labels, criterion)


def matchability_aware_labels(self, outputs, targets, indices, num_boxes):
    """DEIM CVPR2025 MAL, gamma=2 and mal_alpha=None (official default).

    A loss-only adaptation: Dense O2O/matching-union changes are not applied.
    See Intellindust-AI-Lab/DEIM engine/deim/deim_criterion.py.
    """
    logits, quality, matched = _matched_quality(self, outputs, targets, indices)
    weight = torch.where(matched, torch.ones_like(logits), logits.sigmoid().detach().square())
    loss = F.binary_cross_entropy_with_logits(logits, quality.square(), weight=weight, reduction='none')
    return {'loss_class': loss.sum()/num_boxes}


def enable_matchability_aware(criterion):
    if criterion.num_classes != 1 or criterion.loss_class_type != 'focal_loss':
        raise ValueError('Expected the existing single-class focal DINO criterion')
    criterion.loss_labels = MethodType(matchability_aware_labels, criterion)
