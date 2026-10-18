"""Opt-in A/B/C losses. Missing GT is never treated as background."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def quality_targets(outputs, batch, sigma):
    if sigma <= 0:
        raise ValueError('quality sigma must be positive')
    with torch.no_grad():
        # Candidate batch indices, NOT point batch indices: N_voxel != N_point.
        gt = batch['target_xyz'][outputs['batch_index']].float()
        distance = torch.linalg.vector_norm(outputs['pred_xyz'].detach().float() - gt, dim=1)
        return torch.exp(-distance.square() / (2 * sigma**2))


def focal_terms(logits, target, alpha, gamma):
    ce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    p = logits.float().sigmoid()
    pt = target * p + (1 - target) * (1 - p)
    return (target * alpha + (1 - target) * (1 - alpha)) * (1 - pt).pow(gamma) * ce


def range_weights(centers, config):
    edges = config.get('edges_m', [30., 60.])
    values = config.get('weights', [1., 1.5, 2.5])
    if len(values) != len(edges) + 1 or sorted(edges) != list(edges) or any(v <= 0 for v in values):
        raise ValueError('range weights require sorted edges and positive bin weights')
    origin = centers.new_tensor(config.get('origin_xyz_m', [0., 0., 0.]))
    ranges = torch.linalg.vector_norm(centers.float() - origin, dim=1)
    bins = torch.bucketize(ranges, ranges.new_tensor(edges), right=False)
    return ranges.new_tensor(values)[bins]


def negative_weights(logits, negative, config):
    """Per-query OHEM; always retain easy negatives with a nonzero weight."""
    weights = logits.new_ones(len(logits), dtype=torch.float32)
    if not config.get('enabled', False):
        return weights, 0
    topk = int(config.get('topk', 200))
    easy = float(config.get('easy_weight', .25))
    hard = float(config.get('hard_weight', 2.))
    if topk < 1 or easy <= 0 or hard < easy:
        raise ValueError('OHEM requires topk>=1 and 0<easy_weight<=hard_weight')
    ids = torch.nonzero(negative, as_tuple=False).flatten()
    weights[ids] = easy
    # No score threshold: early training must not silently remove every negative.
    chosen = ids[torch.argsort(logits[ids].detach().float(), descending=True, stable=True)[:topk]]
    weights[chosen] = hard
    return weights, len(chosen)


def dense_loss(owner, outputs, batch):
    cfg = owner.ablation
    mode = cfg.get('quality_mode', 'none')
    if mode not in ('none', 'soft', 'head'):
        raise ValueError(f'unknown quality mode {mode}')
    pos, ignore, neg, target = owner.labels(outputs, batch)
    logits = outputs['logits'].float()
    zero = logits.sum() * 0
    quality = quality_targets(outputs, batch, float(cfg.get('quality_sigma_m', 1.))) if mode != 'none' else None
    range_cfg = cfg.get('range', {})
    rw = range_weights(outputs['voxel_centers'], range_cfg) if range_cfg.get('enabled', False) else torch.ones_like(logits)
    cls_values, reg_values, qual_values = [], [], []
    no_support = hard_count = 0
    for b in range(int(batch['num_samples'])):
        if not bool(batch['target_valid'][b]):
            continue
        ids = torch.nonzero(outputs['batch_index'] == b, as_tuple=False).flatten()
        p, n = pos[ids], neg[ids]
        np_, nn_ = int(p.sum()), int(n.sum())
        no_support += int(np_ == 0)
        if np_ == 0 and (not cfg.get('negative_only', True) or nn_ == 0):
            continue
        z = logits[ids]
        t = p.float()
        if mode == 'soft':
            t = torch.where(p, quality[ids], torch.zeros_like(z))
        if mode == 'soft' or cfg.get('binary_bce', False):
            # Continuous target uses BCE. Do not reuse binary focal alpha/pt
            # with a soft target without explicitly defining another objective.
            terms = F.binary_cross_entropy_with_logits(z, t, reduction='none')
        else:
            terms = focal_terms(z, t, owner.alpha, owner.gamma)
        nw, hard_n = negative_weights(z, n, cfg.get('ohem', {}))
        hard_count += hard_n
        weight = torch.where(p, rw[ids], nw)
        # Keep original positive-count normalization, including with OHEM.
        # No-positive queries normalize by their own negative count.
        cls = (terms[p | n] * weight[p | n]).sum() / max(1, np_ if np_ else nn_)
        if not np_:
            cls = cls * float(cfg.get('negative_only_weight', .25))
        reg = zero
        if np_:
            reg = (F.smooth_l1_loss(outputs['residual_xyz'][ids[p]].float(), target[ids[p]].float(),
                                   beta=owner.beta, reduction='none').sum(1) * rw[ids[p]]).sum() / np_
        qual = zero
        if mode == 'head':
            qz = outputs['quality_logits'][ids].float()
            # Supervise negatives as zero too: positive-only quality is
            # unconstrained precisely on the false positives of interest.
            qt = torch.where(p, quality[ids], torch.zeros_like(z))
            qt_terms = F.binary_cross_entropy_with_logits(qz, qt, reduction='none')
            pieces = []
            if np_:
                pieces.append(qt_terms[p].mean())
            if nn_:
                pieces.append(float(cfg.get('quality_negative_weight', .25)) * qt_terms[n].mean())
            qual = sum(pieces, zero)
        cls_values.append(cls); reg_values.append(reg); qual_values.append(qual)
    mean = lambda values: torch.stack(values).mean() if values else zero
    cls, reg, qual = mean(cls_values), mean(reg_values), mean(qual_values)
    return dict(loss=cls + owner.reg_weight * reg + float(cfg.get('quality_loss_weight', 1.)) * qual,
                loss_cls=cls, loss_reg=reg, loss_quality=qual,
                num_pos=int(pos.sum()), num_neg=int(neg.sum()), num_ignore=int(ignore.sum()),
                num_supervised_samples=len(cls_values), num_no_current_support=no_support,
                num_hard_negatives=hard_count, positive_mask=pos, ignore_mask=ignore, negative_mask=neg)


def query_loss(owner, outputs, batch):
    cfg = owner.ablation.get('query', {})
    cls_weight = float(cfg.get('matching_class_weight', 1.))
    xyz_weight = float(cfg.get('matching_xyz_weight', 1.))
    scale = float(cfg.get('coordinate_scale_m', 50.))
    bg_weight = float(cfg.get('background_weight', .1))
    if scale <= 0 or not 0 < bg_weight <= 1:
        raise ValueError('query loss requires positive scale and background weight in (0,1]')
    z = outputs['class_logits'].float()  # class 0=UAV, class 1=background
    zero = z.sum() * 0
    cls_values, reg_values = [], []
    matched = torch.zeros(len(z), device=z.device, dtype=torch.bool)
    negative = torch.zeros_like(matched)
    for b in range(int(batch['num_samples'])):
        if not bool(batch['target_valid'][b]):
            continue
        ids = torch.nonzero(outputs['batch_index'] == b, as_tuple=False).flatten()
        if not len(ids):
            continue
        pred = outputs['pred_xyz'][ids].float()
        gt = batch['target_xyz'][b].float()
        # Single valid GT -> one assignment. This assumes target presence,
        # not that a supporting return must lie within 1m of the GT.
        with torch.no_grad():
            cost = -cls_weight * z[ids].softmax(-1)[:, 0] + xyz_weight * (pred - gt).abs().sum(1) / scale
            best = cost.argmin()
        labels = torch.ones(len(ids), device=z.device, dtype=torch.long)
        labels[best] = 0
        cls_values.append(F.cross_entropy(z[ids], labels, weight=z.new_tensor([1., bg_weight])))
        # Regression remains in metres, matching the dense baseline's units.
        reg_values.append(F.smooth_l1_loss(pred[best], gt, beta=owner.beta, reduction='sum'))
        negative[ids] = True; negative[ids[best]] = False; matched[ids[best]] = True
    mean = lambda values: torch.stack(values).mean() if values else zero
    cls, reg = mean(cls_values), mean(reg_values)
    return dict(loss=cls + owner.reg_weight * reg, loss_cls=cls, loss_reg=reg,
                num_pos=int(matched.sum()), num_neg=int(negative.sum()), num_ignore=0,
                num_supervised_samples=len(cls_values), num_no_current_support=0,
                positive_mask=matched, negative_mask=negative, ignore_mask=torch.zeros_like(matched))
