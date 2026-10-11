"""GT-free centroid change statistics on the existing sparse hierarchy."""
from __future__ import annotations

import math
import torch
from torch import nn


class TemporalCentroidFeatures(nn.Module):
    """Five fixed bins; adjacent supported pairs only, never bridge empty bins."""

    def __init__(self, config):
        super().__init__()
        self.history_s = float(config.get("history_s", 1.0))
        self.num_bins = int(config.get("num_bins", 5))
        self.min_gap_s = float(config.get("min_gap_s", .05))
        self.reference = float(config.get("speed_reference_mps", 1.0))
        self.reliability_only = bool(config.get("reliability_only", False))
        if self.history_s != 1.0 or self.num_bins != 5:
            raise ValueError("MTCE-3 requires a 1-second history and five bins")
        if not math.isfinite(self.min_gap_s) or not 0 < self.min_gap_s < .4:
            raise ValueError("min_gap_s must be finite and in (0, .4)")
        if not math.isfinite(self.reference) or self.reference <= 0:
            raise ValueError("speed_reference_mps must be finite and positive")

    @torch.no_grad()
    def forward(self, batch, hierarchy, return_debug=False):
        # Statistics need no gradients: only the three downstream projections learn.
        with torch.autocast(device_type=batch["points"].device.type, enabled=False):
            points = batch["points"].float()
            dt = batch["delta_t"].float()
            if dt.shape != (len(points),) or not torch.isfinite(dt).all():
                raise ValueError("MTCE requires one finite delta_t per point")
            if not torch.isfinite(points).all() or ((dt < -self.history_s) | (dt > 0)).any():
                raise ValueError("MTCE requires finite points and delta_t in [-1, 0]")
            # Internal boundaries go to the newer bin; both -1 and 0 are included.
            edges = torch.arange(1, self.num_bins, device=dt.device, dtype=torch.float32)
            edges = -self.history_s + edges * (self.history_s / self.num_bins)
            bins = torch.bucketize(dt.contiguous(), edges, right=True)
            map0 = hierarchy.point_to_l0
            map1 = hierarchy.parent_l0_to_l1[map0]
            map2 = hierarchy.parent_l1_to_l2[map1]
            features, diagnostics = [], []
            for level, point_map in zip(hierarchy.levels, (map0, map1, map2)):
                n = len(level.coords)
                index = point_map * self.num_bins + bins
                count = torch.bincount(index, minlength=n * self.num_bins).reshape(n, self.num_bins)
                if not torch.equal(count.sum(1), level.point_count):
                    raise AssertionError("MTCE point counts do not match hierarchy")
                denom = count.clamp_min(1).float()
                xyz_sum = points.new_zeros((n * self.num_bins, 3)).index_add_(0, index, points)
                time_sum = dt.new_zeros(n * self.num_bins).index_add_(0, index, dt)
                centroid = xyz_sum.reshape(n, self.num_bins, 3) / denom[..., None]
                mean_time = time_sum.reshape(n, self.num_bins) / denom
                supported = (count[:, :-1] > 0) & (count[:, 1:] > 0)
                gap = mean_time[:, 1:] - mean_time[:, :-1]
                valid = supported & (gap >= self.min_gap_s)
                speed = torch.linalg.vector_norm(centroid[:, 1:] - centroid[:, :-1], dim=-1)
                speed = torch.where(valid, speed / gap.clamp_min(self.min_gap_s), 0.)
                pairs = valid.sum(1)
                average = speed.sum(1) / pairs.clamp_min(1)
                pair_points = torch.minimum(count[:, :-1], count[:, 1:])
                sentinel = torch.iinfo(count.dtype).max
                minimum = pair_points.masked_fill(~valid, sentinel).amin(1)
                minimum = torch.where(pairs > 0, minimum, 0)
                motion = torch.stack((torch.log1p(average / self.reference),
                                      pairs.float() / (self.num_bins - 1),
                                      torch.log1p(minimum.float())), dim=1)
                if self.reliability_only:
                    motion[:, 0] = 0.
                if not torch.isfinite(motion).all():
                    raise FloatingPointError("Non-finite MTCE features")
                features.append(motion)
                diagnostic = {"supported_pair_count": supported.sum(),
                              "rejected_small_gap_count": (supported & ~valid).sum(),
                              "valid_pair_count": pairs.sum(),
                              "voxels_with_motion_support": (pairs > 0).sum()}
                if return_debug:
                    diagnostic.update(bin_count=count, bin_centroid=centroid,
                                      bin_mean_time=mean_time, pair_valid=valid,
                                      pair_gap_s=gap, pair_speed_mps=speed,
                                      motion=motion)
                diagnostics.append(diagnostic)
        return features, diagnostics


def load_a2_warmstart(detector, checkpoint):
    """Load every original radar tensor; allow only new motion projection keys."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    prefix = "network.lidar.detector."
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"A2 warmstart has no {prefix} keys: {checkpoint}")
    own = detector.state_dict()
    new = {"voxel_embed.motion_proj.weight", "merge01.motion_proj.weight", "merge12.motion_proj.weight"}
    if not new.issubset(own):
        raise ValueError("A2 warmstart requires the enabled MTCE-3 detector")
    missing = set(own) - set(selected)
    unexpected = set(selected) - set(own)
    mismatched = [k for k in set(own) & set(selected) if own[k].shape != selected[k].shape]
    if missing != new or unexpected or mismatched:
        raise ValueError(f"A2 warmstart mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={mismatched}")
    for key in new:
        if torch.count_nonzero(own[key]):
            raise ValueError(f"Warmstart requires zero-initialized {key}")
    result = detector.load_state_dict(selected, strict=False)
    if set(result.missing_keys) != new or result.unexpected_keys:
        raise AssertionError("Unexpected warmstart load result")
    return {"checkpoint": str(checkpoint), "loaded_keys": len(selected),
            "new_zero_initialized_keys": sorted(new)}


def set_radar_train_mode(detector, mode):
    if mode not in ("full", "motion_only"):
        raise ValueError(f"Unknown radar_train_mode: {mode}")
    detector.requires_grad_(mode == "full")
    if mode == "motion_only":
        projections = [getattr(module, "motion_proj", None) for module in
                       (detector.voxel_embed, detector.merge01, detector.merge12)]
        if any(module is None for module in projections):
            raise ValueError("motion_only requires all three MTCE projections")
        for module in projections:
            module.requires_grad_(True)
        if sum(p.numel() for p in detector.parameters() if p.requires_grad) != 1152:
            raise AssertionError("MTCE-3 motion_only must train exactly 1152 parameters")
