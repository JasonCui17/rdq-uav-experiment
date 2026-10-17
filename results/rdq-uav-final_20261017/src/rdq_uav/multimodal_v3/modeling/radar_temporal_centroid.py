"""GT-free centroid change statistics on the existing sparse hierarchy."""
from __future__ import annotations

import math
from dataclasses import dataclass
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
                    sensor_sum = points.new_zeros(n * self.num_bins).index_add_(
                        0, index, batch["sensor_id"].float())
                    diagnostic.update(bin_count=count, bin_centroid=centroid,
                                      bin_mean_time=mean_time, pair_valid=valid,
                                      pair_gap_s=gap, pair_speed_mps=speed,
                                      bin_sensor_ratio=sensor_sum.reshape(n, self.num_bins) / denom,
                                      motion=motion)
                diagnostics.append(diagnostic)
        return features, diagnostics


@dataclass(frozen=True)
class MotionContext:
    """GT-free, fixed-voxel observation motion; all physical values are FP32.

    This is not object velocity or pointwise instance segmentation. A missing
    fit is unknown, not static. No observations are associated across voxels.
    """
    descriptor: torch.Tensor       # [N, 11], dimensionless network input
    velocity_xyz: torch.Tensor     # [N, 3], meters/second
    reference_dt: torch.Tensor     # [N], weighted observation time relative to query
    reliability: torch.Tensor      # [N], [0, 1]
    valid: torch.Tensor            # [N], accepted continuous fit
    dynamic: torch.Tensor          # [N], reliable speed above configured noise threshold
    fit_error_m: torch.Tensor      # [N], weighted centroid RMS residual


class GatedTemporalMotion(nn.Module):
    """One motion context drives multiscale conditioning and L0-only offsets.

    The legacy MTCE bin engine is reused once per forward. Geometry/statistics
    have no gradient; conditioners and offset acceptance are trainable.
    """
    DESCRIPTOR_DIM = 11

    def __init__(self, config, dim=128):
        super().__init__()
        self.statistics = TemporalCentroidFeatures(config)
        self.feature_gate = bool(config.get("feature_gate", True))
        self.velocity_guidance = bool(config.get("velocity_guidance", True))
        self.min_bins = int(config.get("min_consecutive_bins", 3))
        self.min_points = int(config.get("min_points_per_bin", 2))
        self.count_cap = float(config.get("count_weight_cap", 8.))
        self.min_span = float(config.get("min_span_s", .3))
        self.error_scale = float(config.get("fit_error_scale_m", .05))
        self.max_error = float(config.get("max_fit_error_m", .1))
        self.max_sensor_change = float(config.get("max_sensor_ratio_change", .5))
        self.min_reliability = float(config.get("min_reliability", .25))
        self.dynamic_speed = float(config.get("dynamic_speed_mps", .15))
        self.max_speed = float(config.get("max_speed_mps", 15.))
        self.max_age = float(config.get("max_latest_age_s", .3))
        self.max_offset = float(config.get("max_offset_m", .5))
        if not 3 <= self.min_bins <= self.statistics.num_bins or self.min_points < 1:
            raise ValueError("motion fit requires 3..5 consecutive bins and positive point support")
        positive = (self.count_cap, self.min_span, self.error_scale, self.max_error,
                    self.dynamic_speed, self.max_speed, self.max_age, self.max_offset)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("motion scales and thresholds must be finite and positive")
        if not 0 <= self.max_sensor_change <= 1 or not 0 < self.min_reliability <= 1:
            raise ValueError("invalid sensor-change/reliability threshold")
        if self.dynamic_speed >= self.max_speed:
            raise ValueError("dynamic threshold must be below maximum accepted speed")
        if config.get("reliability_only", False):
            raise ValueError("reliability_only belongs to legacy MTCE, not gated_velocity")
        width = dim + self.DESCRIPTOR_DIM
        self.conditioners = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2 * dim))
            for _ in range(3)
        ]) if self.feature_gate else nn.ModuleList()
        self.offset_gate = (nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
                            if self.velocity_guidance else None)
        self.reset_identity()

    def reset_identity(self):
        # Zero only the final projections: original A2 forward is exactly kept.
        for block in list(self.conditioners) + ([self.offset_gate] if self.offset_gate is not None else []):
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    @torch.no_grad()
    def _fit(self, debug):
        count, c, t = debug["bin_count"], debug["bin_centroid"], debug["bin_mean_time"]
        n, bins = count.shape
        supported = count >= self.min_points
        # Choose longest consecutive run, newest on ties; never bridge a hole
        # or a too-small timestamp gap. Loop is over five bins, not voxels.
        run = torch.zeros(n, device=count.device, dtype=torch.long)
        best = run.clone(); end = run.clone()
        for k in range(bins):
            connected = debug["pair_valid"][:, k-1] if k else torch.zeros_like(supported[:, 0])
            run = torch.where(supported[:, k], torch.where(connected, run + 1, 1), 0)
            replace = (run > 0) & (run >= best)
            best = torch.where(replace, run, best)
            end = torch.where(replace, k, end)
        positions = torch.arange(bins, device=count.device)[None]
        mask = (positions <= end[:, None]) & (positions > (end-best)[:, None])
        weight = count.float().clamp(max=self.count_cap).sqrt() * mask
        total = weight.sum(1).clamp_min(1.)
        tref = (weight*t).sum(1) / total
        cref = (weight[..., None]*c).sum(1) / total[:, None]
        tau = t-tref[:, None]
        time_var = (weight*tau.square()).sum(1)
        velocity = (weight[..., None]*tau[..., None]*(c-cref[:, None])).sum(1) / time_var.clamp_min(1e-8)[:, None]
        fitted = cref[:, None] + tau[..., None]*velocity[:, None]
        error = ((weight*(c-fitted).square().sum(-1)).sum(1)/total).clamp_min(0).sqrt()
        tmin = t.masked_fill(~mask, float("inf")).amin(1)
        tmax = t.masked_fill(~mask, -float("inf")).amax(1)
        span = torch.where(best > 0, tmax-tmin, 0.)
        age = torch.where(best > 0, -tmax, self.statistics.history_s)
        ratio = debug["bin_sensor_ratio"]
        sensor_change = (ratio.masked_fill(~mask, -float("inf")).amax(1)
                         - ratio.masked_fill(~mask, float("inf")).amin(1))
        sensor_change = torch.where(best > 0, sensor_change, 0.)
        speed = velocity.norm(dim=1)
        valid = ((best >= self.min_bins) & (span >= self.min_span)
                 & (error <= self.max_error) & (speed <= self.max_speed)
                 & (age <= self.max_age) & (sensor_change <= self.max_sensor_change)
                 & torch.isfinite(velocity).all(1) & torch.isfinite(error))
        reliability = ((best.float()/bins) * (span/(2*self.min_span)).clamp(0, 1)
                       * torch.exp(-error/self.error_scale) * (1-sensor_change))
        reliability = torch.where(valid, reliability.clamp(0, 1), 0.)
        dynamic = valid & (reliability >= self.min_reliability) & (speed >= self.dynamic_speed)
        velocity = torch.where(valid[:, None], velocity, 0.)
        speed = velocity.norm(dim=1)
        # Dimensionless, bounded/log-scaled descriptors; distinguish unknown
        # fits from reliable zero speed using explicit validity and reliability.
        reference = self.statistics.reference
        descriptor = torch.cat((
            torch.sign(velocity)*torch.log1p(velocity.abs()/reference),
            torch.stack((torch.log1p(speed/reference), best.float()/bins,
                         torch.log1p((count.float()*mask).sum(1)),
                         (error/self.error_scale).clamp(max=10),
                         age.clamp(0, self.statistics.history_s)/self.statistics.history_s,
                         sensor_change, reliability, valid.float()), dim=1)), dim=1)
        if not torch.isfinite(descriptor).all():
            raise FloatingPointError("non-finite gated MTCE context")
        return MotionContext(descriptor, velocity, tref, reliability, valid, dynamic, error)

    def forward(self, batch, hierarchy):
        _, bins = self.statistics(batch, hierarchy, return_debug=True)
        with torch.autocast(device_type=batch["points"].device.type, enabled=False):
            contexts = [self._fit(item) for item in bins]
        diagnostics = []
        for context, item in zip(contexts, bins):
            diagnostics.append({key: value for key, value in item.items() if key in (
                "valid_pair_count", "voxels_with_motion_support", "rejected_small_gap_count")})
            diagnostics[-1].update(valid_fit_count=context.valid.sum(),
                                   dynamic_count=context.dynamic.sum(),
                                   voxel_count=context.valid.new_tensor(len(context.valid), dtype=torch.long))
        return contexts, diagnostics

    def condition(self, feature, context, level):
        if not self.feature_gate:
            return feature
        inputs = torch.cat((feature, context.descriptor.to(feature.dtype)), 1)
        gamma, beta = self.conditioners[level](inputs).chunk(2, dim=1)
        delta = torch.tanh(gamma)*feature + beta
        return feature + context.reliability[:, None].to(feature.dtype)*delta

    def offset(self, feature, context):
        if self.offset_gate is None:
            return feature.new_zeros((len(feature), 3), dtype=torch.float32)
        coefficient = torch.tanh(self.offset_gate(torch.cat(
            (feature, context.descriptor.to(feature.dtype)), 1))[:, 0].float())
        with torch.autocast(device_type=feature.device.type, enabled=False):
            prior = context.velocity_xyz*(-context.reference_dt).clamp(0, self.statistics.history_s)[:, None]
            offset = coefficient[:, None]*context.reliability[:, None]*prior
            # Bound displacement magnitude, not each axis separately.
            norm = offset.norm(dim=1, keepdim=True).clamp_min(1e-8)
            offset = offset*(self.max_offset/norm).clamp(max=1.)
            return torch.where(context.dynamic[:, None], offset, 0.)


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
