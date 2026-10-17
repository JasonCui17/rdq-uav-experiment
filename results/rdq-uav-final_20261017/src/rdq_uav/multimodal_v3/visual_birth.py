"""Camera-supported initialization over the unchanged frozen target KF.

Only births/rebirths use image evidence. Established tracks retain the original
radar association, including camera-unknown frames. Frustum-style projection
association is adapted from CenterFusion; this is not its learned architecture.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
import torch

from rdq_uav.multimodal_v3.modeling.geometry import project_omni_radtan
from .target_tracker import TargetTracker
from .tracking import Observations


@dataclass(frozen=True)
class VisualHint:
    image_time: float
    box: tuple
    score: float

    def __post_init__(self):
        b = np.asarray(self.box, dtype=float)
        if not math.isfinite(self.image_time) or not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError('Invalid visual time/score')
        if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
            raise ValueError('Invalid visual box')


def _subset(obs, keep):
    return Observations(obs.xyz[keep], obs.score[keep], obs.rank_value[keep], obs.token_id[keep])


class VisualBirthTracker(TargetTracker):
    def __init__(self, config, projection, *, visual_lookup):
        super().__init__(config)
        self.projection = projection
        self.visual_lookup = visual_lookup

    def step(self, timestamp, frame):
        t = float(timestamp)
        cfg = self.config
        needs_birth = (self.kf is None or t-self.last_update > cfg.filter.max_prediction_age_s or
                       t-self.last_time > cfg.filter.reset_gap_s)
        available, count = False, None
        if needs_birth:
            hint = self.visual_lookup(t)
            if hint is not None and hint.image_time > t:
                raise ValueError('Future image cannot initialize a causal radar track')
            available = hint is not None and hint.score >= cfg.birth_score and t-hint.image_time <= 1.0
            keep = np.zeros(len(frame.dense.xyz), dtype=bool)
            if available:
                x = frame.dense.xyz
                uv, valid = project_omni_radtan(torch.as_tensor(x, dtype=torch.float32),
                    torch.zeros(len(x), dtype=torch.long), self.projection)
                uv, box = uv.numpy(), np.asarray(hint.box)
                outside = np.maximum(np.maximum(box[:2]-uv, uv-box[2:]), 0)
                # Reuse the existing inverse-range geometry gate (16px@20m,
                # clipped to 8..48px); no target GT, height prior or tuned ROI.
                margin = np.clip(16*20/np.maximum(np.linalg.norm(x, axis=1), 1e-6), 8, 48)
                keep = valid.numpy() & (np.linalg.norm(outside, axis=1) <= margin)
            ids = set(frame.dense.token_id[keep].tolist())
            frame = replace(frame, dense=_subset(frame.dense, keep),
                global_pool=_subset(frame.global_pool, np.array([i in ids for i in frame.global_pool.token_id], bool)),
                baseline=_subset(frame.baseline, np.array([i in ids for i in frame.baseline.token_id], bool)))
            count = int(sum(frame.dense.score >= cfg.birth_score))
        result = super().step(t, frame)
        return {**result, 'birth_requires_vision': needs_birth,
                'birth_visual_available': available, 'birth_visual_candidates': count,
                'birth_visual_verified': result['event'] == 'birth' and available}
