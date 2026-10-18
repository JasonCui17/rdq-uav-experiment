"""Bounded single-target adaptation of BYTE association in 3D.

Reuses the existing CVKalman. High confidence observations start a target;
low confidence dense candidates may only continue an established target.
No semantic background deletion and no competition between persistent tracks.
This adapts BYTE's association principle, not its complete box MOT algorithm.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .tracking import CVKalman, TrackerConfig


@dataclass(frozen=True)
class TargetConfig:
    filter: TrackerConfig = field(default_factory=TrackerConfig)
    birth_score: float = .9
    high_score: float = .5
    fallback_radius_m: float = 1.
    quality_weight: float = 2.
    high_rank_first: bool = False

    def __post_init__(self):
        if not 0 < self.high_score <= self.birth_score <= 1:
            raise ValueError('Invalid confidence thresholds')
        if not math.isfinite(self.fallback_radius_m) or self.fallback_radius_m <= 0:
            raise ValueError('Invalid fallback radius')
        if not math.isfinite(self.quality_weight) or self.quality_weight < 0:
            raise ValueError('Invalid quality weight')
        if type(self.high_rank_first) is not bool:
            raise ValueError('high_rank_first must be boolean')


class TargetTracker:
    def __init__(self, config=None):
        self.config = config or TargetConfig()
        self.kf = None
        self.last_time = self.last_update = None

    def step(self, timestamp, frame):
        cfg, t = self.config, float(timestamp)
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Timestamps must be finite and strictly increasing')
        dt = None if self.last_time is None else t - self.last_time
        if self.kf is not None and (dt > cfg.filter.reset_gap_s or
                                   t - self.last_update > cfg.filter.max_prediction_age_s):
            self.kf = None
        self.last_time = t
        obs = frame.dense
        selected = None
        event = 'warmup'
        if self.kf is None:
            eligible = np.flatnonzero(obs.score >= cfg.birth_score)
            rank = obs.rank_value if cfg.high_rank_first else obs.score
            order = eligible[np.lexsort((obs.token_id[eligible], -rank[eligible]))]
            if len(order):
                selected = int(order[0])
                self.kf = CVKalman(obs.xyz[selected], cfg.filter)
                self.last_update = t
                event = 'birth'
        else:
            self.kf.predict(dt)
            delta, d2 = self.kf.innovations(obs.xyz)
            distance = np.linalg.norm(delta, axis=1)
            allowed = (d2 <= cfg.filter.mahalanobis_gate_sq) & (distance <= cfg.filter.max_innovation_m)
            strong = np.flatnonzero(allowed & (obs.score >= cfg.high_score))
            weak = np.flatnonzero(allowed & (distance <= cfg.fallback_radius_m) &
                                  (obs.score < cfg.high_score))
            for candidates, is_strong in ((strong, True), (weak, False)):
                cost = .5 * d2 - cfg.quality_weight * np.log(np.clip(obs.score, 1e-8, 1.))
                # Scores saturate near 1: using -log(score) with motion cost
                # then systematically favors the old prediction. Preserve the
                # detector's original logit ranking among gated strong returns.
                key = -obs.rank_value if is_strong and cfg.high_rank_first else cost
                order = candidates[np.lexsort((obs.token_id[candidates], key[candidates]))]
                for idx in order:
                    x, _ = self.kf.posterior(obs.xyz[idx])
                    if np.linalg.norm(x[3:]) <= cfg.filter.max_speed_mps:
                        selected = int(idx)
                        self.kf.update(obs.xyz[selected])
                        self.last_update = t
                        event = 'high_update' if is_strong else 'low_update'
                        break
                if selected is not None:
                    break
            if selected is None:
                event = 'predict'
        return {
            'xyz': None if self.kf is None else self.kf.x[:3].tolist(),
            'velocity_mps': None if self.kf is None else self.kf.x[3:].tolist(),
            'event': event,
            'selected_token_id': None if selected is None else int(obs.token_id[selected]),
            'observation_xyz': None if selected is None else obs.xyz[selected].tolist(),
            'measurement_age_s': None if self.kf is None else t - self.last_update,
        }
