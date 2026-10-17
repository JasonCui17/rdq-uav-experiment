"""Bounded lookahead RTS adapter using Stone Soup's existing smoother.

Inputs contain filtered state/covariance only, never ground truth. Leading
warmup may be back-extrapolated from an anchor inside the declared lookahead.
Timeout gaps are not filled across a new target birth.
"""
from datetime import datetime, timezone
import math

import numpy as np
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.smoother.kalman import KalmanSmoother
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate

from .tracking import CVKalman


def fixed_lag_outputs(times, snapshots, config, lag_s):
    """Return estimates using no observation newer than query_time + lag_s.

    snapshots are None or (x, P, segment_id). Segment IDs change at each birth.
    This batch adapter evaluates a bounded lookahead algorithm; it is not a
    real-time emission API. End-of-sequence outputs use shorter lookahead.
    """
    times = np.asarray(times, dtype=float)
    if (not math.isfinite(lag_s) or lag_s < 0 or len(times) != len(snapshots)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid lag, timestamps or snapshot count')
    smoother = KalmanSmoother(CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(config.process_noise_density) for _ in range(3)]))
    order = [0, 3, 1, 4, 2, 5]
    first = next((i for i, s in enumerate(snapshots) if s is not None), len(times))
    outputs = []
    for i, t in enumerate(times):
        end = int(np.searchsorted(times, t + lag_s, side='right'))
        start = first if i < first else i
        if start >= end or snapshots[start] is None:
            outputs.append({'xyz': None, 'event': 'unavailable', 'lookahead_s': 0.})
            continue
        states = []
        previous = None
        for j in range(start, end):
            s = snapshots[j]
            if s is None or s[2] != snapshots[start][2]:
                break
            x, cov, _ = s
            if previous is None:
                prior_x, prior_cov = x, cov
            else:
                kf = CVKalman(previous[0][:3], config)
                kf.x, kf.P = previous[0].copy(), previous[1].copy()
                kf.predict(times[j] - times[j - 1])
                prior_x, prior_cov = kf.x, kf.P
            stamp = datetime.fromtimestamp(times[j], timezone.utc)
            prediction = GaussianStatePrediction(prior_x[order, None],
                                                  prior_cov[np.ix_(order, order)], timestamp=stamp)
            states.append(GaussianStateUpdate(x[order, None], cov[np.ix_(order, order)],
                                              SingleHypothesis(prediction, None), timestamp=stamp))
            previous = s
        smoothed = smoother.smooth(Track(states))
        x = np.asarray(smoothed[0].state_vector).reshape(-1)[np.argsort(order)]
        xyz = x[:3] + x[3:] * (t - times[start])
        outputs.append({'xyz': xyz.tolist(), 'event': 'startup_backfill' if i < first else 'smoothed',
                        'lookahead_s': float(times[start + len(states) - 1] - t)})
    return outputs
