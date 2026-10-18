import numpy as np
import pytest

from rdq_uav.multimodal_v3.target_smoothing import fixed_lag_outputs
from rdq_uav.multimodal_v3.tracking import CVKalman, TrackerConfig


def snapshots():
    cfg = TrackerConfig()
    kf = CVKalman([0, 0, 0], cfg)
    result = []
    for i in range(6):
        if i:
            kf.predict(.1)
            kf.update([i * .1, 0, 0])
        result.append((kf.x.copy(), kf.P.copy(), 1))
    return cfg, result


def test_zero_lag_equals_causal_filter():
    cfg, s = snapshots()
    result = fixed_lag_outputs(np.arange(6) * .1, s, cfg, 0)
    np.testing.assert_allclose([r['xyz'] for r in result], [x[:3] for x, _, _ in s])


def test_output_cannot_see_beyond_declared_horizon():
    cfg, s = snapshots()
    t = np.arange(6) * .1
    before = fixed_lag_outputs(t, s, cfg, .21)
    s[3] = (np.ones(6) * 100, s[3][1], 1)
    after = fixed_lag_outputs(t, s, cfg, .21)
    np.testing.assert_array_equal(before[0]['xyz'], after[0]['xyz'])
    assert all(r['lookahead_s'] <= .21 for r in before)


def test_startup_backfill_requires_anchor_within_horizon():
    cfg = TrackerConfig()
    x = np.array([1., 0, 0, 2., 0, 0])
    s = [None, None, (x, np.eye(6), 1)]
    result = fixed_lag_outputs([0., .1, .2], s, cfg, .15)
    assert result[0]['xyz'] is None
    assert result[1]['event'] == 'startup_backfill'
    np.testing.assert_allclose(result[1]['xyz'], [.8, 0, 0])


def test_no_smoothing_or_backfill_across_target_rebirth():
    cfg, s = snapshots()
    t = np.arange(6) * .1
    s[2] = None
    s[3:] = [(np.ones(6) * 100, cov, 2) for _, cov, _ in s[3:]]
    result = fixed_lag_outputs(t, s, cfg, .5)
    assert result[2]['xyz'] is None
    assert np.linalg.norm(result[0]['xyz']) < 1


@pytest.mark.parametrize('lag', [-1, float('nan'), float('inf')])
def test_invalid_lag(lag):
    with pytest.raises(ValueError):
        fixed_lag_outputs([], [], TrackerConfig(), lag)
