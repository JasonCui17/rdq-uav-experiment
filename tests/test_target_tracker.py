"""Behavioral checks for bounded radar target association."""
import numpy as np
import pytest

from rdq_uav.multimodal_v3.candidate_adapter import CandidateFrame
from rdq_uav.multimodal_v3.target_tracker import TargetConfig, TargetTracker
from rdq_uav.multimodal_v3.tracking import Observations, TrackerConfig


def frame(xyz=(), scores=(), ranks=None):
    obs = Observations(np.array(xyz).reshape(-1, 3), scores,
                       scores if ranks is None else ranks, np.arange(len(scores)))
    return CandidateFrame(obs, obs, obs, {})


def test_empty_start_is_missing_not_fabricated():
    assert TargetTracker().step(1., frame())['xyz'] is None


def test_low_confidence_cannot_start_background_track():
    assert TargetTracker().step(1., frame([[20, 0, 0]], [.2]))['xyz'] is None


def test_far_high_score_cannot_steal_existing_target():
    t = TargetTracker()
    t.step(1., frame([[0, 0, 0]], [.99]))
    result = t.step(1.05, frame([[30, 0, 0], [.1, 0, 0]], [1., .05]))
    assert result['event'] == 'low_update'
    assert result['selected_token_id'] == 1
    assert np.linalg.norm(result['xyz']) < .2


def test_saturated_scores_use_detector_rank_at_birth_and_update():
    t = TargetTracker(TargetConfig(high_rank_first=True))
    assert t.step(1., frame([[1, 0, 0], [0, 0, 0]], [1, 1], [10, 20]))['selected_token_id'] == 1
    assert t.step(1.05, frame([[0, 0, 0], [.1, 0, 0]], [1, 1], [10, 20]))['selected_token_id'] == 1


def test_speed_rejected_strong_falls_back_to_weak():
    t = TargetTracker(TargetConfig(filter=TrackerConfig(max_speed_mps=.01)))
    t.step(1., frame([[0, 0, 0]], [.99]))
    result = t.step(1.05, frame([[1, 0, 0], [0, 0, 0]], [.99, .1]))
    assert result['event'] == 'low_update'
    assert result['selected_token_id'] == 1


def test_prediction_then_timeout():
    t = TargetTracker()
    t.step(1., frame([[0, 0, 0]], [.99]))
    assert t.step(1.1, frame())['event'] == 'predict'
    assert t.step(1.6, frame())['xyz'] is None


def test_time_gap_requires_new_trustworthy_birth():
    t = TargetTracker()
    t.step(1., frame([[0, 0, 0]], [.99]))
    assert t.step(5., frame([[0, 0, 0]], [.2]))['xyz'] is None


@pytest.mark.parametrize('timestamp', [1., 0., float('nan'), float('inf')])
def test_invalid_or_nonchronological_times(timestamp):
    t = TargetTracker()
    t.step(1., frame())
    with pytest.raises(ValueError):
        t.step(timestamp, frame())
