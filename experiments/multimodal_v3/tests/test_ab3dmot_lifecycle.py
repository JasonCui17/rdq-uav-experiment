"""AB3DMOT lifecycle rule contract, implemented with real Stone Soup filters."""
import json

import numpy as np
import pytest

from rdq_uav.multimodal_v3.stone_tracker import MultiFilterConfig
from test_multi_tracking import frame, tracker, _make_cache, SCRIPT
from test_track_lifecycle import run_eval


def ab_tracker(**kwargs):
    return tracker(lifecycle_mode='ab3dmot', capacity_pruning_enabled=False, **kwargs)


def latest_track(result):
    assert len(result['track_states']) == 1
    return result['track_states'][0]


def test_birth_counts_as_first_hit_and_min_hits_gates_output():
    state = ab_tracker(min_hits=3, confirm_hits=1, max_age=4)
    for i in range(3):
        result = state.step(i * .2, frame([[.1 * i, 0, 0]], [.9], [7]))
        assert latest_track(result)['hits'] == i + 1
        assert latest_track(result)['time_since_update'] == 0
        assert (result['xyz'] is not None) == (i == 2)
    assert result['hits'] == 3
    assert result['active_track_count'] == 1
    assert result['tentative_track_count'] == 0


@pytest.mark.parametrize('min_hits', [1, 3])
def test_delete_at_exact_max_age_for_both_mature_and_immature(min_hits):
    state = ab_tracker(min_hits=min_hits, max_age=3)
    state.step(0., frame([[0, 0, 0]], [.9]))
    for i in (1, 2):
        result = state.step(.2 * i, frame([], []))
        assert latest_track(result)['hits'] == 1
        assert latest_track(result)['time_since_update'] == i
    expired = state.step(.6, frame([], []))
    assert not expired['track_states']
    assert expired['removed_tracks'] == [{'track_id': 1, 'reason': 'max_age'}]
    assert state.history[1][-1]['time_since_update'] == 3


def test_match_resets_miss_counter_and_hits_are_cumulative():
    state = ab_tracker(min_hits=3, max_age=3)
    for t, present, expected_hits, expected_age in (
        (0., True, 1, 0), (.2, False, 1, 1), (.4, True, 2, 0),
        (.6, False, 2, 1), (.8, True, 3, 0),
    ):
        result = state.step(t, frame([[0, 0, 0]], [.9]) if present else frame([], []))
        row = latest_track(result)
        assert (row['hits'], row['time_since_update']) == (expected_hits, expected_age)
    assert result['track_id'] == 1


def test_age_counts_queries_not_elapsed_seconds():
    state = ab_tracker(min_hits=1, max_age=3)
    state.step(0., frame([[0, 0, 0]], [.9]))
    result = state.step(1., frame([], []))
    assert result['active_track_count'] == 1
    assert latest_track(result)['time_since_update'] == 1
    result = state.step(1.05, frame([], []))
    assert result['active_track_count'] == 1
    assert latest_track(result)['time_since_update'] == 2
    assert not state.step(1.1, frame([], []))['track_states']


def test_association_happens_before_max_age_deletion():
    state = ab_tracker(min_hits=1, max_age=2)
    state.step(0., frame([[0, 0, 0]], [.9]))
    state.step(.2, frame([], []))
    result = state.step(.4, frame([[0, 0, 0]], [.9]))
    assert result['track_id'] == 1
    assert result['hits'] == 2
    assert result['time_since_update'] == 0
    assert not result['removed_tracks']


def test_late_target_can_join_ten_old_mature_tracks():
    state = ab_tracker(min_hits=3, max_age=6)
    bg = np.c_[np.arange(10) * 10., np.zeros((10, 2))]
    for i in range(4):
        state.step(i * .2, frame(bg, [.6] * 10, np.arange(10)))
    for i in range(3):
        result = state.step(.8 + i * .2, frame(np.vstack((bg, [200, 0, 0])),
                                              [.6] * 10 + [.95], np.arange(11)))
    assert result['active_track_count'] == 11
    assert len(state.tracker.tracks) == 11
    assert result['birth_candidate_count'] == 10
    assert not any(item['reason'] == 'capacity' for item in result['removed_tracks'])


def test_duplicate_removal_updates_physical_bank():
    state = ab_tracker(min_hits=1, max_age=4, birth_nms_radius_m=.1)
    result = state.step(0., frame([[0, 0, 0], [.3, 0, 0]], [.9, .8]))
    assert result['active_track_count'] == len(state.tracker.tracks) == 1
    assert any(r['reason'] == 'duplicate' for r in result['removed_tracks'])
    result = state.step(.2, frame([[0, 0, 0]], [.9]))
    assert len(state.tracker.tracks) == 1
    assert result['hits'] == 2


@pytest.mark.parametrize('settings', [dict(min_hits=0), dict(max_age=0),
                                    dict(max_age=1.2), dict(min_hits=True),
                                    dict(lifecycle_mode='unknown'),
                                    dict(capacity_pruning_enabled=True)])
def test_invalid_configuration_rejected(settings):
    config = dict(lifecycle_mode='ab3dmot', capacity_pruning_enabled=False)
    config.update(settings)
    with pytest.raises(ValueError):
        MultiFilterConfig(**config)


def test_counter_contract_against_reference_rules():
    # AB3DMOT Filter starts at hits=1, time_since_update=0.
    # prediction increments age; successful matching resets age and adds a hit;
    # output deletion runs AFTER matching. No third-party KF is reimplemented.
    state = ab_tracker(min_hits=3, max_age=3)
    hits, age = 0, 0
    for i, matched in enumerate((True, False, True, True, False, False, False)):
        if i == 0:
            hits, age = 1, 0
        else:
            age += 1
            if matched:
                hits += 1
                age = 0
        result = state.step(i * .1, frame([[0, 0, 0]], [.9]) if matched else frame([], []))
        if age >= 3:
            assert not result['track_states']
        else:
            row = latest_track(result)
            assert (row['hits'], row['time_since_update']) == (hits, age)
            assert (row['status'] == 'confirmed') == (hits >= 3)


def test_cli_config_and_gt_independence(tmp_path):
    normal, altered = tmp_path / 'normal', tmp_path / 'altered'
    _make_cache(normal, schema=2)
    _make_cache(altered, schema=2, mutate_gt=True)
    config = SCRIPT.parent / 'multi_tracker_ab3dmot.json'
    first = run_eval(normal, tmp_path / 'first', config)
    run_eval(altered, tmp_path / 'second', config)
    protocol = first['protocol']['multi_track_lifecycle']
    assert protocol['mode'] == 'ab3dmot'
    assert protocol['min_hits'] == 3
    assert protocol['max_age_queries'] == 24
    for artifact in ('tracking_events.jsonl', 'track_scores.jsonl', 'all_track_history.jsonl'):
        assert (tmp_path / 'first' / artifact).read_bytes() == (
            tmp_path / 'second' / artifact).read_bytes()
