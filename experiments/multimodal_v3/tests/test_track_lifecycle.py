"""Lifecycle regressions use synthetic candidates, never network inference."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rdq_uav.multimodal_v3.stone_tracker import MultiFilterConfig
from test_multi_tracking import frame, tracker, _make_cache, SCRIPT


@pytest.mark.parametrize('capacity_enabled, expected', [(True, 10), (False, 11)])
def test_late_target_confirmation_with_ten_old_backgrounds(capacity_enabled, expected):
    state = tracker(capacity_pruning_enabled=capacity_enabled)
    bg = np.c_[np.arange(10) * 10., np.zeros((10, 2))]
    for i in range(4):
        state.step(i * .2, frame(bg, [.6] * 10, np.arange(10)))
    assert len(state.tracker.tracks) == 10
    for i in range(3):
        xyz = np.vstack((bg, [200. + i * .1, 0, 0]))
        result = state.step(.8 + i * .2, frame(xyz, [.6] * 10 + [.95], np.arange(11)))
        assert result['birth_candidate_count'] == 10
    assert result['active_track_count'] == expected
    assert any(item['reason'] == 'capacity' for item in result['removed_tracks']) == capacity_enabled
    target_rows = [r for r in result['track_states'] if r['birth_token_id'] == 10]
    assert bool(target_rows) == (not capacity_enabled)
    if not capacity_enabled:
        assert target_rows[0]['status'] == 'confirmed'
        assert target_rows[0]['birth_time'] == pytest.approx(.8)
        assert target_rows[0]['observation_xyz'] == pytest.approx([200.2, 0, 0])
        continuation = state.step(1.4, frame(np.vstack((bg, [200.3, 0, 0])),
                                            [.6] * 10 + [.95], np.arange(11)))
        assert continuation['active_track_count'] == 11
        assert not continuation['new_track_ids']


def test_relaxed_tentative_timeout_allows_intermittent_confirmation():
    state = tracker(capacity_pruning_enabled=False,
                    max_tentative_age_s=.8, max_prediction_age_s=1.2)
    first = state.step(0., frame([[0, 0, 0]], [.9], [7]))
    identity = first['new_track_ids'][0]
    state.step(.2, frame([], []))
    third = state.step(.4, frame([], []))
    assert third['active_track_count'] == 0  # predictions do not confirm
    assert third['tentative_track_count'] == 1
    state.step(.6, frame([[.1, 0, 0]], [.9], [8]))
    state.step(.8, frame([], []))
    result = state.step(1., frame([[.2, 0, 0]], [.9], [9]))
    assert result['track_id'] == identity
    assert result['active_track_count'] == 1
    assert result['track_states'][0]['birth_token_id'] == 7


def test_old_tentative_timeout_expires_during_same_gap():
    state = tracker(capacity_pruning_enabled=False)
    state.step(0., frame([[0, 0, 0]], [.9]))
    state.step(.2, frame([], []))
    result = state.step(.4, frame([], []))
    assert result['tentative_track_count'] == 0
    assert result['removed_tracks'][0]['reason'] == 'expired'


def test_relaxed_tentative_still_expires_without_observation():
    state = tracker(capacity_pruning_enabled=False, max_tentative_age_s=.8)
    state.step(0., frame([[0, 0, 0]], [.9]))
    assert state.step(.6, frame([], []))['tentative_track_count'] == 1
    result = state.step(.9, frame([], []))
    assert result['tentative_track_count'] == 0
    assert result['removed_tracks'][0]['reason'] == 'expired'


def test_confirmed_retention_is_longer_but_finite():
    state = tracker(capacity_pruning_enabled=False,
                    max_tentative_age_s=.8, max_prediction_age_s=1.2)
    for i in range(3):
        state.step(i * .2, frame([[0, 0, 0]], [.9]))
    retained = state.step(1., frame([], []))  # last observation was .4
    assert retained['active_track_count'] == 1
    assert retained['output_kind'] == 'predicted'
    assert retained['observation_xyz'] is None
    expired = state.step(1.7, frame([], []))
    assert expired['active_track_count'] == 0
    assert expired['removed_tracks'][0]['reason'] == 'expired'


def test_capacity_disabled_does_not_disable_duplicate_removal():
    state = tracker(capacity_pruning_enabled=False, confirm_hits=1,
                    birth_nms_radius_m=.1, duplicate_radius_m=.5)
    result = state.step(0., frame([[0, 0, 0], [.3, 0, 0]], [.9, .8]))
    assert result['active_track_count'] == 1
    assert any(item['reason'] == 'duplicate' for item in result['removed_tracks'])


def test_capacity_flag_requires_boolean():
    with pytest.raises(ValueError, match='boolean'):
        MultiFilterConfig(capacity_pruning_enabled='false')


def test_more_than_twelve_tentative_tracks_can_coexist_without_exceeding_birth_budget():
    state = tracker(capacity_pruning_enabled=False)
    xyz = np.c_[np.arange(20) * 10., np.zeros((20, 2))]
    first = state.step(0., frame(xyz[:10], [.6] * 10, np.arange(10)))
    second = state.step(.2, frame(xyz, [.6] * 10 + [.95] * 10, np.arange(20)))
    assert first['birth_candidate_count'] == second['birth_candidate_count'] == 10
    assert second['tentative_track_count'] == 20
    assert second['active_track_count'] == 0
    assert len(second['new_track_ids']) == 10
    assert not second['removed_tracks']


def run_eval(cache, output, config):
    subprocess.run([sys.executable, str(SCRIPT), '--cache', str(cache),
                    '--output', str(output), '--methods', 'multi',
                    '--multi-config', str(config)], check=True, capture_output=True)
    return json.loads((output / 'summary.json').read_text())


def test_new_configs_cli_and_gt_independence(tmp_path):
    cache, altered = tmp_path / 'cache', tmp_path / 'altered'
    _make_cache(cache, schema=2)
    _make_cache(altered, schema=2, mutate_gt=True)
    configs = SCRIPT.parent
    baseline = run_eval(cache, tmp_path / 'legacy', configs / 'multi_tracker_defaults.json')
    for stem in ('multi_tracker_no_capacity', 'multi_tracker_lifecycle'):
        result = run_eval(cache, tmp_path / stem, configs / (stem + '.json'))
        assert result['independent_top1'] == baseline['independent_top1']
        lifecycle = result['multi_candidate_kf_select']['lifecycle']
        assert lifecycle['removal_counts'].get('capacity', 0) == 0
        assert lifecycle['confirmed_pool_success_1m_count'] == 4
        assert (tmp_path / stem / 'sequence_lifecycle.csv').is_file()
        rows = [json.loads(line) for line in (tmp_path / stem / 'per_query.jsonl').read_text().splitlines()]
        last = rows[-1]
        assert last['multi_candidate_kf_select_observation_error_m'] == pytest.approx(0.)
    run_eval(altered, tmp_path / 'mutated', configs / 'multi_tracker_lifecycle.json')
    for artifact in ('tracking_events.jsonl', 'track_scores.jsonl', 'all_track_history.jsonl'):
        assert (tmp_path / 'multi_tracker_lifecycle' / artifact).read_bytes() == (
            tmp_path / 'mutated' / artifact).read_bytes()
