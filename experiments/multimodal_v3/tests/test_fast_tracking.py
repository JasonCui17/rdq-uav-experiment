import json
from unittest.mock import patch
import subprocess
import sys

import numpy as np
import pytest

from rdq_uav.multimodal_v3.stone_tracker import (
    MultiFilterConfig, SequenceMultiTrackers, GatedQualityHypothesiser,
)
from rdq_uav.multimodal_v3.track_selection import SelectionConfig
from test_multi_tracking import frame, _make_cache, SCRIPT, evidence


def compare(left, right):
    # Scores are mappings keyed by deterministic local track ID.
    assert left == right


@pytest.mark.parametrize('mode', ['stone_soup', 'ab3dmot'])
def test_shared_bank_matches_independent_views(mode):
    cfg = MultiFilterConfig(lifecycle_mode=mode, capacity_pruning_enabled=False, use_kdtree=True)
    selection = SelectionConfig()
    bank = SequenceMultiTrackers(cfg, selection, select_evidence=False)
    view = SequenceMultiTrackers(cfg, selection, select_evidence=True, shared_bank=bank)
    independent = SequenceMultiTrackers(cfg, selection, select_evidence=True)
    for i, t in enumerate((0., .1, .2, .3, .9, 1., 1.1, 3.3, 3.4, 3.5)):
        current = (frame([], []) if i == 4 else frame(
            [[20, 0, 0], [i * .1, 0, 0]], [.95, .7], [1, 2], evidence={
                1: evidence(120, speed=0.), 2: evidence(2, dynamic=True, speed=1.)}))
        bank.step('a', t, current)
        compare(view.step('a', t, current), independent.step('a', t, current))
        assert view.states['a'].history is bank.states['a'].history
        assert view.states['a'].tracker is bank.states['a'].tracker


def test_shared_view_never_steps_filter_twice():
    cfg = MultiFilterConfig(use_kdtree=True)
    bank = SequenceMultiTrackers(cfg, SelectionConfig(), select_evidence=False)
    view = SequenceMultiTrackers(cfg, SelectionConfig(), select_evidence=True, shared_bank=bank)
    current = frame([[0, 0, 0]], [.8])
    bank.step('a', 0., current)
    backend = bank.states['a'].tracker
    with patch.object(type(backend), 'update_tracker', wraps=backend.update_tracker) as update:
        view.step('a', 0., current)
        assert update.call_count == 0
    with pytest.raises(ValueError, match='before'):
        view.step('a', .1, current)


@pytest.mark.parametrize('mode', ['stone_soup', 'ab3dmot'])
def test_kdtree_preserves_full_scan_outputs_and_reduces_pairs(mode):
    xy = np.c_[np.arange(16) * 10., np.zeros((16, 2))]
    # Dense candidates include many points too far to ever match a given track.
    cfgs = [MultiFilterConfig(lifecycle_mode=mode, capacity_pruning_enabled=False,
                              use_kdtree=fast, birth_max_per_step=20)
            for fast in (False, True)]
    managers = [SequenceMultiTrackers(cfg, SelectionConfig(), select_evidence=True) for cfg in cfgs]
    totals = []
    original = GatedQualityHypothesiser.hypothesise
    outputs = []
    for manager in managers:
        counts = []
        def counted(self, track, detections, timestamp, **kwargs):
            counts.append(len(detections))
            return original(self, track, detections, timestamp, **kwargs)
        sequence = []
        with patch.object(GatedQualityHypothesiser, 'hypothesise', counted):
            for i in range(5):
                sequence.append(manager.step('a', i * .1,
                    frame(xy + [i * .01, 0, 0], np.linspace(.9, .6, 16), np.arange(16))))
        totals.append(sum(counts))
        outputs.append(sequence)
    compare(outputs[0], outputs[1])
    assert totals[1] < totals[0] / 5


def test_exact_four_meter_boundary_still_reaches_original_gate():
    # A permissive Mahalanobis gate exposes the independent Euclidean boundary.
    configs = [MultiFilterConfig(confirm_hits=1, use_kdtree=fast,
                                mahalanobis_gate_sq=1e6, max_speed_mps=1e6,
                                initial_velocity_std_mps=100, missed_cost=1e6)
               for fast in (False, True)]
    outputs = []
    for config in configs:
        manager = SequenceMultiTrackers(config, SelectionConfig(), select_evidence=False)
        manager.step('a', 0., frame([[0, 0, 0]], [.9]))
        outputs.append(manager.step('a', .1, frame([[4., 0, 0]], [.9])))
    compare(outputs[0], outputs[1])


def test_cli_shared_and_unshared_results_identical(tmp_path):
    cache = tmp_path / 'cache'
    _make_cache(cache, schema=2)
    for name, config, flags in (
        ('old', 'multi_tracker_ab3dmot.json', []),
        ('fast', 'multi_tracker_ab3dmot_fast.json', ['--shared-multi-bank']),
    ):
        subprocess.run([sys.executable, str(SCRIPT), '--cache', str(cache),
                        '--output', str(tmp_path / name), '--methods', 'multi',
                        '--multi-config', str(SCRIPT.parent / config), *flags],
                       check=True, capture_output=True)
    old = json.loads((tmp_path / 'old/summary.json').read_text())
    fast = json.loads((tmp_path / 'fast/summary.json').read_text())
    for method in ('independent_top1', 'multi_candidate_kf', 'multi_candidate_kf_select'):
        compare(old[method], fast[method])
    for filename in ('per_query.jsonl', 'tracking_events.jsonl', 'all_track_history.jsonl', 'track_scores.jsonl'):
        assert (tmp_path / 'old' / filename).read_bytes() == (tmp_path / 'fast' / filename).read_bytes()


def test_pilot_limit_is_explicit_and_does_not_modify_cache(tmp_path):
    cache = tmp_path / 'cache'
    _make_cache(cache, schema=2)
    before = (cache / 'manifest.json').read_bytes()
    out = tmp_path / 'pilot'
    subprocess.run([sys.executable, str(SCRIPT), '--cache', str(cache),
                    '--output', str(out), '--methods', 'multi',
                    '--multi-config', str(SCRIPT.parent / 'multi_tracker_ab3dmot_fast.json'),
                    '--shared-multi-bank', '--max-queries', '3', '--progress-every', '1'],
                   check=True, capture_output=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['protocol']['partial_cache_evaluation'] is True
    assert summary['protocol']['evaluated_queries'] == 3
    assert summary['protocol']['full_cache_queries'] == 6
    assert summary['independent_top1']['queries'] == 3
    assert (cache / 'manifest.json').read_bytes() == before


def test_shared_selector_resets_after_empty_bank_time_gap():
    cfg = MultiFilterConfig(confirm_hits=1)
    bank = SequenceMultiTrackers(cfg, SelectionConfig(), select_evidence=False)
    view = SequenceMultiTrackers(cfg, SelectionConfig(), select_evidence=True, shared_bank=bank)
    for time, current in ((0., frame([[0, 0, 0]], [.8])),
                          (.6, frame([], [])), (3., frame([], []))):
        bank.step('a', time, current)
        view.step('a', time, current)
    assert view.states['a'].selector.query_times == [3.]
    assert bank.states['a'].selector.query_times == [3.]


def test_kdtree_mode_rejects_non_boolean():
    with pytest.raises(ValueError, match='boolean'):
        MultiFilterConfig(use_kdtree='true')
