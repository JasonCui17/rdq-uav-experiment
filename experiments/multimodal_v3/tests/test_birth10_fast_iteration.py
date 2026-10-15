"""Ten spatial birth candidates must survive confirmation; no training/data."""
from pathlib import Path
import json
import numpy as np

from rdq_uav.multimodal_v3.candidate_adapter import CandidateFrame
from rdq_uav.multimodal_v3.stone_tracker import MultiCandidateTracker, MultiFilterConfig
from rdq_uav.multimodal_v3.track_selection import SelectionConfig
from rdq_uav.multimodal_v3.tracking import Observations


def test_sequence_backend_is_constructed_only_once_per_sequence():
    from unittest.mock import patch
    from rdq_uav.multimodal_v3 import stone_tracker
    observations = Observations([[0., 0., 0.]], [.8], [.8], [0])
    frame = CandidateFrame(observations, observations, observations, {})
    manager = stone_tracker.SequenceMultiTrackers(
        MultiFilterConfig(confirm_hits=1), SelectionConfig(), select_evidence=False)
    with patch.object(stone_tracker, 'MultiCandidateTracker', wraps=MultiCandidateTracker) as factory:
        manager.step('a', 0., frame)
        manager.step('a', .1, frame)
        manager.step('b', 0., frame)
        assert factory.call_count == 2


def test_default_birth_and_active_capacity_support_ten_tracks():
    config = MultiFilterConfig()
    saved = json.loads((Path(__file__).resolve().parents[1] / 'multi_tracker_defaults.json').read_text())['filter']
    assert config.birth_max_per_step == saved['birth_max_per_step'] == 10
    assert config.max_active_tracks == saved['max_active_tracks'] == 10
    assert config.max_tentative_tracks >= 10
    state = MultiCandidateTracker(config, SelectionConfig(), select_evidence=False)
    # Ten distinct spatial hypotheses. The tenth is persistently lowest ranked.
    xyz = np.c_[np.arange(10) * 10., np.zeros((10, 2))]
    score = np.linspace(.99, .6, 10)
    observations = Observations(xyz, score, score, np.arange(10))
    frame = CandidateFrame(observations, observations, observations.take([0]), {})
    for step in range(config.confirm_hits):
        result = state.step(step * .1, frame)
        assert result['birth_candidate_count'] == 10
    assert result['active_track_count'] == 10
    assert {row['selected_token_id'] for row in result['track_states']
            if row['status'] == 'confirmed'} == set(range(10))


def test_fast_mode_is_declared():
    script = Path(__file__).resolve().parents[1] / 'evaluate_tracking.py'
    # CLI integration and output parity should also be checked with synthetic
    # caches through the existing evaluation test helpers.
    from test_multi_tracking import _make_cache
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache = root / 'cache'
        _make_cache(cache, schema=2)
        for mode in ('all', 'multi'):
            subprocess.run([sys.executable, str(script), '--cache', str(cache),
                            '--output', str(root / mode), '--methods', mode],
                           check=True, capture_output=True)
        full = json.loads((root / 'all/summary.json').read_text())
        fast = json.loads((root / 'multi/summary.json').read_text())
        assert 'top1_kf' not in fast and 'candidate_kf' not in fast
        for method in ('independent_top1', 'multi_candidate_kf', 'multi_candidate_kf_select'):
            assert full[method] == fast[method]
