from types import SimpleNamespace

import numpy as np
import torch

from rdq_uav.multimodal_v3 import causal_tracking, visual_birth
from rdq_uav.multimodal_v3.candidate_adapter import CandidateFrame
from rdq_uav.multimodal_v3.target_tracker import TargetConfig
from rdq_uav.multimodal_v3.tracking import Observations
from rdq_uav.multimodal_v3.visual_birth import VisualHint


def frame(score=.99):
    obs = Observations([[0., 0., 10.]], [score], [score], [0])
    return CandidateFrame(obs, obs, obs, {})


def tracker(monkeypatch, pairs, *, bearing=False):
    def project(points, *_):
        return 50. + 10. * points[:, :2], torch.ones(len(points), dtype=torch.bool)

    monkeypatch.setattr(visual_birth, 'project_omni_radtan', project)
    monkeypatch.setattr(causal_tracking, 'project_omni_radtan', project)
    projection = SimpleNamespace(rotation_camera_from_radar=torch.eye(3)[None],
                                 translation_camera_from_radar_m=torch.zeros(1, 3),
                                 image_size_wh=torch.tensor([[100, 100]]))
    return causal_tracking.CausalVisualTracker(
        TargetConfig(), projection,
        visual_lookup=lambda t: VisualHint(t, (49, 49, 57, 51), .99),
        lidar_pair_lookup=pairs.__getitem__, bearing=bearing)


def test_only_new_strong_measurements_refresh_the_timeout(monkeypatch):
    tr = tracker(monkeypatch, {
        1.: (0, -1), 1.125: (0, -1), 1.25: (1, -1), 1.625: (2, -1), 1.75: (2, -1)
    })
    initial = tr.step(1., frame())
    assert initial['event'] == 'birth'
    assert tr.step(1.125, frame())['event'] == 'predict'
    low = tr.step(1.25, frame(.1))
    assert low['event'] == 'predict' and low['measurement_age_s'] == .25
    lost = tr.step(1.625, frame())
    assert lost['event'] == 'lost' and lost['xyz'] is None
    assert initial.keys() == lost.keys()
    birth = tr.step(1.75, frame())
    assert birth['event'] == 'birth' and birth['segment_id'] == 2


def test_visual_correction_preserves_range_and_kf_state(monkeypatch):
    pairs = {1.: (0, -1), 1.0625: (1, -1)}
    base = tracker(monkeypatch, pairs)
    bearing = tracker(monkeypatch, pairs, bearing=True)
    assert base.step(1., frame())['xyz'] == bearing.step(1., frame())['xyz']
    old = base.step(1.0625, frame())
    new = bearing.step(1.0625, frame())
    assert old['event'] == new['event'] == 'high_update'
    assert new['readout']['base_xyz'] == old['xyz'] and new['xyz'][0] > old['xyz'][0]
    np.testing.assert_allclose(np.linalg.norm(new['xyz']), np.linalg.norm(old['xyz']), atol=1e-10)
    np.testing.assert_array_equal(bearing.kf.x, base.kf.x)
    np.testing.assert_array_equal(bearing.kf.P, base.kf.P)


def test_scan_pairs_use_the_exact_sensor_timestamp(tmp_path):
    scans = tmp_path / 'seq0001' / 'livox_avia'
    scans.mkdir(parents=True)
    (scans / '1700.0000002.npy').touch()
    (scans / '1700.0000004.npy').touch()
    pairs = causal_tracking.build_lidar_pairs([
        {'sequence_id': 'seq0001', 'query_time': 1700., 'sample_id': 'seq0001_sensor_1700.000000300'}
    ], tmp_path)
    assert pairs[1700.] == (0, -1)
