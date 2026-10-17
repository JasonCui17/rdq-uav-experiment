import numpy as np
import pytest
import torch

from rdq_uav.multimodal_v3 import visual_birth
from rdq_uav.multimodal_v3.visual_birth import VisualBirthTracker, VisualHint
from rdq_uav.multimodal_v3.target_tracker import TargetConfig
from rdq_uav.multimodal_v3.tracking import Observations
from rdq_uav.multimodal_v3.candidate_adapter import CandidateFrame
from rdq_uav.multimodal_v3.pipeline.evaluate_visual_birth import build_lookup


def frame(include_target=True):
    xyz = np.array([[0., 0., 1.], [0., 0., 25.]])[:2 if include_target else 1]
    obs = Observations(xyz, np.array([.99, .95])[:len(xyz)], np.array([10., 9.])[:len(xyz)],
                       np.arange(len(xyz)))
    return CandidateFrame(obs, obs, obs, {})


def projection(points, *args):
    uv = torch.tensor([[200., 200.] if p[2] < 10 else [15., 15.] for p in points]).reshape(-1, 2)
    return uv, torch.ones(len(points), dtype=torch.bool)


def tracker(monkeypatch, lookup):
    monkeypatch.setattr(visual_birth, 'project_omni_radtan', projection)
    return VisualBirthTracker(TargetConfig(high_rank_first=True), None, visual_lookup=lookup)


def test_highest_radar_background_cannot_birth_without_image_support(monkeypatch):
    tr = tracker(monkeypatch, lambda t: VisualHint(t, (10, 10, 20, 20), .97))
    assert tr.step(1., frame(False))['xyz'] is None
    out = tr.step(1.05, frame())
    assert out['event'] == 'birth' and out['selected_token_id'] == 1
    assert out['birth_visual_verified']


def test_unknown_camera_blocks_birth_but_preserves_radar_continuation(monkeypatch):
    hint = [None]
    tr = tracker(monkeypatch, lambda t: hint[0])
    assert tr.step(1., frame())['xyz'] is None
    hint[0] = VisualHint(1.05, (10, 10, 20, 20), .97)
    assert tr.step(1.05, frame())['event'] == 'birth'
    hint[0] = None
    out = tr.step(1.1, frame())
    assert out['event'] == 'high_update' and out['selected_token_id'] == 1
    assert not out['birth_requires_vision']


def test_timeout_cannot_rebirth_without_visual_verification(monkeypatch):
    hint = [VisualHint(1., (10, 10, 20, 20), .97)]
    tr = tracker(monkeypatch, lambda t: hint[0])
    assert tr.step(1., frame())['event'] == 'birth'
    hint[0] = None
    assert tr.step(2., frame(False))['xyz'] is None


def test_future_or_low_confidence_camera_cannot_authorize_birth(monkeypatch):
    tr = tracker(monkeypatch, lambda t: VisualHint(t, (10, 10, 20, 20), .5))
    assert tr.step(1., frame())['xyz'] is None
    tr = tracker(monkeypatch, lambda t: VisualHint(t+1, (10, 10, 20, 20), .97))
    with pytest.raises(ValueError, match='Future'):
        tr.step(1., frame())


def inputs():
    frames = [{'sample_id': 'a', 'sequence_id': 's', 'query_time': 10.}]
    samples = [{**frames[0], 'has_image': True, 'image_path': '/any/s/Image/9.95.png',
                'image_time': 9.95, 'gt_box_xyxy_px': [100, 100, 101, 101]}]
    raw = [{'image_path': 's/Image/9.95.png', 'views': {'full':
        {'boxes': [[0, 0, 1, 1], [10, 10, 20, 20]], 'scores': [.2, .97]}}}]
    return frames, samples, raw


def test_lookup_ignores_annotations_and_chooses_model_top1():
    frames, samples, raw = inputs()
    before, missing = build_lookup(frames, samples, raw)
    samples[0]['gt_box_xyxy_px'] = [0, 0, 1, 1]
    samples[0]['has_gt2d'] = False
    samples[0]['gt_xyz'] = [1e8, 1e8, 1e8]
    after, _ = build_lookup(frames, samples, raw)
    assert before == after and before[10.].box == (10, 10, 20, 20) and missing == 0


def test_lookup_records_missing_cached_images_and_rejects_mixed_identity():
    frames, samples, raw = inputs()
    hints, missing = build_lookup(frames, samples, [])
    assert hints[10.] is None and missing == 1
    samples[0]['sequence_id'] = 'other'
    with pytest.raises(ValueError):
        build_lookup(frames, samples, raw)
