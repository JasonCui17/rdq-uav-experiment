"""Synthetic release-layout tests for query-time Sample/Batch/model contracts."""
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from rdq_uav.multimodal_v2.data import (
    MultimodalV2Dataset, build_datasets, collate_multimodal_v2, prepare_model_batch, validate_sample,
)
from rdq_uav.multimodal_v2.geometry import ProjectionContext, project_omni_radtan
from rdq_uav.multimodal_v2.contracts import CandidateBatch, validate_batch
from rdq_uav.multimodal_v2.interaction import CandidateCrossAttention
from rdq_uav.multimodal_v2.model import MultimodalV2
from rdq_uav.multimodal_v2.scoring import CandidateScoring


def release(root, seq, *, queries=(10.0,), events=(), images=(), n=2):
    for directory in ('ground_truth', 'livox_avia', 'lidar_360', 'Image', '2d_detect'):
        (root/seq/directory).mkdir(parents=True, exist_ok=True)
    for tq in queries:
        np.save(root/seq/'ground_truth'/f'{tq}.npy', np.array([1., 2., 3.]))
    for timestamp, sensor in events:
        points = np.tile(np.array([[1., 0., 2.]], dtype=np.float32), (n, 1))
        np.save(root/seq/('livox_avia' if sensor == 0 else 'lidar_360')/f'{timestamp}.npy', points)
    for timestamp in images:
        Image.new('RGB', (16, 8), 'white').save(root/seq/'Image'/f'{timestamp}.png')


def dataset(root, seqs=('seq0001',), **kwargs):
    return MultimodalV2Dataset(root, seqs, camera_wh=(16, 8), short_edge=8, max_size=16, **kwargs)


def projection():
    return ProjectionContext(torch.eye(3)[None], torch.zeros(1, 3),
                             torch.tensor([[0., 5., 5., 8., 4.]]), torch.zeros(1, 4),
                             torch.tensor([[16., 8.]]), torch.ones(1, 2))


class FakeDINO:
    def preprocess_image(self, inputs):
        # Add genuine padding to exercise pixel masks and retain 0..255 input.
        images = torch.stack([x['image'] for x in inputs])
        return SimpleNamespace(tensor=torch.nn.functional.pad(images, (0, 2, 0, 2)),
                               image_sizes=[(8, 16)]*len(inputs))


def test_radar_inclusive_window_and_both_sensors(tmp_path):
    release(tmp_path, 'seq0001', events=((8.9, 0), (9., 0), (9.5, 1), (10., 0), (10.1, 1)))
    s = dataset(tmp_path)[0]
    assert s['event_timestamps'] == [9., 9.5, 10.]
    assert s['points'].shape == (6, 3)
    assert s['sensor_id'].tolist() == [0, 0, 1, 1, 0, 0]
    assert s['delta_t'].min() == -1 and s['delta_t'].max() == 0


def test_sequence_isolation(tmp_path):
    release(tmp_path, 'seq0001', events=((9.5, 0),))
    release(tmp_path, 'seq0002', queries=(20.,), events=((9.5, 1),))
    d = dataset(tmp_path, ('seq0001', 'seq0002'))
    assert d[0]['event_sequence_ids'] == ['seq0001']
    assert d[0]['sensor_id'].tolist() == [0, 0]


def test_nearest_historical_image_rejects_closer_future_and_relative_time(tmp_path):
    release(tmp_path, 'seq0001', events=((9.6, 0),), images=(9.8, 10.1, 10.9))
    s = dataset(tmp_path)[0]
    assert s['image_time'] == 9.8 and s['m_V']
    assert s['vision_delta_t'] == pytest.approx(-.2)
    assert s['delta_t'].tolist() == pytest.approx([-.4, -.4])


def test_image_gap_masks_placeholder(tmp_path):
    release(tmp_path, 'seq0001', images=(8.8,))
    s = dataset(tmp_path)[0]
    assert not s['m_V'] and not s['gt_2d_valid']
    assert s['left_image_path'] is None and s['image_time'] is None
    assert not s['image_uint8'].any()


def test_batch_variable_counts(tmp_path):
    release(tmp_path, 'seq0001', queries=(10., 20.), events=((9.5, 0),), images=(10., 20.), n=100)
    np.save(tmp_path/'seq0001'/'lidar_360'/'19.5.npy', np.ones((200, 3), np.float32))
    d = dataset(tmp_path)
    b = collate_multimodal_v2([d[0], d[1]])
    assert b['points'].shape == (300, 3)
    for key in ('delta_t', 'sensor_id', 'point_batch_index'):
        assert b[key].shape == (300,)
    assert b['point_counts'].tolist() == [100, 200]
    assert b['point_batch_index'].tolist() == [0]*100+[1]*200
    assert b['image_uint8'].shape == (2, 3, 8, 16)


def test_filter_double_missing_at_init_including_invalid_cloud(tmp_path):
    release(tmp_path, 'seq0001', queries=(10., 20., 30.), events=((9.5, 0),), images=(20.,))
    np.save(tmp_path/'seq0001'/'livox_avia'/'9.5.npy', np.array([[0., 0., 0.], [np.nan, 1., 2.]]))
    train = dataset(tmp_path, filter_empty=True)
    assert train.filtered_empty_queries == 2 and len(train) == 1
    assert train[0]['m_V'] and not train[0]['m_R']
    assert len(dataset(tmp_path, filter_empty=False)) == 3


def test_no_event_or_point_cap_and_custom_window(tmp_path):
    events = [(9.+i*.02, i % 2) for i in range(40)]
    release(tmp_path, 'seq0001', events=events, n=3)
    s = dataset(tmp_path)[0]
    assert s['event_count'] == 40 and len(s['points']) == 120
    s = dataset(tmp_path, radar_history_s=.3)[0]
    assert all(9.7 <= t <= 10. for t in s['event_timestamps'])


def test_gt_matches_selected_filename_only(tmp_path):
    release(tmp_path, 'seq0001', images=(9.8, 10.1))
    labels = tmp_path/'seq0001'/'2d_detect'
    (labels/'10.1.txt').write_text('0 0.5 0.5 0.25 0.5\n')
    assert not dataset(tmp_path)[0]['gt_2d_valid']
    (labels/'9.8.txt').write_text('0 0.25 0.5 0.25 0.5\n')
    s = dataset(tmp_path)[0]
    assert s['gt_2d_valid'] and s['gt_box_xyxy_px'].tolist() == [2., 2., 6., 6.]
    assert s['target_xyz'].tolist() == [1., 2., 3.]
    assert 'calibration_handle' not in s and 'projection' not in s


def test_prepare_contract_padding_and_no_identity_features(tmp_path):
    release(tmp_path, 'seq0001', queries=(10., 20.), events=((9.5, 0),), images=(10.,))
    d = dataset(tmp_path)
    b = collate_multimodal_v2([d[0], d[1]])
    moved, images, mask, proj, target, transforms = prepare_model_batch(b, FakeDINO(), projection(), torch.device('cpu'))
    assert validate_batch(moved) == 2
    assert not {'query_time', 'sample_id', 'sequence_id', 'image_time', 'left_image_path'} & moved.keys()
    assert images.shape == (2, 3, 10, 18) and mask.shape == (2, 10, 18)
    assert not mask[:, :8, :16].any() and mask[:, 8:, :].all() and mask[:, :, 16:].all()
    assert moved['m_V'].tolist() == [True, False]
    assert target.has_xyz.tolist() == [True, True] and not target.has_box.any()
    assert len(transforms) == 2 and proj.batch_size == 2


def test_duplicate_query_time_rejected(tmp_path):
    release(tmp_path, 'seq0001')
    release(tmp_path, 'seq0002')
    with pytest.raises(ValueError, match='duplicate query_time'):
        dataset(tmp_path, ('seq0001', 'seq0002'))


def test_integrity_rejects_sequence_future_and_validity_mismatch(tmp_path):
    release(tmp_path, 'seq0001', events=((9.5, 0),))
    s = dataset(tmp_path)[0]
    for changed in ({'event_sequence_ids': ['seq0002']}, {'event_timestamps': [10.1]}, {'m_R': False}):
        with pytest.raises(ValueError):
            validate_sample({**s, **changed})


def test_indices_cached_without_glob_per_getitem(tmp_path, monkeypatch):
    release(tmp_path, 'seq0001', events=((9.5, 0),), images=(10.,))
    d = dataset(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('filesystem index rescanned during getitem')
    monkeypatch.setattr(type(tmp_path), 'glob', forbidden)
    assert d[0]['m_R'] and d[0]['m_V']


def test_projection_identical_to_previous_math():
    from rdq_uav.multimodal_v1.interaction.geometry_local import project_omni_radtan as old
    p = projection()
    points = torch.tensor([[0., 0., 2.], [1., 0., 2.], [0., 0., -1.]])
    ids = torch.zeros(3, dtype=torch.long)
    new_xy, new_valid = project_omni_radtan(points, ids, p)
    old_xy, old_valid = old(points, ids, p)
    assert torch.equal(new_xy, old_xy) and torch.equal(new_valid, old_valid)


def test_build_datasets_new_pair_and_ignores_max_events(tmp_path):
    release(tmp_path, 'seq0001', events=((9.5, 0),), images=(10.,))
    release(tmp_path, 'seq0002', queries=(20.,), images=(20.,))
    (tmp_path/'split.json').write_text(json.dumps({'train': ['seq0001'], 'val': ['seq0002']}))
    (tmp_path/'geometry.json').write_text('{"time_offset_s": 0.0}')
    (tmp_path/'camera.yaml').write_text('cameras:\n  left:\n    resolution: [16, 8]\n')
    cfg = {'data': dict(root='.', split_file='split.json', train_split='train', val_split='val',
                        geometry_calibration='geometry.json', camera_config='camera.yaml',
                        dino_short_edge=8, dino_max_size=16,
                        max_events=0)}
    train, val = build_datasets(cfg, tmp_path)
    assert train[0]['m_R'] and len(val) == 1
    (tmp_path/'geometry.json').write_text('{"time_offset_s": 0.2}')
    with pytest.raises(ValueError, match='time_offset_s=0'):
        build_datasets(cfg, tmp_path)


class FakeCandidateProducer(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
    def forward(self, *args):
        source = self.source
        cand = CandidateBatch(torch.zeros(1, 128), torch.tensor([.8]),
                              torch.tensor([[0., 0., 2.]]), torch.tensor([source == 'R']),
                              torch.tensor([[6., 2., 10., 6.]]), torch.tensor([source == 'V']),
                              torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), source,
                              torch.zeros(1, 2), torch.tensor([False]))
        raw = {} if source == 'R' else {'pyramid': SimpleNamespace(features=(torch.ones(1, 96, 4, 8), torch.ones(1, 192, 2, 4)))}
        return raw, cand


def test_current_model_chain_gates_placeholder_candidates(tmp_path):
    release(tmp_path, 'seq0001', events=((9.5, 0),))
    b = collate_multimodal_v2([dataset(tmp_path)[0]])
    moved, images, masks, proj, _, _ = prepare_model_batch(b, FakeDINO(), projection(), torch.device('cpu'))
    model = MultimodalV2(FakeCandidateProducer('R'), FakeCandidateProducer('V'), CandidateCrossAttention(), CandidateScoring())
    out = model(moved, images, masks, proj)
    assert out.radar_candidates.n == 1 and out.vision_candidates.n == 0
    assert not out.evidence_valid_3d.any() and torch.equal(out.score_3d_before, out.score_3d_after)


def test_lidar_detector_and_frozen_loss_accept_new_batch(tmp_path):
    from pathlib import Path
    import yaml
    from rdq_uav.lidar_v2.model import LiDARUAVDetector
    from rdq_uav.lidar_v2.loss import CandidateLoss
    release(tmp_path, 'seq0001', queries=(10., 20.), events=((9.5, 0),), n=5)
    d = dataset(tmp_path)
    b = collate_multimodal_v2([d[0], d[1]])
    moved, _, _, _, _, _ = prepare_model_batch(b, FakeDINO(), projection(), torch.device('cpu'))
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root/'configs/lidar_uav_v2.yaml').read_text())
    detector = LiDARUAVDetector(cfg).eval()
    with torch.no_grad():
        out = detector(moved)
        loss = CandidateLoss(cfg)(out, moved)
    assert out['pred_xyz'].shape[-1] == 3
    assert torch.isfinite(out['pred_xyz']).all() and torch.isfinite(loss['loss'])
    assert not (out['batch_index'] == 1).any()


def test_current_model_valid_visual_evidence_with_direct_projection(tmp_path):
    release(tmp_path, 'seq0001', events=((9.5, 0),), images=(9.9,))
    b = collate_multimodal_v2([dataset(tmp_path)[0]])
    moved, images, masks, proj, targets, _ = prepare_model_batch(b, FakeDINO(), projection(), torch.device('cpu'))
    model = MultimodalV2(FakeCandidateProducer('R'), FakeCandidateProducer('V'), CandidateCrossAttention(), CandidateScoring())
    out = model(moved, images, masks, proj)
    assert out.radar_candidates.n == out.vision_candidates.n == 1
    assert out.evidence_valid_3d.any()
    assert torch.equal(out.score_3d_before, out.score_3d_after)
    from rdq_uav.multimodal_v2.loss import CandidateRankingLoss
    values = CandidateRankingLoss()(out, targets)
    assert torch.isfinite(values['loss'])


@pytest.mark.parametrize('contents', ['', '   \n'])
def test_empty_label_does_not_create_negative(tmp_path, contents):
    release(tmp_path, 'seq0001', images=(10.,))
    (tmp_path/'seq0001'/'2d_detect'/'10.0.txt').write_text(contents)
    s = dataset(tmp_path)[0]
    assert s['m_V'] and not s['gt_2d_valid']


@pytest.mark.parametrize('contents', ['0 0.5 0.5 0.2', '0 nan 0.5 0.2 0.2',
                                     '0 0.1 0.5 0.9 0.2',
                                     '0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.1 0.1'])
def test_bad_or_multiple_labels_raise(tmp_path, contents):
    release(tmp_path, 'seq0001', images=(10.,))
    (tmp_path/'seq0001'/'2d_detect'/'10.0.txt').write_text(contents)
    with pytest.raises(ValueError):
        dataset(tmp_path)[0]


def test_yolo_uses_left_crop_size_not_stereo_png_size(tmp_path):
    release(tmp_path, 'seq0001', images=(10.,))
    Image.new('RGB', (32, 8)).save(tmp_path/'seq0001'/'Image'/'10.0.png')
    (tmp_path/'seq0001'/'2d_detect'/'10.0.txt').write_text('0 0.5 0.5 0.25 0.5')
    assert dataset(tmp_path)[0]['gt_box_xyxy_px'].tolist() == [6., 2., 10., 6.]



def test_future_only_image_is_missing(tmp_path):
    release(tmp_path, 'seq0001', images=(10.1, 10.9))
    s = dataset(tmp_path)[0]
    assert not s['m_V'] and s['image_time'] is None and s['left_image_path'] is None
    assert not s['gt_2d_valid'] and not s['image_uint8'].any()


@pytest.mark.parametrize('image_time,valid', [(9., True), (10., True), (8.999, False), (10.001, False)])
def test_historical_image_inclusive_boundaries(tmp_path, image_time, valid):
    release(tmp_path, 'seq0001', images=(image_time,))
    s = dataset(tmp_path)[0]
    assert s['m_V'] == valid
    assert s['image_time'] == (image_time if valid else None)
    if valid:
        assert -1. <= s['vision_delta_t'] <= 0.


def test_historical_image_configurable_gap(tmp_path):
    release(tmp_path, 'seq0001', images=(9.5,))
    assert dataset(tmp_path, max_image_gap_s=.5)[0]['m_V']
    assert not dataset(tmp_path, max_image_gap_s=.25)[0]['m_V']


def test_validation_rejects_future_image(tmp_path):
    release(tmp_path, 'seq0001', images=(9.8,))
    s = dataset(tmp_path)[0]
    s.update(image_time=10.1, vision_delta_t=.1)
    with pytest.raises(ValueError, match='causal history window'):
        validate_sample(s)
