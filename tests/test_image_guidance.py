import numpy as np
import pytest

from rdq_uav.multimodal_v3.image_guidance import ImageTrackPrior, centered_crop, restore_crop_boxes


def track(t, x, **extra):
    return {'sequence_id': 'a', 'query_time': t, 'xyz': [x, 0, 0], 'velocity_mps': [2, 0, 0],
            'lagged_xyz': [x, 0, 0], 'lookahead_s': .5, 'event': 'high_update', **extra}


def test_causal_alignment_ignores_future_and_gt():
    rows = [track(1., 0), track(1.1, 50, gt_xyz=[float('nan')] * 3)]
    prior = ImageTrackPrior(rows).estimate('a', 1.05, 'causal')
    np.testing.assert_allclose(prior['xyz'], [.1, 0, 0])
    assert prior['image_lookahead_s'] == 0
    assert ImageTrackPrior(rows).estimate('b', 1.05, 'causal') is None


def test_delayed_alignment_discloses_actual_image_delay():
    prior = ImageTrackPrior([track(1, 0), track(1.1, 1)]).estimate('a', 1.04, 'lag05')
    np.testing.assert_allclose(prior['xyz'], [.4, 0, 0])
    assert prior['image_lookahead_s'] == pytest.approx(.56)


def test_missing_start_and_stale_prior_remain_missing():
    prior = ImageTrackPrior([track(1., 0)])
    assert prior.estimate('a', .9, 'causal') is None
    assert prior.estimate('a', 1.3, 'causal') is None
    assert prior.estimate('a', 1.1, 'lag05') is None


def test_no_interpolation_across_rebirth():
    prior = ImageTrackPrior([track(1., 0), track(1.1, 20, event='birth')])
    assert prior.estimate('a', 1.05, 'lag05') is None


def test_delayed_startup_backfill_allows_same_target_interpolation():
    prior = ImageTrackPrior([track(1., 0, xyz=None, event='warmup'), track(1.1, 1, event='birth')])
    np.testing.assert_allclose(prior.estimate('a', 1.05, 'lag05')['xyz'], [.5, 0, 0])


def test_timeout_gap_is_not_interpolated_from_new_birth():
    prior = ImageTrackPrior([track(1., 0, xyz=None, lagged_xyz=None, event='warmup'),
                            track(1.1, 10, event='birth')])
    assert prior.estimate('a', 1.05, 'lag05') is None


def test_delayed_mode_requires_auditable_source_lookahead():
    prior = ImageTrackPrior([track(1., 0, lookahead_s=None)])
    with pytest.raises(ValueError):
        prior.estimate('a', 1., 'lag05')


def test_duplicate_track_times_rejected():
    with pytest.raises(ValueError):
        ImageTrackPrior([track(1., 0), track(1., 1)])


@pytest.mark.parametrize('center,expected', [([5, 5], [0, 0, 64, 64]), ([127, 90], [64, 32, 128, 96])])
def test_crop_shifts_inside_view_without_shrinking(center, expected):
    assert centered_crop(center, [128, 96], 64) == expected


def test_restore_crop_coordinates_accepts_numpy():
    np.testing.assert_array_equal(restore_crop_boxes([[1, 2, 3, 4]], np.array([20, 30, 84, 94])),
                                  [[21, 32, 23, 34]])


def test_empty_crop_predictions():
    assert restore_crop_boxes([], [20, 30, 84, 94]).shape == (0, 4)


def test_prior_export_cannot_read_gt_coordinates_or_boxes():
    import torch
    from rdq_uav.multimodal_v3.pipeline.prepare_image_guidance import build_priors
    from rdq_uav.multimodal_v3.modeling.geometry import ProjectionContext
    projection = ProjectionContext(torch.eye(3)[None], torch.zeros(1, 3),
                                   torch.tensor([[1., 10., 10., 50., 50.]]), torch.zeros(1, 4),
                                   torch.tensor([[100., 100.]]), torch.ones(1, 2))
    tracks = [track(1., 0, xyz=[0, 0, 10], lagged_xyz=[0, 0, 10]),
              track(1.1, 0, xyz=[0, 0, 10], lagged_xyz=[0, 0, 10])]
    sample = {'sample_id': 'a', 'sequence_id': 'a', 'query_time': 1.1,
              'image_time': 1.05, 'image_path': 'a/Image/1.05.png',
              'gt_xyz': [0, 0, 10], 'gt_box_xyxy_px': [45, 45, 55, 55]}
    cfg = {'max_prior_gap_s': .25, 'crop_side_px': 32}
    before = build_priors([sample], tracks, projection, [100, 100], cfg)
    sample.update(gt_xyz='forbidden GT', gt_box_xyxy_px='forbidden GT')
    for r in tracks:
        r.update(gt_xyz='forbidden GT', target_error_m='forbidden GT')
    assert build_priors([sample], tracks, projection, [100, 100], cfg) == before
