import json
import pytest
import torch

from rdq_uav.multimodal_v3.pipeline.extract_observations import image_sample_metadata, load_validation_reference


def batch(image=True, labeled=True):
    return dict(num_samples=1, sample_id=['a'], sequence_id=['seq0021'],
                query_time=torch.tensor([10.], dtype=torch.float64), m_V=torch.tensor([image]),
                image_time=torch.tensor([9.9 if image else float('nan')], dtype=torch.float64),
                left_image_path=['seq0021/Image/9.9.png' if image else None],
                image_source_wh=torch.tensor([[1280, 960]]) if image else torch.empty((0, 2)),
                gt_2d_valid=torch.tensor([labeled and image]), gt_box_xyxy_px=torch.tensor([[1., 2., 3., 4.]]))


def test_metadata_keeps_image_identity_without_changing_batch():
    original = batch()
    out = image_sample_metadata(original)
    assert out['has_image'] and out['has_gt2d'] and out['image_time'] == 9.9
    assert out['image_source_wh'] == [1280, 960]
    assert out['gt_box_xyxy_px'] == [1., 2., 3., 4.]
    assert out['gt_box_width_px'] == out['gt_box_height_px'] == 2.
    assert bool(original['m_V'][0])
    json.dumps(out, allow_nan=False)


def test_missing_image_has_no_nan_and_unlabeled_image_is_retained():
    out = image_sample_metadata(batch(image=False))
    assert out['image_time'] is None and out['image_path'] is None and not out['has_gt2d']
    json.dumps(out, allow_nan=False)
    out = image_sample_metadata(batch(labeled=False))
    assert out['has_image'] and not out['has_gt2d']


def test_test_split_never_reads_validation_reference(tmp_path):
    (tmp_path/'eval_best').mkdir()
    (tmp_path/'eval_best/per_query.jsonl').write_text('not valid JSON')
    assert load_validation_reference(tmp_path/'config.yaml', 'heldout_test_sub') is None
    (tmp_path/'eval_best/per_query.jsonl').write_text('{"sample_id":"a","final_error":1.}\n'.replace('1.', '1.0'))
    assert load_validation_reference(tmp_path/'config.yaml', 'validation_sub')['a']['final_error'] == 1.


def test_metadata_requires_extractor_single_sample_contract():
    original = batch(); original['num_samples'] = 2
    with pytest.raises(ValueError, match='batch_size=1'):
        image_sample_metadata(original)


def test_exported_samples_run_through_prior_preparation(tmp_path, monkeypatch):
    import sys
    from rdq_uav.multimodal_v3.pipeline import prepare_image_guidance
    samples = tmp_path/'samples.jsonl'
    samples.write_text(json.dumps(image_sample_metadata(batch())) + '\n')
    tracks = tmp_path/'tracks.jsonl'
    tracks.write_text('')
    output = tmp_path/'priors'
    monkeypatch.setattr(sys, 'argv', ['prepare_image_guidance.py', '--samples', str(samples),
        '--tracks', str(tracks), '--output', str(output)])
    prepare_image_guidance.main()
    assert json.loads((output/'diagnosis.json').read_text())['unique_labeled_images'] == 1
    row = json.loads((output/'image_priors.jsonl').read_text())
    assert row['priors'] == {'causal': None, 'lag05': None}
    assert 'gt_box_xyxy_px' not in row
