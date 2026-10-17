import numpy as np
from PIL import Image
import pytest
import torch

from rdq_uav.multimodal_v3.modeling.dino_supervision import build_dino_targets
from rdq_uav.multimodal_v3.modeling.data import ViewTransform
from rdq_uav.multimodal_v3.crop_training import crop_training_sample, MixedCropDataset, wrap_crop_training


def sample(tmp_path):
    path = tmp_path / 'stereo.png'
    pixels = np.zeros((960, 2560, 3), dtype=np.uint8)
    pixels[:, :1280, 0] = 255
    pixels[:, 1280:, 2] = 255
    Image.fromarray(pixels).save(path)
    return {'m_V': True, 'gt_2d_valid': True, 'image_source_wh': torch.tensor([1280, 960]),
            'gt_box_xyxy_px': torch.tensor([500., 400., 512., 412.]),
            'left_image_path': str(path), 'sequence_id': 'train', 'target_xyz': torch.ones(3)}


def test_crop_target_and_pixels_share_same_coordinates(tmp_path):
    original = sample(tmp_path)
    out = crop_training_sample(original, [506, 406])
    assert out['image_source_wh'].tolist() == [256, 256]
    assert out['image_view_wh'].tolist() == [800, 800]
    assert out['gt_box_xyxy_px'].tolist() == [122., 122., 134., 134.]
    assert torch.all(out['image_uint8'][0] == 255)
    assert torch.all(out['image_uint8'][2] == 0)
    assert original['gt_box_xyxy_px'].tolist() == [500., 400., 512., 412.]
    targets = build_dino_targets(out['gt_box_xyxy_px'][None], torch.tensor([True]),
                                 [ViewTransform((256, 256), (800, 800), False)])
    torch.testing.assert_close(targets[0]['boxes'], torch.tensor([[.5, .5, 12/256, 12/256]]))


def test_crop_does_not_clip_or_discard_annotation(tmp_path):
    original = sample(tmp_path)
    assert crop_training_sample(original, [0, 0]) is original
    original['m_V'] = False
    assert crop_training_sample(original, [506, 406]) is original


def test_mixed_dataset_keeps_identities_length_and_full_control(tmp_path):
    original = sample(tmp_path)
    ds = MixedCropDataset([original], probability=0)
    assert len(ds) == 1 and ds[0] is original
    torch.manual_seed(42)
    ds = MixedCropDataset([original], probability=1)
    assert ds[0]['sequence_id'] == 'train'
    cfg = {'experiment': {'stage': 'B4'}}
    with pytest.raises(ValueError, match='B1'):
        wrap_crop_training([original], cfg)


def test_real_collate_unpads_mixed_views_before_dino_and_masks_padding(tmp_path):
    from types import SimpleNamespace
    from tests.test_v2_data import release, dataset, projection
    from rdq_uav.multimodal_v3.modeling.data import collate_multimodal_v2, prepare_model_batch
    release(tmp_path, 'seq0001', queries=(10., 11.), images=(10., 11.))
    for stamp in ('10.0', '11.0'):
        (tmp_path/'seq0001'/'2d_detect'/f'{stamp}.txt').write_text('0 .5 .5 .25 .25\n')
    ds = dataset(tmp_path)
    full = ds[0]
    cropped = crop_training_sample(ds[1], [8, 4], side=8, short_edge=8, max_size=16)
    batch = collate_multimodal_v2([full, cropped])
    seen = []
    class DINO:
        def preprocess_image(self, inputs):
            seen.extend([tuple(row['image'].shape) for row in inputs])
            return SimpleNamespace(tensor=torch.stack([
                torch.nn.functional.pad(row['image'], (0, 16-row['image'].shape[2], 0, 0))
                for row in inputs]), image_sizes=[(8, 16), (8, 8)])
    _, _, masks, _, _, transforms = prepare_model_batch(batch, DINO(), projection(), torch.device('cpu'))
    assert seen == [(3, 8, 16), (3, 8, 8)]
    assert not masks[0].any() and not masks[1, :, :8].any() and masks[1, :, 8:].all()
    targets = build_dino_targets(batch['gt_box_xyxy_px'], batch['gt_2d_valid'], transforms)
    torch.testing.assert_close(targets[1]['boxes'], torch.tensor([[.5, .5, .5, .25]]))
