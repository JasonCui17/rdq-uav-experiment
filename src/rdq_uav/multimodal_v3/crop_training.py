"""B1-only mixed full/crop supervision; validation and inference stay GT-free.

Training GT defines a positive crop with up to 64px center jitter. Original
source pixels are re-read before resizing, so no detail is lost by cropping an
already-downsampled full image. No new split, model, or loss is introduced.
"""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from rdq_uav.multimodal_v3.modeling.data import resize_wh
from .image_guidance import centered_crop


def crop_training_sample(sample, center_xy, side=256, short_edge=800, max_size=1333):
    if not sample['m_V'] or not sample['gt_2d_valid']:
        return sample
    source_wh = tuple(map(int, sample['image_source_wh'].tolist()))
    crop = centered_crop(center_xy, source_wh, side)
    gt = sample['gt_box_xyxy_px']
    # Retain full image if its annotation cannot fit; never clip training GT.
    if any(gt[i] < crop[i] for i in (0, 1)) or any(gt[i] > crop[i] for i in (2, 3)):
        return sample
    with Image.open(sample['left_image_path']) as image:
        if image.width < source_wh[0] or image.height < source_wh[1]:
            raise ValueError('Image smaller than source left view')
        view = image.convert('RGB').crop(crop)
        crop_wh = view.size
        view_wh = resize_wh(crop_wh, short_edge, max_size)
        view = view.resize(view_wh, Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(view, dtype=np.uint8).copy()).permute(2, 0, 1)
    source = torch.tensor(crop_wh, dtype=torch.long)
    resized = torch.tensor(view_wh, dtype=torch.long)
    return {**sample, 'image_uint8': tensor, 'image_source_wh': source,
            'image_view_wh': resized, 'image_scale_xy': resized.float() / source,
            'gt_box_xyxy_px': gt - gt.new_tensor([crop[0], crop[1], crop[0], crop[1]]),
            'training_crop_xyxy': crop}


class MixedCropDataset(Dataset):
    def __init__(self, dataset, *, side=256, probability=.5, short_edge=800, max_size=1333):
        self.dataset = dataset
        self.side, self.probability = side, probability
        self.short_edge, self.max_size = short_edge, max_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not sample['m_V'] or not sample['gt_2d_valid'] or torch.rand(()).item() >= self.probability:
            return sample
        box = sample['gt_box_xyxy_px']
        center = (box[:2] + box[2:]) / 2 + (torch.rand(2) - .5) * (self.side / 2)
        return crop_training_sample(sample, center.tolist(), self.side, self.short_edge, self.max_size)


def wrap_crop_training(dataset, config):
    if config['experiment']['stage'] != 'B1':
        raise ValueError('Crop-coordinate supervision is restricted to standalone B1')
    policy = {'side_px': 256, 'probability': .5, 'center_jitter_max_px': 64,
              'training_GT_used_for_crop': True, 'validation_unchanged': True,
              'short_edge': config['data']['dino_short_edge'],
              'max_size': config['data']['dino_max_size']}
    config['training']['image_crop_policy'] = policy
    print('Mixed image crop training:', policy, flush=True)
    return MixedCropDataset(dataset, short_edge=policy['short_edge'], max_size=policy['max_size'])
