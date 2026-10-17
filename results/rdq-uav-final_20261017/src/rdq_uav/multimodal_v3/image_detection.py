"""Reuse the trained DINO on full images and GT-free guided crops."""
import numpy as np
from PIL import Image
import torch

from rdq_uav.multimodal_v3.modeling.candidate_builders import _cxcywh_norm_to_source_xyxy, _nms_xyxy
from rdq_uav.multimodal_v3.modeling.data import resize_wh
from .image_guidance import centered_crop, restore_crop_boxes


@torch.no_grad()
def detect_view(vision, image, device, config):
    """Return raw detector boxes/scores in this view's unresized pixels."""
    source_wh = image.size
    resized = image.resize(resize_wh(source_wh, config['short_edge'], config['max_size']),
                           Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    detector = vision.detector
    images = detector.preprocess_image([{'image': tensor, 'height': resized.height, 'width': resized.width}])
    mask = torch.ones((1, *images.tensor.shape[-2:]), dtype=torch.bool, device=device)
    h, w = images.image_sizes[0]
    mask[0, :h, :w] = False
    raw = vision.dino.forward_from_pyramid(vision.dino.swin(images.tensor), mask)
    scores = raw['pred_logits'][0, :, 0].float().sigmoid()
    boxes = _cxcywh_norm_to_source_xyxy(raw['pred_boxes'].float(),
                                       torch.tensor([source_wh], device=device))[0]
    # Preserve the original RGBCandidateBuilder convention, including boxes
    # extending beyond a view. Clipping here would change the raw visual
    # baseline and confound the comparison with B4 joint scoring.
    valid = torch.isfinite(boxes).all(1) & torch.isfinite(scores) & (boxes[:, 2:] > boxes[:, :2]).all(1)
    ids = torch.nonzero(valid).flatten()
    ids = ids[torch.argsort(scores[ids], descending=True, stable=True)[:config['topk']]]
    return {'boxes': boxes[ids].cpu().tolist(), 'scores': scores[ids].cpu().tolist()}


def merge_detections(parts, nms_iou, topk):
    boxes = torch.tensor([b for part in parts for b in part['boxes']], dtype=torch.float32).reshape(-1, 4)
    scores = torch.tensor([s for part in parts for s in part['scores']], dtype=torch.float32)
    keep = _nms_xyxy(boxes, scores, nms_iou)[:topk] if len(boxes) else torch.empty(0, dtype=torch.long)
    return {'boxes': boxes[keep].tolist(), 'scores': scores[keep].tolist()}


def merge_guided_detections(full, local, prior, config):
    """Associate 2D boxes with a radar projection, then reuse existing NMS.

    The optional gate uses the same fixed inverse-range margin as visual birth.
    It never changes box coordinates or confidence. Missing priors fall back to
    full-view NMS. Inputs contain predictions and sensor geometry only.
    """
    enabled = config.get('radar_box_gate', False)
    if enabled not in (False, 'crop_only', 'all_views'):
        raise ValueError('radar_box_gate must be false, crop_only or all_views')
    if prior is None:
        return merge_detections([full], config['nms_iou'], config['topk'])
    parts = [full, local]
    if enabled:
        center = np.asarray(prior['projected_xy_px'], dtype=float)
        xyz = np.asarray(prior['xyz'], dtype=float)
        if center.shape != (2,) or xyz.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(xyz).all():
            raise ValueError('Invalid radar projection geometry')
        radius = np.clip(16*20/max(np.linalg.norm(xyz), 1e-6), 8, 48)
        parts = []
        for part in (full, local):
            if part is full and enabled == 'crop_only':
                parts.append(part)
                continue
            boxes = np.asarray(part['boxes'], dtype=float).reshape(-1, 4)
            scores = np.asarray(part['scores'], dtype=float)
            if scores.shape != (len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError('Invalid box/score arrays')
            outside = np.maximum(np.maximum(boxes[:, :2]-center, center-boxes[:, 2:]), 0)
            keep = np.linalg.norm(outside, axis=1) <= radius
            parts.append({'boxes': boxes[keep].tolist(), 'scores': scores[keep].tolist()})
    return merge_detections(parts, config['nms_iou'], config['topk'])


def infer_image_variants(vision, image, priors, device, config, *, crop_vision=None, raw_views=None):
    """Inference boundary takes only image pixels and geometry priors, no GT.

    Four passes: full view; visual-Top1 crop control; causal crop; delayed crop.
    Missing priors fall back to the full-view baseline, never drop an image.
    """
    full = detect_view(vision, image, device, config)
    if raw_views is not None:
        raw_views['full'] = full
    nms = merge_detections([full], config['nms_iou'], config['topk'])
    outputs = {'vision_top50': {'boxes': full['boxes'][:50], 'scores': full['scores'][:50]},
               'vision_top100': full, 'vision_nms': nms}
    center = ((np.array(full['boxes'][0][:2]) + full['boxes'][0][2:]) / 2
              if full['boxes'] else np.array(image.size) / 2)
    crops = {'vision_crop': centered_crop(center, image.size, config['crop_side_px'])}
    crops.update({f'track_crop_{mode}': None if priors[mode] is None else priors[mode]['crop_xyxy']
                  for mode in ('causal', 'lag05')})
    for name, crop in crops.items():
        if crop is None:
            outputs[name] = nms
            continue
        if len(crop) != 4 or not (0 <= crop[0] < crop[2] <= image.width and
                                  0 <= crop[1] < crop[3] <= image.height):
            raise ValueError('Prior crop outside source image')
        local = detect_view(vision if crop_vision is None else crop_vision,
                            image.crop(crop), device, config)
        local['boxes'] = restore_crop_boxes(local['boxes'], crop).tolist()
        if raw_views is not None:
            raw_views[name] = {'crop_xyxy': crop, **local}
        outputs[name] = (merge_detections([full, local], config['nms_iou'], config['topk'])
                         if name == 'vision_crop' else
                         merge_guided_detections(full, local, priors[name.removeprefix('track_crop_')], config))
    return outputs
