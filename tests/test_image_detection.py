from PIL import Image
import numpy as np
import pytest

from rdq_uav.multimodal_v3 import image_detection
from rdq_uav.multimodal_v3.detection_metrics import image_detection_metrics


def test_raw_detector_preserves_original_unclipped_box_convention():
    from types import SimpleNamespace
    import torch
    def preprocess(inputs):
        image = inputs[0]['image']
        return SimpleNamespace(tensor=image[None], image_sizes=[image.shape[-2:]])
    vision = SimpleNamespace(
        detector=SimpleNamespace(preprocess_image=preprocess),
        dino=SimpleNamespace(swin=lambda tensor: tensor,
            forward_from_pyramid=lambda pyramid, mask: {
                'pred_logits': torch.tensor([[[2.]]]),
                'pred_boxes': torch.tensor([[[.02, .5, .2, .2]]])}))
    result = image_detection.detect_view(vision, Image.new('RGB', (100, 100)), 'cpu',
                                         {'short_edge': 100, 'max_size': 100, 'topk': 100})
    np.testing.assert_allclose(result['boxes'], [[-8., 40., 12., 60.]], atol=1e-5)


def test_missing_track_prior_preserves_full_image_detections(monkeypatch):
    calls = []
    def detect(vision, image, device, config):
        calls.append(image.size)
        return {'boxes': [[10, 10, 20, 20]], 'scores': [.9]}
    monkeypatch.setattr(image_detection, 'detect_view', detect)
    result = image_detection.infer_image_variants(None, Image.new('RGB', (128, 96)),
              {'causal': None, 'lag05': None}, 'cpu', {'nms_iou': .5, 'topk': 100, 'crop_side_px': 64})
    assert result['track_crop_causal'] == result['track_crop_lag05'] == result['vision_nms']
    assert len(calls) == 2  # Full image and visual control, no fabricated radar crops.


def test_guided_crop_detection_returns_source_pixels(monkeypatch):
    monkeypatch.setattr(image_detection, 'detect_view', lambda *args:
                        {'boxes': [[1, 2, 11, 12]], 'scores': [.9]})
    result = image_detection.infer_image_variants(None, Image.new('RGB', (128, 96)),
              {'causal': {'crop_xyxy': [64, 32, 128, 96]}, 'lag05': None}, 'cpu',
              {'nms_iou': .5, 'topk': 100, 'crop_side_px': 64})
    assert [65., 34., 75., 44.] in result['track_crop_causal']['boxes']
    assert [1., 2., 11., 12.] in result['track_crop_causal']['boxes']


def test_separate_crop_detector_leaves_full_baseline_and_caches_raw_views(monkeypatch):
    full_model, crop_model = object(), object()
    calls = []
    def detect(model, image, *args):
        calls.append(model)
        return {'boxes': [[1, 2, 11, 12]], 'scores': [.9 if model is full_model else .8]}
    monkeypatch.setattr(image_detection, 'detect_view', detect)
    raw = {}
    result = image_detection.infer_image_variants(full_model, Image.new('RGB', (128, 96)),
        {'causal': {'crop_xyxy': [64, 32, 128, 96]}, 'lag05': None}, 'cpu',
        {'nms_iou': .5, 'topk': 100, 'crop_side_px': 64}, crop_vision=crop_model, raw_views=raw)
    assert calls == [full_model, crop_model, crop_model]
    assert result['vision_top50'] == raw['full']
    assert result['track_crop_lag05'] == result['vision_nms']
    assert raw['track_crop_causal']['boxes'] == [[65., 34., 75., 44.]]
    assert raw['track_crop_causal']['scores'] == [.8]
    assert 'track_crop_lag05' not in raw


def row(boxes, scores, image='a', t=1.):
    return {'has_gt2d': True, 'sample_id': image + str(t), 'sequence_id': 's', 'image_path': image,
            'image_time': 1., 'query_time': t, 'image_source_wh': [100, 100],
            'gt_box_xyxy_px': [10, 10, 20, 20], 'pred_boxes_xyxy_px': boxes, 'pred_box_scores': scores}


def test_precision_counts_duplicate_as_fp_and_missing_as_fn():
    rows = [row([[10, 10, 20, 20], [10, 10, 20, 20], [60, 60, 70, 70]], [.9, .8, .1]),
            row([], [], 'b')]
    m = image_detection_metrics(rows, .25)
    assert (m['TP'], m['FP'], m['FN']) == (1, 1, 1)
    assert m['Precision'] == m['Recall'] == .5
    assert m['mAP50'] > 0 and m['mAP50_95'] > 0


def test_image_dedup_does_not_select_better_prediction_using_gt():
    # Closest-time row is missing; a better but later query must not replace it.
    m = image_detection_metrics([row([], [], t=1.), row([[10, 10, 20, 20]], [.99], t=1.1)])
    assert m['unique_labeled_images'] == 1
    assert m['Recall'] == 0 and m['mAP50'] == 0


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        image_detection_metrics([], float('nan'))
