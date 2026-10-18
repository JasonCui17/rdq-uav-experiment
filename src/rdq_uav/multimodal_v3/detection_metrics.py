"""Image-level COCO AP plus explicit fixed-threshold precision and recall."""
import numpy as np

from rdq_uav.multimodal_v3.modeling.paper_metrics import coco_bbox_metrics, deduplicate_images


def image_detection_metrics(rows, confidence_threshold=.25, output=None):
    if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError('Invalid confidence threshold')
    result = coco_bbox_metrics(rows, output)
    result['prediction_scope'] = 'provided_2D_predictions_in_source_left_view_pixels'
    images = deduplicate_images(rows)
    tp = fp = fn = 0
    for row in images:
        boxes = np.asarray(row['pred_boxes_xyxy_px'], dtype=float).reshape(-1, 4)
        scores = np.asarray(row['pred_box_scores'], dtype=float)
        ids = np.argsort(-scores, kind='stable')[:100]
        ids = ids[scores[ids] >= confidence_threshold]
        gt = np.asarray(row['gt_box_xyxy_px'])
        inter = np.maximum(0., np.minimum(boxes[ids, 2:], gt[2:]) - np.maximum(boxes[ids, :2], gt[:2])).prod(1)
        area = (boxes[ids, 2:] - boxes[ids, :2]).prod(1)
        iou = inter / np.maximum(area + np.prod(gt[2:] - gt[:2]) - inter, 1e-12)
        hit = bool(np.any(iou >= .5))
        tp += int(hit); fn += int(not hit); fp += len(ids) - int(hit)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    result.update(mAP50=result['AP50'], mAP50_95=result['AP'], Precision=precision, Recall=recall,
                  F1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                  TP=tp, FP=fp, FN=fn, confidence_threshold=confidence_threshold,
                  precision_recall_iou_threshold=.5, precision_recall_max_dets=100,
                  precision_recall_scope='one annotated UAV per labeled image; unlabeled images excluded')
    return result
