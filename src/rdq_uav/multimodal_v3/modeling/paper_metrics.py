"""Offline paper metrics: conditional 3D RMSE and image-level COCO bbox AP."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np


def localization_metrics(rows, prediction_key='pred_xyz'):
    labeled = [r for r in rows if r['has_gt3d']]
    errors = []
    for row in labeled:
        gt = np.asarray(row['gt_xyz'], dtype=np.float64)
        if gt.shape != (3,) or not np.isfinite(gt).all():
            raise ValueError('GT XYZ must be finite [3]')
        pred = row.get(prediction_key)
        if pred is None:
            continue
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != (3,) or not np.isfinite(pred).all():
            raise ValueError('prediction XYZ must be finite [3] or None')
        errors.append(pred - gt)
    n, p = len(labeled), len(errors)
    error = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    distances = np.linalg.norm(error, axis=1)
    axis = np.sqrt(np.mean(error ** 2, axis=0)).tolist() if p else [None] * 3
    return {
        'gt_queries': n, 'predicted_queries': p, 'missing_queries': n-p,
        'coverage': p/n if n else None,
        'rmse_denominator': p, 'rmse_scope': 'finite_top1_outputs_only',
        'rmse_x_m': axis[0], 'rmse_y_m': axis[1], 'rmse_z_m': axis[2],
        'rmse_3d_m': float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))) if p else None,
        'mean_error_m': float(distances.mean()) if p else None,
        'median_error_m': float(np.median(distances)) if p else None,
        **{f'success_{radius:g}m': int((distances <= radius).sum())/n if n else None
           for radius in (.5, 1., 2.)},
    }


def assign_conditions(rows, mapping=None):
    if mapping is None:
        return [dict(r, condition=None) for r in rows]
    if not isinstance(mapping, dict) or set(mapping)-{'sequences', 'samples'}:
        raise ValueError('conditions JSON must contain only sequences/samples maps')
    sequences, samples = mapping.get('sequences', {}), mapping.get('samples', {})
    if not isinstance(sequences, dict) or not isinstance(samples, dict):
        raise ValueError('condition maps must be objects')
    if any(v not in ('day', 'night') for v in [*sequences.values(), *samples.values()]):
        raise ValueError('condition labels must be day or night')
    result = []
    for row in rows:
        condition = samples.get(row['sample_id'], sequences.get(row['sequence_id']))
        if condition is None:
            raise ValueError(f"missing day/night label: {row['sample_id']}")
        result.append(dict(row, condition=condition))
    return result


def deduplicate_images(rows):
    """Choose query closest to selected image time, without using model quality/GT."""
    images = {}
    for row in rows:
        if not row.get('has_gt2d'):
            continue
        path, time = row.get('image_path'), row.get('image_time')
        if not path or time is None or row.get('image_source_wh') is None:
            raise ValueError('labeled image requires path, time and source dimensions')
        key = (row['sequence_id'], path)
        old = images.get(key)
        if old is not None:
            if (old['gt_box_xyxy_px'] != row['gt_box_xyxy_px']
                    or old['image_source_wh'] != row['image_source_wh']
                    or old.get('condition') != row.get('condition')):
                raise ValueError('same image has inconsistent GT, dimensions or condition')
        priority = lambda r: (abs(r['query_time']-r['image_time']), r['query_time'], r['sample_id'])
        if old is None or priority(row) < priority(old):
            images[key] = row
    return [images[key] for key in sorted(images)]


def _xywh(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('COCO boxes must be finite, positive-area xyxy')
    return [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]


def coco_bbox_metrics(rows, output=None):
    images = deduplicate_images(rows)
    result = {'labeled_queries': sum(bool(r.get('has_gt2d')) for r in rows),
              'unique_labeled_images': len(images),
              'deduplication': 'closest_query_to_image_time_then_query_time_then_sample_id',
              'prediction_scope': 'final_V_and_RV_candidates_in_source_left_view_pixels',
              'AP': None, 'AP50': None, 'AP75': None, 'AP_small': None, 'AR100': None}
    if not images:
        return dict(result, status='not_applicable_no_labeled_images')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dataset = {'info': {}, 'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'UAV'}]}
    predictions, selected = [], []
    for image_id, row in enumerate(images, 1):
        w, h = row['image_source_wh']
        if min(w, h) <= 0:
            raise ValueError('image source dimensions must be positive')
        dataset['images'].append({'id': image_id, 'width': int(w), 'height': int(h),
                                  'file_name': row['image_path']})
        box = _xywh(row['gt_box_xyxy_px'])
        dataset['annotations'].append({'id': image_id, 'image_id': image_id,
                                       'category_id': 1, 'bbox': box, 'area': box[2]*box[3], 'iscrowd': 0})
        boxes, scores = row['pred_boxes_xyxy_px'], row['pred_box_scores']
        if len(boxes) != len(scores):
            raise ValueError('prediction box/score lengths must match')
        for box, score in zip(boxes, scores):
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('prediction scores must be finite probabilities')
            predictions.append({'image_id': image_id, 'category_id': 1,
                                'bbox': _xywh(box), 'score': float(score)})
        selected.append({'image_id': image_id, 'sample_id': row['sample_id'],
                         'sequence_id': row['sequence_id'], 'image_path': row['image_path']})
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        for name, value in [('coco_gt.json', dataset), ('coco_predictions.json', predictions),
                            ('coco_selected_queries.json', selected)]:
            (output/name).write_text(json.dumps(value, indent=2, allow_nan=False))
    # COCO.loadRes([]) fails in some releases. Build an empty detection dataset explicitly.
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = dataset; gt.createIndex()
        if predictions:
            detections = gt.loadRes(predictions)
        else:
            detections = COCO(); detections.dataset = dict(dataset, annotations=[]); detections.createIndex()
        evaluator = COCOeval(gt, detections, 'bbox')
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    for key, index in [('AP', 0), ('AP50', 1), ('AP75', 2), ('AP_small', 3), ('AR100', 8)]:
        value = float(evaluator.stats[index])
        result[key] = value if value >= 0 else None
    result.update(status='evaluated', max_predictions_per_image=max(len(r['pred_box_scores']) for r in images),
                  coco_max_dets=[1, 10, 100], unit='fraction_0_to_1')
    return result


def paper_report(rows, mode, conditions=None, coco_output=None, compute_coco=True):
    rows = assign_conditions(rows, conditions)
    result = {'schema_version': 1, 'mode': mode,
              'localization': None if mode == 'B1' else localization_metrics(rows),
              'day_night': {'status': 'disabled_no_labels'},
              'vision_coco': coco_bbox_metrics(rows, coco_output) if compute_coco else {'status': 'disabled_by_user'}}
    if conditions is not None:
        grouped = {condition: [r for r in rows if r['condition'] == condition] for condition in ('day', 'night')}
        result['day_night'] = {'status': 'enabled', **{
            condition: {'queries': len(group),
                        'localization': None if mode == 'B1' else localization_metrics(group),
                        'vision_coco': coco_bbox_metrics(group, Path(coco_output)/condition if coco_output else None)
                        if compute_coco else {'status': 'disabled_by_user'}}
            for condition, group in grouped.items()}}
        day, night = (result['day_night'][c]['localization'] for c in ('day', 'night'))
        result['day_night']['macro_mean_day_night_rmse_m'] = (
            (day['rmse_3d_m']+night['rmse_3d_m'])/2
            if day and night and day['rmse_3d_m'] is not None and night['rmse_3d_m'] is not None else None)
    return result


def write_paper_csv(report, path, *, method, modality, bandwidth=''):
    total = report['localization'] or {}
    group = report['day_night']
    row = {'Method': method, 'Modality': modality, 'Training': 'Supervised',
           'Bandwidth': bandwidth, 'Split': report['split']}
    for condition in ('day', 'night'):
        loc = group.get(condition, {}).get('localization') or {}
        for axis in ('x', 'y', 'z'):
            row[f'{condition}_RMSE_D{axis}_m'] = loc.get(f'rmse_{axis}_m')
        row[f'{condition}_RMSE_3D_m'] = loc.get('rmse_3d_m')
    for key in ('rmse_x_m', 'rmse_y_m', 'rmse_z_m', 'rmse_3d_m', 'coverage',
                'gt_queries', 'predicted_queries', 'success_0.5m', 'success_1m', 'success_2m'):
        row[key] = total.get(key)
    row['macro_mean_day_night_RMSE_m'] = group.get('macro_mean_day_night_rmse_m')
    for key in ('AP', 'AP50', 'AP75', 'AP_small', 'AR100'):
        row[key] = report['vision_coco'].get(key)
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
