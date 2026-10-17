#!/usr/bin/env python3
"""CPU diagnosis and GT-free image priors from frozen trajectory outputs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from rdq_uav.multimodal_v3.modeling.geometry import load_left_projection_context, project_omni_radtan
from rdq_uav.multimodal_v3.modeling.paper_metrics import deduplicate_images
from rdq_uav.multimodal_v3.image_guidance import ImageTrackPrior, centered_crop


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def image_rows(samples, source_wh, include_unlabeled=False):
    rows = []
    for row in samples:
        if row.get('has_image') and (include_unlabeled or row.get('has_gt2d')):
            image_name = Path(row['image_path']).name
            rows.append({**row, 'image_path': row['sequence_id'] + '/Image/' + image_name,
                         'image_source_wh': list(source_wh)})
    if not include_unlabeled:
        return deduplicate_images(rows)
    # Sensor-frame selection must not depend on label availability or quality.
    selected = {}
    priority = lambda r: (abs(r['query_time']-r['image_time']), r['query_time'], r['sample_id'])
    for row in rows:
        key = (row['sequence_id'], row['image_path'])
        if key not in selected or priority(row) < priority(selected[key]):
            selected[key] = row
    return sorted(selected.values(), key=lambda r: (r['sequence_id'], r['image_time']))


def build_priors(rows, tracks, projection, source_wh, config):
    # Only sequence/time/path identity is read from samples during inference.
    trajectory = ImageTrackPrior(tracks, config['max_prior_gap_s'])
    result = []
    for row in rows:
        record = {k: row[k] for k in ('sample_id', 'sequence_id', 'query_time', 'image_time', 'image_path')}
        record.update(image_source_wh=list(source_wh), crop_side_px=config['crop_side_px'])
        record['priors'] = {}
        for mode in ('causal', 'lag05'):
            prior = trajectory.estimate(row['sequence_id'], row['image_time'], mode)
            if prior is not None:
                uv, valid = project_omni_radtan(torch.tensor([prior['xyz']], dtype=torch.float32),
                                               torch.zeros(1, dtype=torch.long), projection)
                if bool(valid[0]):
                    prior.update(projected_xy_px=uv[0].tolist(),
                                 crop_xyxy=centered_crop(uv[0].tolist(), source_wh, config['crop_side_px']))
                else:
                    prior = None
            record['priors'][mode] = prior
        result.append(record)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--metrics', type=Path, help='Optional B4 training metrics.csv')
    p.add_argument('--config', type=Path, default=ROOT/'configs/final/image_guidance.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all-available-images', action='store_true',
                   help='Build priors for all query-associated sensor images, including unlabeled frames')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    camera = ROOT / config['camera_config']
    geometry = ROOT / config['geometry_calibration']
    source_wh = yaml.safe_load(camera.read_text())['cameras']['left']['resolution']
    projection = load_left_projection_context(camera, geometry, image_scale_xy=torch.ones(1, 2))
    samples, tracks = read_jsonl(args.samples), read_jsonl(args.tracks)
    rows = image_rows(samples, source_wh, args.all_available_images)
    priors = build_priors(rows, tracks, projection, source_wh, config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'image_priors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in priors))
    labeled_pairs = [(r, p) for r, p in zip(rows, priors) if r.get('has_gt2d')]
    labeled_rows = [r for r, _ in labeled_pairs]
    summary = {'queries': len(samples), 'unique_labeled_images': len(labeled_rows),
               'unique_sensor_images': len(rows), 'all_available_images': args.all_available_images,
               'image_sequences': dict(Counter(r['sequence_id'] for r in rows)),
               'source_wh': source_wh, 'config': config, 'GT_used_to_construct_priors': False,
               'scope': 'diagnostic coverage, not measured detection AP improvement',
               'crop_recall_denominator': 'all_unique_labeled_images; unlabeled sensor frames excluded from GT diagnostics',
               'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.samples, args.tracks, args.config, camera, geometry)}}
    diagnostics = []
    for mode in ('causal', 'lag05'):
        errors, delay, covers, center_inside = [], [], [], []
        for row, prior_row in labeled_pairs:
            prior = prior_row['priors'][mode]
            if prior is None:
                continue
            box = np.asarray(row['gt_box_xyxy_px']); crop = np.asarray(prior['crop_xyxy'])
            uv = np.asarray(prior['projected_xy_px']); center = (box[:2] + box[2:]) / 2
            error = float(np.linalg.norm(uv - center))
            covered = bool(np.all(box[:2] >= crop[:2]) and np.all(box[2:] <= crop[2:]))
            inside = bool(np.all(uv >= box[:2]) and np.all(uv <= box[2:]))
            errors.append(error); delay.append(prior['image_lookahead_s']); covers.append(covered); center_inside.append(inside)
            diagnostics.append({'sample_id': row['sample_id'], 'sequence_id': row['sequence_id'],
                                'mode': mode, 'center_error_px': error, 'gt_box_fully_in_crop': covered,
                                'projected_point_inside_gt_box': inside,
                                'image_lookahead_s': prior['image_lookahead_s']})
        summary[mode] = {'priors': len(errors), 'missing_priors': len(labeled_rows)-len(errors),
                         'center_error_px_quantiles_50_90_95_max': np.quantile(errors, [.5, .9, .95, 1]).tolist() if errors else None,
                         'gt_box_fully_in_crop': sum(covers),
                         'crop_recall_all_images': sum(covers)/len(labeled_rows) if labeled_rows else None,
                         'projected_point_inside_gt_box': sum(center_inside),
                         'max_image_lookahead_s': max(delay, default=0.)}
    widths = [r['gt_box_width_px'] for r in labeled_rows]
    heights = [r['gt_box_height_px'] for r in labeled_rows]
    summary['box_width_px_quantiles_10_50_90'] = np.quantile(widths, [.1, .5, .9]).tolist() if labeled_rows else None
    summary['small_box_fraction_area_lt_32sq'] = float(np.mean(np.asarray(widths)*heights < 32**2)) if labeled_rows else None
    if args.metrics:
        epochs = [r for r in csv.DictReader(args.metrics.open()) if r.get('val/AP')]
        summary['b4_epochs'] = [{k: float(r[k]) for k in ('epoch', 'val/AP', 'val/AP50', 'val/AP75', 'val/AR100', 'val/joint_success')} for r in epochs]
        summary['best_ap_epoch'] = max(summary['b4_epochs'], key=lambda r: r['val/AP']) if epochs else None
    (args.output / 'diagnosis.json').write_text(json.dumps(summary, indent=2))
    if diagnostics:
        with (args.output / 'projection_diagnostics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, diagnostics[0]); writer.writeheader(); writer.writerows(diagnostics)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('b4_epochs', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
