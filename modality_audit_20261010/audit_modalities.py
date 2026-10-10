#!/usr/bin/env python3
"""Offline audit of enriched V2 per_query.jsonl; no training or GT-based selection."""
import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
# Works both in diagnostics/ and from an unpacked delivery directory in the repo.
for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
    if (parent / 'src/rdq_uav/multimodal_v2/paper_metrics.py').exists():
        ROOT = parent
        break
spec = importlib.util.spec_from_file_location('audit_paper_metrics', ROOT / 'src/rdq_uav/multimodal_v2/paper_metrics.py')
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
coco_bbox_metrics = metrics_module.coco_bbox_metrics
deduplicate_images = metrics_module.deduplicate_images
localization_metrics = metrics_module.localization_metrics


def overlap(boxes, gt):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt, dtype=float)
    inter = np.maximum(0, np.minimum(boxes[:, 2:], gt[2:])-np.maximum(boxes[:, :2], gt[:2])).prod(1)
    union = np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(1) + np.prod(gt[2:]-gt[:2]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def quantiles(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(zip(('p10', 'median', 'p90'), map(float, np.quantile(a, [.1, .5, .9])))) if len(a) else None


def image_record(row):
    gt = np.asarray(row['gt_box_xyxy_px'], dtype=float)
    boxes = np.asarray(row['pred_boxes_xyxy_px'], dtype=float).reshape(-1, 4)
    scores = np.asarray(row['pred_box_scores'], dtype=float)
    if len(boxes) != len(scores):
        raise ValueError('box/score count mismatch')
    ious = overlap(boxes, gt)
    # Preserve exported final score order/tie handling; do not rerank using GT.
    top = 0 if len(boxes) else None
    oracle = int(ious.argmax()) if len(boxes) else None
    ti, oi = (float(ious[top]), float(ious[oracle])) if top is not None else (0., 0.)
    category = ('no_output' if top is None else 'success' if ti >= .5 else
                'ranking_failure' if oi >= .5 else 'candidate_miss')
    wh = gt[2:] - gt[:2]
    result = {k: row[k] for k in ('sequence_id', 'sample_id', 'query_time', 'image_time', 'image_path')}
    result.update(category=category, candidate_count=len(boxes), top1_iou=ti, oracle_iou=oi,
                  top1_score=float(scores[top]) if top is not None else None,
                  oracle_score=float(scores[oracle]) if oracle is not None else None,
                  gt_width_px=float(wh[0]), gt_height_px=float(wh[1]), gt_area_px2=float(wh.prod()))
    for key in ('center_dx_px', 'center_dy_px', 'center_error_px', 'normalized_center_error',
                'width_ratio', 'height_ratio'):
        result[key] = None
    if top is not None:
        d = (boxes[top, :2]+boxes[top, 2:]-gt[:2]-gt[2:])/2
        pw = boxes[top, 2:]-boxes[top, :2]
        result.update(center_dx_px=float(d[0]), center_dy_px=float(d[1]),
                      center_error_px=float(np.linalg.norm(d)),
                      normalized_center_error=float(np.linalg.norm(d/wh)),
                      width_ratio=float(pw[0]/wh[0]), height_ratio=float(pw[1]/wh[1]))
    return result


def draw_cases(rows, records, folder, count, data_root):
    from PIL import Image, ImageDraw
    lookup = {r['sample_id']: r for r in rows}
    groups = defaultdict(list)
    for r in records:
        groups[(r['sequence_id'], r['category'])].append(r)
    failures = []
    for (seq, category), group in sorted(groups.items()):
        for rank, record in enumerate(sorted(group, key=lambda r:(r['top1_iou'], r['sample_id']))[:count]):
            row = lookup[record['sample_id']]
            path = Path(row['image_path'])
            if data_root:
                path = data_root / seq / 'Image' / path.name
            try:
                with Image.open(path) as handle:
                    image = handle.convert('RGB')
                w, h = map(int, row['image_source_wh'])
                if image.width < w or image.height < h:
                    raise ValueError('image smaller than source left view')
                image = image.crop((0, 0, w, h))
                gt = np.asarray(row['gt_box_xyxy_px'], float)
                boxes = np.asarray(row['pred_boxes_xyxy_px'], float).reshape(-1,4)
                canvas = ImageDraw.Draw(image)
                canvas.rectangle(gt.tolist(), outline='lime', width=2)
                if len(boxes):
                    canvas.rectangle(boxes[0].tolist(), outline='red', width=2)
                    canvas.rectangle(boxes[int(overlap(boxes,gt).argmax())].tolist(), outline='cyan', width=1)
                image.thumbnail((1280, 960))
                # Magnified ROI for small UAV boxes, retaining all relevant boxes.
                relevant = np.concatenate([gt.reshape(1,4), boxes[:1]])
                lo = np.maximum(0, relevant[:, :2].min(0)-30)
                hi = np.minimum([w,h], relevant[:, 2:].max(0)+30)
                with Image.open(path) as handle:
                    roi = handle.convert('RGB').crop((int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])))
                dr = ImageDraw.Draw(roi)
                dr.rectangle((gt-np.tile(lo,2)).tolist(), outline='lime', width=2)
                if len(boxes):
                    dr.rectangle((boxes[0]-np.tile(lo,2)).tolist(), outline='red', width=2)
                roi.thumbnail((600,600))
                if max(roi.size) < 600:
                    scale = min(4.,600/max(roi.size))
                    roi = roi.resize((int(roi.width*scale),int(roi.height*scale)))
                out = folder / seq / category
                out.mkdir(parents=True,exist_ok=True)
                image.save(out/f'{rank:02d}_full.png')
                roi.save(out/f'{rank:02d}_zoom.png')
            except (OSError, ValueError) as e:
                failures.append(dict(sample_id=row['sample_id'], path=str(path), error=str(e)))
    return failures


def radar_summary(rows):
    result = localization_metrics(rows)
    errors = np.asarray([np.asarray(r['pred_xyz'])-r['gt_xyz'] for r in rows
                         if r['has_gt3d'] and r.get('pred_xyz') is not None], float).reshape(-1,3)
    for i, axis in enumerate('xyz'):
        result[f'bias_{axis}_m'] = float(errors[:,i].mean()) if len(errors) else None
        result[f'mae_{axis}_m'] = float(np.abs(errors[:,i]).mean()) if len(errors) else None
    result['distance_quantiles_m'] = quantiles(np.linalg.norm(errors,axis=1))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['B0','B1'], required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--visualize-per-category', type=int, default=4)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError('empty input')
    required = {'sequence_id','sample_id','has_gt3d','gt_xyz','pred_xyz','has_gt2d'}
    if any(required-set(r) for r in rows):
        raise ValueError('Old per-query schema; rerun current evaluate.py first')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'mode':args.mode, 'source':str(args.input.resolve()), 'groups':{}}
    groups = {'overall':rows}
    for seq in sorted({r['sequence_id'] for r in rows}):
        groups[seq] = [r for r in rows if r['sequence_id']==seq]
    records = []
    for name, group in groups.items():
        if args.mode == 'B0':
            result = radar_summary(group)
        else:
            selected = deduplicate_images(group)
            metrics = [image_record(r) for r in selected]
            if name == 'overall':
                records = metrics
            n = len(metrics)
            result = coco_bbox_metrics(group, args.output/'coco'/name)
            result.update(top1_iou50=sum(r['top1_iou']>=.5 for r in metrics)/n if n else None,
                          final_candidate_oracle_iou50=sum(r['oracle_iou']>=.5 for r in metrics)/n if n else None,
                          failure_counts={c:sum(r['category']==c for r in metrics) for c in
                                          ('success','ranking_failure','candidate_miss','no_output')})
            for key in ('gt_width_px','gt_height_px','gt_area_px2','top1_score','oracle_score',
                        'center_error_px','normalized_center_error','width_ratio','height_ratio'):
                result[key+'_quantiles'] = quantiles([r[key] for r in metrics])
        report['groups'][name] = result
    if records:
        with (args.output/'per_image.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        report['visualization_errors'] = draw_cases(deduplicate_images(rows), records, args.output/'examples',
                                                    max(0,args.visualize_per_category),args.data_root)
    (args.output/'audit_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    flat = [{ 'group':name, **{k:v for k,v in r.items() if not isinstance(v,(dict,list))}}
            for name,r in report['groups'].items()]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with (args.output/'by_sequence.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(flat)
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
