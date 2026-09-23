#!/usr/bin/env python3
"""P4-A diagnostic: replay frozen GT->left-camera projection against manually verified V2 boxes.

Important scope:
  - Uses annotation.initial_projection.xyz (stored GT reference), NOT predicted LiDAR candidates.
  - Manual boxes may overlap calibration fitting data: engineering diagnosis ONLY.
  - This report cannot mark the formal P4 geometry gate PASS and cannot establish
    positive HCI impact. A later independent set and P4-B intervention are required.
  - Does not fit or change R, t, dt, and does not write into existing audit assets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RADII = (8, 16, 32, 64)


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def load_projection(camera_path: Path, geometry_path: Path) -> dict[str, Any]:
    cam = yaml.safe_load(camera_path.read_text(encoding='utf8'))['cameras']['left']
    geometry = json.loads(geometry_path.read_text(encoding='utf8'))
    if cam.get('model') != 'omni' or cam.get('distortion_model') != 'radtan':
        raise ValueError('expected omni/radtan left camera')
    if geometry.get('time_convention') != 'gt_query_time = image_time + time_offset_s':
        raise ValueError('unrecognized time convention')
    extrinsic = geometry['cameras']['left']
    R = np.asarray(extrinsic['rotation_camera_from_gt'], dtype=np.float64)
    t = np.asarray(extrinsic['translation_camera_from_gt_m'], dtype=np.float64)
    intr = np.asarray(cam['intrinsics'], dtype=np.float64)
    dist = np.asarray(cam['distortion_coeffs'], dtype=np.float64)
    wh = np.asarray(cam['resolution'], dtype=np.int64)
    if R.shape != (3, 3) or t.shape != (3,) or intr.shape != (5,) or dist.shape != (4,) or wh.shape != (2,):
        raise ValueError('invalid camera/extrinsic tensor dimensions')
    if not all(np.isfinite(x).all() for x in (R, t, intr, dist)):
        raise ValueError('non-finite camera calibration')
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-3 or np.linalg.det(R) < 0.99:
        raise ValueError('rotation not orthonormal')
    return dict(R=R, t=t, intr=intr, dist=dist, wh=wh,
                dt=float(geometry['time_offset_s']))


def project_gt(xyz: Any, c: dict[str, Any]) -> tuple[float | None, float | None]:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return None, None
    x, y, z = c['R'] @ xyz + c['t']
    xi, fu, fv, pu, pv = c['intr']
    k1, k2, p1, p2 = c['dist']
    norm = math.sqrt(x*x + y*y + z*z)
    denom = z + xi * norm
    if not math.isfinite(denom) or denom <= 1e-12:
        return None, None
    xn, yn = x / denom, y / denom
    r2 = xn*xn + yn*yn
    radial = 1 + k1*r2 + k2*r2*r2
    xd = xn*radial + 2*p1*xn*yn + p2*(r2+2*xn*xn)
    yd = yn*radial + p1*(r2+2*yn*yn) + 2*p2*xn*yn
    u, v = fu*xd + pu, fv*yd + pv
    if not all(map(math.isfinite, (u, v))) or not (0 <= u < c['wh'][0] and 0 <= v < c['wh'][1]):
        return None, None
    return float(u), float(v)


def point_box_distance(u: float, v: float, box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(x1-u, 0.0, u-x2), max(y1-v, 0.0, v-y2))


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r['projection_valid']]
    distances = [r['distance_to_bbox_px'] for r in valid]
    center = [r['distance_to_center_px'] for r in valid]
    ans: dict[str, Any] = {
        'n': len(rows),
        'valid': len(valid),
        'projection_valid_rate': len(valid)/len(rows) if rows else None,
    }
    if not valid:
        ans.update(inside_bbox_rate=None, bbox_distance_median_px=None,
                   bbox_distance_p95_px=None, center_distance_median_px=None,
                   center_distance_p95_px=None,
                   **{f'coverage_bbox_at_{rad}px': None for rad in RADII})
    else:
        ans.update(
            inside_bbox_rate=sum(r['inside_bbox'] for r in valid)/len(valid),
            bbox_distance_median_px=float(np.median(distances)),
            bbox_distance_p95_px=float(np.percentile(distances, 95)),
            center_distance_median_px=float(np.median(center)),
            center_distance_p95_px=float(np.percentile(center, 95)),
            **{f'coverage_bbox_at_{rad}px':sum(d <= rad for d in distances)/len(valid) for rad in RADII},
        )
    return ans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations-root', type=Path,
                    default=ROOT/'outputs/multimodal_v1/manual_calibration')
    ap.add_argument('--camera-config', type=Path,
                    default=ROOT/'configs/calibration/mmaud_v1_omni.yaml')
    ap.add_argument('--geometry-calibration', type=Path,
                    default=ROOT/'calibration/official_left_p4_current_geometry.json')
    ap.add_argument('--output-dir', type=Path,
                    default=ROOT/'outputs/own_multimodal_research/p4_manual161_replay')
    args = ap.parse_args()
    camera_path, geometry_path = resolve(args.camera_config), resolve(args.geometry_calibration)
    c = load_projection(camera_path, geometry_path)
    files = sorted(resolve(args.annotations_root).glob('seq*/bbox_annotations.json'))
    if not files:
        raise RuntimeError('no manual bbox_annotations.json files found')
    rows: list[dict[str, Any]] = []
    unique = set()
    for path in files:
        anns = json.loads(path.read_text(encoding='utf8'))
        if not isinstance(anns, list):
            raise ValueError(f'{path} must contain a list')
        for ann in anns:
            source = str(ann.get('annotation_source',''))
            if not source.startswith('manual'):
                raise ValueError(f'unverified annotation source at {path}: {source}')
            seq, name = str(ann['sequence']), str(ann['image_name'])
            if seq != path.parent.name or (seq, name) in unique:
                raise ValueError(f'sequence mismatch or duplicate annotation: {seq}/{name}')
            unique.add((seq, name))
            if [ann['image_width'], ann['image_height']] != c['wh'].tolist():
                raise ValueError(f'annotation image resolution mismatches camera: {seq}/{name}')
            box = list(map(float, ann['bbox_xyxy']))
            w, h = map(int, c['wh'])
            if len(box)!=4 or not all(map(math.isfinite,box)) or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
                raise ValueError(f'invalid bbox: {seq}/{name}')
            init = ann['initial_projection']
            xyz = np.asarray(init['xyz'], dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f'invalid GT XYZ: {seq}/{name}')
            u, v = project_gt(xyz, c)
            dist_m = float(np.linalg.norm(xyz))
            range_bin = ('0-15m' if dist_m < 15 else '15-30m' if dist_m < 30 else '30m+')
            valid = u is not None
            dc = math.hypot(u - (box[0]+box[2])/2, v-(box[1]+box[3])/2) if valid else None
            db = point_box_distance(u, v, box) if valid else None
            gt_image_abs_gap = abs(float(ann['image_time']) - float(init['gt_time'])) if init.get('gt_time') is not None else None
            rows.append({
                'sequence':seq, 'image_name':name, 'range_m':dist_m, 'range_bin':range_bin,
                'annotation_source':source,'gt_mode':init.get('mode'),
                'gt_image_abs_gap_s':gt_image_abs_gap,
                'bbox_x1':box[0], 'bbox_y1':box[1], 'bbox_x2':box[2], 'bbox_y2':box[3],
                'projected_u':u,'projected_v':v,'projection_valid':valid,
                'inside_bbox':(db==0.0) if valid else None,
                'distance_to_bbox_px':db,'distance_to_center_px':dc,
            })
    by_bin = {b:stats([r for r in rows if r['range_bin']==b]) for b in ('0-15m','15-30m','30m+')}
    by_seq = {seq:stats([r for r in rows if r['sequence']==seq]) for seq in sorted({r['sequence'] for r in rows})}
    gaps = [r['gt_image_abs_gap_s'] for r in rows if r['gt_image_abs_gap_s'] is not None]
    report = {
        'status':'ENGINEERING_DIAGNOSTIC_ONLY_NOT_FORMAL_P4_PASS',
        'note':'Uses stored 3D GT reference and existing manual boxes; NOT raw LiDAR candidate coverage, NOT independent calibration validation, NOT proof of HCI benefit.',
        'camera_config':str(camera_path),'geometry_calibration':str(geometry_path),
        'time_offset_s':c['dt'],'manual_annotation_files':len(files),
        'overall':stats(rows),'range_bins':by_bin,'sequences':by_seq,
        'gt_image_gap_median_s':float(np.median(gaps)) if gaps else None,
        'gt_image_gap_p95_s':float(np.percentile(gaps,95)) if gaps else None,
    }
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    with (out_dir/'per_image.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k!='sequences'},indent=2,ensure_ascii=False))
    print('CSV:',out_dir/'per_image.csv')
    print('JSON:',out_dir/'summary.json')

if __name__ == '__main__':
    main()
