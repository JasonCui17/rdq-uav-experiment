#!/usr/bin/env python3
"""Read-only real MMAUD Sample smoke; no model or checkpoint is required."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
import yaml
from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from rdq_uav.multimodal_v2.data import MultimodalV2Dataset, collate_multimodal_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    resolve = lambda path: resolve_project_path(path, ROOT)
    cfg = apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    data = cfg['data']
    root = resolve(data['root'])
    if not (root/args.sequence/'ground_truth').is_dir():
        raise FileNotFoundError(f'real sequence GT unavailable: {root/args.sequence}')
    geometry = json.loads(resolve(data['geometry_calibration']).read_text())
    if float(geometry.get('time_offset_s', 0.0)) != 0:
        raise ValueError('raw image-time contract requires zero time_offset_s')
    camera = yaml.safe_load(resolve(data['camera_config']).read_text())
    ds = MultimodalV2Dataset(root, [args.sequence],
                            camera_wh=tuple(camera['cameras']['left']['resolution']),
                            short_edge=int(data['dino_short_edge']), max_size=int(data['dino_max_size']),
                            radar_history_s=float(data.get('radar_history_s', 1.0)),
                            max_image_gap_s=float(data.get('max_image_gap_s', 1.0)),
                            label_directory=str(data.get('label_directory', '2d_detect')))
    if args.samples < 3 or args.start_index < 0 or args.start_index+args.samples > len(ds):
        raise ValueError(f'smoke requires at least 3 available queries; dataset has {len(ds)}')
    samples = [ds[i] for i in range(args.start_index, args.start_index+args.samples)]
    rows = []
    for s in samples:
        times, dt = s['event_timestamps'], s['delta_t']
        row = dict(sequence_id=s['sequence_id'], query_time=s['query_time'],
                   radar_event_count=s['event_count'],
                   radar_event_min_time=min(times) if times else None,
                   radar_event_max_time=max(times) if times else None,
                   radar_delta_t_min=float(dt.min()) if len(dt) else None,
                   radar_delta_t_max=float(dt.max()) if len(dt) else None,
                   number_of_points=len(s['points']), radar_valid=s['m_R'],
                   image_time=s['image_time'], image_delta_t=s['vision_delta_t'] if s['m_V'] else None,
                   image_valid=s['m_V'], image_path=s['left_image_path'],
                   target_xyz=s['target_xyz'].tolist(), target_valid=s['target_valid'],
                   gt_box=s['gt_box_xyxy_px'].tolist(), gt_2d_valid=s['gt_2d_valid'])
        rows.append(row)
    batch = collate_multimodal_v2(samples)
    report = {'status': 'PASS', 'kind': 'real_sequence_data_smoke', 'sequence': args.sequence,
              'radar_history_s': ds.radar_history_s, 'max_image_gap_s': ds.max_image_gap_s,
              'time_offset_s': geometry.get('time_offset_s'), 'samples': rows,
              'batch_shapes': {k: list(batch[k].shape) for k in
                               ('points', 'delta_t', 'sensor_id', 'point_batch_index', 'point_counts', 'image_uint8')}}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
