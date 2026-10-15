#!/usr/bin/env python3
"""Evaluate bounded target tracking on the same immutable candidate cache."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rdq_uav.multimodal_v3.candidate_adapter import from_cache
from rdq_uav.multimodal_v3.target_tracker import TargetConfig, TargetTracker
from rdq_uav.multimodal_v3.tracking import TrackerConfig
from evaluate_tracking import metrics, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lag-s', type=float, default=0., help='Bounded future lookahead; zero is causal')
    args = p.parse_args()
    if not np.isfinite(args.lag_s) or args.lag_s < 0:
        p.error('--lag-s must be finite and nonnegative')
    config = json.loads(args.config.read_text())
    cfg = TargetConfig(**{**config, 'filter': TrackerConfig(**config.get('filter', {}))})
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Incomplete cache')
    frames = [json.loads(line) for line in (args.cache / 'index.jsonl').open()]
    if len(frames) != manifest['queries'] or len({f['sample_id'] for f in frames}) != len(frames):
        raise ValueError('Cache index mismatch')
    frames.sort(key=lambda f: (f['sequence_id'], f['query_time']))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trackers, rows, snapshots, segments = {}, [], {}, {}
    for f in frames:
        with np.load(args.cache / f['file'], allow_pickle=False) as data:
            frame = from_cache(data, manifest['schema_version'])
            tracker = trackers.setdefault(f['sequence_id'], TargetTracker(cfg))
            result = tracker.step(f['query_time'], frame)
            seq = f['sequence_id']
            if result['event'] == 'birth':
                segments[seq] = segments.get(seq, 0) + 1
            snapshots.setdefault(seq, []).append(None if tracker.kf is None else
                (tracker.kf.x.copy(), tracker.kf.P.copy(), segments[seq]))
            # GT accessed only after the inference step.
            gt, has_gt = data['gt_xyz'].tolist(), bool(data['has_gt'])
        xyz = result['xyz']
        error = float(np.linalg.norm(np.array(xyz) - gt)) if has_gt and xyz is not None else None
        baseline = frame.baseline.xyz[0].tolist() if len(frame.baseline.xyz) else None
        rows.append({**f, **result, 'gt_xyz': gt, 'has_gt': has_gt, 'target_error_m': error,
                     'independent_top1_error_m': float(np.linalg.norm(np.array(baseline) - gt))
                     if has_gt and baseline is not None else None})
        if len(rows) % 400 == 0:
            print(f'target {len(rows)}/{len(frames)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    summary = {m: metrics(rows, m) for m in ('independent_top1', 'target')}
    if args.lag_s > 0:
        from rdq_uav.multimodal_v3.target_smoothing import fixed_lag_outputs
        for seq in trackers:
            seq_rows = [r for r in rows if r['sequence_id'] == seq]
            outputs = fixed_lag_outputs([r['query_time'] for r in seq_rows], snapshots[seq], cfg.filter, args.lag_s)
            for r, estimate in zip(seq_rows, outputs):
                r['lagged_xyz'] = estimate['xyz']
                r['lagged_event'] = estimate['event']
                r['lookahead_s'] = estimate['lookahead_s']
                r['lagged_error_m'] = (float(np.linalg.norm(np.array(estimate['xyz']) - r['gt_xyz']))
                                       if r['has_gt'] and estimate['xyz'] is not None else None)
        summary['lagged'] = metrics(rows, 'lagged')
        summary['lagged_sequences'] = {s: metrics([r for r in rows if r['sequence_id'] == s], 'lagged') for s in trackers}
        summary['lagged_events'] = dict(Counter(r['lagged_event'] for r in rows))
    summary['sequences'] = {s: metrics([r for r in rows if r['sequence_id'] == s], 'target') for s in trackers}
    summary['events'] = dict(Counter(r['event'] for r in rows))
    summary['protocol'] = {'target_causal': True, 'GT_used_for_inference': False,
                           'split': manifest['split'], 'schedule': manifest['protocol'],
                           'validation_tuned': True, 'held_out_test': False}
    summary['protocol']['lagged'] = {'causal': args.lag_s == 0, 'max_lookahead_s': args.lag_s,
                                     'batch_evaluation': True,
                                     'startup_completion': 'bounded backward extrapolation, not radar detection'}
    summary['runtime_s'] = time.monotonic() - started
    summary['reproducibility'] = {
        'argv': sys.argv,
        'python': sys.version,
        'numpy': np.__version__,
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(),
                                       ROOT / 'src/rdq_uav/multimodal_v3/target_tracker.py',
                                       ROOT / 'src/rdq_uav/multimodal_v3/target_smoothing.py',
                                       ROOT / 'src/rdq_uav/multimodal_v3/tracking.py',
                                       ROOT / 'src/rdq_uav/multimodal_v3/candidate_adapter.py',
                                       ROOT / 'experiments/multimodal_v3/evaluate_tracking.py')},
        'cache_index_sha256': hashlib.sha256((args.cache / 'index.jsonl').read_bytes()).hexdigest()}
    (args.output / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    (args.output / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'per_query.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_csv(args.output / 'per_query.csv', rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
