#!/usr/bin/env python3
"""CPU replay of visual-supported births; reuse the frozen target evaluator.

Provided old labeled-image caches are exploratory subsets. For full sensor
coverage use extract_observations.py --vision-checkpoint, which never filters
images by their labels. GT is only scored by the original evaluator after step.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(Path(__file__).resolve().parent)]
import torch
from rdq_uav.multimodal_v3.modeling.geometry import load_left_projection_context
from rdq_uav.multimodal_v3.visual_birth import VisualBirthTracker, VisualHint
from rdq_uav.multimodal_v3.pipeline import evaluate_target
from rdq_uav.multimodal_v3.pipeline.extract_observations import sha


def read(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def build_lookup(frames, samples, raw_views):
    metadata = {r['sample_id']: r for r in samples}
    views = {r['image_path']: r for r in raw_views}
    if len(metadata) != len(samples) or len(views) != len(raw_views):
        raise ValueError('Duplicate sample/image identities')
    if {r['sample_id'] for r in frames} != set(metadata):
        raise ValueError('Sample set differs from radar query set')
    hints, missing = {}, 0
    for frame in frames:
        sample = metadata[frame['sample_id']]
        t = frame['query_time']
        if sample['sequence_id'] != frame['sequence_id'] or sample['query_time'] != t or t in hints:
            raise ValueError('Sequence/time identity mismatch or duplicate time')
        hints[t] = None
        if not sample.get('has_image'):
            continue
        path = sample['sequence_id']+'/Image/'+Path(sample['image_path']).name
        raw = views.get(path)
        if raw is None:
            missing += 1
            continue
        image_time = float(sample['image_time'])
        if image_time > t or float(Path(path).stem) != image_time or raw.get('image_time', image_time) != image_time:
            raise ValueError('Invalid/future image timestamp')
        full = raw['views']['full']
        if len(full['boxes']) != len(full['scores']):
            raise ValueError('Visual box/score lengths differ')
        if full['scores']:
            # Never use annotation or an IoU oracle to choose the visual hint.
            best = max(range(len(full['scores'])), key=lambda i: full['scores'][i])
            hints[t] = VisualHint(image_time, tuple(full['boxes'][best]), full['scores'][best])
    return hints, missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--visual-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/final/tracker.json')
    args = p.parse_args()
    frames = read(args.cache/'index.jsonl')
    hints, missing = build_lookup(frames, read(args.samples), read(args.visual_cache))
    manifest = json.loads((args.cache/'manifest.json').read_text())
    scope = (manifest.get('vision_cache_scope') if args.visual_cache.parent.resolve() == args.cache.resolve()
             else 'provided_subset; may_exclude_unlabeled_images')
    if scope == 'all_available_images_no_label_filter' and missing:
        raise ValueError('Complete visual cache is missing available sensor images')
    camera = ROOT/'configs/calibration/mmaud_v1_omni.yaml'
    geometry = ROOT/'calibration/official_left_p4_current_geometry.json'
    projection = load_left_projection_context(camera, geometry, image_scale_xy=torch.ones(1, 2))
    # Injection is process-local, restored afterward. Frozen sources/hashes stay
    # intact; record the actual policy implementation in the extended summary.
    previous_factory, previous_argv = evaluate_target.TargetTracker, sys.argv
    evaluate_target.TargetTracker = partial(VisualBirthTracker, projection=projection,
                                            visual_lookup=hints.get)
    previous_metrics = evaluate_target.metrics
    def inference_safe_metrics(rows, method):
        if not any(r['has_gt'] for r in rows):
            return {'status':'not_scored_no_GT_in_inference_cache','queries':0,'outputs':None if method=='independent_top1' else sum(r.get('xyz' if method=='target' else 'lagged_xyz') is not None for r in rows),'rmse_3d_m':None}
        return previous_metrics(rows,method)
    evaluate_target.metrics=inference_safe_metrics
    try:
        sys.argv = [str(Path(evaluate_target.__file__)), '--cache', str(args.cache),
            '--config', str(args.config), '--lag-s', '0.5', '--output', str(args.output)]
        evaluate_target.main()
    finally:
        evaluate_target.TargetTracker, sys.argv = previous_factory, previous_argv
        evaluate_target.metrics=previous_metrics
    path = args.output/'summary.json'
    result = json.loads(path.read_text())
    result['frozen_evaluator_protocol'] = result['protocol'].copy()
    result['protocol'].update(policy='camera_supported_birth; original_radar_continuation',
        GT_used_for_birth=False, held_out_test=manifest['split'] == 'heldout_test_sub',
        post_test_exploration=manifest['split'] == 'heldout_test_sub',
        diagnostic_cache_subset=bool(manifest.get('diagnostic_subset', False)),
        missing_cached_image_queries=missing,
        camera_cache_scope=scope)
    result['visual_birth_reproducibility'] = {'argv': previous_argv,
        'input_sha256': {str(p): sha(p) for p in (args.samples, args.visual_cache, camera, geometry)},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__).resolve(), ROOT/'src/rdq_uav/multimodal_v3/visual_birth.py')}}
    path.write_text(json.dumps(result, indent=2))
    print('VISUAL BIRTH RESULTS:', path, flush=True)
    print(json.dumps({k: result[k] for k in ('target', 'lagged', 'protocol')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
