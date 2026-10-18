#!/usr/bin/env python3
"""Manual GPU ablation: full DINO, visual crop control, track-guided crops."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from rdq_uav.multimodal_v3.pipeline.prepare_image_guidance import image_rows, read_jsonl
from rdq_uav.multimodal_v3.detection_metrics import image_detection_metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Trained B1 or B4 Lightning checkpoint')
    p.add_argument('--crop-checkpoint', type=Path,
                   help='Optional separate trained detector for crop passes; full view uses --checkpoint')
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--priors', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/final/image_guidance.json')
    p.add_argument('--b4-predictions', type=Path, help='Optional original B4 evaluate.py per_query.jsonl')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-images', type=int, help='Pilot only, explicitly marked partial')
    p.add_argument('--all-available-images', action='store_true', help='Infer every query-associated available image, including unlabeled frames')
    p.add_argument('--visual-refinement', action='store_true', help='Also evaluate single-UAV visual motion refinement from causal predictions')
    args = p.parse_args()
    if args.all_available_images and args.b4_predictions:
        p.error('Legacy B4 comparison supports labeled-image evaluation only')
    if args.max_images is not None and args.max_images < 1:
        p.error('--max-images must be positive')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; execute this experiment on the GPU server')
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    config = json.loads(args.config.read_text())
    if config['topk'] != 100:
        raise ValueError('This ablation fixes candidate budgets to 50/100')
    source_wh = yaml.safe_load((ROOT / config['camera_config']).read_text())['cameras']['left']['resolution']
    samples = image_rows(read_jsonl(args.samples), source_wh, args.all_available_images)
    prior_rows = read_jsonl(args.priors)
    priors = {r['image_path']: r for r in prior_rows}
    if not samples or len(priors) != len(prior_rows):
        raise ValueError('Empty labeled image set or duplicate priors')
    if any(r['image_source_wh'] != source_wh or r['crop_side_px'] != config['crop_side_px'] for r in prior_rows):
        raise ValueError('Prepared prior geometry differs from evaluation config')
    if len(priors) != len(samples) or set(priors) != {r['image_path'] for r in samples}:
        raise ValueError('Prior image set must exactly match the complete labeled image set')
    total_images = len(samples)
    total_labeled_images = sum(bool(r.get('has_gt2d')) for r in samples)
    samples = samples[:args.max_images] if args.max_images is not None else samples
    # Reuse the B1 runtime only to build/load the identical frozen DINO branch.
    # B4 adapters and joint scores are deliberately not part of these ablations.
    from rdq_uav.multimodal_v3.modeling.training import build_runtime, _load_trained_branch, synchronize_dino_device
    from rdq_uav.multimodal_v3.image_detection import infer_image_variants
    cfg = yaml.safe_load((ROOT / 'configs/final/vision_coco.yaml').read_text())
    from rdq_uav.multimodal_v3.runtime_paths import apply_runtime_path_overrides
    cfg = apply_runtime_path_overrides(cfg)
    # The trained checkpoint is loaded strictly below; do not require an
    # unrelated COCO initialization asset just to construct the detector.
    cfg['initialization'].pop('dino_checkpoint', None)
    runtime = build_runtime(cfg, ROOT, device, initialize_from_baselines=False)
    _load_trained_branch(runtime.dino_detector, args.checkpoint, 'network.vision.dino.detector.')
    synchronize_dino_device(runtime, device)
    runtime.model.eval().requires_grad_(False)
    crop_runtime = None
    if args.crop_checkpoint is not None:
        crop_runtime = build_runtime(copy.deepcopy(cfg), ROOT, device, initialize_from_baselines=False)
        _load_trained_branch(crop_runtime.dino_detector, args.crop_checkpoint,
                             'network.vision.dino.detector.')
        synchronize_dino_device(crop_runtime, device)
        crop_runtime.model.eval().requires_grad_(False)
    args.output.mkdir(parents=True)
    (args.output / 'prediction_status.json').write_text(json.dumps({
        'complete': False, 'requested_images': len(samples), 'full_image_set': total_images,
        'partial_evaluation': len(samples) != total_images}, indent=2))
    grouped, started = {}, time.monotonic()
    with (args.output / 'prediction_cache.jsonl').open('w') as cache, \
            (args.output / 'raw_view_cache.jsonl').open('w') as raw_cache:
        for index, row in enumerate(samples):
            image_path = args.data_root / row['image_path']
            with Image.open(image_path) as handle:
                image = handle.convert('RGB')
            if image.width < source_wh[0] or image.height < source_wh[1]:
                raise ValueError('Image smaller than calibrated left view')
            image = image.crop((0, 0, *source_wh))
            prior = priors[row['image_path']]
            if prior['sequence_id'] != row['sequence_id'] or prior['image_time'] != row['image_time']:
                raise ValueError('Prior image identity/time mismatch')
            raw_views = {}
            outputs = infer_image_variants(runtime.model.vision, image, prior['priors'], device, config,
                crop_vision=None if crop_runtime is None else crop_runtime.model.vision,
                raw_views=raw_views)
            raw_cache.write(json.dumps({'image_path': row['image_path'], 'views': raw_views},
                                       allow_nan=False) + '\n')
            # Attach annotations only after all image/geometry-only predictions.
            base = {k: row[k] for k in ('sample_id', 'sequence_id', 'query_time', 'image_time',
                                        'image_path', 'image_source_wh', 'has_gt2d', 'gt_box_xyxy_px')}
            for name, pred in outputs.items():
                grouped.setdefault(name, []).append({**base, 'pred_boxes_xyxy_px': pred['boxes'],
                                                     'pred_box_scores': pred['scores']})
            cache.write(json.dumps({'image': base, 'predictions': outputs}) + '\n'); cache.flush()
            if (index + 1) % 100 == 0 or index + 1 == len(samples):
                print(f'image-guidance {index+1}/{len(samples)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    if args.visual_refinement:
        from rdq_uav.multimodal_v3.visual_refinement import refine_visual_tracks, refine_visual_detections
        source_rows = grouped['track_crop_causal']
        frames = [{k: row[k] for k in ('sequence_id', 'image_time', 'image_path')} for row in source_rows]
        for frame, row in zip(frames, source_rows):
            frame.update(boxes=row['pred_boxes_xyxy_px'], scores=row['pred_box_scores'],
                         prior=priors[row['image_path']]['priors']['causal'])
        for name, delay, support in (('track_visual_top1', 0., 10**9),
                                     ('track_visual_causal', 0., 3), ('track_visual_lag05', .5, 3),
                                     ('track_visual_all_causal', 0., 3), ('track_visual_all_lag05', .5, 3)):
            refine = refine_visual_detections if name.startswith('track_visual_all_') else refine_visual_tracks
            options = {'nms_iou': config['nms_iou'], 'topk': config['topk']} if name.startswith('track_visual_all_') else {}
            refined = refine(frames, lookahead_s=delay, min_support=support, **options)
            grouped[name] = [{**row, 'pred_boxes_xyxy_px': pred['boxes'], 'pred_box_scores': pred['scores'],
                'visual_refinement': {k: pred[k] for k in ('support', 'image_lookahead_s')}} for row, pred in zip(source_rows, refined)]
    # Persist costly GPU predictions before any CPU metric work/comparison;
    # summarize_image_guidance.py can recover metrics without rerunning DINO.
    for name, rows in grouped.items():
        (args.output / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (args.output / 'prediction_status.json').write_text(json.dumps({
        'complete': True, 'predicted_images': len(samples), 'full_image_set': total_images,
        'partial_evaluation': len(samples) != total_images}, indent=2))
    if args.b4_predictions:
        original = read_jsonl(args.b4_predictions)
        for row in original:
            if row.get('image_path'):
                row['image_path'] = row['sequence_id'] + '/Image/' + Path(row['image_path']).name
        from rdq_uav.multimodal_v3.modeling.paper_metrics import deduplicate_images
        original = {r['image_path']: r for r in deduplicate_images(original)}
        if any(row['image_path'] not in original for row in samples):
            raise ValueError('B4 comparison is missing evaluated images')
        for row in samples:
            baseline = original[row['image_path']]
            if (baseline['image_source_wh'] != row['image_source_wh'] or
                    baseline['gt_box_xyxy_px'] != row['gt_box_xyxy_px']):
                raise ValueError('B4 and image-guidance annotations/source dimensions differ')
        grouped['b4_joint'] = [original[row['image_path']] for row in samples]
        (args.output / 'b4_joint.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in grouped['b4_joint']))
    method_metrics = {}
    for name, rows in grouped.items():
        metric = image_detection_metrics(rows, config['confidence_threshold'], args.output / name / 'coco')
        metric['prediction_scope'] = ('original_B4_final_V_and_RV' if name == 'b4_joint' else
                                      'frozen_visual_detector_full_and_or_crop_predictions_in_source_left_pixels')
        if name.startswith('track_visual_'):
            metric['prediction_scope'] = ('radar_conditioned_visual_motion_refinement; other_detections_retained; original_NMS_top100'
                if name.startswith('track_visual_all_') else 'single_UAV_track_conditioned_visual_motion_refinement; max_one_output_per_sensor_image')
            metric['max_image_lookahead_s'] = max(r['visual_refinement']['image_lookahead_s'] for r in rows)
        method_metrics[name] = metric
    summary = {'methods': method_metrics,
               'protocol': {'unique_labeled_images': sum(bool(r.get('has_gt2d')) for r in samples), 'complete_image_set': total_images,
                            'unique_sensor_images': len(samples), 'all_available_images': args.all_available_images,
                            'complete_labeled_image_set': total_labeled_images,
                            'visual_refinement': args.visual_refinement,
                            'visual_refinement_scope': 'one_target_refined; other_detections_retained_in_all_arms; matched_max_one_controls; causal_input; optional_0.5s_lookahead' if args.visual_refinement else None,
                            'partial_evaluation': len(samples) != total_images,
                            'precision': 'fp32', 'GT_used_for_inference': False,
                            'new_training': False,
                            'same_detector_weights_for_all_crop_arms': True,
                            'same_detector_weights_for_full_and_crop': args.crop_checkpoint is None or
                                sha(args.crop_checkpoint) == sha(args.checkpoint),
                            'crop_weights': str(args.crop_checkpoint or args.checkpoint),
                            'raw_views_saved_before_cross_view_NMS': True,
                            'all_images_retained_when_prior_missing': True,
                            'lag05_image_delay': 'recorded in input priors; may exceed 0.5 due to timestamp alignment',
                            'selection_objective': 'image mAP50_95, mAP50, Precision and Recall; not joint_success'},
               'config': config, 'runtime_s': time.monotonic() - started,
               'argv': sys.argv, 'torch': torch.__version__,
               'input_sha256': {str(path): sha(path) for path in (args.checkpoint, args.samples, args.priors, args.config)},
               'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in (
                   Path(__file__).resolve(), ROOT / 'src/rdq_uav/multimodal_v3/image_guidance.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/image_detection.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/visual_refinement.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/detection_metrics.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/modeling/dino_adapter.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/modeling/training.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/modeling/candidate_builders.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/modeling/data.py',
                   ROOT / 'src/rdq_uav/multimodal_v3/modeling/paper_metrics.py')}}
    if args.b4_predictions:
        summary['input_sha256'][str(args.b4_predictions)] = sha(args.b4_predictions)
    if args.crop_checkpoint:
        summary['input_sha256'][str(args.crop_checkpoint)] = sha(args.crop_checkpoint)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        summary['gpu_memory'] = {
            'name': torch.cuda.get_device_name(device),
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 1024**3,
            'free_at_end_gib': free_bytes / 1024**3, 'total_gib': total_bytes / 1024**3,
            'scope': 'model construction and sequential inference; no backward or optimizer'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
