#!/usr/bin/env python3
"""Fine-tune the image detector from trained B1/B4 weights using the V2 trainer.

Usage: train_image.py --warmstart TRAINED.ckpt [V2 train.py options]
This trains the image detector; --crop-training mixes full images and positive
training crops. --quality-focal optionally teaches classification scores box IoU. The track-guided inference policy itself is not learned.
"""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(path))


def _keep_backbone_eval(swin, inputs):
    # The adapter calls Swin stages directly, bypassing backbone.forward().
    swin.backbone.eval()


def build_image_runtime(config, root, device, *, initialize_from_baselines=True, warmstart=None,
                        freeze_backbone=False, quality_focal=False, matchability_aware=False):
    from rdq_uav.multimodal_v3.modeling.training import build_runtime, _load_trained_branch
    from rdq_uav.multimodal_v3.runtime_paths import resolve_project_path
    if config['experiment']['stage'] != 'B1':
        raise ValueError('MMV3 image fine-tuning requires stage B1')
    if initialize_from_baselines and warmstart is None:
        raise ValueError('Provide --warmstart with a trained B1 or B4 checkpoint')
    path = None if warmstart is None else resolve_project_path(warmstart, root)
    if initialize_from_baselines and not path.is_file():
        raise FileNotFoundError(path)
    # Adapt to one UAV class first; then strictly load its trained head.
    config['initialization'].pop('dino_checkpoint', None)
    runtime = build_runtime(config, root, device, initialize_from_baselines=False)
    if freeze_backbone:
        runtime.dino_detector.backbone.requires_grad_(False)
        runtime.dino_detector.backbone.eval()
        runtime.model.vision.dino.swin.register_forward_pre_hook(_keep_backbone_eval)
    config.setdefault('training', {})['image_freeze_backbone'] = bool(freeze_backbone)
    if quality_focal and matchability_aware:
        raise ValueError('Choose one classification loss')
    if quality_focal or matchability_aware:
        from rdq_uav.multimodal_v3.quality_focal import enable_quality_focal, enable_matchability_aware
        (enable_matchability_aware if matchability_aware else enable_quality_focal)(runtime.dino_detector.criterion)
        config['training']['image_classification_policy'] = {
            'loss': 'matchability_aware' if matchability_aware else 'quality_focal',
            'beta': 2, 'target': 'detached_matched_box_IoU_squared' if matchability_aware else 'detached_matched_box_IoU',
            'applied_to': 'decoder_aux_encoder_and_denoising', 'matching_changed': False,
            'source_sha256': hashlib.sha256((root/'src/rdq_uav/multimodal_v3/quality_focal.py').read_bytes()).hexdigest()}
        print('Image classification policy:', config['training']['image_classification_policy'], flush=True)
    if initialize_from_baselines:
        _load_trained_branch(runtime.dino_detector, path, 'network.vision.dino.detector.')
        _load_trained_branch(runtime.model.vision.builder.feature_proj, path,
                             'network.vision.builder.feature_proj.')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        config['initialization']['image_warmstart'] = {
            'checkpoint': str(path), 'sha256': digest.hexdigest(), 'strict': True,
            'optimizer_and_epoch_restored': False}
        print('Loaded trained image branch:', config['initialization']['image_warmstart'], flush=True)
    return runtime


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--warmstart', type=Path)
    p.add_argument('--crop-training', action='store_true',
                   help='B1-only mixed full/crop training; keep validation on full images')
    p.add_argument('--freeze-backbone', action='store_true',
                   help='Freeze Swin; still train neck, transformer and detection heads')
    loss_flags = p.add_mutually_exclusive_group()
    loss_flags.add_argument('--matchability-aware', action='store_true',
                   help='Opt-in DEIM MAL classification loss only; unchanged DINO matcher')
    loss_flags.add_argument('--quality-focal', action='store_true',
                   help='Opt-in single-class IoU quality focal classification; unchanged matcher and box losses')
    args, rest = p.parse_known_args()
    if '--help' in rest or '-h' in rest:
        print(__doc__)
    from rdq_uav.multimodal_v3.training import train
    from rdq_uav.multimodal_v3.crop_training import wrap_crop_training
    sys.argv = [sys.argv[0], '--config', str(ROOT/'configs/final/vision_train.yaml'), *rest]
    train.main(runtime_builder=partial(build_image_runtime, warmstart=args.warmstart,
                                      freeze_backbone=args.freeze_backbone, quality_focal=args.quality_focal,
                                      matchability_aware=args.matchability_aware),
               train_dataset_transform=wrap_crop_training if args.crop_training else None)


if __name__ == '__main__':
    main()
