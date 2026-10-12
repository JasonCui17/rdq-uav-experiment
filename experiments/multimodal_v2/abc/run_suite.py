#!/usr/bin/env python3
"""Generate auditable A/B/C configs; run independent gates, training, evaluation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('baseline', 'control', 'a_soft', 'a_head', 'a_bce', 'b_range', 'b_ohem', 'b_both', 'b_meta', 'c_query')


def ablation_config(base, name):
    cfg = copy.deepcopy(base)
    if name == 'baseline':
        return cfg
    a = {'enabled': True, 'quality_mode': 'none', 'negative_only': True,
         'negative_only_weight': .25, 'quality_sigma_m': 1., 'quality_loss_weight': 1.,
         'quality_negative_weight': .25}
    cfg['loss']['ablation'] = a
    if name in ('a_soft', 'a_head'):
        a['quality_mode'] = name.removeprefix('a_')
    if name == 'a_bce':
        a['binary_bce'] = True
    if name == 'a_head':
        cfg['model']['quality_head'] = {'enabled': True}
    if name in ('b_range', 'b_both'):
        a['range'] = {'enabled': True, 'edges_m': [30., 60.], 'weights': [1., 1.5, 2.5],
                      'origin_xyz_m': [0., 0., 0.]}
    if name in ('b_ohem', 'b_both'):
        a['ohem'] = {'enabled': True, 'topk': 200, 'easy_weight': .25, 'hard_weight': 2.}
    if name == 'b_meta':
        cfg['model']['head_metadata'] = True
    if name == 'c_query':
        cfg['model']['query_decoder'] = {'enabled': True, 'num_queries': 32, 'layers': 1,
                                         'heads': 4, 'coordinate_scale_m': 50.}
        a['query'] = {'coordinate_scale_m': 50., 'matching_class_weight': 1.,
                      'matching_xyz_weight': 1., 'background_weight': .1}
    return cfg


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=('prepare', 'gate', 'train', 'eval', 'all'), required=True)
    p.add_argument('--checkpoint', type=Path, help='A2 Lightning warmstart, required for prepare')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--base-config', type=Path, default=ROOT/'experiments/multimodal_v2/configs/b0_a2_learned_sbe32.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--split', choices=('validation_sub', 'heldout_test_sub'), default='validation_sub')
    args = p.parse_args()
    root = args.output.resolve()
    manifest_path = root/'manifest.json'
    if args.action in ('prepare', 'all'):
        if manifest_path.exists():
            raise FileExistsError('suite exists; use gate/train/eval with its saved manifest')
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError('provide an existing --checkpoint from the original A2 run')
        base = yaml.safe_load(args.base_config.read_text())
        radar_path = Path(base['initialization']['lidar_config'])
        if not radar_path.is_absolute(): radar_path = ROOT/radar_path
        radar = yaml.safe_load(radar_path.read_text())
        # Reject an already modified starting design; do not hide combined changes.
        if radar['model'].get('query_decoder', {}).get('enabled') or radar['model'].get('quality_head', {}).get('enabled') or radar['model']['voxel'].get('motion_centroid', {}).get('enabled') or radar['model'].get('head_metadata') or radar['loss'].get('ablation', {}).get('enabled'):
            raise ValueError('use an unmodified A2 base for this suite')
        if base['experiment']['stage'] != 'B0': raise ValueError('ABC suite requires a B0 base')
        root.mkdir(parents=True, exist_ok=True)
        if args.data_root: base['data']['root'] = str(args.data_root.resolve())
        base['initialization'].pop('b0_warmstart_checkpoint', None)
        base['initialization']['abc_warmstart_checkpoint'] = str(args.checkpoint.resolve())
        base['training']['epochs'] = args.epochs
        base['training']['radar_train_mode'] = 'full'
        base['data']['num_workers'] = args.num_workers
        # Primary endpoint is fixed before looking at any results.
        base['checkpoint'] = {'monitor': 'val/success_1m', 'mode': 'max', 'filename': 'best'}
        base['lightning']['precision'] = '32-true'
        runs = []
        for seed in args.seeds:
            for name in args.variants:
                folder = root/f'{name}_seed{seed}'
                folder.mkdir()
                rcfg = ablation_config(radar, name)
                rpath = folder/'radar.yaml'; rpath.write_text(yaml.safe_dump(rcfg, sort_keys=False))
                cfg = copy.deepcopy(base)
                cfg['initialization']['lidar_config'] = str(rpath)
                cfg['experiment'].update(name=f'abc_{name}', seed=seed, output_dir=str(folder/'train'))
                path = folder/'config.yaml'; path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                runs.append({'name': name, 'seed': seed, 'config': str(path), 'folder': str(folder),
                             'config_sha256': sha256(path), 'radar_sha256': sha256(rpath)})
        split = Path(base['data']['split_file']); split = split if split.is_absolute() else ROOT/split
        dependency_paths = [base['initialization']['candidate_config'], base['data']['camera_config'], base['data']['geometry_calibration']]
        dependencies = {}
        for value in dependency_paths:
            path = Path(value); path = path if path.is_absolute() else ROOT/path
            dependencies[str(path)] = sha256(path)
        manifest = {'dependencies_sha256': dependencies, 'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    'warmstart': str(args.checkpoint.resolve()), 'warmstart_sha256': sha256(args.checkpoint),
                    'split_file': str(split), 'split_sha256': sha256(split),
                    'primary_metric': 'success_1m', 'checkpoint_selection': base['checkpoint'],
                    'epochs': args.epochs, 'runs': runs}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest = json.loads(manifest_path.read_text())
    if sha256(Path(manifest['split_file'])) != manifest['split_sha256']:
        raise ValueError('split changed since suite preparation')
    for path, expected in manifest.get('dependencies_sha256', {}).items():
        if sha256(Path(path)) != expected: raise ValueError(f'dependency changed: {path}')
    if args.action in ('gate', 'train', 'all') and sha256(Path(manifest['warmstart'])) != manifest['warmstart_sha256']:
        raise ValueError('warmstart checkpoint changed')
    for run in manifest['runs']:
        if run['name'] not in args.variants or run['seed'] not in args.seeds: continue
        config, folder = Path(run['config']), Path(run['folder'])
        cfg = yaml.safe_load(config.read_text())
        if sha256(config) != run['config_sha256'] or sha256(Path(cfg['initialization']['lidar_config'])) != run['radar_sha256']:
            raise ValueError('configuration changed; prepare a new suite for a new experiment')
        cpu = args.device == 'cpu'
        devices = '1' if cpu else args.device.split(':')[-1] + ','
        train = [sys.executable, str(ROOT/'experiments/multimodal_v2/train.py'), '--config', str(config),
                 '--accelerator', 'cpu' if cpu else 'gpu', '--devices', devices]
        commands = []
        if args.action in ('gate', 'all'):
            commands.append(train + ['--output', str(folder/'gate'), '--max-updates','2','--train-limit','32','--val-limit','32'])
        if args.action in ('train', 'all'):
            cmd = train + ['--output', str(folder/'train')]
            if args.resume: cmd += ['--resume','auto']
            commands.append(cmd)
        if args.action in ('eval', 'all'):
            commands.append([sys.executable, str(ROOT/'experiments/multimodal_v2/evaluate.py'),
                             '--config',str(config),'--mode','B0','--checkpoint',str(folder/'train/checkpoints/best.ckpt'),
                             '--output',str(folder/args.split),'--device',args.device,'--split',args.split,
                             '--num-workers',str(args.num_workers),'--skip-coco'])
        for index, cmd in enumerate(commands):
            log = folder/f'{args.action}_{args.split}_{index}.log'
            if log.exists() and not args.resume:
                raise FileExistsError(f'{log}: use a new suite or explicit --resume')
            print('Running', run['name'], 'seed', run['seed'], 'log:', log, flush=True)
            with log.open('a') as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True,
                               env={k:v for k,v in os.environ.items() if not k.startswith('RDQ_')})
    print('Manifest:', manifest_path)


if __name__ == '__main__': main()
