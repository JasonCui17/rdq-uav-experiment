#!/usr/bin/env python3
"""Four fixed-budget motion ablations; reuse existing train/eval/compare tools."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('baseline', 'gate', 'velocity', 'joint')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=('prepare','gate','train','eval','compare'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-mode', choices=('heads','lora','full'), default='lora')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--split', choices=('validation_sub','heldout_test_sub'), default='validation_sub')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    output = a.output.resolve(); manifest_path = output/'manifest.json'
    if a.action == 'prepare':
        if output.exists():
            raise FileExistsError('output already exists; choose a new suite directory')
        if not a.checkpoint or not a.checkpoint.is_file() or not a.data_root or not a.data_root.is_dir():
            raise ValueError('prepare requires existing --checkpoint and --data-root')
        if a.epochs < 1 or a.num_workers < 0:
            raise ValueError('epochs must be positive, workers nonnegative')
        if 'baseline' not in a.variants:
            raise ValueError('include baseline in prepared suite for paired comparison')
        base = yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/b0_a2_learned_sbe32.yaml').read_text())
        radar = yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/radar_a2_gated_velocity.yaml').read_text())
        base['data'].update(root=str(a.data_root.resolve()), num_workers=a.num_workers)
        base['training'].update(epochs=a.epochs, radar_train_mode=a.train_mode)
        base['initialization']['abc_warmstart_checkpoint'] = str(a.checkpoint.resolve())
        base['checkpoint'] = dict(monitor='val/success_1m',mode='max',filename='best')
        base['efficiency'] = dict(enabled=True)
        dependencies = {str((ROOT/base['data'][k]).resolve()):sha(ROOT/base['data'][k])
                        for k in ('split_file','camera_config','geometry_calibration')}
        cp = ROOT/base['initialization']['candidate_config']; dependencies[str(cp.resolve())] = sha(cp)
        runs = []
        for variant in a.variants:
            folder = output/f'{variant}_seed{a.seed}'; folder.mkdir(parents=True)
            rcfg, cfg = deepcopy(radar), deepcopy(base)
            motion = rcfg['model']['voxel']['motion_centroid']
            motion.update(enabled=variant!='baseline',feature_gate=variant in ('gate','joint'),
                          velocity_guidance=variant in ('velocity','joint'))
            rcfg['model']['lora'] = dict(enabled=a.train_mode=='lora',rank=8,alpha=16.,dropout=0.,seed=a.seed)
            rp = folder/'radar.yaml'; rp.write_text(yaml.safe_dump(rcfg,sort_keys=False))
            cfg['initialization']['lidar_config'] = str(rp)
            cfg['experiment'].update(name='mtce_'+variant,seed=a.seed,output_dir=str(folder/'train'))
            config = folder/'config.yaml'; config.write_text(yaml.safe_dump(cfg,sort_keys=False))
            runs.append(dict(name=variant,seed=a.seed,folder=str(folder),config=str(config),
                             config_sha256=sha(config),radar_sha256=sha(rp)))
        sources = {str(path.resolve()):sha(path) for path in (ROOT/'src/rdq_uav/multimodal_v2').glob('*.py')}
        for path in (ROOT/'experiments/multimodal_v2/train.py',ROOT/'experiments/multimodal_v2/evaluate.py',Path(__file__)):
            sources[str(path.resolve())] = sha(path)
        manifest = dict(base_commit='01f2ac5427a92a944376fbba13a76ee2f4a73cac',
                        epochs=a.epochs,train_mode=a.train_mode,warmstart=str(a.checkpoint.resolve()),
                        warmstart_sha256=sha(a.checkpoint),dependencies_sha256=dependencies,
                        source_sha256=sources,runs=runs,
                        note='Fixed-voxel observation motion, not object velocity. No cross-voxel tracking.')
        manifest_path.write_text(json.dumps(manifest,indent=2))
        print('Prepared:',manifest_path); return
    manifest = json.loads(manifest_path.read_text())
    if a.action == 'compare':
        subprocess.run([sys.executable,str(ROOT/'experiments/multimodal_v2/abc/compare.py'),
                        '--suite',str(output),'--reference','baseline','--split',a.split],check=True)
        return
    for path, expected in {**manifest['dependencies_sha256'],**manifest['source_sha256']}.items():
        if sha(path) != expected:
            raise ValueError(f'dependency/source changed since prepare: {path}; create a new suite')
    if a.action in ('gate','train') and sha(manifest['warmstart']) != manifest['warmstart_sha256']:
        raise ValueError('warmstart changed')
    if a.device != 'cpu' and not (a.device.startswith('cuda:') and a.device[5:].isdigit()):
        raise ValueError('--device must be cpu or cuda:<index>')
    env = {k:v for k,v in os.environ.items() if not k.startswith('RDQ_')}
    env['PYTHONPATH'] = str(ROOT/'src')
    for run in manifest['runs']:
        if run['name'] not in a.variants: continue
        folder, config = Path(run['folder']), Path(run['config'])
        cfg = yaml.safe_load(config.read_text())
        if sha(config) != run['config_sha256'] or sha(cfg['initialization']['lidar_config']) != run['radar_sha256']:
            raise ValueError('saved configuration changed; create a new suite')
        log = folder/f'{a.action}_{a.split}.log'
        if log.exists() and not a.resume:
            raise FileExistsError(f'{log} exists; inspect it or explicitly use --resume')
        if a.action == 'eval':
            cmd = [sys.executable,str(ROOT/'experiments/multimodal_v2/evaluate.py'),
                   '--config',str(config),'--mode','B0','--checkpoint',str(folder/'train/checkpoints/best.ckpt'),
                   '--output',str(folder/a.split),'--split',a.split,'--device',a.device,
                   '--num-workers',str(a.num_workers),'--skip-coco']
        else:
            cmd = [sys.executable,str(ROOT/'experiments/multimodal_v2/train.py'),
                   '--config',str(config),'--output',str(folder/a.action),
                   '--accelerator','cpu' if a.device=='cpu' else 'gpu',
                   '--devices','1' if a.device=='cpu' else a.device[5:]+',']
            if a.action == 'gate':
                cmd += ['--max-updates','2','--train-limit','32','--val-limit','8']
            elif a.resume: cmd += ['--resume','auto']
        print('Running',run['name'],'log:',log,flush=True)
        with log.open('a') as handle:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print('Finished:',a.action,flush=True)


if __name__ == '__main__': main()
