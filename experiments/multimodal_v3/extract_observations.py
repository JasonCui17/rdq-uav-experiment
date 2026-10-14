#!/usr/bin/env python3
"""One frozen B0 inference pass. Cache all dense positions without GT filtering."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--limit',type=int,help='Smoke test only')
    args=p.parse_args();root=args.repo.resolve();os.chdir(root)
    sys.path[:0]=[str(root),str(root/'src')]
    def resolve(path):return path.resolve() if path.is_absolute() else (root/path).resolve()
    config,checkpoint,cache=map(resolve,(args.config,args.checkpoint,args.cache))
    if cache.exists():raise FileExistsError(f'Use a fresh cache directory: {cache}')
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader,Subset
    from rdq_uav.multimodal_v2.training import build_runtime,_stage_batch,synchronize_dino_device
    from rdq_uav.multimodal_v2.data import build_split_dataset,collate_multimodal_v2,prepare_model_batch
    from rdq_uav.multimodal_v2.geometry import load_left_projection_context
    from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
    from rdq_uav.runtime_paths import apply_runtime_path_overrides
    from experiments.multimodal_v2.evaluate import load_network
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    cfg=apply_runtime_path_overrides(yaml.safe_load(config.read_text()))
    reference_path=config.parent/'eval_best/per_query.jsonl'
    reference={r['sample_id']:r for r in map(json.loads,reference_path.read_text().splitlines())} if reference_path.is_file() else None
    if cfg['experiment']['stage']!='B0':raise ValueError('This first version only supports radar-only B0')
    if args.data_root:cfg['data']['root']=str(resolve(args.data_root))
    cfg['model'].update(interaction_enabled=False,vision_reads_radar=False,vision_scoring_enabled=False)
    radar_path=resolve(Path(cfg['initialization']['lidar_config']))
    if not radar_path.is_file():radar_path=config.parent/'radar.yaml'
    cfg['initialization']['lidar_config']=str(radar_path)
    runtime=build_runtime(cfg,root,torch.device('cpu'),initialize_from_baselines=False)
    info=load_network(runtime.model,checkpoint)
    runtime.model.to(device).eval().requires_grad_(False);synchronize_dino_device(runtime,device)
    runtime.projection_base=load_left_projection_context(runtime.camera_config,runtime.geometry_calibration,
        image_scale_xy=torch.ones((1,2),device=device),device=device)
    dataset=build_split_dataset(cfg,root,'validation_sub')
    # Explicitly process chronological observations. No labels used to select queries.
    order=sorted(range(len(dataset)),key=lambda i:(dataset.query_records[i].sequence_id,dataset.query_records[i].query_time))
    if args.limit is not None:order=order[:args.limit]
    if not order:raise ValueError('Empty dataset')
    loader=DataLoader(Subset(dataset,order),batch_size=1,shuffle=False,num_workers=args.num_workers,collate_fn=collate_multimodal_v2)
    cache.mkdir(parents=True);(cache/'frames').mkdir()
    (cache/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    selector=runtime.model.lidar.builder.selector
    manifest=dict(schema_version=1,complete=False,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                  checkpoint_info=info,config=str(config),config_sha256=sha(config),
                  lidar_config_sha256=sha(radar_path),candidate_config_sha256=sha(resolve(Path(cfg['initialization']['candidate_config']))),
                  split_file_sha256=sha(resolve(Path(cfg['data']['split_file']))),
                  data_root=cfg['data']['root'],split='validation_sub',queries=len(order),
                  torch=torch.__version__,precision='fp32',
                  code_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
                  selector={'raw_topk':selector.raw_topk,'pre_nms_topk':selector.pre,'final_topk':selector.final,'nms_radius_m':selector.radius},
                  protocol='existing GT-timestamp query schedule, not continuous raw sensor tracking',
                  independent_reference_checked=reference is not None,
                  smoke_test=args.limit is not None)
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2))
    started=time.monotonic()
    with (cache/'index.jsonl').open('w') as index_file,torch.no_grad():
        for idx,batch in enumerate(loader):
            batch=_stage_batch(batch,'B0')
            moved,images,masks,projection,targets,_=prepare_model_batch(batch,runtime.dino_detector,runtime.projection_base,device)
            with torch.autocast(device_type=device.type,enabled=False):output=runtime.model(moved,images,masks,projection)
            if output.vision_candidates.n or bool((output.hypothesis_type!=1).any()):raise RuntimeError('Unexpected non-R hypothesis')
            torch.testing.assert_close(output.score_3d_before,output.score_3d_after,rtol=0,atol=0)
            raw=output.diagnostics.get('lidar_raw');radar=output.radar_candidates
            def cpu(x):return x.detach().cpu().numpy()
            if raw is not None:
                if raw.get('prediction_kind','dense')!='dense':raise ValueError('Expected dense voxel predictions')
                dense_xyz=cpu(raw['pred_xyz'].float());dense_score=cpu(CandidateSelector.ranking_scores(raw))
                dense_rank=cpu(CandidateSelector.ranking_scores(raw) if 'quality_logits' in raw else raw['logits'].float())
                tokens=cpu(raw['source_token_id'])
            else:dense_xyz=np.empty((0,3),np.float32);dense_score=dense_rank=np.empty(0,np.float32);tokens=np.empty(0,np.int64)
            top=output.top3d_indices(1)[0]
            arrays=dict(dense_xyz=dense_xyz,dense_score=dense_score,dense_rank=dense_rank,dense_token_id=tokens,
                        global_xyz=cpu(radar.xyz_m.float()),global_score=cpu(radar.score.float()),global_token_id=cpu(radar.source_index),
                        baseline_xyz=cpu(output.xyz_m[top].float()),baseline_score=cpu(output.score_3d_after[top].float()),
                        baseline_token_id=cpu(output.radar_source_index[top]),gt_xyz=cpu(targets.xyz_m[0].float()),
                        has_gt=np.asarray(bool(targets.has_xyz[0])))
            for key in ('dense_xyz','dense_score','dense_rank','global_xyz','global_score','baseline_xyz'):
                if not np.isfinite(arrays[key]).all():raise ValueError(f'nonfinite {key}')
            filename=f'frames/{idx:06d}.npz';np.savez_compressed(cache/filename,**arrays)
            row=dict(sequence_id=batch['sequence_id'][0],sample_id=batch['sample_id'][0],query_time=float(batch['query_time'][0]),file=filename)
            if reference is not None:
                previous=reference.get(row['sample_id'])
                if previous is None:raise ValueError('Query absent from existing evaluation')
                error=float(np.linalg.norm(arrays['baseline_xyz'][0]-arrays['gt_xyz'])) if len(top) else None
                old=previous['final_error']
                if (error is None)!=(old is None) or (error is not None and abs(error-old)>max(1e-4,1e-5*abs(old))):
                    raise RuntimeError(f'Independent reference mismatch {row["sample_id"]}: {error} vs {old}')
            index_file.write(json.dumps(row)+'\n');index_file.flush()
            if (idx+1)%20==0 or idx+1==len(order):
                elapsed=time.monotonic()-started
                print(f'extract {idx+1}/{len(order)} elapsed={elapsed/60:.1f}min ETA={elapsed/(idx+1)*(len(order)-idx-1)/60:.1f}min',flush=True)
    manifest['complete']=True;manifest['extraction_seconds']=time.monotonic()-started
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'CACHE READY: {cache}',flush=True)

if __name__=='__main__':main()
