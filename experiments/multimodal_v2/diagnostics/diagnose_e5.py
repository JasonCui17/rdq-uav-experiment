#!/usr/bin/env python3
"""Full E5 Best/Last candidate diagnosis; inference only, always FP32."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
for path in (ROOT,ROOT/'src'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
from rdq_uav.runtime_paths import apply_runtime_path_overrides,resolve_project_path

def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--best',type=Path,required=True);p.add_argument('--last',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--expected-best-success-1m',type=float);p.add_argument('--expected-last-success-1m',type=float);p.add_argument('--expected-tolerance',type=float,default=.01);return p.parse_args()
def resolve(x):return resolve_project_path(x,ROOT)
def load(module,path):
    payload=torch.load(path,map_location='cpu',weights_only=False);state=payload['state_dict'];prefix='network.'
    network={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)};module.network.load_state_dict(network,strict=True)
    return {'epoch':payload.get('epoch'),'global_step':payload.get('global_step')}
def summary(rows,key):
    count=len(rows);values=np.asarray([r[key] if r[key] is not None else np.inf for r in rows]);finite=values[np.isfinite(values)]
    result={'n_gt':count,'n_output':int(len(finite)),'coverage':float(len(finite)/count) if count else 0.}
    for radius in (.5,1.,2.):result[f'success_{radius:g}m']=float(np.mean(values<=radius)) if count else 0.
    for name,fn in [('mean_error_m',np.mean),('median_error_m',np.median),('p90_error_m',lambda x:np.percentile(x,90))]:result[name]=float(fn(finite)) if len(finite) else float('inf')
    return result
@torch.no_grad()
def run(name,path,cfg,train_lidar,dataset,device,workers):
    from rdq_uav.multimodal_v1.lightning_system import MultimodalV1LightningModule
    from tools.train_multimodal_v1_full import build_runtime,collate_e5,prepare_batch
    runtime=build_runtime(cfg,train_lidar,torch.device('cpu'));module=MultimodalV1LightningModule(runtime,cfg);info=load(module,path)
    module.to(device).eval();module._synchronize_external_runtime_device();module._refresh_projection_context()
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=workers,collate_fn=collate_e5)
    rows=[];types=Counter();wrong_visual=0;radar_exists_but_final_fails=0
    for batch in loader:
        lidar,images,masks,context,targets,_=prepare_batch(batch,runtime,device)
        with torch.autocast(device_type=device.type,enabled=False):out=runtime.model(lidar,images,context,image_padding_mask=masks,return_aux=True,return_diagnostics=False)
        if not bool(targets.gt_3d_valid[0]):continue
        gt=targets.gt_xyz[0].float();radar=out.aux['radar_candidates'];rids=torch.nonzero(radar.batch_index==0).flatten();rids=rids[torch.argsort(radar.score[rids].float(),descending=True,stable=True)]
        rd=torch.linalg.vector_norm(radar.xyz[rids].float()-gt,dim=1).cpu().tolist() if len(rids) else []
        ids=torch.nonzero(out.batch_index==0).flatten();ids=ids[torch.argsort(out.fused_score[ids].float(),descending=True,stable=True)]
        fd=torch.linalg.vector_norm(out.xyz[ids].float()-gt,dim=1).cpu().tolist() if len(ids) else []
        top_type=None if not len(ids) else int(out.hypothesis_type[ids[0]]);types[top_type]+=1
        r_oracle=min(rd) if rd else None;ferr=fd[0] if fd else None
        failure=r_oracle is not None and r_oracle<=1 and (ferr is None or ferr>1)
        radar_exists_but_final_fails+=int(failure);wrong_visual+=int(failure and top_type==2)
        rows.append({'sample_id':batch['sample_id'][0],'sequence_id':batch['sequence_id'][0],'radar_top1_error':rd[0] if rd else None,'radar_oracle_error':r_oracle,'final_error':ferr,'top_type':top_type,'radar_distances':rd,'final_distances':fd})
    return rows,{'checkpoint':name,'path':str(path),'checkpoint_info':info,'queries':len(rows),'final':summary(rows,'final_error'),'radar_top1_pre_fusion':summary(rows,'radar_top1_error'),'radar_oracle_pre_fusion':summary(rows,'radar_oracle_error'),'top_type_counts':{'RV':types[0],'R':types[1],'V':types[2]},'failure_diagnostics':{'radar_candidate_within_1m_but_final_failed':radar_exists_but_final_fails,'of_which_top1_v_only':wrong_visual}}
def main():
    args=parse_args();cfg=apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()));cfg['validation']['precision']='fp32';device=torch.device(args.device)
    from tools.train_multimodal_v1_lightning import build_datasets
    train_lidar,_,dataset=build_datasets(cfg,1,args.limit);best_rows,best=run('best',resolve(args.best),cfg,train_lidar,dataset,device,args.num_workers);last_rows,last=run('last',resolve(args.last),cfg,train_lidar,dataset,device,args.num_workers)
    paired=sum(1 for a,b in zip(best_rows,last_rows) if a['final_error'] is not None and a['final_error']<=1 and (b['final_error'] is None or b['final_error']>1))
    report={'precision':'fp32','validation_queries':len(dataset),'best':best,'last':last,'paired_best_success_last_failure_1m':paired}
    if args.limit is None:
        for expected,actual,label in [(args.expected_best_success_1m,best['final']['success_1m'],'best'),(args.expected_last_success_1m,last['final']['success_1m'],'last')]:
            if expected is not None and abs(actual-expected)>args.expected_tolerance:raise RuntimeError(f'{label} metric mismatch: {actual} vs {expected}')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=True))
    with (args.output/'per_query.jsonl').open('w') as handle:
        for checkpoint,rows in [('best',best_rows),('last',last_rows)]:
            for row in rows:handle.write(json.dumps({'checkpoint':checkpoint,**row})+'\n')
    print(json.dumps(report,indent=2,allow_nan=True))
if __name__=='__main__':main()
