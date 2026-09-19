#!/usr/bin/env python3
"""Fixed-real-case V2 learnability gate; never starts an epoch scheduler."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (LiDARUAVDataset,LiDARUAVDetector,QueryCausalLoss,
    TemporalQueryClipDataset,collate_temporal_queries)
from rdq_uav.lidar_v2.contracts import validate_frozen_v2_config
from rdq_uav.lidar_v2.runtime import move_batch,optimizer_groups
from rdq_uav.utils.seed import seed_everything


GRADIENT_PARAMETERS={
    'sbe_vqsa':'voxel_embed.vqsa.slot_embed.0.weight',
    'l0_spatial':'encoder0.blocks.0.qkv.weight',
    'l2_spatial':'encoder2.blocks.0.qkv.weight',
    'candidate_objectness':'head.cls.2.weight',
    'candidate_xyz':'head.reg.2.weight',
    'query_pool':'query_pool.xyz_embed.2.weight',
    'temporal_transformer':'temporal_transformer.blocks.0.qkv.weight',
    'temporal_xyz_head':'temporal_head.net.2.weight',
}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/lidar_uav_v2.yaml')
    p.add_argument('--train-root',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/train'))
    p.add_argument('--output',type=Path,default=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/learnability_smoke')
    p.add_argument('--device',default='0');p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--group-a',type=int,default=4);p.add_argument('--group-b',type=int,default=4);p.add_argument('--group-c',type=int,default=2)
    p.add_argument('--select-only',action='store_true')
    return p.parse_args()


def write_csv(path,rows):
    if not rows:return
    fieldnames=[]
    for row in rows:
        for key in row:
            if key not in fieldnames:fieldnames.append(key)
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)


def finite(name,value):
    tensor=value if torch.is_tensor(value) else torch.as_tensor(value)
    if not torch.isfinite(tensor).all():raise FloatingPointError(f'non-finite {name}')


def endpoint_support(query):
    recent=query['supervision_recent_mask'];points=query['points'];gt=query['target_xyz']
    near=recent&(torch.linalg.vector_norm(points-gt,dim=1)<=1.) if len(points) else recent
    if not bool(near.any()):return 0
    return int(torch.unique(torch.floor(points[near]/.5).long(),dim=0).shape[0])


def select_cases(queries,clips,a_count,b_count,c_count,max_clip_l0=64000):
    """Scan sequence-local records and freeze compact A/B/C endpoint sets."""
    by_sequence=defaultdict(list)
    for i,record in enumerate(queries.records):by_sequence[record['sequence_id']].append(i)
    support={};points={};l0_counts={};targets={};empty=[];a_candidates=defaultdict(list);b_candidates=defaultdict(list)
    for sequence,indices in sorted(by_sequence.items()):
        history=[]
        for ordinal,index in enumerate(indices):
            if ordinal>=100:break
            query=queries[index];positive=endpoint_support(query);current=positive>0
            support[index]=current;points[index]=len(query['points']);targets[index]=query['target_xyz'].tolist()
            l0_counts[index]=int(torch.unique(torch.floor(query['points']/.5).long(),dim=0).shape[0]) if len(query['points']) else 0
            clip_l0=sum(l0_counts[i] for i in indices[max(0,ordinal-7):ordinal+1])
            executable=clip_l0<=max_clip_l0
            if current and ordinal>=7 and executable:a_candidates[sequence].append(index)
            if (not current) and any(history[-7:]) and executable:b_candidates[sequence].append(index)
            if not len(query['points']):empty.append(index)
            history.append(current)
            if (len(a_candidates[sequence])>=a_count and len(b_candidates[sequence])>=b_count and len(empty)>=c_count):break
        if len(empty)>=c_count and any(len(v)>=a_count for v in a_candidates.values()) and any(len(v)>=b_count for v in b_candidates.values()):break
    def compact(candidates,count):
        choices=[]
        for sequence,indices in candidates.items():
            if len(indices)>=count:
                window=min((indices[i:i+count] for i in range(len(indices)-count+1)),key=lambda x:x[-1]-x[0])
                choices.append((window[-1]-window[0],sequence,window))
        if choices:return min(choices)[2]
        merged=[i for _,values in sorted(candidates.items()) for i in values]
        if len(merged)<count:raise RuntimeError(f'Could not find {count} eligible fixed cases')
        return merged[:count]
    chosen={'A':compact(a_candidates,a_count),'B':compact(b_candidates,b_count),'C':empty[:c_count]}
    if len(chosen['C'])<c_count:raise RuntimeError(f'Could not find {c_count} empty observations')
    clip_by_anchor={row['anchor_sample_id']:i for i,row in enumerate(clips.clip_metadata)}
    cases=[]
    for group,indices in chosen.items():
        for index in indices:
            record=queries.records[index];clip_index=clip_by_anchor[record['sample_id']]
            window=clips.windows[clip_index];history_support=sum(bool(support[i]) for i in window[:-1])
            positive=endpoint_support(queries[index])
            cases.append(dict(group=group,dataset_index=index,clip_index=clip_index,sequence_id=record['sequence_id'],
                query_time=float(record['query_time']),sample_id=record['sample_id'],current_support=positive>0,
                positive_count=positive,history_current_support_count=history_support,
                history_has_support=history_support>0,point_count=points[index],target_xyz=targets[index]))
            cases[-1]['clip_l0_voxel_count']=sum(l0_counts[i] for i in window)
    return cases


def materialize(clips,queries,cases):
    cache={};items=[]
    for case in cases:
        ids=clips.windows[case['clip_index']]
        qs=[]
        for i in ids:
            if i not in cache:cache[i]=queries[i]
            qs.append(cache[i])
        items.append({'queries':qs,'clip_length':clips.clip_length,'case_group':case['group']})
    return items


def endpoint_layout(batch,groups):
    B,T=batch['query_valid_mask'].shape;occ=[]
    for b in range(B):occ.append(b*T+int(batch['query_valid_mask'][b].sum())-1)
    occ=torch.tensor(occ,device=batch['occurrence_to_unique'].device)
    return occ,batch['occurrence_to_unique'][occ],groups


@torch.no_grad()
def measure(model,batch,criterion,groups):
    model.eval();out=model(batch);loss=criterion(out,batch);occ,unique,groups=endpoint_layout(batch,groups)
    pos,_,neg,_=criterion.labels(out,batch);spatial_errors=[];temporal_errors=[];positive=[];negative=[];diagnostics=[]
    B,T=batch['query_valid_mask'].shape
    for b,(occurrence,spatial_query,group) in enumerate(zip(occ.tolist(),unique.tolist(),groups)):
        token=out['batch_index']==spatial_query;gt=batch['target_xyz'][occurrence]
        if bool(token.any()):
            ids=torch.nonzero(token).flatten();top=ids[torch.argmax(out['logits'][ids].float())]
            spatial_error=float(torch.linalg.vector_norm(out['pred_xyz'][top].float()-gt));weights=torch.softmax(out['logits'][ids].float(),0)
            xyz=out['pred_xyz'][ids].float();pooled=(weights[:,None]*xyz).sum(0);reference=xyz[torch.argmax(out['logits'][ids].float())]
            entropy=float(-(weights*weights.clamp_min(1e-12).log()).sum());near=float(weights[torch.linalg.vector_norm(xyz-gt,dim=1)<=1.].sum())
            confidence=float(torch.sigmoid(out['logits'][ids].float().max()))
            if bool((pos&token).any()):positive.extend(out['logits'][pos&token].float().tolist())
            if bool((neg&token).any()):negative.append(float(out['logits'][neg&token].float().max()))
            diagnostics.append(dict(group=group,pool_entropy=entropy,max_objectness_probability=confidence,
                gt_near_pool_weight=near,pooled_xyz_error=float(torch.linalg.vector_norm(pooled-gt)),
                reference_xyz_error=float(torch.linalg.vector_norm(reference-gt)),spatial_top1_error=spatial_error))
        else:
            spatial_error=float('inf');diagnostics.append(dict(group=group,pool_entropy=0.,max_objectness_probability=0.,
                gt_near_pool_weight=0.,pooled_xyz_error=float('inf'),reference_xyz_error=float('inf'),spatial_top1_error=float('inf')))
        spatial_errors.append(spatial_error)
        t=int(batch['query_valid_mask'][b].sum())-1;te=float(torch.linalg.vector_norm(out['temporal_pred_xyz'][b,t].float()-gt));temporal_errors.append(te);diagnostics[-1]['temporal_error']=te
    def median(values):
        finite_values=[x for x in values if math.isfinite(x)]
        return float(np.median(finite_values)) if finite_values else float('inf')
    result=dict(total_loss=float(loss['loss']),objectness_loss=float(loss['loss_cls']),spatial_xyz_loss=float(loss['loss_reg']),
        temporal_loss=float(loss['temporal_loss']),spatial_median_error=median(spatial_errors),temporal_median_error=median(temporal_errors),
        positive_logit=float(np.mean(positive)) if positive else float('nan'),best_negative_logit=float(np.mean(negative)) if negative else float('nan'),
        objectness_margin=(float(np.mean(positive))-float(np.mean(negative))) if positive and negative else float('nan'),
        spatial_success_0p5=float(np.mean(np.asarray(spatial_errors)<=.5)),spatial_success_1=float(np.mean(np.asarray(spatial_errors)<=1.)),
        spatial_success_2=float(np.mean(np.asarray(spatial_errors)<=2.)),temporal_success_0p5=float(np.mean(np.asarray(temporal_errors)<=.5)),
        temporal_success_1=float(np.mean(np.asarray(temporal_errors)<=1.)),temporal_success_2=float(np.mean(np.asarray(temporal_errors)<=2.)),
        endpoint_spatial_supervised=sum(bool((pos&(out['batch_index']==int(i))).any()) for i in unique),
        endpoint_count=len(unique),num_supervised_occurrences=loss['num_supervised_samples'],num_temporal_supervised=loss['num_temporal_supervised'])
    return result,diagnostics,out


def grad_norms(model):
    params=dict(model.named_parameters());result={}
    for label,name in GRADIENT_PARAMETERS.items():
        grad=params[name].grad
        result[label]=None if grad is None else float(torch.linalg.vector_norm(grad.float()))
        if grad is not None:finite(f'gradient {label}',grad)
    return result


def train_smoke(cfg,batch,groups,device,lr,steps,max_steps,log_every,name):
    seed_everything(42);model=LiDARUAVDetector(cfg).to(device).train();criterion=QueryCausalLoss(cfg)
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg['train']['weight_decay']),lr=lr,
        betas=tuple(cfg['train']['betas']),eps=cfg['train']['eps'])
    initial,initial_diag,_=measure(model,batch,criterion,groups);curve=[];gradient_rows=[]
    def row(step,metrics,grads=None):
        result={'step':step,**metrics};result.update({f'grad_{k}':v for k,v in (grads or {}).items()});return result
    curve.append(row(0,initial))
    target_steps=steps;started=time.perf_counter()
    progress=tqdm(total=target_steps,desc=f'Smoke-{name}',unit='step',dynamic_ncols=True)
    for step in range(1,max_steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);out=model(batch);loss=criterion(out,batch);finite('loss',loss['loss']);loss['loss'].backward()
        grads=grad_norms(model);torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip_norm']);optimizer.step()
        progress.update(1)
        if step%log_every==0 or step==target_steps:
            metrics,_,_=measure(model,batch,criterion,groups);curve.append(row(step,metrics,grads));gradient_rows.append({'experiment':name,'step':step,**grads})
            progress.set_postfix(loss=f"{metrics['total_loss']:.4f}",spatial=f"{metrics['spatial_median_error']:.3f}m",temporal=f"{metrics['temporal_median_error']:.3f}m")
        if step==target_steps:
            final=curve[-1]
            spatial_ratio=final['spatial_median_error']/max(initial['spatial_median_error'],1e-12)
            temporal_ratio=final['temporal_median_error']/max(initial['temporal_median_error'],1e-12)
            if name=='A':passed=(spatial_ratio<=.5 and temporal_ratio<=.5 and final['objectness_loss']<initial['objectness_loss'] and final['spatial_xyz_loss']<initial['spatial_xyz_loss'] and final['temporal_loss']<initial['temporal_loss'] and final['objectness_margin']>initial['objectness_margin'])
            elif name=='B':passed=(temporal_ratio<=.5 and final['temporal_loss']<initial['temporal_loss'] and final['endpoint_spatial_supervised']==0)
            else:passed=True
            if passed or target_steps>=max_steps:break
            target_steps=max_steps;progress.total=max_steps;progress.refresh()
    progress.close()
    final,final_diag,_=measure(model,batch,criterion,groups)
    return model,curve,gradient_rows,initial,final,initial_diag,final_diag,target_steps,time.perf_counter()-started


def combine_measurements(measured):
    """Reproduce joint-batch sample normalization from separately forwarded groups."""
    metrics=[x[0] for x in measured];diagnostics=sum((x[1] for x in measured),[])
    ns=sum(x['num_supervised_occurrences'] for x in metrics);nt=sum(x['num_temporal_supervised'] for x in metrics)
    weighted=lambda key,weight:sum(x[key]*x[weight] for x in metrics)/max(1,sum(x[weight] for x in metrics))
    spatial=weighted('objectness_loss','num_supervised_occurrences')+2*weighted('spatial_xyz_loss','num_supervised_occurrences')
    temporal=weighted('temporal_loss','num_temporal_supervised')
    spatial_errors=[x['spatial_top1_error'] for x in diagnostics if math.isfinite(x['spatial_top1_error'])]
    temporal_errors=[x['temporal_error'] for x in diagnostics]
    positive=[x['positive_logit'] for x in metrics if math.isfinite(x['positive_logit'])]
    negative=[x['best_negative_logit'] for x in metrics if math.isfinite(x['best_negative_logit'])]
    combined=dict(total_loss=spatial+temporal,objectness_loss=weighted('objectness_loss','num_supervised_occurrences'),
        spatial_xyz_loss=weighted('spatial_xyz_loss','num_supervised_occurrences'),temporal_loss=temporal,
        spatial_median_error=float(np.median(spatial_errors)) if spatial_errors else float('inf'),
        temporal_median_error=float(np.median(temporal_errors)),positive_logit=float(np.mean(positive)) if positive else float('nan'),
        best_negative_logit=float(np.mean(negative)) if negative else float('nan'),
        objectness_margin=float(np.mean(positive)-np.mean(negative)) if positive and negative else float('nan'),
        endpoint_spatial_supervised=sum(x['endpoint_spatial_supervised'] for x in metrics),endpoint_count=sum(x['endpoint_count'] for x in metrics),
        num_supervised_occurrences=ns,num_temporal_supervised=nt)
    return combined,diagnostics


def train_smoke_accum(cfg,batches,groups_by_batch,device,lr,steps,max_steps,log_every,name):
    """One optimizer update accumulates exact joint loss from 2-clip microbatches."""
    seed_everything(42);model=LiDARUAVDetector(cfg).to(device).train();criterion=QueryCausalLoss(cfg)
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg['train']['weight_decay']),lr=lr,
        betas=tuple(cfg['train']['betas']),eps=cfg['train']['eps'])
    measured=[measure(model,b,criterion,g) for b,g in zip(batches,groups_by_batch)];initial,initial_diag=combine_measurements(measured)
    ns=[x[0]['num_supervised_occurrences'] for x in measured];nt=[x[0]['num_temporal_supervised'] for x in measured]
    curve=[{'step':0,**initial}];gradient_rows=[];started=time.perf_counter()
    target_steps=steps
    progress=tqdm(total=target_steps,desc=f'Smoke-{name}',unit='step',dynamic_ncols=True)
    for step in range(1,max_steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        for batch,n,t in zip(batches,ns,nt):
            out=model(batch);loss=criterion(out,batch)
            objective=loss['spatial_loss']*(n/max(1,sum(ns)))+loss['temporal_loss']*(t/max(1,sum(nt)))
            finite('mix loss',objective);objective.backward()
        grads=grad_norms(model);torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip_norm']);optimizer.step()
        progress.update(1)
        if step%log_every==0 or step==target_steps:
            current,current_diag=combine_measurements([measure(model,b,criterion,g) for b,g in zip(batches,groups_by_batch)])
            curve.append({'step':step,**current,**{f'grad_{k}':v for k,v in grads.items()}});gradient_rows.append({'experiment':name,'step':step,**grads})
            progress.set_postfix(loss=f"{current['total_loss']:.4f}",spatial=f"{current['spatial_median_error']:.3f}m",temporal=f"{current['temporal_median_error']:.3f}m")
        if step==target_steps:
            current=curve[-1]
            spatial_ratio=current['spatial_median_error']/max(initial['spatial_median_error'],1e-12)
            temporal_ratio=current['temporal_median_error']/max(initial['temporal_median_error'],1e-12)
            if name=='A':passed=(spatial_ratio<=.5 and temporal_ratio<=.5 and current['objectness_loss']<initial['objectness_loss'] and current['spatial_xyz_loss']<initial['spatial_xyz_loss'] and current['temporal_loss']<initial['temporal_loss'] and current['objectness_margin']>initial['objectness_margin'])
            elif name=='B':passed=(temporal_ratio<=.5 and current['temporal_loss']<initial['temporal_loss'] and current['endpoint_spatial_supervised']==0)
            else:passed=True
            if passed or target_steps>=max_steps:break
            target_steps=max_steps;progress.total=max_steps;progress.refresh()
    progress.close()
    measured=[measure(model,b,criterion,g) for b,g in zip(batches,groups_by_batch)];final,final_diag=combine_measurements(measured)
    return model,curve,gradient_rows,initial,final,initial_diag,final_diag,target_steps,time.perf_counter()-started


def uqp_equivalence(cfg,items,device):
    seed_everything(42);model=LiDARUAVDetector(cfg).to(device).eval();criterion=QueryCausalLoss(cfg)
    ref=move_batch(collate_temporal_queries(items,False),device);packed=move_batch(collate_temporal_queries(items,True),device)
    with torch.no_grad():a=model(ref);b=model(packed)
    diffs={key:0. for key in ('logits','pred_xyz','fine_features')}
    for occurrence,unique in enumerate(packed['occurrence_to_unique'].tolist()):
        if unique<0:continue
        left=a['batch_index']==occurrence;right=b['batch_index']==unique
        for key in diffs:
            if bool(left.any()):diffs[key]=max(diffs[key],float((a[key][left]-b[key][right]).abs().max()))
    diffs.update(query_token=float((a['query_token']-b['query_token']).abs().max()),temporal_xyz=float((a['temporal_pred_xyz']-b['temporal_pred_xyz']).abs().max()))
    la=criterion(a,ref);lb=criterion(b,packed);diffs['loss']=abs(float(la['loss'])-float(lb['loss']))
    return diffs


def temporal_endpoint_gradient(cfg,batches,device):
    seed_everything(42);model=LiDARUAVDetector(cfg).to(device).train();criterion=QueryCausalLoss(cfg);model.zero_grad(set_to_none=True)
    losses=[]
    for batch in batches:
        out=model(batch);endpoint=torch.zeros_like(batch['target_valid_clip'])
        for b in range(len(endpoint)):endpoint[b,int(batch['query_valid_mask'][b].sum())-1]=True
        diagnostic=dict(batch);diagnostic['target_valid_clip']=endpoint
        loss=criterion.temporal(out,diagnostic);losses.append(float(loss.detach()));(loss/len(batches)).backward()
    return float(np.mean(losses)),grad_norms(model)


def empty_robustness(cfg,batches,device):
    seed_everything(42);model=LiDARUAVDetector(cfg).to(device).train();criterion=QueryCausalLoss(cfg);model.zero_grad(set_to_none=True)
    tokens=[];supervised=0;losses=[];forward_finite=True
    for batch in batches:
        out=model(batch);loss=criterion(out,batch);finite('empty loss',loss['loss']);(loss['loss']/len(batches)).backward();losses.append(float(loss['loss'].detach()))
        _,unique,_=endpoint_layout(batch,['C']*len(batch['query_valid_mask']))
        tokens.extend(int((out['batch_index']==i).sum()) for i in unique)
        supervised+=measure(model,batch,criterion,['C']*len(batch['query_valid_mask']))[0]['endpoint_spatial_supervised']
        forward_finite=forward_finite and all(bool(torch.isfinite(out[k]).all()) for k in ('logits','pred_xyz','temporal_pred_xyz'))
    grad=model.query_pool.missing_lidar_token.grad
    return dict(forward_finite=forward_finite,loss_finite=all(math.isfinite(x) for x in losses),
        backward_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
        endpoint_spatial_tokens=tokens,endpoint_spatial_supervised=supervised,
        missing_token_gradient=None if grad is None else float(torch.linalg.vector_norm(grad.float())))


def summarize_diagnostics(rows):
    result={}
    for group in ('A','B'):
      result[group]={}
      for stage in ('initial','final'):
        selected=[r for r in rows if r['group']==group and r['stage']==stage]
        result[group][stage]={key:float(np.mean([r[key] for r in selected])) for key in ('pool_entropy','max_objectness_probability','gt_near_pool_weight','pooled_xyz_error','reference_xyz_error','spatial_top1_error','temporal_error')} if selected else {}
    return result


def group_errors(rows):
    result={}
    for group in ('A','B'):
        selected=[r for r in rows if r['group']==group]
        result[group]=dict(spatial_median_error=float(np.median([r['spatial_top1_error'] for r in selected])),
            temporal_median_error=float(np.median([r['temporal_error'] for r in selected]))) if selected else {}
    return result


def main():
    a=parse_args();cfg=yaml.safe_load(a.config.read_text());validate_frozen_v2_config(cfg);seed_everything(42)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    queries=LiDARUAVDataset(a.train_root,ROOT/cfg['data']['split_file'],cfg['data']['train_split'],cfg['data']['num_merged_frames'])
    clips=TemporalQueryClipDataset(queries,cfg['temporal']['clip_length'],stride=1)
    cases=select_cases(queries,clips,a.group_a,a.group_b,a.group_c)
    (a.output/'learnability_cases.json').write_text(json.dumps(cases,indent=2))
    if a.select_only:
        print(json.dumps({'cases':len(cases),'groups':{g:sum(c['group']==g for c in cases) for g in 'ABC'},'output':str(a.output)},indent=2));return
    if a.device=='cpu':device=torch.device('cpu')
    else:
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the full smoke; use --select-only for CPU case audit')
        device=torch.device(f'cuda:{a.device}')
    grouped={g:[c for c in cases if c['group']==g] for g in 'ABC'}
    items={g:materialize(clips,queries,grouped[g]) for g in 'ABC'}
    # A high-density 2-clip spatial batch exceeds the current CUDA VQSA grid
    # limit. Keep every full 8-query clip intact and accumulate the exact
    # occurrence-normalized group loss across one-clip microbatches.
    batch_groups={g:[move_batch(collate_temporal_queries([item],unique_query_packing=True),device)
                     for item in items[g]] for g in 'ABC'}
    # Two adjacent two-query prefixes share one real query (4 occurrences ->
    # 3 unique). This exercises the exact UQP boundary while keeping the
    # intentionally duplicated reference below the CUDA VQSA grid limit.
    equivalence_items=[{'queries':items['A'][0]['queries'][-2:]},
                       {'queries':items['A'][1]['queries'][-2:]}]
    equivalence=uqp_equivalence(cfg,equivalence_items,device)
    if max(equivalence.values())>1e-5:raise AssertionError(f'UQP equivalence failed: {equivalence}')
    endpoint_temp_loss,b_temporal_grads=temporal_endpoint_gradient(cfg,batch_groups['B'],device)
    seed_everything(42);initial=LiDARUAVDetector(cfg);torch.save({'model_state':initial.state_dict(),'seed':42,'lr':a.lr},a.output/'initial_state.pt');del initial
    experiments={};curves={};gradient_rows=[];all_diagnostics=[];final_states={}
    for name in ('A','B'):
        groups=[['A' if name=='A' else 'B']*len(batch['query_valid_mask']) for batch in batch_groups[name]]
        model,curve,gradients,start,end,start_diag,end_diag,steps,seconds=train_smoke_accum(cfg,batch_groups[name],groups,device,a.lr,a.steps,a.max_steps,a.log_every,name)
        curves[name]=curve;gradient_rows.extend(gradients);all_diagnostics.extend([dict(stage='initial',**x) for x in start_diag]);all_diagnostics.extend([dict(stage='final',**x) for x in end_diag])
        experiments[name]=dict(initial=start,final=end,steps=steps,seconds=seconds)
        final_states[name]={k:v.detach().cpu() for k,v in model.state_dict().items()};del model
        if device.type=='cuda':torch.cuda.empty_cache()
    mix_batches=batch_groups['A']+batch_groups['B']
    mix_groups=[['A']*len(batch['query_valid_mask']) for batch in batch_groups['A']]+[['B']*len(batch['query_valid_mask']) for batch in batch_groups['B']]
    model,curve,gradients,start,end,start_diag,end_diag,steps,seconds=train_smoke_accum(cfg,mix_batches,mix_groups,
        device,a.lr,a.steps,a.steps,a.log_every,'MIX')
    curves['MIX']=curve;gradient_rows.extend(gradients);all_diagnostics.extend([dict(stage='initial',**x) for x in start_diag]);all_diagnostics.extend([dict(stage='final',**x) for x in end_diag])
    experiments['MIX']=dict(initial=start,final=end,steps=steps,seconds=seconds,
        group_metrics=dict(initial=group_errors(start_diag),final=group_errors(end_diag)))
    final_states['MIX']={k:v.detach().cpu() for k,v in model.state_dict().items()};del model
    if device.type=='cuda':torch.cuda.empty_cache()
    c_result=empty_robustness(cfg,batch_groups['C'],device)
    torch.save({'models':final_states,'steps':{k:v['steps'] for k,v in experiments.items()},'lr':a.lr},a.output/'final_smoke_state.pt')
    for name in ('A','B','MIX'):write_csv(a.output/f'smoke_{name.lower()}_curve.csv',curves[name])
    module_health={name:dict(finite=all(v is None or math.isfinite(v) for v in [row.get(name) for row in gradient_rows]),
        nonzero=any((row.get(name) or 0)>0 for row in gradient_rows),max_norm=max((row.get(name) or 0 for row in gradient_rows),default=0.)) for name in GRADIENT_PARAMETERS}
    gradient_health=dict(records=gradient_rows,module_health=module_health,b_endpoint_temporal_only_loss=endpoint_temp_loss,
        b_endpoint_temporal_only_gradients=b_temporal_grads,
        all_recorded_gradients_finite=all(v is None or math.isfinite(v) for row in gradient_rows for k,v in row.items() if k not in ('experiment','step')))
    diagnostics=summarize_diagnostics(all_diagnostics)
    (a.output/'gradient_health.json').write_text(json.dumps(gradient_health,indent=2))
    (a.output/'query_readout_diagnostics.json').write_text(json.dumps({'summary':diagnostics,'records':all_diagnostics},indent=2))
    def reduction(exp,key):return exp['final'][key]/max(exp['initial'][key],1e-12)
    pass_a=(reduction(experiments['A'],'spatial_median_error')<=.5 and reduction(experiments['A'],'temporal_median_error')<=.5 and experiments['A']['final']['objectness_loss']<experiments['A']['initial']['objectness_loss'] and experiments['A']['final']['spatial_xyz_loss']<experiments['A']['initial']['spatial_xyz_loss'] and experiments['A']['final']['temporal_loss']<experiments['A']['initial']['temporal_loss'] and experiments['A']['final']['objectness_margin']>experiments['A']['initial']['objectness_margin'])
    pass_b=(experiments['B']['final']['endpoint_spatial_supervised']==0 and reduction(experiments['B'],'temporal_median_error')<=.5 and experiments['B']['final']['temporal_loss']<experiments['B']['initial']['temporal_loss'])
    pass_c=(c_result['forward_finite'] and c_result['loss_finite'] and c_result['backward_finite'] and not any(c_result['endpoint_spatial_tokens']) and c_result['endpoint_spatial_supervised']==0)
    report=dict(status='PASS' if pass_a and pass_b and pass_c else 'FAIL',base_commit='5549e3b',parameters=1328075,
        groups={g:dict(samples=len(grouped[g]),sequences=len({x['sequence_id'] for x in grouped[g]}),positive_count=sum(x['positive_count'] for x in grouped[g]),history_support_count=sum(x['history_current_support_count'] for x in grouped[g])) for g in 'ABC'},
        smoke_a={**experiments['A'],'pass':pass_a},smoke_b={**experiments['B'],'pass':pass_b,'endpoint_temporal_only_gradient':b_temporal_grads},
        smoke_mix=experiments['MIX'],empty_observation={**c_result,'pass':pass_c},uqp_equivalence=equivalence,
        query_readout=diagnostics,smoke_only_lr=a.lr,optimizer_steps={k:v['steps'] for k,v in experiments.items()},
        optimizer='AdamW formal betas/eps/weight_decay, fixed LR',scheduler_steps=0,model_structure_modified=False,formal_loss_modified=False,
        execution_note='Full 8-query clips; one clip per microbatch with exact group-normalized gradient accumulation because dense two-clip VQSA exceeds the current CUDA grid limit')
    (a.output/'smoke_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'groups':report['groups'],'steps':report['optimizer_steps'],'output':str(a.output)},indent=2))


if __name__=='__main__':main()
