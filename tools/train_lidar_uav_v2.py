#!/usr/bin/env python3
"""Manual V2 spatial-candidate training entry. Use --precheck-only for data checks."""
import argparse,functools,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (LiDARUAVDetector,LiDARUAVDataset,LiDARUAVValidationDataset,
    TemporalQueryClipDataset,collate_temporal_queries,CandidateLoss,CandidateSelector,
    EpochCyclicQuerySampler,OverlapAwareBatchSampler,planned_epoch_stats)
from rdq_uav.lidar_v2.runtime import move_batch,optimizer_groups,UpdateScheduler
from rdq_uav.lidar_v2.training import (inspect_dataset_timing,write_csv,validate,finite_or_raise,save_checkpoint,
    spatial_selection_metrics,better_spatial)
from rdq_uav.lidar_v2.contracts import effective_config,validate_frozen_v2_config
from rdq_uav.utils.seed import seed_everything

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/lidar_uav_v2.yaml')
    p.add_argument('--train-root',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/train'))
    p.add_argument('--val-root',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/val'))
    p.add_argument('--val-reference',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv'))
    p.add_argument('--epochs',type=int);p.add_argument('--device',default='0');p.add_argument('--output',type=Path)
    p.add_argument('--precheck-only',action='store_true');p.add_argument('--max-updates',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--batch-size',type=int);p.add_argument('--accumulate',type=int)
    return p.parse_args()

def main():
    args=parse_args();cfg=yaml.safe_load(args.config.read_text());validate_frozen_v2_config(cfg);seed_everything(cfg['experiment']['seed'])
    if int(os.environ.get('WORLD_SIZE','1'))>1:raise RuntimeError('DDP not implemented; use single device')
    output=(args.output or ROOT/cfg['experiment']['output_dir']/'spatial_candidates_v1').resolve()
    protected=(ROOT/'outputs/own_multimodal_research/lidar_uav_v1').resolve()
    if output==protected or protected in output.parents:raise ValueError('V1 output directory is frozen')
    if output.exists() and not args.precheck_only and not args.resume:
        # A failure before the first optimizer update leaves only reproducible
        # precheck/config artifacts. Allow that exact bootstrap state to restart;
        # never overwrite a run that reached metrics or a checkpoint.
        bootstrap={'effective_config.json','effective_config.yaml','precheck_samples.csv','resolved_config.yaml'}
        existing={p.name for p in output.iterdir()}
        if not existing.issubset(bootstrap):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    cfg['data'].update(root=str(args.train_root),val_root=str(args.val_root),val_reference=str(args.val_reference))
    device=torch.device('cpu' if args.device=='cpu' else f'cuda:{args.device}')
    effective_cfg,training_precision,evaluation_precision=effective_config(cfg,device)
    (output/'effective_config.yaml').write_text(yaml.safe_dump(effective_cfg,sort_keys=False))
    (output/'effective_config.json').write_text(json.dumps(effective_cfg,indent=2))
    trainq=LiDARUAVDataset(args.train_root,ROOT/cfg['data']['split_file'],cfg['data']['train_split'],cfg['data']['num_merged_frames'])
    valq=LiDARUAVValidationDataset(args.val_root,args.val_reference,cfg['data']['num_merged_frames'])
    clip_length=int(cfg['data']['query_clip_length']);clip_stride=int(cfg['data']['query_clip_stride'])
    train=TemporalQueryClipDataset(trainq,clip_length,stride=clip_stride);val=TemporalQueryClipDataset(valq,clip_length,validation=True)
    eqs=cfg['train']['query_subsampling']
    if eqs['offset_policy']!='cyclic_epoch' or eqs['anchor']!='clip_end' or not eqs['dense_clip_context']:
        raise ValueError('EQS-v1 requires cyclic_epoch, clip_end, and dense_clip_context=true')
    query_stride=int(eqs['stride']) if eqs['enabled'] else 1
    train_sampler=EpochCyclicQuerySampler(train,query_stride,cfg['experiment']['seed'],eqs['shuffle_selected'])
    checks=[]
    for name,ds in (('train',trainq),('val',valq)):
        rows=inspect_dataset_timing(ds,np.linspace(0,len(ds)-1,min(32,len(ds)),dtype=int))
        checks.extend(dict(source=name,**r) for r in rows)
    write_csv(output/'precheck_samples.csv',checks)
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(effective_cfg,sort_keys=False))
    train_sampler.set_epoch(1)
    print(f'SPATIAL CANDIDATE PRECHECK PASS: train queries={len(trainq)}, full clips={len(train)}, '
          f'epoch1 selected clips={len(train_sampler)}, val endpoints={len(val)}, future events=0; '
          'query and target times are independent interfaces')
    print(f'Training precision: {training_precision.effective}\nEvaluation precision: {evaluation_precision.effective}')
    if args.precheck_only:return
    model=LiDARUAVDetector(cfg).to(device);criterion=CandidateLoss(cfg);selector=CandidateSelector(cfg)
    batch_size=args.batch_size or cfg['train']['per_gpu_batch_size'];accum=args.accumulate or cfg['train']['single_gpu_accumulate']
    if batch_size<1 or accum<1:raise ValueError('Invalid batch/accumulation')
    uqp=cfg['train']['unique_query_packing']
    if uqp['dedup_key']!='sequence_query_uid':raise ValueError('UQP-v1 requires sequence_query_uid')
    collate=functools.partial(collate_temporal_queries,unique_query_packing=bool(uqp['enabled']))
    overlap_sampler=OverlapAwareBatchSampler(train_sampler,batch_size,cfg['experiment']['seed'],eqs['shuffle_selected']) if uqp['overlap_aware_batching'] else None
    if overlap_sampler is not None:
        loader=DataLoader(train,batch_sampler=overlap_sampler,num_workers=args.num_workers,collate_fn=collate)
    else:
        loader=DataLoader(train,batch_size=batch_size,shuffle=False,sampler=train_sampler,num_workers=args.num_workers,collate_fn=collate)
    vloader=DataLoader(val,batch_size=cfg['evaluation']['batch_size'],shuffle=False,num_workers=args.num_workers,collate_fn=collate_temporal_queries)
    opt=torch.optim.AdamW(optimizer_groups(model,cfg['train']['weight_decay']),lr=cfg['train']['lr'],betas=tuple(cfg['train']['betas']),eps=cfg['train']['eps'])
    epochs=args.epochs or cfg['train']['epochs']
    update_plan=planned_epoch_stats(train_sampler,epochs,batch_size,accum)
    total_planned_updates=sum(row['optimizer_updates'] for row in update_plan)
    sched=UpdateScheduler(opt,total_planned_updates,cfg['train']['warmup_fraction'],cfg['train']['lr'],cfg['train']['final_lr']);sched.prepare_first_update()
    first=update_plan[0]
    print(f"Query stride : {query_stride}\nQuery offset : {first['offset']}\n"
          f"Selected clips : {first['selected_clips']} / {len(train)}\n"
          f"Selection ratio : {first['selected_clips']/len(train):.2%}\n"
          f"Unique query packing : {bool(uqp['enabled'])}\nOverlap-aware batching : {bool(overlap_sampler is not None)}\n"
          f"Validation endpoints : {len(val)} (full)\nPlanned optimizer updates : {total_planned_updates}")
    start=step=0;best_spatial=None;history=[]
    if args.resume:
        state=torch.load(args.resume,map_location=device);model.load_state_dict(state['model_state'],strict=True)
        opt.load_state_dict(state['optimizer_state']);sched.load_state_dict(state['scheduler_state'])
        start=state['epoch'];step=state.get('global_optimizer_step',state['global_step'])
        selection=state.get('checkpoint_selection',{});best_spatial=selection.get('spatial')
        if (output/'metrics.json').exists():history=json.loads((output/'metrics.json').read_text())
        for record in history:
            record.setdefault('query_stride',None);record.setdefault('query_offset',None)
            record.setdefault('selected_clips',None);record.setdefault('full_clips',None)
            record.setdefault('selection_ratio',None);record.setdefault('valid_query_slots',None)
            record.setdefault('query_occurrences',None);record.setdefault('unique_spatial_queries',None)
            record.setdefault('spatial_dedup_ratio',None)
    completed=start;git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();run_id=output.name
    try:
        for epoch in range(start+1,epochs+1):
            if overlap_sampler is not None:overlap_sampler.set_epoch(epoch)
            else:train_sampler.set_epoch(epoch)
            selected_clips=len(train_sampler);valid_query_slots=sum(train.clip_metadata[i]['valid_query_slots'] for i in train_sampler.selected_indices(shuffle=False))
            names=('loss','loss_cls','loss_reg')
            model.train();opt.zero_grad(set_to_none=True);pending=0;totals=np.zeros(len(names));batches=0;occurrences=unique_queries=0;stop=False;begin=time.perf_counter()
            offset=train_sampler.offset_for_epoch();ratio=selected_clips/len(train)
            bar=tqdm(loader,desc=f'TRAIN {epoch}/{epochs} EQS s={query_stride} o={offset} clips={selected_clips}/{len(train)}',dynamic_ncols=True)
            for raw in bar:
                occurrences+=int(raw['query_valid_mask'].sum());unique_queries+=int(raw['spatial_num_samples'])
                batch=move_batch(raw,device)
                with training_precision.context(device):out=model(batch)
                for key in ('logits','pred_xyz'):finite_or_raise(key,out[key])
                loss=criterion(out,batch);finite_or_raise('loss',loss['loss'])
                if loss['num_supervised_samples']==0:continue
                (loss['loss']/accum).backward();pending+=1;batches+=1
                totals+=np.array([float(loss[k].detach()) for k in names])
                if pending==accum:
                    for p in model.parameters():
                        if p.grad is not None:finite_or_raise('gradient',p.grad)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip_norm'])
                    opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1;pending=0
                bar.set_postfix_str(' '.join(f'{k}={v/batches:.4f}' for k,v in zip(names,totals))+
                    f' UQP={1-unique_queries/max(1,occurrences):.1%}')
                if args.max_updates and step>=args.max_updates:stop=True;break
            if pending:
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(accum/pending);finite_or_raise('gradient',p.grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip_norm'])
                opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1
            metrics,rows,health=validate(model,vloader,criterion,selector,device,evaluation_precision)
            spatial_candidate=spatial_selection_metrics(metrics,epoch)
            improved_spatial=better_spatial(spatial_candidate,best_spatial)
            if improved_spatial:best_spatial=spatial_candidate
            record=dict(epoch=epoch,global_step=step,query_stride=query_stride,query_offset=offset,
                        selected_clips=selected_clips,full_clips=len(train),selection_ratio=ratio,
                        valid_query_slots=valid_query_slots,query_occurrences=occurrences,unique_spatial_queries=unique_queries,
                        spatial_dedup_ratio=1-unique_queries/max(1,occurrences),train=dict(zip(names,(totals/max(1,batches)).tolist())),
                        validation=metrics,val_loss=health,seconds=time.perf_counter()-begin)
            history.append(record);(output/'metrics.json').write_text(json.dumps(history,indent=2))
            write_csv(output/'metrics.csv',[dict(epoch=r['epoch'],query_stride=r.get('query_stride'),query_offset=r.get('query_offset'),
                selected_clips=r.get('selected_clips'),full_clips=r.get('full_clips'),selection_ratio=r.get('selection_ratio'),
                valid_query_slots=r.get('valid_query_slots'),query_occurrences=r.get('query_occurrences'),
                unique_spatial_queries=r.get('unique_spatial_queries'),spatial_dedup_ratio=r.get('spatial_dedup_ratio'),
                **r['train'],**r['validation']['all']) for r in history])
            write_csv(output/f'validation_epoch_{epoch:03d}.csv',rows)
            selection=dict(spatial=best_spatial)
            metadata=dict(git_commit=git_commit,run_id=run_id,eqs=dict(stride=query_stride,current_offset=offset,
                completed_coverage_cycles=epoch//query_stride))
            save_checkpoint(output/'last.pt',model,opt,sched,epoch,step,selection,effective_cfg,metadata)
            if improved_spatial:save_checkpoint(output/'best_spatial.pt',model,opt,sched,epoch,step,selection,effective_cfg,metadata)
            print('ALL',metrics['all']);print('NO_CURRENT_SUPPORT',metrics['no_current_support']);completed=epoch
            if stop:break
    except KeyboardInterrupt:
        selection=dict(spatial=best_spatial)
        metadata=dict(git_commit=git_commit,run_id=run_id,eqs=dict(stride=query_stride,
            current_offset=train_sampler.offset_for_epoch(max(1,completed)),completed_coverage_cycles=completed//query_stride))
        save_checkpoint(output/'interrupt.pt',model,opt,sched,completed,step,selection,effective_cfg,metadata)
        print('Interrupted; checkpoint saved. Resume restarts unfinished epoch.')

if __name__=='__main__':main()
