#!/usr/bin/env python3
"""UQP exactness, full-epoch static efficiency, and real no-grad benchmark."""
from __future__ import annotations

import argparse,csv,importlib.util,json,subprocess,sys,time,unittest
from pathlib import Path

import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (EpochCyclicQuerySampler,LiDARUAVDataset,LiDARUAVDetector,
    OverlapAwareBatchSampler,CandidateLoss,TemporalQueryClipDataset,
    collate_temporal_queries,planned_epoch_stats)


def load_test(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def move(batch,device):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def benchmark(model,batch,device,repeats):
    with torch.no_grad():
        for _ in range(3):result=model(batch);del result
        synchronize(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for i in range(repeats):
            synchronize(device);start=time.perf_counter();result=model(batch);synchronize(device)
            times.append((time.perf_counter()-start)*1000);del result
            if (i+1)%5==0:print(f'  {i+1}/{repeats}',flush=True)
    return dict(median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
        repeats=repeats,warmup=3,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)


def compare_outputs(reference,packed):
    spatial={key:0. for key in ('logits','pred_xyz','fine_features')}
    mapping=packed['occurrence_to_unique']
    for occurrence,unique in enumerate(mapping.tolist()):
        if unique<0:continue
        left=reference['batch_index']==occurrence;right=packed['output']['batch_index']==unique
        for key in spatial:
            diff=float((reference[key][left]-packed['output'][key][right]).abs().max()) if bool(left.any()) else 0.
            spatial[key]=max(spatial[key],diff)
    return dict(spatial=spatial)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/unique_query_packing_audit')
    args=parser.parse_args()
    if args.repeats<20:raise ValueError('At least 20 benchmark repeats required')
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
    queries=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'])
    clips=TemporalQueryClipDataset(queries,cfg['data']['query_clip_length'],stride=cfg['data']['query_clip_stride'])
    eqs=EpochCyclicQuerySampler(clips,4,cfg['experiment']['seed'],True);batcher=OverlapAwareBatchSampler(eqs,2,cfg['experiment']['seed'],True);batcher.set_epoch(1)
    batches=batcher.batches_for_epoch(shuffle=False);examples=[];total_occurrences=total_unique=0;same_overlap=mixed=0
    typical=None
    for number,indices in enumerate(batches):
        records=[clips.clip_metadata[i] for i in indices];keys=[]
        for record in records:keys.extend((record['sequence_id'],query_index) for query_index in record['query_indices'])
        unique=len(set(keys));occurrences=len(keys);sequences=sorted({r['sequence_id'] for r in records})
        total_occurrences+=occurrences;total_unique+=unique
        if len(sequences)==1 and unique<occurrences:same_overlap+=1
        if len(sequences)>1:mixed+=1
        examples.append(dict(batch_index=number,clip_indices=json.dumps(indices),sequence_ids=json.dumps(sequences),
            anchor_ordinals=json.dumps([r['anchor_query_ordinal'] for r in records]),query_occurrences=occurrences,
            unique_queries=unique,deduplicated_occurrences=occurrences-unique,dedup_ratio=1-unique/occurrences))
        if typical is None and len(indices)==2 and len(sequences)==1 and occurrences==16 and unique==12:typical=indices
    if typical is None:raise AssertionError('No typical 16-to-12 real batch')
    items=[clips[i] for i in typical]
    materialized_points=sum(len(q['points']) for item in items for q in item['queries'])
    reference=collate_temporal_queries(items,unique_query_packing=False);packed=collate_temporal_queries(items,unique_query_packing=True)
    if int(reference['query_valid_mask'].sum())!=16 or packed['spatial_num_samples']!=12:raise AssertionError('Unexpected typical mapping')
    torch.manual_seed(42);model=LiDARUAVDetector(cfg).eval();criterion=CandidateLoss(cfg)
    with torch.no_grad():
        ref_out=model(reference);uqp_out=model(packed)
        comparison=compare_outputs(ref_out,dict(packed,output=uqp_out))
        ref_loss=criterion(ref_out,reference);uqp_loss=criterion(uqp_out,packed)
    loss_diffs={key:abs(float(ref_loss[key])-float(uqp_loss[key])) for key in ('loss','loss_cls','loss_reg')}
    if max((*comparison['spatial'].values(),*loss_diffs.values()))>1e-6:
        raise AssertionError(dict(comparison=comparison,loss_diffs=loss_diffs))
    test_names=('test_lidar_uav_v2_uqp','test_lidar_uav_v2_eqs','test_lidar_uav_v2_sequence_isolation','test_lidar_uav_v2_query_causal','test_lidar_uav_v2_sbe')
    modules=[load_test(name) for name in test_names]
    with (out/'test_report.txt').open('w') as handle:
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules]))
        spatial=load_test('test_lidar_uav_v2');spatial_names=[]
        for name,fn in vars(spatial).items():
            if name.startswith('test_') and callable(fn):fn();spatial_names.append(name);handle.write(name+' PASS\n')
    if not result.wasSuccessful():raise AssertionError('Regression failure')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model.to(device)
    ref_device=move(reference,device);packed_device=move(packed,device)
    print(f'Benchmark reference on {device}',flush=True);ref_timing=benchmark(model,ref_device,device,args.repeats)
    print(f'Benchmark UQP on {device}',flush=True);uqp_timing=benchmark(model,packed_device,device,args.repeats)
    plan=planned_epoch_stats(eqs,100,2,2)
    gradient=modules[0].RESULTS
    report=dict(status='V2 UNIQUE QUERY PACKING READY',typical_batch=dict(clip_indices=typical,
        query_occurrences=16,unique_queries=12,dedup_ratio=.25,dataset_materialized_points=materialized_points,
        reference_packed_points=len(reference['points']),uqp_spatial_packed_points=len(packed['points']),
        point_disk_io_deduplicated=False,model_input_points_deduplicated=True),
        epoch1=dict(selected_clips=len(eqs.selected_indices(1,False)),batches=len(batches),same_sequence_overlap_batches=same_overlap,
            mixed_sequence_leftover_batches=mixed,query_occurrences=total_occurrences,unique_spatial_queries=total_unique,
            deduplicated_occurrences=total_occurrences-total_unique,dedup_ratio=1-total_unique/total_occurrences),
        compute_funnel=dict(dense_stride1_spatial_queries=sum(r['valid_query_slots'] for r in clips.clip_metadata),
            eqs_stride4_spatial_queries=total_occurrences,eqs_uqp_spatial_queries=total_unique,
            dense_to_eqs_reduction=1-total_occurrences/sum(r['valid_query_slots'] for r in clips.clip_metadata),
            eqs_to_uqp_reduction=1-total_unique/total_occurrences,
            dense_to_uqp_reduction=1-total_unique/sum(r['valid_query_slots'] for r in clips.clip_metadata)),
        exact_output=comparison,loss_diffs=loss_diffs,gradient_equivalence=gradient.get('gradient_diffs'),
        regressions=dict(uqp='PASS',eqs='PASS',sequence_isolation='PASS',future_leakage='PASS',sbe='PASS',
            arbitrary_query='PASS',tests=result.testsRun+len(spatial_names),causal_results=modules[3].RESULTS,
            isolation_results=modules[2].RESULTS),scheduler_dry_run=dict(planned_optimizer_updates=sum(r['optimizer_updates'] for r in plan),
            warmup_updates=round(sum(r['optimizer_updates'] for r in plan)*cfg['train']['warmup_fraction'])),
        benchmark=dict(hardware=str(device),mode='CUDA_FP32' if device.type=='cuda' else 'CPU_FP32',
            reference=ref_timing,uqp=uqp_timing,speedup=ref_timing['median_ms']/uqp_timing['median_ms']),
        remaining_redundancy='different-query overlapping latest20-event windows are not eliminated',
        optimizer_steps=0,scheduler_steps=0,training_epochs=0,checkpoint_optimization=False,
        git_head_before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_csv(out/'uqp_batch_examples.csv',examples)
    write_csv(out/'uqp_epoch_efficiency.csv',[dict(epoch=1,**report['epoch1'],**report['compute_funnel'])])
    (out/'uqp_gradient_equivalence.json').write_text(json.dumps(dict(gradients=gradient.get('gradient_diffs')),indent=2))
    (out/'uqp_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
