#!/usr/bin/env python3
"""Metadata-only EQS audit, coverage analysis, and scheduler dry-run."""
from __future__ import annotations

import argparse,csv,importlib.util,json,math,subprocess,sys,unittest
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (EpochCyclicQuerySampler,LiDARUAVDataset,
    LiDARUAVValidationDataset,TemporalQueryClipDataset,planned_epoch_stats)


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=int(a.max()))


def appearances(dataset,indices):
    count=Counter()
    for index in indices:count.update(dataset.clip_metadata[index]['query_indices'])
    return count


def load_test(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/query_subsampling_audit')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text());torch.set_num_threads(4)
    train_queries=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'])
    train=TemporalQueryClipDataset(train_queries,cfg['temporal']['clip_length'],stride=1)
    official=Path(cfg['data']['root']).parent
    val_queries=LiDARUAVValidationDataset(official/'val',official/'validation_ref_new (for your ref).csv')
    validation=TemporalQueryClipDataset(val_queries,cfg['temporal']['clip_length'],validation=True)
    full=len(train);sequence_full=Counter(r['sequence_id'] for r in train.clip_metadata)
    stride_summary={};per_sequence=[]
    appearance_rows={1:[],4:[]};appearance_summary={}
    for stride in (1,2,4,8):
        sampler=EpochCyclicQuerySampler(train,stride,cfg['experiment']['seed'],True)
        offsets=[]
        for offset in range(stride):
            epoch=offset+1;indices=sampler.selected_indices(epoch,False)
            slots=sum(train.clip_metadata[i]['valid_query_slots'] for i in indices)
            by_sequence=Counter(train.clip_metadata[i]['sequence_id'] for i in indices)
            row=dict(offset=offset,epoch=epoch,selected_clips=len(indices),selection_ratio=len(indices)/full,
                     valid_query_slots=slots,estimated_spatial_query_forwards=slots,estimated_lidar_sample_builds=slots)
            offsets.append(row)
            for sequence in sorted(sequence_full):
                per_sequence.append(dict(stride=stride,offset=offset,sequence_id=sequence,
                    full_clips=sequence_full[sequence],selected_clips=by_sequence[sequence]))
            if stride in appearance_rows:
                count=appearances(train,indices)
                values=[]
                for query_index,record in enumerate(train_queries.records):
                    value=count[query_index];values.append(value)
                    appearance_rows[stride].append(dict(stride=stride,offset=offset,query_index=query_index,
                        sequence_id=record['sequence_id'],sample_id=record['sample_id'],appearances=value))
                appearance_summary[f'stride{stride}_offset{offset}']=distribution(values)
        stride_summary[str(stride)]=offsets
    sampler4=EpochCyclicQuerySampler(train,4,cfg['experiment']['seed'],True)
    selected=[];coverage_rows=[];supervised=set()
    for epoch in range(1,5):
        for index in sampler4.selected_indices(epoch,False):
            selected.append(index);record=train.clip_metadata[index];supervised.update(record['query_indices'])
            coverage_rows.append(dict(epoch=epoch,offset=epoch-1,dataset_index=index,sequence_id=record['sequence_id'],
                anchor_query_ordinal=record['anchor_query_ordinal'],anchor_query_time=record['anchor_query_time'],
                anchor_sample_id=record['anchor_sample_id'],valid_query_slots=record['valid_query_slots']))
    selected_count=Counter(selected)
    anchor_coverage=dict(anchor_coverage_ratio=len(selected_count)/full,
        duplicate_anchor_selection=sum(value-1 for value in selected_count.values() if value>1),
        missing_anchor_count=full-len(selected_count),query_supervision_coverage=len(supervised)/len(train_queries),
        missing_supervised_query_count=len(train_queries)-len(supervised))
    batch_size=int(cfg['train']['per_gpu_batch_size']);accumulate=int(cfg['train']['single_gpu_accumulate']);epochs=int(cfg['train']['epochs'])
    plan=planned_epoch_stats(sampler4,epochs,batch_size,accumulate)
    resume_a=[]
    for epoch in range(1,18):sampler4.set_epoch(epoch);resume_a.append(list(sampler4))
    resumed=EpochCyclicQuerySampler(train,4,cfg['experiment']['seed'],True);resumed.set_epoch(17)
    resume_deterministic=resume_a[-1]==list(resumed)
    stride1=EpochCyclicQuerySampler(train,1,42,False)
    report=dict(status='V2 EQS STRUCTURE READY',full_training_clips=full,train_sequences=len(sequence_full),
        dataset_rebuilt_each_epoch=False,strides=stride_summary,appearance_summary=appearance_summary,
        stride4_four_epoch=anchor_coverage,validation=dict(before=len(validation),after=len(validation),sampler='NONE',full_denominator=True),
        stride1_equivalence=set(stride1)==set(range(full)),resume_deterministic=resume_deterministic,
        scheduler_dry_run=dict(status='PASS',epochs=epochs,batch_size_clips=batch_size,accumulate=accumulate,
            total_planned_optimizer_updates=sum(r['optimizer_updates'] for r in plan),warmup_updates=round(sum(r['optimizer_updates'] for r in plan)*cfg['train']['warmup_fraction']),
            per_epoch=plan),sequence_isolation='PASS',optimizer_steps=0,scheduler_steps=0,training_epochs=0,
        definitions=dict(event='one LiDAR frame',query='query_time plus latest 20 causal merged events',
            clip='up to 8 consecutive same-sequence queries',batch='B complete clips',eqs_scope='clip selection only'))
    modules=[load_test(name) for name in ('test_lidar_uav_v2_eqs','test_lidar_uav_v2_sequence_isolation','test_lidar_uav_v2_query_causal','test_lidar_uav_v2_sbe')]
    with (out/'test_report.txt').open('w') as handle:
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules]))
        spatial=load_test('test_lidar_uav_v2');spatial_names=[]
        for name,fn in vars(spatial).items():
            if name.startswith('test_') and callable(fn):fn();spatial_names.append(name);handle.write(name+' PASS\n')
    if not result.wasSuccessful():raise AssertionError('Regression failed')
    report['tests']=dict(unittest=result.testsRun,spatial=len(spatial_names),total=result.testsRun+len(spatial_names),
        causal_results=modules[2].RESULTS,isolation_results=modules[1].RESULTS)
    report['git_head_before']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write_csv(out/'query_subsampling_per_sequence.csv',per_sequence)
    write_csv(out/'query_appearance_stride1.csv',appearance_rows[1])
    write_csv(out/'query_appearance_stride4.csv',appearance_rows[4])
    write_csv(out/'stride4_four_epoch_coverage.csv',coverage_rows)
    (out/'query_subsampling_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='scheduler_dry_run'},indent=2))
    print('planned_optimizer_updates',report['scheduler_dry_run']['total_planned_optimizer_updates'])


if __name__=='__main__':main()
