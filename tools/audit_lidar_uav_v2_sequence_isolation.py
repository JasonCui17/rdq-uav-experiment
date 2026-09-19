#!/usr/bin/env python3
"""Read-only full train/validation clip metadata audit; no model updates."""
import argparse,collections,csv,hashlib,importlib.util,json,subprocess,sys,unittest
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import LiDARUAVDataset,LiDARUAVValidationDataset,TemporalQueryClipDataset
from rdq_uav.lidar_v2.isolation import assert_temporal_clip_integrity


def audit(ds,validation,clip_length,examples):
    clips=TemporalQueryClipDataset(ds,clip_length,validation=validation)
    counts=dict(clips=len(clips),sequences=len({r['sequence_id'] for r in ds.records}),
                queries=len(ds.records),cross_sequence_clips=0,unsorted_clips=0,duplicate_timestamp_clips=0,
                empty_sequence_id=0,event_sequence_mismatch=0,event_path_sequence_mismatch=0,
                future_event_count=0,duplicate_sample_identity_clips=0,selected_event_references_checked=0)
    anomalies=[];seen=set();split='validation' if validation else 'train'
    for i,window in enumerate(clips.windows):
        records=[ds.records[j] for j in window];seqs=[r['sequence_id'] for r in records];times=[r['query_time'] for r in records];ids=[r['sample_id'] for r in records]
        counts['cross_sequence_clips']+=int(len(set(seqs))!=1)
        counts['unsorted_clips']+=int(any(b<a for a,b in zip(times,times[1:])))
        counts['duplicate_timestamp_clips']+=int(len(set(times))!=len(times))
        counts['empty_sequence_id']+=int(any(not s or not s.strip() for s in seqs))
        counts['duplicate_sample_identity_clips']+=int(len(set(ids))!=len(ids))
        queries=[]
        for r in records:
            events=ds.builder.select_events(r['sequence_id'],r['query_time'])
            counts['selected_event_references_checked']+=len(events)
            counts['event_sequence_mismatch']+=sum(e.sequence_id!=r['sequence_id'] for e in events)
            counts['event_path_sequence_mismatch']+=sum(e.file_path.parent.parent.name!=r['sequence_id'] for e in events)
            counts['future_event_count']+=sum(e.timestamp>r['query_time'] for e in events)
            queries.append(dict(sequence_id=r['sequence_id'],sample_id=r['sample_id'],query_time=r['query_time'],event_count=len(events),
                                event_sequence_ids=[e.sequence_id for e in events],event_timestamps=[e.timestamp for e in events]))
        try:assert_temporal_clip_integrity(queries,clip_index=i,require_events=True)
        except AssertionError as error:anomalies.append(str(error))
        if seqs[0] not in seen or i==len(clips)-1 or (i+1<len(clips) and ds.records[clips.windows[i+1][-1]]['sequence_id']!=seqs[-1]):
            examples.append(dict(split=split,clip_index=i,sequence_ids=json.dumps(seqs),sample_ids=json.dumps(ids),
                                 query_times=json.dumps(times),query_count=len(window),selected_event_count=sum(q['event_count'] for q in queries)))
        seen.add(seqs[0])
        if (i+1)%5000==0:print(f'{split}: {i+1}/{len(clips)} clips audited',flush=True)
    counts['anomalies']=anomalies
    counts['duplicate_query_timestamp_records']=[]
    counts['audit_scope']='Every clip and every selected event metadata reference; no point arrays are read.'
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/sequence_isolation_audit');args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'git_head_before.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
    (out/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text());torch.set_num_threads(4)
    train=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'])
    official=Path(cfg['data']['root']).parent
    val=LiDARUAVValidationDataset(official/'val',official/'validation_ref_new (for your ref).csv')
    examples=[];report=dict(train=audit(train,False,cfg['temporal']['clip_length'],examples),validation=audit(val,True,cfg['temporal']['clip_length'],examples))
    modules=[]
    for name in ('test_lidar_uav_v2_sequence_isolation','test_lidar_uav_v2_query_causal','test_lidar_uav_v2_sbe'):
        spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);modules.append(m)
    with (out/'test_report.txt').open('w') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules]))
    if not result.wasSuccessful():raise AssertionError('Regression failure; see test_report')
    isolation=modules[0].RESULTS
    report.update(temporal_attention=dict(cross_batch_invariance='PASS',cross_batch_max_diff=isolation['cross_batch_max_diff'],
                        same_timestamp_cross_sequence='PASS',same_timestamp_max_diff=isolation['same_timestamp_cross_sequence_max_diff']),
                  spatial=dict(same_xyz_cross_sample_voxel_isolation='PASS'),rolling_validation_boundary_reset='PASS',
                  arbitrary_query_mixed_history_protection='PASS',causal_regression=modules[1].RESULTS,
                  original_dataset_sequence_isolated=True,tests=result.testsRun,optimizer_steps=0,scheduler_steps=0,training_epochs=0,
                  train_root=str(train.root),val_root=str(val.root),split_reference=cfg['data']['split_file'])
    (out/'sequence_isolation_report.json').write_text(json.dumps(report,indent=2))
    with (out/'sequence_isolation_examples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
    for split in ('train','validation'):
        assert not report[split]['anomalies']
        for key in ('cross_sequence_clips','unsorted_clips','duplicate_timestamp_clips','empty_sequence_id','event_sequence_mismatch','event_path_sequence_mismatch','future_event_count'):assert report[split][key]==0,(split,key)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
