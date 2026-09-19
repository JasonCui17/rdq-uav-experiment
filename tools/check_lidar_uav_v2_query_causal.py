#!/usr/bin/env python3
"""Structure-only tests and real no_grad smoke. Never creates an optimizer."""
import hashlib,importlib.util,json,subprocess,sys,types,unittest
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (LiDARUAVDetector,LiDARUAVDataset,LiDARQueryBuilder,
    collate_temporal_queries,collate_lidar_samples,CandidateSelector)
from rdq_uav.multimodal.merged_lidar import select_last_history

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    output=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/query_causal_v1_structure_smoke'
    output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('query_tests',ROOT/'tests/test_lidar_uav_v2_query_causal.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    with (output/'test_report.txt').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():raise AssertionError('Query causal tests failed')
    spatial_spec=importlib.util.spec_from_file_location('spatial_tests',ROOT/'tests/test_lidar_uav_v2.py')
    spatial_tests=importlib.util.module_from_spec(spatial_spec);spatial_spec.loader.exec_module(spatial_tests)
    passed=[]
    for name,fn in vars(spatial_tests).items():
        if name.startswith('test_') and callable(fn):fn();passed.append(name)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
    ds=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'],sequence_limit=1)
    # First full-history query with seven following queries in the same sequence.
    start=next(i for i,r in enumerate(ds.records[:-7]) if len(select_last_history(ds.builder.stream(r['sequence_id']),r['query_time'],20))==20)
    queries=[ds[i] for i in range(start,start+8)];batch=collate_temporal_queries([{'queries':queries}])
    # Load frozen V2-base model class from committed source without importing V1.
    baseline_commit='38f939a7c4be8215a5f53ae5d7d0d84f29679c56'
    source=subprocess.check_output(['git','show',f'{baseline_commit}:src/rdq_uav/lidar_v2/model.py'],cwd=ROOT,text=True)
    mod=types.ModuleType('rdq_uav.lidar_v2._frozen_base');mod.__package__='rdq_uav.lidar_v2'
    exec(compile(source,'frozen_v2_base_model.py','exec'),mod.__dict__)
    base=mod.LiDARUAVDetector(cfg).eval();current=LiDARUAVDetector(cfg).eval()
    state=torch.load(ROOT/'outputs/own_multimodal_research/lidar_uav_v1_frozen_20260919/pilot_3epoch_seed42/best.pt',map_location='cpu')['model_state']
    base.load_state_dict(state,strict=True);load=current.load_spatial_v2_base_weights(state)
    one=collate_lidar_samples([queries[0]]);old=dict(one,gt_xyz=one['target_xyz'])
    with torch.no_grad():
        ref=base(old);new=current(one)
        full=current(batch)
    keys=('logits','residual_xyz','pred_xyz','fine_features','voxel_centers')
    diffs={k:float((ref[k]-new[k]).abs().max()) for k in keys}
    assert all(d<=1e-6 for d in diffs.values()),diffs
    tq=(queries[0]['query_time']+queries[1]['query_time'])/2
    arbitrary=ds.builder.build_inference_query(queries[0]['sequence_id'],tq)
    free=collate_lidar_samples([arbitrary])
    for k in ('gt_xyz','gt_timestamp','target_xyz','target_timestamp'):assert k not in free
    with torch.no_grad():a=current(free);candidates=CandidateSelector(cfg)(a)
    assert torch.isfinite(a['temporal_pred_xyz']).all()
    assert all(t<=tq for t in arbitrary['event_timestamps']) and bool((arbitrary['delta_t']<=0).all())
    per_query=[]
    for i,q in enumerate(queries):
        per_query.append(dict(sample_id=q['sample_id'],query_time=q['query_time'],event_count=q['event_count'],
            point_count=len(q['points']),has_observation=q['has_observation'],
            token_counts=[int((level.batch_index==i).sum()) for level in full['layouts'].levels]))
    params=sum(p.numel() for p in current.parameters());spatial=sum(p.numel() for p in base.parameters())
    report=dict(status='V2 QUERY-CAUSAL STRUCTURE READY',query_tests=result.testsRun,spatial_tests=len(passed),
        spatial_test_names=passed,causal_tests=tests.RESULTS,params=dict(total=params,spatial=spatial,temporal_added=params-spatial),
        spatial_compatibility=dict(status='PASS',base_commit=baseline_commit,max_abs_diff=diffs,
            token_counts_equal=ref['aux_stats']['token_counts']==new['aux_stats']['token_counts'],missing_keys=load.missing_keys,unexpected_keys=load.unexpected_keys),
        arbitrary_query=dict(status='PASS',query_time=tq,target_provided=False,event_count=arbitrary['event_count'],raw_candidates=len(candidates[0]['raw']['xyz'])),
        real_smoke=dict(status='PASS',device='cpu',dtype='float32',clip_length=8,queries=per_query,
            total_points=len(batch['points']),token_counts=full['aux_stats']['token_counts'],
            query_token_shape=list(full['query_token'].shape),temporal_hidden_shape=list(full['temporal_hidden'].shape),
            temporal_pred_xyz_shape=list(full['temporal_pred_xyz'].shape),future_event_count=0),
        optimizer_updates=0,training_epochs=0,checkpoint_optimization=False,
        gradient_test_only='synthetic test E backward; no optimizer or scheduler constructed')
    (output/'structure_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
