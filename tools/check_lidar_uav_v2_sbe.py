#!/usr/bin/env python3
"""SBE-Lite structure tests and no_grad benchmarks; never creates an optimizer."""
import argparse,copy,hashlib,importlib.util,json,subprocess,sys,time,types,unittest
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import LiDARUAVDetector,LiDARUAVDataset,collate_temporal_queries,collate_lidar_samples,CandidateSelector,CandidateLoss
from rdq_uav.lidar_v2.runtime import move_batch
from rdq_uav.multimodal.merged_lidar import select_last_history


def load_tests(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def benchmark(fn,device,repeats,warmup=3):
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(warmup):result=fn();del result
        sync()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for i in range(repeats):
            sync();start=time.perf_counter();result=fn();sync()
            times.append((time.perf_counter()-start)*1000);del result
            if (i+1)%5==0:print(f'  timed {i+1}/{repeats}',flush=True)
    return dict(mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                repeats=repeats,warmup=warmup,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/sbe_lite_v1_structure_smoke')
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    if args.repeats<20:raise ValueError('At least 20 timed repetitions required')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'structure_report.json').exists():raise FileExistsError('Use new output directory to preserve results')
    torch.set_num_threads(args.threads);torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
    sbtests=load_tests('test_lidar_uav_v2_sbe');qtests=load_tests('test_lidar_uav_v2_query_causal')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (sbtests,qtests)])
    with (out/'test_report.txt').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    if not result.wasSuccessful():raise AssertionError('Regression failed; see test_report.txt')
    spatial=load_tests('test_lidar_uav_v2');spatial_names=[]
    for name,fn in vars(spatial).items():
        if name.startswith('test_') and callable(fn):fn();spatial_names.append(name)
    print(f'Tests PASS: {result.testsRun} SBE/query + {len(spatial_names)} spatial',flush=True)
    pre=(out/'pre_sbe_commit.txt').read_text().strip() if (out/'pre_sbe_commit.txt').exists() else '7c5ae42'
    source=subprocess.check_output(['git','show',f'{pre}:src/rdq_uav/lidar_v2/model.py'],cwd=ROOT,text=True)
    old_module=types.ModuleType('rdq_uav.lidar_v2._pre_sbe');old_module.__package__='rdq_uav.lidar_v2'
    exec(compile(source,'pre_sbe_model.py','exec'),old_module.__dict__)
    old=old_module.LiDARUAVDetector(cfg).eval();new=LiDARUAVDetector(cfg).eval()
    load_report=new.load_pre_sbe_weights(old.state_dict())
    # No trained Query-Causal checkpoint is assumed: this is state_dict inheritance.
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items() if not k.startswith('voxel_embed.'))
    old_params=sum(p.numel() for p in old.parameters());assert old_params==1330445
    old_embed=sum(p.numel() for p in old.voxel_embed.parameters());new_embed=sum(p.numel() for p in new.voxel_embed.parameters())
    new_params=sum(p.numel() for p in new.parameters())
    prefixes=('query_pool.','time_encoding.','presence_embedding.','temporal_transformer.','temporal_head.')
    temporal=sum(p.numel() for k,p in new.named_parameters() if k.startswith(prefixes))
    ds=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'],sequence_limit=1)
    start=next(i for i,r in enumerate(ds.records[:-7]) if len(select_last_history(ds.builder.stream(r['sequence_id']),r['query_time'],20))==20)
    queries=[ds[i] for i in range(start,start+8)];batch=collate_temporal_queries([{'queries':queries}])
    P=len(batch['points']);h=new.hierarchy(batch['points'],batch['point_batch_index']);V=len(h.levels[0].coords)
    linear_calls={}
    for name,model in (('legacy',old),('sbe',new)):
        calls=[]
        def record(module,inputs):calls.append(dict(in_features=module.in_features,out_features=module.out_features,input_shape=list(inputs[0].shape),input_elements=inputs[0].numel()))
        hooks=[m.register_forward_pre_hook(record) for m in model.voxel_embed.modules() if isinstance(m,torch.nn.Linear)]
        with torch.no_grad():model.voxel_embed(batch,h)
        for hook in hooks:hook.remove()
        linear_calls[name]=calls
    assert len(linear_calls['sbe'])==1 and linear_calls['sbe'][0]['input_shape']==[V,88]
    print('Real 8-query eval/no_grad forward...',flush=True)
    with torch.no_grad():
        stats=new.voxel_embed.slot_statistics(batch,h);embedded=new.voxel_embed(batch,h)
        full=new(batch)
        for key,value in full.items():
            if torch.is_tensor(value) and value.is_floating_point():assert torch.isfinite(value).all(),key
        old_full=old(batch)
        # Labels must be unchanged even though logits/features intentionally differ.
        criterion=CandidateLoss(cfg)
        for x,y in zip(criterion.labels(old_full,batch),criterion.labels(full,batch)):assert torch.equal(x,y)
        assert old_full['aux_stats']['token_counts']==full['aux_stats']['token_counts']
    tq=(queries[0]['query_time']+queries[1]['query_time'])/2
    query=ds.builder.build_inference_query(queries[0]['sequence_id'],tq);free=collate_lidar_samples([query])
    assert all(k not in free for k in ('gt_xyz','gt_timestamp','target_xyz','target_timestamp'))
    with torch.no_grad():a=new(free);candidates=CandidateSelector(cfg)(a)
    assert torch.isfinite(a['temporal_pred_xyz']).all()
    future=sum(t>q['query_time'] for q in queries+[query] for t in q['event_timestamps'])
    assert future==0 and all(bool((q['delta_t']<=0).all()) for q in queries+[query])
    real=dict(query_count=8,sequence_id=queries[0]['sequence_id'],sample_ids=[q['sample_id'] for q in queries],
              query_times=[q['query_time'] for q in queries],points=P,points_per_query=[len(q['points']) for q in queries],
              event_counts=[q['event_count'] for q in queries],token_counts=full['aux_stats']['token_counts'],
              slot_stats_shape=list(stats.shape),flatten_shape=list(stats.flatten(1).shape),l0_token_shape=list(embedded.shape),
              query_token_shape=list(full['query_token'].shape),temporal_hidden_shape=list(full['temporal_hidden'].shape),
              temporal_pred_xyz_shape=list(full['temporal_pred_xyz'].shape),future_event_count=future,all_outputs_finite=True,
              spatial_labels_unchanged=True)
    del full,old_full,a,stats,embedded
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    timings={}
    for name,model in (('legacy',old),('sbe',new)):
        model.to(device);b=move_batch(batch,device);layout=model.hierarchy(b['points'],b['point_batch_index'])
        print(f'Benchmark {name} embedding on {device}',flush=True)
        embedding=benchmark(lambda:model.voxel_embed(b,layout),device,args.repeats)
        del layout
        print(f'Benchmark {name} full model on {device}',flush=True)
        complete=benchmark(lambda:model(b),device,args.repeats)
        timings[name]=dict(embedding=embedding,full=complete)
        model.cpu();del b
        if device.type=='cuda':torch.cuda.empty_cache()
    preserved={}
    for name in ('data.py','geometry.py','loss.py','runtime.py','training.py','selector.py','temporal.py'):
        previous=subprocess.check_output(['git','show',f'{pre}:src/rdq_uav/lidar_v2/{name}'],cwd=ROOT)
        preserved[name]=previous==(ROOT/'src/rdq_uav/lidar_v2'/name).read_bytes()
    assert all(preserved.values())
    report=dict(status='V2 SBE-LITE STRUCTURE READY',pre_sbe_commit=pre,
        tests=dict(sbe=14,query_causal=13,spatial=len(spatial_names),causal_results=qtests.RESULTS),
        parameters=dict(old_total=old_params,legacy_embedding=old_embed,sbe_embedding=new_embed,spatial=new_params-temporal,
                        temporal=temporal,total=new_params,reduction=old_params-new_params,reduction_fraction=(old_params-new_params)/old_params),
        downstream_load=dict(status='PASS',source='pre-SBE class initialized state_dict; no trained temporal weights claimed',**load_report),
        arbitrary_query=dict(status='PASS',query_time=tq,target_present=False,spatial_candidate_count=len(candidates[0]['raw']['xyz'])),
        real_smoke=real,compute=dict(point_count=P,l0_voxels=V,point_to_voxel_ratio=P/V,linear_calls=linear_calls,
                                    sbe_pointwise_learned_calls=0),
        benchmark=dict(device=str(device),dtype='float32',threads=args.threads,torch=torch.__version__,
                       cuda_status='TESTED' if device.type=='cuda' else 'CUDA_NOT_TESTED',timings=timings,
                       embedding_speedup=timings['legacy']['embedding']['median_ms']/timings['sbe']['embedding']['median_ms'],
                       full_speedup=timings['legacy']['full']['median_ms']/timings['sbe']['full']['median_ms']),
        unchanged_modules=preserved,optimizer_steps=0,scheduler_steps=0,training_epochs=0,checkpoint_optimization=False)
    (out/'structure_report.json').write_text(json.dumps(report,indent=2))
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
