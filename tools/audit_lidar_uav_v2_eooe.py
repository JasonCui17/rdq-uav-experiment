#!/usr/bin/env python3
"""Read-only EOOE structure audit on one real eight-query MMAUD clip."""
from __future__ import annotations

import csv,importlib.util,json,subprocess,sys,unittest
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import LiDARUAVDataset,LiDARUAVDetector,TemporalQueryClipDataset,collate_temporal_queries


def load_test(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write_csv(path,rows,fieldnames=None):
    fieldnames=fieldnames or list(rows[0])
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)


def occupancy_audit(name,occupancy):
    cpu=occupancy.to(torch.int64).cpu();counts=cpu.sum(1).numpy();patterns=[''.join(map(str,row.tolist())) for row in cpu]
    frequency=Counter(patterns);by_count=defaultdict(set)
    for pattern,count in zip(patterns,counts):by_count[int(count)].add(pattern)
    ambiguous=sum(1 for count in counts if len(by_count[int(count)])>1)
    summary=dict(level=name,parent_count=len(patterns),mean_occupied_slots=float(np.mean(counts)),
        median_occupied_slots=float(np.median(counts)),p90_occupied_slots=float(np.percentile(counts,90)),
        max_occupied_slots=int(np.max(counts)),unique_pattern_count=len(frequency),
        all_8_occupied_ratio=float(np.mean(counts==8)),single_child_ratio=float(np.mean(counts==1)),
        parents_in_multi_pattern_child_count_ratio=ambiguous/len(patterns))
    distribution=[]
    for pattern,n in frequency.most_common():
        distribution.append(dict(level=name,pattern=pattern,occupied_slots=pattern.count('1'),count=n,frequency=n/len(patterns)))
    count_rows=[]
    for child_count in sorted(by_count):
        count_rows.append(dict(level=name,child_count=child_count,parent_count=int(np.sum(counts==child_count)),
            unique_occupancy_patterns=len(by_count[child_count]),patterns=';'.join(sorted(by_count[child_count]))))
    return summary,distribution,count_rows


def main():
    output=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/eooe_structure_audit'
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
    queries=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'])
    clips=TemporalQueryClipDataset(queries,cfg['data']['query_clip_length'],stride=cfg['data']['query_clip_stride'])
    clip_index=next(i for i,r in enumerate(clips.clip_metadata)
        if r['sequence_id']=='seq0001' and r['valid_query_slots']==cfg['data']['query_clip_length'])
    item=clips[clip_index];batch=collate_temporal_queries([item],unique_query_packing=True)
    model=LiDARUAVDetector(cfg).eval()
    with torch.no_grad():result=model(batch)
    hierarchy=result['layouts'];l0,l1,l2=hierarchy.levels
    input01,occupancy01,_=model.merge01.parent_input(result['fine_features'].new_zeros((len(l0.coords),128)),l0,l1,hierarchy.parent_l0_to_l1)
    input12,occupancy12,_=model.merge12.parent_input(result['fine_features'].new_zeros((len(l1.coords),128)),l1,l2,hierarchy.parent_l1_to_l2)
    summaries=[];distribution=[];count_rows=[]
    for name,occupancy in (('L0_to_L1',occupancy01),('L1_to_L2',occupancy12)):
        summary,patterns,counts=occupancy_audit(name,occupancy);summaries.append(summary);distribution.extend(patterns);count_rows.extend(counts)
    test_names=('test_lidar_uav_v2_eooe','test_lidar_uav_v2_uqp','test_lidar_uav_v2_eqs',
        'test_lidar_uav_v2_sequence_isolation','test_lidar_uav_v2_query_causal','test_lidar_uav_v2_sbe')
    modules=[load_test(name) for name in test_names]
    with (output/'test_report.txt').open('w') as handle:
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
        tests=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
        spatial=load_test('test_lidar_uav_v2');spatial_names=[]
        for name,fn in vars(spatial).items():
            if name.startswith('test_') and callable(fn):fn();spatial_names.append(name);handle.write(name+' PASS\n')
    if not tests.wasSuccessful():raise AssertionError('Regression test failure')
    params=sum(p.numel() for p in model.parameters());spatial=params
    future_events=sum(sum(t>q['query_time'] for t in q['event_timestamps']) for q in item['queries'])
    if future_events:raise AssertionError(f'Future events: {future_events}')
    uqp=modules[1].RESULTS;causal=modules[4].RESULTS;isolation=modules[3].RESULTS
    report=dict(status='V2 EOOE STRUCTURE READY',old_sparse_merge_input=dict(fields=['max_child_feature[128]',
        'mean_child_feature[128]','log1p(raw_point_count)[1]','child_voxel_count/8[1]'],dimension=258),
        removed_fields=['child_voxel_count/8'],new_sparse_merge_input=dict(fields=['max_child_feature[128]',
        'mean_child_feature[128]','log1p(raw_point_count)[1]','explicit_octant_occupancy[8]'],dimension=265),
        slot_convention='4*x + 2*y + z; identical to SBE-Lite',raw_point_count_propagation='L0 raw points; L1/L2 sum child raw_point_count',
        parameters=dict(before=1322891,after=params,delta=params-1322891,spatial=spatial,
            merge01_parent=list(model.merge01.parent.weight.shape),merge12_parent=list(model.merge12.parent.weight.shape)),
        real_smoke=dict(sequence_id='seq0001',clip_index=clip_index,query_count=len(item['queries']),
            query_times=[q['query_time'] for q in item['queries']],points=len(batch['points']),future_event_count=future_events,
            token_shapes=dict(L0=[len(l0.coords),128],L1=[len(l1.coords),128],L2=[len(l2.coords),128]),
            occupancy_shapes=dict(L0_to_L1=list(occupancy01.shape),L1_to_L2=list(occupancy12.shape)),
            parent_input_shapes=dict(L0_to_L1=list(input01.shape),L1_to_L2=list(input12.shape)),
            all_outputs_finite=all(torch.isfinite(result[k]).all().item()
                for k in ('logits','pred_xyz','fine_features'))),
        occupancy=summaries,regressions=dict(tests=tests.testsRun+len(spatial_names),eooe='PASS',
            permutation_invariance='PASS',topology_sensitivity='PASS',uqp_exact='PASS',eqs='PASS',sequence_isolation='PASS',
            causal_input_window='PASS',sbe='PASS',uqp_output_diffs=dict(spatial=uqp.get('spatial')),uqp_loss_diffs=uqp.get('loss_diffs'),
            uqp_gradient_diffs=uqp.get('gradient_diffs'),future_leakage=causal,sequence_isolation_details=isolation),
        optimizer_steps=0,scheduler_steps=0,training_epochs=0,checkpoint_optimization=False,
        git_head_before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_csv(output/'eooe_occupancy_statistics.csv',count_rows)
    write_csv(output/'eooe_pattern_distribution.csv',distribution)
    (output/'eooe_structure_report.json').write_text(json.dumps(report,indent=2))
    (output/'eooe_regression_report.json').write_text(json.dumps(report['regressions'],indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
