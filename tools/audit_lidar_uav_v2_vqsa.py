#!/usr/bin/env python3
"""Read-only VQSA real-clip structure, attention, complexity, and regression audit."""
from __future__ import annotations

import csv,importlib.util,json,subprocess,sys,unittest
from pathlib import Path

import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import LiDARUAVDataset,LiDARUAVDetector,TemporalQueryClipDataset,collate_temporal_queries


def load_test(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    output=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/vqsa_structure_audit'
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
    queries=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'])
    clips=TemporalQueryClipDataset(queries,cfg['data']['query_clip_length'],stride=cfg['data']['query_clip_stride'])
    clip_index=next(i for i,r in enumerate(clips.clip_metadata) if r['sequence_id']=='seq0001' and r['valid_query_slots']==8)
    item=clips[clip_index];batch=collate_temporal_queries([item],unique_query_packing=True)
    model=LiDARUAVDetector(cfg).eval();hierarchy=model.hierarchy(batch['points'],batch['point_batch_index'])
    with torch.no_grad():l0_token,debug=model.voxel_embed(batch,hierarchy,True);result=model(batch)
    occupied=debug['slot_count']>0;occupied_count=occupied.sum(1).cpu().numpy();attention=debug['attention'].cpu()
    visible=int(occupied.sum());v0=len(occupied);heads=model.voxel_embed.vqsa.heads
    examples=[]
    for i in range(min(32,v0)):
        examples.append(dict(voxel_index=i,batch_index=int(hierarchy.levels[0].batch_index[i]),
            voxel_coord=' '.join(map(str,hierarchy.levels[0].coords[i].tolist())),
            occupancy=''.join(map(str,occupied[i].to(torch.int).tolist())),occupied_slots=int(occupied_count[i]),
            head0=' '.join(f'{x:.9g}' for x in attention[i,0,0].tolist()),
            head1=' '.join(f'{x:.9g}' for x in attention[i,1,0].tolist())))
    with (output/'vqsa_attention_examples.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
    # Find two different real inputs and verify their debug attention is not identical.
    dynamic_attention_diff=0.
    for i in range(1,min(v0,1000)):
        if not torch.equal(occupied[0],occupied[i]) or not torch.equal(debug['slot_stats'][0],debug['slot_stats'][i]):
            dynamic_attention_diff=float((attention[0]-attention[i]).abs().max());break
    if dynamic_attention_diff==0.:raise AssertionError('Real-input attention did not change')
    complexity=dict(per_voxel_mac=dict(slot_projection_8x14x16=1792,mha_q_projection_1x16x16=256,
        mha_k_projection_8x16x16=2048,mha_v_projection_8x16x16=2048,mha_output_projection_1x16x16=256,
        qk_attention_2x1x8x8=128,av_aggregation_2x1x8x8=128,final_projection_increment_16x128=2048,
        total_increment_over_fixed_sbe=8704),attention_design=dict(query_length=1,key_value_length=8,heads=2,
        head_dim=8,full_slot_self_attention=False),pointwise_high_dimensional_learned_expansion=False)
    (output/'vqsa_complexity_report.json').write_text(json.dumps(complexity,indent=2))
    test_names=('test_lidar_uav_v2_vqsa','test_lidar_uav_v2_sbe','test_lidar_uav_v2_eooe','test_lidar_uav_v2_uqp',
        'test_lidar_uav_v2_eqs','test_lidar_uav_v2_sequence_isolation','test_lidar_uav_v2_recent_contract','test_lidar_uav_v2_query_causal')
    modules=[load_test(name) for name in test_names]
    with (output/'test_report.txt').open('w') as handle:
        tests=unittest.TextTestRunner(stream=handle,verbosity=2).run(unittest.TestSuite(
            [unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules]))
        spatial=load_test('test_lidar_uav_v2');spatial_names=[]
        for name,fn in vars(spatial).items():
            if name.startswith('test_') and callable(fn):fn();spatial_names.append(name);handle.write(name+' PASS\n')
    if not tests.wasSuccessful():raise AssertionError('Regression test failure')
    uqp=modules[3].RESULTS;recent=modules[6].RESULTS;vqsa=modules[0].RESULTS
    parameters=sum(p.numel() for p in model.parameters())
    all_finite=all(torch.isfinite(result[key]).all().item() for key in ('logits','pred_xyz','fine_features'))
    report=dict(status='V2 VQSA STRUCTURE READY',parameters=dict(before=1324683,after=parameters,delta=parameters-1324683,
        vqsa_and_sbe=sum(p.numel() for p in model.voxel_embed.parameters())),complexity=complexity,
        real_smoke=dict(sequence_id='seq0001',clip_index=clip_index,raw_points=len(batch['points']),queries=8,
            token_shapes=dict(slot_stats=list(debug['slot_stats'].shape),slot_input=list(debug['slot_input'].shape),
                slot_tokens=list(debug['slot_tokens'].shape),voxel_query=list(debug['voxel_query'].shape),
                dynamic_attention_output=list(debug['dynamic_sequence'].shape),dynamic_summary=list(debug['dynamic_summary'].shape),
                L0=list(l0_token.shape),L1=[len(hierarchy.levels[1].coords),128],L2=[len(hierarchy.levels[2].coords),128]),
            occupied_slots=dict(mean=float(np.mean(occupied_count)),median=float(np.median(occupied_count)),
                p90=float(np.percentile(occupied_count,90)),max=int(np.max(occupied_count))),voxel_queries=v0,
            visible_slot_tokens=visible,logical_visible_attention_pairs=visible,dense_fixed_attention_logits=v0*8*heads,
            attention_heads=heads,real_attention_max_diff=dynamic_attention_diff,future_event_count=sum(
                sum(t>q['query_time'] for t in q['event_timestamps']) for q in item['queries']),all_outputs_finite=all_finite),
        regressions=dict(tests=tests.testsRun+len(spatial_names),empty_slot_mask='PASS',single_slot_attention='PASS',
            position_sensitivity='PASS',point_permutation='PASS',vqsa_permutation_max_diff=vqsa.get('permutation_max_diff'),
            uqp='PASS',uqp_output_diffs=dict(spatial=uqp.get('spatial')),
            uqp_loss_diffs=uqp.get('loss_diffs'),uqp_gradient_diffs=uqp.get('gradient_diffs'),sbe='PASS',eooe='PASS',
            eqs='PASS',sequence_isolation='PASS',recent_contract='PASS',recent_gradient_max_diff=max(recent['gradient_diffs'].values()),
            causal_input_window='PASS'),pointwise_learned_expansion=False,optimizer_steps=0,scheduler_steps=0,
        training_epochs=0,checkpoint_optimization=False,git_head_before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (output/'vqsa_structure_report.json').write_text(json.dumps(report,indent=2))
    (output/'vqsa_regression_report.json').write_text(json.dumps(report['regressions'],indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
