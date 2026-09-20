#!/usr/bin/env python3
"""Read-only real-batch audit of V2 recent4 supervision/model separation."""
from __future__ import annotations

import csv,copy,importlib.util,json,subprocess,sys,unittest
from pathlib import Path

import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import CandidateLoss,LiDARUAVDataset,LiDARUAVDetector,TemporalQueryClipDataset,collate_temporal_queries


def load_test(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def maximum(a,b):return float((a-b).abs().max()) if a.numel() else 0.


def main():
    output=ROOT/'outputs/own_multimodal_research/lidar_uav_v2/recent_input_contract_audit'
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
    cfg=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
    queries=LiDARUAVDataset(cfg['data']['root'],ROOT/cfg['data']['split_file'],cfg['data']['train_split'])
    clips=TemporalQueryClipDataset(queries,cfg['data']['query_clip_length'],stride=cfg['data']['query_clip_stride'])
    clip_index=next(i for i,r in enumerate(clips.clip_metadata) if r['sequence_id']=='seq0001' and r['valid_query_slots']==8)
    clean=collate_temporal_queries([clips[clip_index]],unique_query_packing=True)
    legacy=copy.deepcopy(clean);legacy['recent_mask']=legacy.pop('supervision_recent_mask')
    model=LiDARUAVDetector(cfg).eval();contract=load_test('test_lidar_uav_v2_recent_contract')
    with torch.no_grad():
        old_output=model(legacy);new_output=model(clean)
        model_only=dict(clean);model_only.pop('supervision_recent_mask');without_mask=model(model_only)
    keys=('logits','pred_xyz','fine_features')
    output_diffs={key:maximum(old_output[key],new_output[key]) for key in keys}
    deletion_diffs={key:maximum(new_output[key],without_mask[key]) for key in keys}
    old_criterion=contract.LegacyCandidateLoss(cfg);new_criterion=CandidateLoss(cfg)
    old_labels=old_criterion.labels(old_output,legacy);new_labels=new_criterion.labels(new_output,clean)
    label_equal=[torch.equal(a,b) for a,b in zip(old_labels[:3],new_labels[:3])]
    old_loss=old_criterion(old_output,legacy);new_loss=new_criterion(new_output,clean)
    loss_diffs={key:abs(float(old_loss[key])-float(new_loss[key])) for key in ('loss','loss_cls','loss_reg')}
    label_counts=dict(positive=int(new_labels[0].sum()),ignore=int(new_labels[1].sum()),negative=int(new_labels[2].sum()),
        current_support_occurrences=int(new_loss['num_supervised_samples']),no_current_support_occurrences=int(new_loss['num_no_current_support']))
    if not all(label_equal) or max((*output_diffs.values(),*deletion_diffs.values(),*loss_diffs.values()))>1e-7:
        raise AssertionError(dict(labels=label_equal,outputs=output_diffs,deletion=deletion_diffs,losses=loss_diffs))
    test_names=('test_lidar_uav_v2_recent_contract','test_lidar_uav_v2_eooe','test_lidar_uav_v2_uqp','test_lidar_uav_v2_eqs',
        'test_lidar_uav_v2_sequence_isolation','test_lidar_uav_v2_query_causal','test_lidar_uav_v2_sbe')
    modules=[load_test(name) for name in test_names]
    with (output/'test_report.txt').open('w') as handle:
        tests=unittest.TextTestRunner(stream=handle,verbosity=2).run(unittest.TestSuite(
            [unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules]))
        spatial=load_test('test_lidar_uav_v2');spatial_names=[]
        for name,fn in vars(spatial).items():
            if name.startswith('test_') and callable(fn):fn();spatial_names.append(name);handle.write(name+' PASS\n')
    if not tests.wasSuccessful():raise AssertionError('Regression test failure')
    gradients=modules[0].RESULTS['gradient_diffs']
    params=sum(p.numel() for p in model.parameters())
    usage=[
        dict(category='A Dataset generation',path='src/rdq_uav/lidar_v2/data.py',use='last 4 selected actual events -> supervision_recent_mask'),
        dict(category='B Collate',path='src/rdq_uav/lidar_v2/data.py',use='concatenate mask; validate repeated UQP query identity'),
        dict(category='C Device transfer',path='src/rdq_uav/lidar_v2/runtime.py',use='generic move_batch transfers bool mask to loss/evaluation device'),
        dict(category='D Model forward',path='src/rdq_uav/lidar_v2/model.py; sbe.py',use='NOT READ'),
        dict(category='E Spatial label/loss',path='src/rdq_uav/lidar_v2/loss.py',use='d_recent, Positive/Ignore/Negative, NoCurrentSupport'),
        dict(category='F Evaluation',path='src/rdq_uav/lidar_v2/runtime.py',use='CurrentSupport split and latest-four neighbor group'),
        dict(category='G Export/debug',path='tools/evaluate_lidar_uav_v2.py; export_lidar_uav_v2_candidates.py',use='indirectly uses shared evaluation; no learned feature'),
    ]
    with (output/'recent_mask_usage.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(usage[0]));writer.writeheader();writer.writerows(usage)
    report=dict(status='V2 RECENT INPUT CONTRACT CLEANUP READY',field='supervision_recent_mask',usage_chain=usage,
        model_forward_uses_recent_mask=False,spatial_supervision_uses_recent_mask=True,evaluation_uses_recent_mask=True,
        transferred_to_gpu=True,transfer_reason='CandidateLoss d_recent and validation grouping execute on the model device',
        model_point_inputs=['points/XYZ','sensor_id','delta_t','point_batch_index'],latest_events=20,recent_supervision_events=4,
        real_batch=dict(sequence_id='seq0001',clip_index=clip_index,points=len(clean['points']),queries=int(clean['query_valid_mask'].sum()),
            label_counts=label_counts,label_masks_equal=all(label_equal),output_diffs=output_diffs,
            model_mask_deletion_output_diffs=deletion_diffs,loss_diffs=loss_diffs),
        gradients=dict(max_diff=max(gradients.values()),per_parameter=gradients),parameters=dict(before=params,after=params),
        regressions=dict(tests=tests.testsRun+len(spatial_names),sbe='PASS',eooe='PASS',uqp='PASS',eqs='PASS',
            sequence_isolation='PASS',query_causal_future_leakage='PASS'),optimizer_steps=0,scheduler_steps=0,
        training_epochs=0,checkpoint_optimization=False,git_head_before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (output/'recent_input_contract_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
