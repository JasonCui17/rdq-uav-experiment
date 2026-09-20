"""Spatial V2 and causal LiDAR-input regressions; no optimizer is constructed."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (CandidateLoss,CandidateSelector,LiDARQueryBuilder,
    LiDARUAVDetector,collate_lidar_samples)
from rdq_uav.lidar_v2.runtime import evaluate_batch,summarize_metrics

CFG=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
RESULTS={}

def query(i,points=None,target=True,valid=True):
    points=torch.tensor([[.2+i*.01,.1,.1],[1.3,.2,.1],[3.1,.1,.2]]) if points is None else points
    n=len(points)
    q=dict(points=points.float(),sensor_id=torch.arange(n)%2,delta_t=torch.full((n,),-.01),
           supervision_recent_mask=torch.ones(n,dtype=torch.bool),sequence_id='synthetic',sample_id=f'q{i}',query_uid=i,
           query_time=100.+i*.07,event_count=1,event_timestamps=[100.+i*.07-.01],event_sequence_ids=['synthetic'],
           has_observation=n>0,target_valid=target)
    if target:q.update(target_xyz=torch.tensor([.2,.1,.1]),target_timestamp=q['query_time'])
    return q

def batch(queries):return collate_lidar_samples(queries)
def model():torch.manual_seed(42);return LiDARUAVDetector(CFG).eval()
def diff(a,b):return float((a-b).abs().max()) if a.numel() else 0.
def occurrence(output,index):
    selected=output['batch_index']==index
    return {key:output[key][selected] for key in ('logits','residual_xyz','pred_xyz','fine_features','voxel_centers','source_token_id')}

class SpatialCandidateTests(unittest.TestCase):
    def test_validation_endpoint_and_export_interface(self):
        from rdq_uav.lidar_v2.training import validate
        loader=[batch([query(0)]),batch([query(1,torch.empty((0,3)))])]
        with tempfile.TemporaryDirectory() as directory:
            metrics,rows,health=validate(model(),loader,CandidateLoss(CFG),CandidateSelector(CFG),
                                       torch.device('cpu'),precision='fp32',export_dir=Path(directory)/'export')
            self.assertEqual(len(rows),2);self.assertEqual(metrics['all']['samples'],2)
            self.assertEqual(metrics['no_current_support']['samples'],1)
            self.assertNotIn('temporal_loss',health);self.assertNotIn('temporal_success_1m',metrics['all'])
            self.assertTrue((Path(directory)/'export/candidate_features.npz').exists())

    def test_future_query_observations_do_not_change_earlier_spatial_outputs(self):
        m=model();qs=[query(i) for i in range(5)];changed=copy.deepcopy(qs)
        for q in changed[3:]:q['points']=q['points']*20+500
        with torch.no_grad():a=m(batch(qs));b=m(batch(changed))
        maximum=0.
        for i in range(3):
            for key in ('logits','pred_xyz','fine_features'):maximum=max(maximum,diff(occurrence(a,i)[key],occurrence(b,i)[key]))
        RESULTS['future_lidar_spatial_max_diff']=maximum;self.assertLessEqual(maximum,1e-6)

    def test_target_metadata_never_enters_model(self):
        m=model();a=batch([query(i) for i in range(3)]);b=copy.deepcopy(a)
        b['target_xyz'][:]=1e8;b['target_timestamp'][:]=0
        with torch.no_grad():x=m(a);y=m(b)
        for key in ('logits','residual_xyz','pred_xyz','fine_features'):self.assertEqual(diff(x[key],y[key]),0.)

    def test_future_query_removal_preserves_existing_spatial_outputs(self):
        m=model();qs=[query(i) for i in range(5)]
        with torch.no_grad():a=m(batch(qs));b=m(batch(qs[:3]))
        for i in range(3):
            for key in ('logits','pred_xyz','fine_features'):self.assertLessEqual(diff(occurrence(a,i)[key],occurrence(b,i)[key]),1e-6)

    def test_gt_free_query_and_independent_target_time(self):
        m=model();b=batch([query(0,target=False),query(1,target=False)])
        for key in ('gt_xyz','gt_timestamp'):self.assertNotIn(key,b)
        with torch.no_grad():out=m(b)
        self.assertTrue(torch.isfinite(out['logits']).all() and torch.isfinite(out['pred_xyz']).all())
        q=query(2);q['target_timestamp']=q['query_time']+99
        self.assertNotEqual(collate_lidar_samples([q])['target_timestamp'][0],q['query_time'])

    def test_no_current_support_skips_spatial_loss(self):
        m=model();b=batch([query(0,torch.tensor([[20.,20.,20.]]))]);out=m(b);loss=CandidateLoss(CFG)(out,b)
        self.assertEqual(loss['num_supervised_samples'],0);self.assertEqual(float(loss['loss']),0.)

    def test_empty_observation_has_no_fake_candidate(self):
        m=model();q=query(0,torch.empty((0,3)));b=collate_lidar_samples([q])
        with torch.no_grad():out=m(b)
        self.assertEqual(len(out['logits']),0);self.assertEqual(len(out['pred_xyz']),0)
        self.assertEqual(CandidateLoss(CFG)(out,b)['num_supervised_samples'],0)

    def test_builder_last20_recent4_and_cleaning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('livox_avia','lidar_360'):(root/'seq'/name).mkdir(parents=True)
            for i in range(30):
                name='livox_avia' if i%2==0 else 'lidar_360'
                np.save(root/'seq'/name/f'{i}.npy',np.array([[i+1.,1,1],[0,0,0],[np.nan,1,1]]))
            q=LiDARQueryBuilder(root).build('seq',25.5,target_xyz=[1,2,3],target_timestamp=99,target_valid=True)
            self.assertEqual(q['event_count'],20);self.assertEqual(q['event_timestamps'],list(map(float,range(6,26))))
            self.assertTrue(bool((q['delta_t']<=0).all()));self.assertEqual(int(q['supervision_recent_mask'].sum()),4)
            self.assertEqual(len(q['points']),20);self.assertEqual(q['sensor_id'].tolist(),[i%2 for i in range(6,26)])

    def test_forward_is_exact_spatial_forward(self):
        m=model();b=batch([query(0),query(1)])
        with torch.no_grad():a=m(b);z=m.spatial_forward(b)
        expected={'logits','residual_xyz','pred_xyz','fine_features','voxel_centers','source_token_id','batch_index','layouts','aux_stats'}
        self.assertEqual(set(a),expected)
        for key in expected-{'layouts','aux_stats'}:self.assertEqual(diff(a[key],z[key]),0.,key)

    def test_gt_free_static_model_and_no_v1_import(self):
        source=(ROOT/'src/rdq_uav/lidar_v2/model.py').read_text()
        for key in ('gt_xyz','gt_timestamp','target_xyz','target_timestamp'):self.assertNotIn(key,source)
        for p in (ROOT/'src/rdq_uav/lidar_v2').glob('*.py'):self.assertNotIn('from rdq_uav.lidar_v1',p.read_text())

    def test_validation_scores_each_query_once_and_spatial_groups(self):
        class D:
            records=[dict(sequence_id=s,query_time=i,sample_id=f'{s}{i}') for s in ('a','b') for i in range(3)]
            def __getitem__(self,i):
                q=query(i%3);q['sequence_id']=self.records[i]['sequence_id'];q['event_sequence_ids']=[q['sequence_id']];q['sample_id']=str(i);return q
        ds=D();self.assertEqual(len(ds.records),6)
        rows=[];m=model()
        with torch.no_grad():
            for i in range(6):
                b=collate_lidar_samples([ds[i]]);rows.extend(evaluate_batch(m(b),b,CandidateSelector(CFG),CandidateLoss(CFG)))
        self.assertEqual(len(rows),6);self.assertEqual(len({r['sample_id'] for r in rows}),6)
        metrics=summarize_metrics(rows)['all'];self.assertIn('nms_recall_at_10_1m',metrics);self.assertNotIn('temporal_success_1m',metrics)

if __name__=='__main__':unittest.main()
