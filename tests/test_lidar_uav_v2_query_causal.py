"""No optimizer is constructed. Only test E uses backward to audit gradient flow."""
import ast,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (LiDARUAVDetector,LiDARQueryBuilder,collate_temporal_queries,
    collate_lidar_samples,CandidateLoss,QueryCausalLoss,TemporalQueryClipDataset)
from rdq_uav.lidar_v2.temporal import causal_mask
from rdq_uav.lidar_v2.runtime import evaluate_batch,summarize_metrics
from rdq_uav.lidar_v2 import CandidateSelector
CFG=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())
RESULTS={}

def query(i,points=None,target=True,valid=True):
    points=torch.tensor([[.2+i*.01,.1,.1],[1.3,.2,.1],[3.1,.1,.2]]) if points is None else points
    n=len(points)
    q=dict(points=points.float(),sensor_id=torch.arange(n)%2,delta_t=torch.full((n,),-.01),
           recent_mask=torch.ones(n,dtype=torch.bool),sequence_id='synthetic',sample_id=f'q{i}',
           query_time=100.+i*.07,event_count=1,event_timestamps=[100.+i*.07-.01],has_observation=n>0,target_valid=target)
    if target:q.update(target_xyz=torch.tensor([.2,.1,.1]),target_timestamp=q['query_time'])
    return q

def batch(queries):return collate_temporal_queries([{'queries':queries}])
def model():torch.manual_seed(42);return LiDARUAVDetector(CFG).eval()
def diff(a,b):return float((a-b).abs().max()) if a.numel() else 0.

class QueryCausalTests(unittest.TestCase):
    def test_validation_endpoint_and_export_interface(self):
        from rdq_uav.lidar_v2.training import validate
        clips=[{'queries':[query(0)],'score_last_only':True},
               {'queries':[query(0),query(1,torch.empty((0,3)))],'score_last_only':True}]
        # Separate batches model rolling validation; context q0 must not be scored twice.
        loader=[collate_temporal_queries([c]) for c in clips]
        with tempfile.TemporaryDirectory() as directory:
            metrics,rows,health=validate(model(),loader,QueryCausalLoss(CFG),CandidateSelector(CFG),
                                       torch.device('cpu'),export_dir=Path(directory)/'export')
            self.assertEqual(len(rows),2)
            self.assertEqual(metrics['all']['samples'],2)
            self.assertEqual(metrics['no_current_support']['samples'],1)
            self.assertEqual(health['num_temporal_supervised'],2)
            self.assertTrue((Path(directory)/'export/candidate_features.npz').exists())
            self.assertIn('temporal_success_1m',metrics['all'])

    def test_a_future_lidar_invariance(self):
        m=model();qs=[query(i) for i in range(5)];changed=copy.deepcopy(qs)
        for q in changed[3:]:q['points']=q['points']*20+500
        with torch.no_grad():a=m(batch(qs));b=m(batch(changed))
        d=diff(a['temporal_pred_xyz'][:,:3],b['temporal_pred_xyz'][:,:3]);RESULTS['future_lidar_max_diff']=d;self.assertLessEqual(d,1e-6)
    def test_b_future_gt_invariance(self):
        m=model();a=batch([query(i) for i in range(5)]);b=copy.deepcopy(a)
        b['target_xyz'][3:]=1e8;b['target_xyz_clip'][:,3:]=-1e8;b['target_timestamp'][3:]=0
        with torch.no_grad():x=m(a);y=m(b)
        d=diff(x['temporal_pred_xyz'][:,:3],y['temporal_pred_xyz'][:,:3]);RESULTS['future_gt_max_diff']=d;self.assertEqual(d,0.)
    def test_c_future_removal(self):
        m=model();qs=[query(i) for i in range(5)]
        with torch.no_grad():a=m(batch(qs));b=m(batch(qs[:3]))
        d=diff(a['temporal_pred_xyz'][:,2],b['temporal_pred_xyz'][:,2]);RESULTS['future_removal_max_diff']=d;self.assertLessEqual(d,1e-6)
    def test_d_gt_free_query_and_independent_target_time(self):
        m=model();b=batch([query(0,target=False),query(1,target=False)])
        for k in ('gt_xyz','gt_timestamp','target_xyz','target_timestamp'):self.assertNotIn(k,b)
        with torch.no_grad():o=m(b)
        self.assertTrue(torch.isfinite(o['temporal_pred_xyz']).all())
        q=query(2);q['target_timestamp']=q['query_time']+99
        self.assertNotEqual(collate_lidar_samples([q])['target_timestamp'][0],q['query_time'])
    def test_e_no_current_support_temporal_gradient(self):
        m=model();b=batch([query(0,torch.tensor([[20.,20.,20.]]))]);o=m(b);loss=QueryCausalLoss(CFG)(o,b)
        self.assertEqual(loss['num_supervised_samples'],0);self.assertEqual(loss['num_temporal_supervised'],1)
        self.assertEqual(float(loss['spatial_loss']),0.);loss['loss'].backward()
        g=m.temporal_head.net[-1].weight.grad
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0.)
        RESULTS['no_current_support_temporal_gradient']='PASS'
    def test_f_empty_observation_and_padding(self):
        m=model();q=query(0,torch.empty((0,3)));b=collate_temporal_queries([{'queries':[q],'clip_length':3}])
        with torch.no_grad():o=m(b)
        self.assertEqual(len(o['logits']),0);self.assertTrue(torch.isfinite(o['temporal_pred_xyz']).all())
        self.assertFalse(bool(o['has_observation'].any()));self.assertTrue(b['query_valid_mask'][0,0]);self.assertFalse(b['query_valid_mask'][0,1])
        self.assertTrue(torch.equal(o['query_token'][0,0],m.query_pool.missing_lidar_token))
        self.assertEqual(QueryCausalLoss(CFG)(o,b)['num_temporal_supervised'],1)
        RESULTS['empty_lidar']='PASS'
    def test_g_causal_mask(self):
        mask=causal_mask(5)
        for i in range(5):
            for j in range(5):self.assertEqual(float(mask[i,j]),float('-inf') if j>i else 0.)
    def test_builder_last20_recent4_and_cleaning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('livox_avia','lidar_360'):(root/'seq'/name).mkdir(parents=True)
            for i in range(30):
                name='livox_avia' if i%2==0 else 'lidar_360'
                np.save(root/'seq'/name/f'{i}.npy',np.array([[i+1.,1,1],[0,0,0],[np.nan,1,1]]))
            q=LiDARQueryBuilder(root).build('seq',25.5,target_xyz=[1,2,3],target_timestamp=99,target_valid=True)
            self.assertEqual(q['event_count'],20);self.assertEqual(q['event_timestamps'],list(map(float,range(6,26))))
            self.assertTrue(bool((q['delta_t']<=0).all()));self.assertEqual(int(q['recent_mask'].sum()),4)
            self.assertEqual(len(q['points']),20);self.assertEqual(q['sensor_id'].tolist(),[i%2 for i in range(6,26)])
    def test_pool_is_query_local_and_soft(self):
        m=model();b=collate_lidar_samples([query(0),query(1)])
        with torch.no_grad():o=m(b)
        self.assertEqual(tuple(o['query_token'].shape),(2,1,128))
        for i in range(2):
            sel=o['batch_index']==i;a=torch.softmax(o['logits'][sel].float(),0)
            expected=m.query_pool.norm((a[:,None]*o['fine_features'][sel]).sum(0)+m.query_pool.xyz_embed((a[:,None]*o['pred_xyz'][sel]).sum(0)/100))
            self.assertLessEqual(diff(expected,o['query_token'][i,0]),1e-6)
    def test_gt_free_static_model_and_no_v1_import(self):
        source=(ROOT/'src/rdq_uav/lidar_v2/model.py').read_text()
        for key in ('gt_xyz','gt_timestamp','target_xyz','target_timestamp'):self.assertNotIn(key,source)
        for p in (ROOT/'src/rdq_uav/lidar_v2').glob('*.py'):self.assertNotIn('from rdq_uav.lidar_v1',p.read_text())
    def test_validation_scores_endpoint_once_and_temporal_groups(self):
        class D:
            records=[dict(sequence_id=s,query_time=i) for s in ('a','b') for i in range(3)]
            def __getitem__(self,i):
                q=query(i%3);q['sequence_id']=self.records[i]['sequence_id'];q['sample_id']=str(i);return q
        ds=TemporalQueryClipDataset(D(),8,validation=True);self.assertEqual(len(ds),6)
        m=model();rows=[]
        with torch.no_grad():
            for i in range(6):
                b=collate_temporal_queries([ds[i]]);o=m(b);rows.extend(evaluate_batch(o,b,CandidateSelector(CFG),QueryCausalLoss(CFG)))
        self.assertEqual(len(rows),6);self.assertEqual(len({r['sample_id'] for r in rows}),6)
        self.assertIn('temporal_success_1m',summarize_metrics(rows)['all'])
    def test_pool_gradient_and_padding_invariance(self):
        m=model();b=batch([query(0),query(1)]);padded=collate_temporal_queries([{'queries':[query(0),query(1)],'clip_length':8}])
        with torch.no_grad():x=m(b);y=m(padded)
        self.assertLessEqual(diff(x['temporal_pred_xyz'],y['temporal_pred_xyz'][:,:2]),1e-6)

if __name__=='__main__':unittest.main()
