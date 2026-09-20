"""Sequence and time jointly define temporal access; no optimizer is created."""
import copy,importlib.util,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('qc',ROOT/'tests/test_lidar_uav_v2_query_causal.py');qc=importlib.util.module_from_spec(spec);spec.loader.exec_module(qc)
from rdq_uav.lidar_v2 import LiDARQueryBuilder,TemporalQueryClipDataset,collate_temporal_queries,build_query_history
from rdq_uav.lidar_v2.geometry import HierarchyBuilder
from rdq_uav.lidar_v2.isolation import assert_temporal_batch_integrity,assert_temporal_clip_integrity
from rdq_uav.multimodal.merged_lidar import LidarFrameEvent
RESULTS={}

def query(seq,i):
    q=qc.query(i);q.update(sequence_id=seq,sample_id=f'{seq}_{i}',query_time=10+i*.1,event_timestamps=[9.99+i*.1],event_sequence_ids=[seq])
    return q

class Queries:
    def __init__(self):
        self.queries=[query(s,i) for i in (3,0,2,1) for s in ('B','A')]
        self.records=[{k:q[k] for k in ('sequence_id','sample_id','query_time')} for q in self.queries]
    def __getitem__(self,i):return self.queries[i]

class IsolationTests(unittest.TestCase):
    def test_a_dataset_groups_before_time(self):
        ds=TemporalQueryClipDataset(Queries(),4)
        for i in range(len(ds)):
            qs=ds[i]['queries'];assert_temporal_clip_integrity(qs,i,True)
            self.assertEqual(len({q['sequence_id'] for q in qs}),1)
        self.assertEqual(len(ds),8)
    def test_b_same_timestamps_and_validation_reset(self):
        ds=TemporalQueryClipDataset(Queries(),4,validation=True)
        self.assertEqual([len(ds[i]['queries']) for i in range(8)],[1,2,3,4,1,2,3,4])
        self.assertEqual(ds[3]['queries'][-1]['sequence_id'],'A');self.assertEqual(ds[4]['queries'][0]['sequence_id'],'B')
        RESULTS['rolling_validation_boundary_reset']='PASS'
    def test_c_event_sequence_checked_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder=LiDARQueryBuilder(tmp)
            builder.streams['A']=[LidarFrameEvent('B',1.,0,'Avia',Path(tmp)/'not_read.npy')]
            with self.assertRaisesRegex(AssertionError,'event_sequence_mismatch'):builder.build('A',2.)
            builder.streams['A']=[LidarFrameEvent('A',1.,0,'Avia',Path(tmp)/'not_read.npy')]
            events=builder.select_events('A',2.);self.assertTrue(all(e.sequence_id=='A' for e in events))
    def test_d_identical_xyz_separate_voxels(self):
        qs=[query(s,0) for s in ('A','B')]
        for q in qs:
            q.update(points=torch.tensor([[1.,2.,3.]]),sensor_id=torch.zeros(1,dtype=torch.long),delta_t=torch.tensor([-.01]),supervision_recent_mask=torch.ones(1,dtype=torch.bool))
        b=collate_temporal_queries([{'queries':[q]} for q in qs]);h=HierarchyBuilder()(b['points'],b['point_batch_index'])
        for level in h.levels:self.assertEqual(len(level.coords),2);self.assertEqual(level.batch_index.tolist(),[0,1])
        self.assertEqual(h.point_to_l0.tolist(),[0,1]);RESULTS['spatial_same_xyz']='PASS'
    def test_e_same_timestamp_cross_sequence_spatial_isolation(self):
        items=[{'queries':[query(s,i) for i in range(4)]} for s in ('A','B')]
        changed=copy.deepcopy(items)
        for q in changed[1]['queries']:q['points']=q['points']*1000+2000
        m=qc.model()
        with torch.no_grad():a=m(collate_temporal_queries(items));b=m(collate_temporal_queries(changed))
        left=a['batch_index']<4;right=b['batch_index']<4
        d=max(float((a[key][left]-b[key][right]).abs().max()) for key in ('logits','pred_xyz','fine_features'))
        self.assertLessEqual(d,1e-6);RESULTS['same_timestamp_cross_sequence_max_diff']=d
    def test_g_reject_mixed_history(self):
        with self.assertRaisesRegex(AssertionError,'cross_sequence_clip'):
            collate_temporal_queries([{'queries':[query('A',0),query('B',1)]}])
        class BadBuilder:
            def build_inference_query(self,seq,t):return query('A' if t==10 else 'B',0 if t==10 else 1)
        with self.assertRaisesRegex(AssertionError,'cross_sequence_clip'):build_query_history(BadBuilder(),'A',[10.,10.1])
        RESULTS['arbitrary_mixed_history']='PASS'
    def test_h_duplicate_unsorted_identity_diagnostics(self):
        for change,code in ((lambda q:q.update(query_time=10.),'duplicate_query_timestamp'),
                            (lambda q:q.update(query_time=9.),'unsorted_query_timestamps'),
                            (lambda q:q.update(sample_id='A_0'),'duplicate_or_missing_sample_identity')):
            qs=[query('A',0),query('A',1)];change(qs[1])
            with self.assertRaisesRegex(AssertionError,code) as error:assert_temporal_clip_integrity(qs,clip_index=17)
            for text in ('clip_index=17','sample_ids','query_times','sequence_ids'):self.assertIn(text,str(error.exception))
        ds=Queries();ds.records[0]['query_time']=ds.records[2]['query_time']
        with self.assertRaisesRegex(AssertionError,'duplicate_query_timestamp'):TemporalQueryClipDataset(ds)
    def test_i_model_rejects_forged_mixed_clip(self):
        b=collate_temporal_queries([{'queries':[query('A',i) for i in range(4)]}]);b['sequence_id'][2]='B'
        with self.assertRaisesRegex(AssertionError,'cross_sequence_clip'):assert_temporal_batch_integrity(b)
    def test_k_inference_sequence_switch_no_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder=LiDARQueryBuilder(tmp)
            a=build_query_history(builder,'A',[10.,10.1]);b=build_query_history(builder,'B',[10.,10.1])
            self.assertEqual(a['sequence_id'],['A','A']);self.assertEqual(b['sequence_id'],['B','B'])

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
