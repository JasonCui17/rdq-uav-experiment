"""Formal spatial-only query batching, EQS, loss, and entrypoint contracts."""
from __future__ import annotations

import sys,unittest
from pathlib import Path
import torch,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (CandidateLoss,CandidateSelector,EpochCyclicQuerySampler,
    LiDARUAVDetector,collate_lidar_samples,collate_temporal_queries,planned_epoch_stats)
from rdq_uav.lidar_v2.runtime import evaluate_batch

CFG=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())


def query(sequence,index,points=None,target_valid=True):
    time=10.+index
    points=torch.tensor([[index+.1,0.,0.]]) if points is None else torch.as_tensor(points,dtype=torch.float32).reshape(-1,3)
    return dict(sequence_id=sequence,sample_id=f'{sequence}_{index}',query_uid=index,query_time=time,
        points=points,sensor_id=torch.zeros(len(points),dtype=torch.long),delta_t=torch.full((len(points),),-.01),
        supervision_recent_mask=torch.ones(len(points),dtype=torch.bool),event_count=1,
        event_timestamps=[time-.01],event_sequence_ids=[sequence],target_valid=target_valid,
        target_timestamp=time,target_xyz=torch.tensor([index+.1,0.,0.]))


class Queries:
    def __init__(self,lengths=(11,7)):
        self.samples=[query(s,i) for s,n in zip(('A','B'),lengths) for i in range(n)]
        self.records=[{k:q[k] for k in ('sequence_id','sample_id','query_time','query_uid')} for q in self.samples]
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return self.samples[index]


class SpatialQueryPipelineTests(unittest.TestCase):
    def test_single_query_collate_has_exactly_b_spatial_samples(self):
        batch=collate_lidar_samples([query('A',0),query('B',0)])
        self.assertEqual(batch['num_samples'],2);self.assertEqual(batch['spatial_num_samples'],2)
        self.assertEqual(set(batch['point_batch_index'].tolist()),{0,1})
        for forbidden in ('clip_batch_index','clip_position','query_valid_mask','query_time_clip',
                          'target_valid_clip','score_last_only','spatial_supervise_mask_occurrence',
                          'occurrence_to_unique','unique_query_packing'):
            self.assertNotIn(forbidden,batch)

    def test_collate_rejects_future_event(self):
        q=query('A',0);q['event_timestamps']=[q['query_time']+.01];q['delta_t']=torch.tensor([.01])
        with self.assertRaises((AssertionError,ValueError)):collate_lidar_samples([q])

    def test_empty_query_is_finite_without_fake_voxel(self):
        batch=collate_lidar_samples([query('A',0,points=[])])
        out=LiDARUAVDetector(CFG)(batch)
        self.assertEqual(tuple(out['logits'].shape),(0,));self.assertTrue(torch.isfinite(out['pred_xyz']).all())

    def test_query_level_eqs_exact_four_epoch_partition(self):
        dataset=Queries();sampler=EpochCyclicQuerySampler(dataset,4,42,False);selected=[]
        for epoch in range(1,5):selected.extend(sampler.selected_indices(epoch,False))
        self.assertEqual(len(selected),len(dataset));self.assertEqual(len(set(selected)),len(dataset))
        self.assertEqual(set(selected),set(range(len(dataset))))
        plan=planned_epoch_stats(sampler,4,2,2)
        self.assertTrue(all('selected_queries' in row and 'selected_clips' not in row for row in plan))

    def test_direct_loss_equals_legacy_single_occurrence_loss(self):
        samples=[query('A',0),query('B',0)];direct=collate_lidar_samples(samples)
        legacy=collate_temporal_queries([{'queries':[q]} for q in samples])
        torch.manual_seed(42);model=LiDARUAVDetector(CFG).eval()
        with torch.no_grad():a=model(direct);b=model(legacy)
        criterion=CandidateLoss(CFG);la=criterion(a,direct);lb=criterion(b,legacy)
        for key in ('positive_mask','ignore_mask','negative_mask'):self.assertTrue(torch.equal(la[key],lb[key]))
        for key in ('loss','loss_cls','loss_reg'):self.assertEqual(float(la[key]),float(lb[key]))

    def test_direct_selector_and_sequence_isolation(self):
        batch=collate_lidar_samples([query('A',0),query('B',0)])
        out=LiDARUAVDetector(CFG)(batch);selected=CandidateSelector(CFG)(out)
        self.assertEqual(len(selected),2)
        self.assertEqual(set(out['batch_index'].tolist()),{0,1})
        rows=evaluate_batch(out,batch,CandidateSelector(CFG),CandidateLoss(CFG))
        self.assertEqual([row['sample_id'] for row in rows],['A_0','B_0'])

    def test_parameter_count_is_frozen(self):
        self.assertEqual(sum(p.numel() for p in LiDARUAVDetector(CFG).parameters()),1045352)

    def test_formal_entrypoints_do_not_use_clip_or_uqp(self):
        for relative in ('tools/train_lidar_uav_v2.py','tools/evaluate_lidar_uav_v2.py','tools/infer_lidar_uav_v2.py'):
            source=(ROOT/relative).read_text()
            self.assertNotIn('TemporalQueryClipDataset',source);self.assertNotIn('collate_temporal_queries',source)
            self.assertNotIn('OverlapAwareBatchSampler',source)
        train=(ROOT/'tools/train_lidar_uav_v2.py').read_text()
        self.assertIn("validation_interval_epochs",train)


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
