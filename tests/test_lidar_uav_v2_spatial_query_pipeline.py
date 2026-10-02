"""Formal full-dataset spatial-query training protocol regressions."""
import math,sys,unittest
from types import SimpleNamespace
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import CandidateLoss,CandidateSelector,LiDARUAVDetector,collate_lidar_samples
from rdq_uav.lidar_v2.runtime import UpdateScheduler,evaluate_batch,summarize_metrics
CFG=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())

def query(sequence,index,points=None,target_valid=True):
    time=10.+index;points=torch.tensor([[index+.1,0.,0.]]) if points is None else torch.as_tensor(points,dtype=torch.float32).reshape(-1,3)
    return dict(sequence_id=sequence,sample_id=f'{sequence}_{index}',query_uid=index,query_time=time,
        points=points,sensor_id=torch.zeros(len(points),dtype=torch.long),delta_t=torch.full((len(points),),-.01),
        event_count=1,event_timestamps=[time-.01],
        event_sequence_ids=[sequence],target_valid=target_valid,target_timestamp=time,target_xyz=torch.tensor([index+.1,0.,0.]))

class SpatialQueryPipelineTests(unittest.TestCase):
    def test_collate_is_exactly_one_sample_per_query(self):
        batch=collate_lidar_samples([query('A',0),query('B',0)])
        self.assertEqual(batch['num_samples'],2);self.assertEqual(set(batch['point_batch_index'].tolist()),{0,1})
        forbidden=('clip_batch_index','clip_position','query_valid_mask','query_time_clip','target_valid_clip','score_last_only',
                   'spatial_supervise_mask_occurrence','occurrence_to_unique','unique_query_packing','spatial_num_samples')
        for key in forbidden:self.assertNotIn(key,batch)
    def test_collate_rejects_future_event(self):
        q=query('A',0);q['event_timestamps']=[q['query_time']+.01];q['delta_t']=torch.tensor([.01])
        with self.assertRaises(AssertionError):collate_lidar_samples([q])
    def test_empty_query_has_no_fake_voxel(self):
        out=LiDARUAVDetector(CFG)(collate_lidar_samples([query('A',0,[])]))
        self.assertEqual(tuple(out['logits'].shape),(0,));self.assertTrue(torch.isfinite(out['pred_xyz']).all())
    def test_full_dataset_protocol_math(self):
        queries=28600;epochs=CFG['train']['epochs'];batch=CFG['train']['per_gpu_batch_size'];accum=CFG['train']['single_gpu_accumulate']
        self.assertEqual(epochs,25);self.assertEqual(queries*epochs,715000)
        micro=math.ceil(queries/batch);updates=math.ceil(micro/accum);self.assertEqual((micro,updates,updates*epochs),(14300,7150,178750))
        parameter=torch.nn.Parameter(torch.zeros(()));optimizer=torch.optim.AdamW([parameter],lr=CFG['train']['lr'])
        scheduler=UpdateScheduler(optimizer,updates*epochs,CFG['train']['warmup_fraction'],CFG['train']['lr'],CFG['train']['final_lr'])
        self.assertEqual(scheduler.warmup,8938)
    def test_loss_selector_and_sequence_isolation(self):
        batch=collate_lidar_samples([query('A',0),query('B',0)]);out=LiDARUAVDetector(CFG)(batch)
        loss=CandidateLoss(CFG)(out,batch);self.assertEqual(loss['num_supervised_samples'],2)
        self.assertEqual(len(CandidateSelector(CFG)(out)),2);self.assertEqual(set(out['batch_index'].tolist()),{0,1})
        self.assertEqual([r['sample_id'] for r in evaluate_batch(out,batch,CandidateSelector(CFG),CandidateLoss(CFG))],['A_0','B_0'])
    def test_labels_use_every_input_point(self):
        # Six independent point/voxel events: early points must remain eligible.
        points=torch.tensor([[.5,0.,0.],[1.,0.,0.],[1.5,0.,0.],[2.,0.,0.],[2.1,0.,0.],[3.,0.,0.]])
        batch=dict(points=points,point_batch_index=torch.zeros(6,dtype=torch.long),
                   target_xyz=torch.zeros(1,3),target_valid=torch.tensor([True]))
        out=dict(logits=torch.zeros(6),pred_xyz=points,voxel_centers=points,
                 batch_index=torch.zeros(6,dtype=torch.long),
                 layouts=SimpleNamespace(point_to_l0=torch.arange(6)))
        positive,ignore,negative,_=CandidateLoss(CFG).labels(out,batch)
        self.assertEqual(positive.tolist(),[True,True,False,False,False,False])
        self.assertEqual(ignore.tolist(),[False,False,True,True,False,False])
        self.assertEqual(negative.tolist(),[False,False,False,False,True,True])
        batch['target_valid']=torch.tensor([False])
        masks=CandidateLoss(CFG).labels(out,batch)[:3]
        self.assertTrue(all(not bool(mask.any()) for mask in masks))

    def test_evaluation_summary_has_no_event_subset_grouping(self):
        batch=collate_lidar_samples([query('A',0)])
        out=LiDARUAVDetector(CFG)(batch)
        rows=evaluate_batch(out,batch,CandidateSelector(CFG),CandidateLoss(CFG))
        self.assertEqual(set(summarize_metrics(rows)),
                         {'all','current_support','no_current_support','per_sequence'})

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in LiDARUAVDetector(CFG).parameters()),1045352)

if __name__=='__main__':unittest.main()
