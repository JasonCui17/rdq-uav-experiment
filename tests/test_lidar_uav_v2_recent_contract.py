"""recent4 is supervision metadata and never a V2 learned input feature."""
from __future__ import annotations

import copy,importlib.util,sys,unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('qc',ROOT/'tests/test_lidar_uav_v2_query_causal.py');qc=importlib.util.module_from_spec(spec);spec.loader.exec_module(qc)
from rdq_uav.lidar_v2 import QueryCausalLoss,collate_temporal_queries

RESULTS={}


class LegacyQueryCausalLoss(QueryCausalLoss):
    """Frozen pre-cleanup label path for exact-equivalence tests only."""
    def labels(self,outputs,batch):
        spatial_gt=batch.get('spatial_target_xyz',batch['target_xyz']);spatial_valid=batch.get('spatial_target_valid',batch['target_valid'])
        h=outputs['layouts'];inv=h.point_to_l0;n=len(outputs['logits']);gt=spatial_gt[outputs['batch_index']]
        point_gt=spatial_gt[batch['point_batch_index']];dist=torch.linalg.vector_norm(batch['points']-point_gt,dim=1)
        all_min=dist.new_full((n,),torch.inf);all_min.scatter_reduce_(0,inv,dist,reduce='amin',include_self=True)
        recent_min=dist.new_full((n,),torch.inf);ridx=inv[batch['recent_mask']];rdist=dist[batch['recent_mask']]
        if len(ridx):recent_min.scatter_reduce_(0,ridx,rdist,reduce='amin',include_self=True)
        positive=recent_min<=1.;ignore=(~positive)&(all_min<=2.);negative=all_min>2.;valid=spatial_valid[outputs['batch_index']]
        return positive&valid,ignore&valid,negative&valid,(gt-outputs['voxel_centers'])/1.


def batches():
    items=[]
    for start in (0,4):
        queries=[]
        for i in range(start,start+8):
            q=qc.query(i);q.update(sequence_id='A',sample_id=f'A_{i}',query_uid=i,event_sequence_ids=['A'])
            queries.append(q)
        items.append({'queries':queries})
    clean=collate_temporal_queries(items,unique_query_packing=True)
    legacy=copy.deepcopy(clean);legacy['recent_mask']=legacy.pop('supervision_recent_mask')
    return legacy,clean


def maximum(a,b):return float((a-b).abs().max()) if a.numel() else 0.


class RecentInputContractTests(unittest.TestCase):
    def test_model_forward_does_not_require_supervision_mask(self):
        _,batch=batches();torch.manual_seed(42);model=qc.model().eval()
        with torch.no_grad():
            normal=model(batch);model_only={k:v for k,v in batch.items() if k!='supervision_recent_mask'};without=model(model_only)
        keys=('logits','pred_xyz','fine_features','query_token','temporal_hidden','temporal_pred_xyz')
        diffs={key:maximum(normal[key],without[key]) for key in keys}
        self.assertEqual(max(diffs.values()),0.);RESULTS['mask_removed_output_diffs']=diffs

    def test_legacy_labels_and_losses_are_exact(self):
        legacy,clean=batches();torch.manual_seed(42);model=qc.model().eval()
        with torch.no_grad():old_output=model(legacy);new_output=model(clean)
        old_criterion=LegacyQueryCausalLoss(qc.CFG);new_criterion=QueryCausalLoss(qc.CFG)
        old_labels=old_criterion.labels(old_output,legacy);new_labels=new_criterion.labels(new_output,clean)
        for old,new in zip(old_labels[:3],new_labels[:3]):self.assertTrue(torch.equal(old,new))
        old_loss=old_criterion(old_output,legacy);new_loss=new_criterion(new_output,clean)
        loss_diffs={key:abs(float(old_loss[key])-float(new_loss[key])) for key in ('loss','spatial_loss','loss_cls','loss_reg','temporal_loss')}
        counts=dict(positive=int(new_labels[0].sum()),ignore=int(new_labels[1].sum()),negative=int(new_labels[2].sum()),
            current=int(new_loss['num_supervised_samples']),no_current=int(new_loss['num_no_current_support']))
        self.assertEqual(max(loss_diffs.values()),0.);RESULTS.update(loss_diffs=loss_diffs,label_counts=counts)

    def test_gradient_exact_equivalence(self):
        legacy,clean=batches();torch.manual_seed(42);old=qc.model().train();new=qc.model().train();new.load_state_dict(old.state_dict())
        LegacyQueryCausalLoss(qc.CFG)(old(legacy),legacy)['loss'].backward();QueryCausalLoss(qc.CFG)(new(clean),clean)['loss'].backward()
        names=('voxel_embed.proj.weight','merge01.parent.weight','encoder0.blocks.0.qkv.weight','head.cls.2.weight',
            'query_pool.xyz_embed.2.weight','temporal_transformer.blocks.0.qkv.weight','temporal_head.net.2.weight')
        diffs={}
        for name in names:
            a=dict(old.named_parameters())[name].grad;b=dict(new.named_parameters())[name].grad
            self.assertIsNotNone(a,name);self.assertIsNotNone(b,name);self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
            diffs[name]=maximum(a,b);self.assertEqual(diffs[name],0.,name)
        RESULTS['gradient_diffs']=diffs

    def test_only_supervision_consumers_reference_mask(self):
        model=(ROOT/'src/rdq_uav/lidar_v2/model.py').read_text();sbe=(ROOT/'src/rdq_uav/lidar_v2/sbe.py').read_text();temporal=(ROOT/'src/rdq_uav/lidar_v2/temporal.py').read_text()
        self.assertNotIn('recent_mask',model);self.assertNotIn('recent_mask',sbe);self.assertNotIn('recent_mask',temporal)
        self.assertIn('supervision_recent_mask',(ROOT/'src/rdq_uav/lidar_v2/loss.py').read_text())
        self.assertIn('supervision_recent_mask',(ROOT/'src/rdq_uav/lidar_v2/runtime.py').read_text())


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
