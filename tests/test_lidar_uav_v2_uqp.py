"""Exact-compute UQP output, loss, gradient, and batching regressions."""
from __future__ import annotations

import copy,importlib.util,sys,unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('qc',ROOT/'tests/test_lidar_uav_v2_query_causal.py');qc=importlib.util.module_from_spec(spec);spec.loader.exec_module(qc)
from rdq_uav.lidar_v2 import (EpochCyclicQuerySampler,OverlapAwareBatchSampler,
    QueryCausalLoss,TemporalPositionLoss,TemporalQueryClipDataset,collate_temporal_queries)

RESULTS={}


def overlapping_items():
    queries=[]
    for i in range(12):
        q=qc.query(i);q.update(sequence_id='A',sample_id=f'A_{i}',query_uid=i,
            event_sequence_ids=['A'])
        queries.append(q)
    return [{'queries':queries[:8]},{'queries':queries[4:12]}]


def packed(enabled):return collate_temporal_queries(overlapping_items(),unique_query_packing=enabled)


def max_diff(a,b):return float((a-b).abs().max()) if a.numel() else 0.


class UQPTests(unittest.TestCase):
    def test_mapping_16_to_12(self):
        batch=packed(True)
        self.assertEqual(int(batch['query_valid_mask'].sum()),16);self.assertEqual(batch['spatial_num_samples'],12)
        self.assertEqual(batch['occurrence_to_unique'].reshape(2,8).tolist(),
            [list(range(8)),list(range(4,12))])
        self.assertEqual(batch['spatial_occurrence_count'].tolist(),[1,1,1,1,2,2,2,2,1,1,1,1])

    def test_key_includes_sequence(self):
        items=overlapping_items();items[1]['queries']=[dict(q,sequence_id='B',event_sequence_ids=['B']) for q in items[1]['queries']]
        batch=collate_temporal_queries(items,unique_query_packing=True)
        self.assertEqual(batch['spatial_num_samples'],16)

    def test_uid_collision_rejected(self):
        items=overlapping_items();items[1]['queries'][0]=copy.deepcopy(items[1]['queries'][0]);items[1]['queries'][0]['points']+=1
        with self.assertRaisesRegex(AssertionError,'query_uid collision'):collate_temporal_queries(items,unique_query_packing=True)

    def test_reference_output_and_loss_equivalence(self):
        torch.manual_seed(42);model=qc.model();reference=packed(False);uqp=packed(True)
        with torch.no_grad():a=model(reference);b=model(uqp)
        spatial={key:0. for key in ('logits','pred_xyz','fine_features')}
        for occurrence,unique in enumerate(uqp['occurrence_to_unique'].tolist()):
            left=a['batch_index']==occurrence;right=b['batch_index']==unique
            for key in spatial:spatial[key]=max(spatial[key],max_diff(a[key][left],b[key][right]))
        query=max_diff(a['query_token'],b['query_token']);hidden=max_diff(a['temporal_hidden'],b['temporal_hidden']);temporal=max_diff(a['temporal_pred_xyz'],b['temporal_pred_xyz'])
        criterion=QueryCausalLoss(qc.CFG);la=criterion(a,reference);lb=criterion(b,uqp)
        losses={key:abs(float(la[key])-float(lb[key])) for key in ('loss','spatial_loss','loss_cls','loss_reg','temporal_loss')}
        self.assertLessEqual(max(spatial.values()),1e-6);self.assertLessEqual(max(query,hidden,temporal),1e-6);self.assertLessEqual(max(losses.values()),1e-6)
        RESULTS.update(spatial=spatial,query_token_max_diff=query,temporal_hidden_max_diff=hidden,
                       temporal_xyz_max_diff=temporal,loss_diffs=losses)

    def test_gradient_equivalence(self):
        torch.manual_seed(42);a=qc.model().train();b=qc.model().train();b.load_state_dict(a.state_dict())
        la=QueryCausalLoss(qc.CFG)(a(packed(False)),packed(False))['loss'];la.backward()
        uqp=packed(True);lb=QueryCausalLoss(qc.CFG)(b(uqp),uqp)['loss'];lb.backward()
        names=('voxel_embed.proj.weight','encoder0.blocks.0.qkv.weight','merge01.parent.weight',
               'head.cls.2.weight','head.reg.2.weight','query_pool.xyz_embed.2.weight',
               'temporal_transformer.blocks.0.qkv.weight','temporal_head.net.2.weight')
        result={}
        for name in names:
            ga=dict(a.named_parameters())[name].grad;gb=dict(b.named_parameters())[name].grad
            self.assertIsNotNone(ga,name);self.assertIsNotNone(gb,name);self.assertTrue(torch.isfinite(ga).all() and torch.isfinite(gb).all())
            absolute=max_diff(ga,gb);relative=absolute/max(float(ga.abs().max()),float(gb.abs().max()),1e-12)
            result[name]=dict(max_abs_diff=absolute,max_relative_diff=relative)
            self.assertLessEqual(absolute,1e-5,name)
        RESULTS['gradient_diffs']=result

    def test_shared_query_receives_both_temporal_context_gradients(self):
        model=qc.model().train();batch=packed(True);captured={}
        def hook(module,args,output):captured.update(unique=output[0]);output[0].retain_grad()
        handle=model.query_pool.register_forward_hook(hook);out=model(batch);out['query_token'].retain_grad()
        loss=TemporalPositionLoss(qc.CFG)(out,batch);loss.backward();handle.remove()
        unique=4;occurrence_grad=out['query_token'].grad.reshape(-1,128)
        expected=occurrence_grad[4]+occurrence_grad[8]
        difference=max_diff(captured['unique'].grad[unique],expected)
        self.assertGreater(float(occurrence_grad[4].abs().sum()),0);self.assertGreater(float(occurrence_grad[8].abs().sum()),0)
        self.assertLessEqual(difference,1e-7);RESULTS['shared_multi_context_gradient_max_diff']=difference

    def test_no_support_weight_and_temporal_occurrences(self):
        items=overlapping_items()
        for clip in items:
            for q in clip['queries']:
                if q['query_uid']==4:
                    q['points']=torch.tensor([[20.,20.,20.]]);q['sensor_id']=torch.zeros(1,dtype=torch.long)
                    q['delta_t']=torch.tensor([-.01]);q['supervision_recent_mask']=torch.ones(1,dtype=torch.bool)
        batch=collate_temporal_queries(items,unique_query_packing=True);out=qc.model()(batch);loss=QueryCausalLoss(qc.CFG)(out,batch)
        self.assertGreaterEqual(loss['num_no_current_support'],2);self.assertEqual(loss['num_temporal_supervised'],16)

    def test_overlap_batch_sampler_preserves_selection(self):
        class Q:
            def __init__(self):
                self.samples=[dict(qc.query(i),sequence_id=s,sample_id=f'{s}{i}',query_uid=i,event_sequence_ids=[s]) for s in ('A','B') for i in range(13)]
                self.records=[{k:q[k] for k in ('sequence_id','sample_id','query_time')} for q in self.samples]
            def __getitem__(self,i):return self.samples[i]
        dataset=TemporalQueryClipDataset(Q(),8,1);eqs=EpochCyclicQuerySampler(dataset,4,42,True);batcher=OverlapAwareBatchSampler(eqs,2,42,True)
        for epoch in (1,2,5,17):
            batcher.set_epoch(epoch);flat=[i for group in batcher for i in group]
            self.assertEqual(set(flat),set(eqs.selected_indices(epoch,False)));self.assertEqual(len(flat),len(set(flat)))
        direct=OverlapAwareBatchSampler(EpochCyclicQuerySampler(dataset,4,42,True),2,42,True);direct.set_epoch(17)
        self.assertEqual(list(batcher),list(direct))


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
