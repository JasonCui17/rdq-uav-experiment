"""VQSA fixed geometry, masking, sensitivity, and invariance tests."""
from __future__ import annotations

import sys,unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2.sbe import VoxelQuerySlotAggregation,ordered_slot_centers
from rdq_uav.lidar_v2.sbe import SBELiteVoxelEmbed
from rdq_uav.lidar_v2.geometry import HierarchyBuilder

RESULTS={}


def build(points,sensors=None,dt=None):
    points=torch.as_tensor(points,dtype=torch.float32).reshape(-1,3);n=len(points)
    batch=dict(points=points,sensor_id=torch.zeros(n,dtype=torch.long) if sensors is None else torch.as_tensor(sensors,dtype=torch.long),
        delta_t=torch.full((n,),-.1) if dt is None else torch.as_tensor(dt,dtype=torch.float32))
    return batch,HierarchyBuilder()(points,torch.zeros(n,dtype=torch.long))


class VQSATests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);self.module=VoxelQuerySlotAggregation()

    def test_slot_centers_and_order(self):
        expected=torch.tensor([[x,y,z] for x in (-.25,.25) for y in (-.25,.25) for z in (-.25,.25)])
        self.assertTrue(torch.equal(ordered_slot_centers(),expected))
        self.assertTrue(torch.equal(self.module.slot_centers,expected))
        self.assertNotIn('slot_centers',dict(self.module.named_parameters()))

    def test_shapes_mask_and_finite(self):
        stats=torch.randn(3,8,11);counts=torch.tensor([[1,0,2,0,0,3,0,1],[0,1,0,0,0,0,0,0],[1]*8])
        dynamic,debug=self.module(stats,counts,True)
        self.assertEqual(tuple(debug['slot_input'].shape),(3,8,14));self.assertEqual(tuple(debug['slot_tokens'].shape),(3,8,16))
        self.assertEqual(tuple(debug['voxel_query'].shape),(3,1,16));self.assertEqual(tuple(debug['dynamic_sequence'].shape),(3,1,16))
        self.assertEqual(tuple(debug['attention'].shape),(3,2,1,8));self.assertEqual(tuple(dynamic.shape),(3,16))
        self.assertEqual(tuple(torch.cat((stats.flatten(1),dynamic),1).shape),(3,104))
        self.assertTrue(torch.isfinite(dynamic).all() and torch.isfinite(debug['attention']).all())
        self.assertEqual(float(debug['attention'].masked_select(debug['key_padding_mask'][:,None,None,:]).abs().max()),0.)

    def test_single_slot_attention_is_one(self):
        stats=torch.randn(2,8,11);counts=torch.zeros(2,8,dtype=torch.long);counts[0,0]=1;counts[1,7]=2
        _,debug=self.module(stats,counts,True);visible=~debug['key_padding_mask'][:,None,None,:]
        torch.testing.assert_close(debug['attention'].masked_select(visible),torch.ones(4))

    def test_reject_all_masked_voxel(self):
        with self.assertRaisesRegex(AssertionError,'all-masked'):
            self.module(torch.zeros(1,8,11),torch.zeros(1,8,dtype=torch.long))

    def test_masked_garbage_invariance(self):
        stats=torch.zeros(1,8,11);counts=torch.zeros(1,8,dtype=torch.long);counts[0,3]=2;stats[0,3]=torch.arange(11)
        changed=stats.clone();changed[0,counts[0]==0]=torch.randn(7,11)*1e6
        with torch.no_grad():a=self.module(stats,counts);b=self.module(changed,counts)
        self.assertEqual(float((a-b).abs().max()),0.)

    def test_attention_batch_chunking_is_exact(self):
        stats=torch.randn(11,8,11);counts=torch.randint(0,4,(11,8));counts[:,0]=1
        module=VoxelQuerySlotAggregation();occupied=counts>0
        tokens=module.slot_embed(torch.cat((stats,module.slot_centers.expand(len(stats),-1,-1)),-1))
        query=module.voxel_query.expand(len(stats),-1,-1)
        a,aw=module._attend(query,tokens,occupied,True,100)
        b,bw=module._attend(query,tokens,occupied,True,3)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8);torch.testing.assert_close(aw,bw,rtol=1e-6,atol=1e-8)

    def test_position_sensitivity(self):
        module=VoxelQuerySlotAggregation()
        with torch.no_grad():
            linear=module.slot_embed[0];linear.weight.zero_();linear.bias.zero_();linear.weight[0,11]=1.
            module.voxel_query.fill_(1.);module.attention.in_proj_weight.zero_();module.attention.in_proj_bias.zero_()
            # Q0 reads query0; K0 and V0 read embedded coordinate channel 0.
            module.attention.in_proj_weight[0,0]=1.;module.attention.in_proj_weight[16,0]=1.;module.attention.in_proj_weight[32,0]=1.
            module.attention.out_proj.weight.zero_();module.attention.out_proj.bias.zero_();module.attention.out_proj.weight[0,0]=1.
        stats=torch.zeros(2,8,11);stats[:,[0,7],0]=1.;counts=torch.zeros(2,8,dtype=torch.long);counts[0,0]=1;counts[1,7]=1
        dynamic,debug=module(stats,counts,True)
        self.assertNotEqual(float(debug['slot_input'][0,0,11]),float(debug['slot_input'][1,7,11]))
        self.assertGreater(float((dynamic[0]-dynamic[1]).abs().max()),0.)

    def test_point_permutation_invariance_with_vqsa(self):
        embed=SBELiteVoxelEmbed();points=torch.rand(200,3)*2-1;sensors=(torch.arange(200)%2).tolist();dt=(-torch.rand(200)).tolist()
        batch,hierarchy=build(points,sensors,dt);order=torch.randperm(200)
        shuffled={key:value[order] for key,value in batch.items()};_,other_hierarchy=build(shuffled['points'],shuffled['sensor_id'].tolist(),shuffled['delta_t'].tolist())
        with torch.no_grad():
            a,da=embed(batch,hierarchy,True);b,db=embed(shuffled,other_hierarchy,True)
        self.assertLessEqual(float((da['slot_stats']-db['slot_stats']).abs().max()),1e-6)
        self.assertLessEqual(float((da['dynamic_summary']-db['dynamic_summary']).abs().max()),1e-6)
        self.assertLessEqual(float((a-b).abs().max()),1e-6)
        RESULTS['permutation_max_diff']=float((a-b).abs().max())

    def test_empty_lidar(self):
        embed=SBELiteVoxelEmbed();batch,hierarchy=build([])
        with torch.no_grad():token,debug=embed(batch,hierarchy,True)
        self.assertEqual(tuple(debug['slot_stats'].shape),(0,8,11));self.assertEqual(tuple(debug['dynamic_summary'].shape),(0,16))
        self.assertEqual(tuple(token.shape),(0,128));self.assertTrue(torch.isfinite(token).all())


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
