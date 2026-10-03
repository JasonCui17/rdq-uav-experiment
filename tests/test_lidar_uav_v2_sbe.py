"""SBE physical-statistics and strict downstream inheritance regressions."""
import copy
import math
import sys
import unittest
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2.model import LiDARUAVDetector,LegacyVoxelEmbed
from rdq_uav.lidar_v2.sbe import SBELiteVoxelEmbed,subvoxel_coordinates,SLOT_DESCRIPTOR
from rdq_uav.lidar_v2.geometry import HierarchyBuilder
CFG=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())

def build(points,sensors=None,dt=None,batch_ids=None):
    points=torch.as_tensor(points,dtype=torch.float32).reshape(-1,3);n=len(points)
    b=dict(points=points,sensor_id=torch.zeros(n,dtype=torch.long) if sensors is None else torch.tensor(sensors),
           delta_t=torch.full((n,),-.1) if dt is None else torch.tensor(dt,dtype=torch.float32))
    ids=torch.zeros(n,dtype=torch.long) if batch_ids is None else torch.tensor(batch_ids)
    return b,HierarchyBuilder()(points,ids)

class SBETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Existing analytical fixtures explicitly exercise a 0.5s half life.
        cfg=copy.deepcopy(CFG['model']['voxel']['sbe']);cfg['time_half_life_s']=.5
        self.embed=SBELiteVoxelEmbed(config=cfg)
    def test_01_octants(self):
        bits=torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)])
        q=bits*.5-.25;slot,r=subvoxel_coordinates(q)
        self.assertTrue(torch.equal(slot,torch.arange(8)));self.assertTrue(torch.equal(r,torch.zeros_like(r)))
        b,h=build(.25+q*.5);s=self.embed.slot_statistics(b,h)
        self.assertEqual(tuple(s.shape),(1,8,11));self.assertTrue(torch.equal(s[0,:,7],torch.ones(8)))
    def test_02_negative_world(self):
        q=torch.tensor([[x,y,z] for x in (-.25,.25) for y in (-.25,.25) for z in (-.25,.25)])
        b,h=build(torch.tensor([-.25,-1.25,-3.25])+q*.5)
        self.assertTrue(torch.equal(h.levels[0].coords,torch.tensor([[-1,-3,-7]])))
        self.assertTrue(torch.equal(self.embed.slot_statistics(b,h)[0,:,7],torch.ones(8)))
    def test_03_boundaries(self):
        q=torch.tensor([[-.5000001,-.5,-.4999999],[-1e-7,0,1e-7],[.4999999,.5,.5000001]])
        slots,_=subvoxel_coordinates(q);self.assertEqual(slots.tolist(),[0,3,7])
    def test_04_residual(self):
        q=torch.tensor([[-.4,-.1,-.2],[.1,.3,.4]])
        _,r=subvoxel_coordinates(q)
        torch.testing.assert_close(r,torch.tensor([[-.15,.15,.05],[-.15,.05,.15]]))
    def test_05_single_point(self):
        b,h=build([[.05,.1,.15]],dt=[-.04]);s=self.embed.slot_statistics(b,h)
        expected=torch.tensor([-.15,-.05,.05,0.,0.,0.]).float()
        torch.testing.assert_close(s[0,0,:6],expected)
        torch.testing.assert_close(s[0,0,6:],torch.tensor([torch.log(torch.tensor(2.)),1.,1.,2**(-.04/.5),0.]))
        self.assertEqual(float(s[0,1:].abs().sum()),0.)
    def test_06_mid360_single(self):
        b,h=build([[.1,.1,.1]],[1]);self.assertEqual(float(self.embed.slot_statistics(b,h)[0,0,8]),0.)
    def test_07_sensor_mixture(self):
        b,h=build([[.1,.1,.1]]*5,[0,1,0,1,1]);self.assertAlmostEqual(float(self.embed.slot_statistics(b,h)[0,0,8]),.4)
    def test_08_time_stats(self):
        b,h=build([[.1,.1,.1]]*3,dt=[-.1,-.2,-.3]);s=self.embed.slot_statistics(b,h)[0,0]
        torch.testing.assert_close(s[9],torch.exp2(b['delta_t']/.5).mean());torch.testing.assert_close(s[10],b['delta_t'].std(unbiased=False))
    def test_09_permutation(self):
        b,h=build(torch.rand(100,3)*2-1,sensors=(torch.arange(100)%2).tolist(),dt=(-torch.rand(100)).tolist())
        perm=torch.randperm(100);b2={k:v[perm] for k,v in b.items()};h2=HierarchyBuilder()(b2['points'],torch.zeros(100,dtype=torch.long))
        with torch.no_grad():
            for x,y in ((self.embed.slot_statistics(b,h),self.embed.slot_statistics(b2,h2)),(self.embed(b,h),self.embed(b2,h2))):
                self.assertLessEqual(float((x-y).abs().max()),1e-6)
    def test_10_empty_slots_and_input(self):
        for points in ([[.1,.1,.1],[10.1,10.1,10.1]],[]):
            b,h=build(points);s=self.embed.slot_statistics(b,h);token=self.embed(b,h)
            self.assertTrue(torch.isfinite(s).all());self.assertTrue(torch.isfinite(token).all())
            self.assertTrue(torch.equal(s[s[:,:,7]==0],torch.zeros_like(s[s[:,:,7]==0])))
            self.assertEqual(tuple(token.shape),(len(h.levels[0].coords),128))
    def test_11_batch_identity(self):
        b,h=build([[.1,.1,.1]]*2,[0,1],[-.1,-.3],[0,1]);s=self.embed.slot_statistics(b,h)
        self.assertEqual(s[:,0,8].tolist(),[1.,0.]);torch.testing.assert_close(s[:,0,9],torch.exp2(torch.tensor([-.1,-.3])/.5))
    def test_12_checkpoint_strictness(self):
        cfg=copy.deepcopy(CFG);cfg['model']['voxel']['embedding']='legacy';old=LiDARUAVDetector(cfg);new=LiDARUAVDetector(CFG)
        state=old.state_dict();report=new.load_pre_sbe_weights(state)
        self.assertEqual(report['unexpected_missing_keys'],[])
        expected=sorted(k for k in new.state_dict() if k.startswith('voxel_embed.'))
        self.assertEqual(report['expected_missing_keys'],expected)
        for k,v in state.items():
            if not k.startswith('voxel_embed.'):self.assertTrue(torch.equal(v,new.state_dict()[k]),k)
        missing=dict(state);del missing['head.cls.0.weight']
        with self.assertRaises(ValueError):new.load_pre_sbe_weights(missing)
        extra=dict(state);extra['unknown.weight']=torch.zeros(1)
        with self.assertRaises(ValueError):new.load_pre_sbe_weights(extra)
        wrong=dict(state);wrong['head.cls.0.weight']=torch.zeros(1)
        with self.assertRaises(ValueError):new.load_pre_sbe_weights(wrong)
    def test_13_no_pointwise_learned_feature(self):
        model=LiDARUAVDetector(CFG)
        self.assertIsInstance(model.voxel_embed,SBELiteVoxelEmbed)
        self.assertFalse(any(isinstance(m,LegacyVoxelEmbed) for m in model.modules()))
        self.assertFalse(any('point_mlp' in n or 'sensor_embedding' in n for n,_ in model.named_parameters()))
        b,h=build(torch.rand(100,3));calls=[]
        hook=model.voxel_embed.proj.register_forward_pre_hook(lambda m,args:calls.append(tuple(args[0].shape)))
        with torch.no_grad():model.voxel_embed(b,h)
        hook.remove();self.assertEqual(calls,[(len(h.levels[0].coords),104)])
        self.assertEqual(len(SLOT_DESCRIPTOR),11)
    def test_14_amp_statistics_fp32(self):
        b,h=build([[.1,.1,.1]]*3,dt=[-.1,-.2,-.3])
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):stats=self.embed.slot_statistics(b,h)
        self.assertEqual(stats.dtype,torch.float32)
        self.assertTrue(torch.equal(stats,self.embed.slot_statistics(b,h)))

class TimeWeightedSBETests(unittest.TestCase):
    def embed(self, half_life=.5):
        cfg=copy.deepcopy(CFG['model']['voxel']['sbe'])
        cfg['time_half_life_s']=half_life
        return SBELiteVoxelEmbed(config=cfg)

    def test_default_half_life_is_point_two_seconds(self):
        self.assertEqual(SBELiteVoxelEmbed().time_half_life_s,.2)
        self.assertEqual(CFG['model']['voxel']['sbe']['time_half_life_s'],.2)
        b,h=build([[.1,.1,.1]],dt=[-1.])
        stats=SBELiteVoxelEmbed().slot_statistics(b,h)
        self.assertEqual(float(stats[0,0,9]),1/32)

    def test_weighted_geometry_analytical(self):
        b,h=build([[.05,.1,.15],[.2,.15,.1]],sensors=[0,1],dt=[-1.,0.])
        stats=self.embed().slot_statistics(b,h)[0,0]
        torch.testing.assert_close(stats[:3],torch.tensor([.09,.03,-.03]))
        torch.testing.assert_close(stats[3:6],torch.tensor([.12,.04,.04]))
        self.assertAlmostEqual(float(stats[6]),math.log(3),places=6)
        self.assertEqual(float(stats[7]),1.)
        self.assertEqual(float(stats[8]),.5)  # Sensor ratio remains unweighted.
        self.assertEqual(float(stats[9]),.625)
        self.assertEqual(float(stats[10]),.5)  # Temporal std remains unweighted.

    def test_equal_times_reduce_to_population_mean_std(self):
        b,h=build([[.05,.1,.15],[.2,.15,.1]],dt=[-.4,-.4])
        stats=self.embed().slot_statistics(b,h)[0,0]
        residual=(b['points']-.125)/.5
        torch.testing.assert_close(stats[:3],residual.mean(0))
        torch.testing.assert_close(stats[3:6],residual.std(0,unbiased=False))
        self.assertAlmostEqual(float(stats[9]),2**(-.4/.5),places=6)

    def test_common_age_shift_only_changes_freshness(self):
        b,h=build([[.05,.1,.15],[.2,.15,.1]],dt=[-.2,0.])
        old={**b,'delta_t':b['delta_t']-.8}
        current_stats=self.embed().slot_statistics(b,h)[0,0]
        old_stats=self.embed().slot_statistics(old,h)[0,0]
        torch.testing.assert_close(current_stats[:6],old_stats[:6])
        torch.testing.assert_close(old_stats[9],current_stats[9]*2**(-.8/.5))
        torch.testing.assert_close(current_stats[10],old_stats[10],atol=1e-6,rtol=1e-5)

    def test_half_life_controls_recent_point_contribution(self):
        b,h=build([[.05,.1,.15],[.2,.15,.1]],dt=[-1.,0.])
        slow=self.embed(1.).slot_statistics(b,h)[0,0]
        fast=self.embed(.25).slot_statistics(b,h)[0,0]
        self.assertGreater(float(fast[0]),float(slow[0]))
        self.assertLess(float(fast[9]),float(slow[9]))

    def test_tiny_half_life_keeps_geometry_finite(self):
        b,h=build([[.05,.1,.15],[.2,.15,.1]],dt=[-1.,-.9])
        stats=self.embed(1e-6).slot_statistics(b,h)[0,0]
        self.assertTrue(torch.isfinite(stats).all())
        torch.testing.assert_close(stats[:3],torch.tensor([.15,.05,-.05]))
        self.assertEqual(float(stats[9]),0.)

    def test_invalid_decay_and_future_time_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.embed(value)
        for value in (.01,float('nan'),float('-inf')):
            b,h=build([[.1,.1,.1]],dt=[value])
            with self.assertRaises(ValueError):self.embed().slot_statistics(b,h)
        cfg=copy.deepcopy(CFG['model']['voxel']['sbe']);cfg['statistics_version']='old'
        with self.assertRaises(ValueError):SBELiteVoxelEmbed(config=cfg)

    def test_internal_crossattention_to_128_has_gradients(self):
        b,h=build([[.05,.1,.15],[.2,.15,.1],[.3,.3,.3]],dt=[-1.,0.,-.2])
        embed=self.embed()
        token,debug=embed(b,h,return_debug=True)
        self.assertEqual(tuple(debug['slot_stats'].shape),(1,8,11))
        self.assertEqual(tuple(debug['slot_input'].shape),(1,8,14))
        self.assertEqual(tuple(debug['slot_tokens'].shape),(1,8,16))
        self.assertEqual(tuple(debug['attention'].shape),(1,2,1,8))
        self.assertEqual(tuple(debug['projection_input'].shape),(1,104))
        self.assertEqual(tuple(token.shape),(1,128))
        empty=debug['key_padding_mask'][0]
        self.assertTrue(torch.equal(debug['attention'][0,:,:,empty],
                                    torch.zeros_like(debug['attention'][0,:,:,empty])))
        (token*torch.arange(128,dtype=token.dtype)).sum().backward()
        for parameter in (embed.vqsa.voxel_query,embed.vqsa.slot_embed[0].weight,
                          embed.vqsa.attention.in_proj_weight,embed.proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
