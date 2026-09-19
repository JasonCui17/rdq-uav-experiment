"""SBE physical-statistics and strict downstream inheritance regressions."""
import copy
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
    def setUp(self):torch.manual_seed(42);self.embed=SBELiteVoxelEmbed()
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
        expected=torch.tensor([-.15,-.05,.05,.15,.05,.05]).float()
        torch.testing.assert_close(s[0,0,:6],expected)
        torch.testing.assert_close(s[0,0,6:],torch.tensor([torch.log(torch.tensor(2.)),1.,1.,.04,0.]))
        self.assertEqual(float(s[0,1:].abs().sum()),0.)
    def test_06_mid360_single(self):
        b,h=build([[.1,.1,.1]],[1]);self.assertEqual(float(self.embed.slot_statistics(b,h)[0,0,8]),0.)
    def test_07_sensor_mixture(self):
        b,h=build([[.1,.1,.1]]*5,[0,1,0,1,1]);self.assertAlmostEqual(float(self.embed.slot_statistics(b,h)[0,0,8]),.4)
    def test_08_time_stats(self):
        b,h=build([[.1,.1,.1]]*3,dt=[-.1,-.2,-.3]);s=self.embed.slot_statistics(b,h)[0,0]
        self.assertAlmostEqual(float(s[9]),.1);torch.testing.assert_close(s[10],b['delta_t'].std(unbiased=False))
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
        self.assertEqual(s[:,0,8].tolist(),[1.,0.]);torch.testing.assert_close(s[:,0,9],torch.tensor([.1,.3]))
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

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
