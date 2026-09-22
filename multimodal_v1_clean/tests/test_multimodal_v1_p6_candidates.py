from __future__ import annotations
import unittest
import torch
from rdq_uav.multimodal_v1.candidate import CandidateSet,associate_candidates,point_to_box_distance,HYP_RV,HYP_R,HYP_V


def radar(batch=(0,0,1)):
    f=torch.zeros(3,128); f[0,0]=1; f[1,1]=1; f[2,:2]=1
    return CandidateSet(torch.tensor([.9,.8,.7]),f,torch.tensor([[0.,0.,1.],[1.,0.,1.],[2.,0.,1.]]),torch.ones(3,dtype=torch.bool),
        torch.zeros(3,4),torch.zeros(3,dtype=torch.bool),torch.tensor(batch), 'radar',torch.tensor([10,11,12]))

def vision(scores=None):
    f=torch.zeros(3,128); f[0,0]=1; f[1,1]=1; f[2,:2]=1
    return CandidateSet(torch.tensor([.6,.5,.4]) if scores is None else scores,f,torch.zeros(3,3),torch.zeros(3,dtype=torch.bool),
        torch.tensor([[9.,9.,11.,11.],[39.,39.,41.,41.],[19.,19.,21.,21.]]),torch.ones(3,dtype=torch.bool),torch.tensor([0,0,1]),'rgb',torch.tensor([20,21,22]))

class TestP6(unittest.TestCase):
    def test_point_to_box_rule(self):
        d=point_to_box_distance(torch.tensor([[10.,10.],[0.,0.]]),torch.tensor([[9.,9.,11.,11.],[3.,4.,5.,6.]]))
        self.assertEqual(float(d[0,0]),0.0); self.assertAlmostEqual(float(d[1,1]),5.0,places=6)

    def test_known_matching_and_conservation(self):
        r,v=radar(),vision(); p=torch.tensor([[10.,10.],[40.,40.],[20.,20.]])
        h=associate_candidates(r,v,projected_radar_xy=p)
        self.assertEqual(int((h.hypothesis_type==HYP_RV).sum()),3)
        self.assertEqual(int(h.m_R.sum()),r.n); self.assertEqual(int(h.m_V.sum()),v.n)

    def test_hard_gate_and_unmatched_preserved(self):
        r,v=radar(),vision(); p=torch.tensor([[100.,100.],[120.,120.],[140.,140.]])
        h=associate_candidates(r,v,projected_radar_xy=p)
        self.assertEqual(int((h.hypothesis_type==HYP_RV).sum()),0)
        self.assertEqual(int((h.hypothesis_type==HYP_R).sum()),3)
        self.assertEqual(int((h.hypothesis_type==HYP_V).sum()),3)

    def test_score_not_in_matching_cost(self):
        r=radar(); p=torch.tensor([[10.,10.],[40.,40.],[20.,20.]])
        a=associate_candidates(r,vision(),projected_radar_xy=p)
        b=associate_candidates(r,vision(torch.tensor([.01,.99,.02])),projected_radar_xy=p)
        self.assertTrue(torch.equal(a.radar_source_index,b.radar_source_index))
        self.assertTrue(torch.equal(a.vision_source_index,b.vision_source_index))


    def test_association_info_keeps_feature_gradient(self):
        r,v=radar(),vision(); r.feature.requires_grad_(True); v.feature.requires_grad_(True)
        h=associate_candidates(r,v,projected_radar_xy=torch.tensor([[10.,10.],[40.,40.],[20.,20.]]))
        rv=torch.nonzero(h.hypothesis_type==HYP_RV).flatten()
        h.association_info[rv,1].sum().backward()
        self.assertIsNotNone(r.feature.grad); self.assertIsNotNone(v.feature.grad)

    def test_batch_isolation(self):
        r,v=radar(),vision(); p=torch.tensor([[20.,20.],[40.,40.],[100.,100.]])
        h=associate_candidates(r,v,projected_radar_xy=p)
        rv=torch.nonzero(h.hypothesis_type==HYP_RV).flatten()
        for i in rv.tolist():
            self.assertEqual(int(h.batch_index[i]),0)  # only sample0 can match here

if __name__=='__main__': unittest.main()
