from __future__ import annotations
import unittest
import torch
from rdq_uav.multimodal_v1.candidate import CandidateSet,associate_candidates,HYP_RV,HYP_R,HYP_V
from rdq_uav.multimodal_v1.candidate.reliability_gate import ReliabilityGate
from rdq_uav.multimodal_v1.candidate.shared_query import TypedSharedQuery
from rdq_uav.multimodal_v1.decoder import FusionTransformerDecoder,FusionPredictionHeads
from rdq_uav.multimodal_v1.loss import FusionLoss,FusionTargets


def make_h():
    torch.manual_seed(2)
    r=CandidateSet(torch.tensor([.9,.8]),torch.randn(2,128),torch.tensor([[1.,2.,3.],[4.,5.,6.]]),torch.ones(2,dtype=torch.bool),torch.zeros(2,4),torch.zeros(2,dtype=torch.bool),torch.tensor([0,1]),'radar',torch.tensor([0,1]))
    v=CandidateSet(torch.tensor([.7,.6]),torch.randn(2,128),torch.zeros(2,3),torch.zeros(2,dtype=torch.bool),torch.tensor([[10.,10.,20.,20.],[30.,30.,40.,40.]]),torch.ones(2,dtype=torch.bool),torch.tensor([0,1]),'rgb',torch.tensor([0,1]))
    return associate_candidates(r,v,projected_radar_xy=torch.tensor([[15.,15.],[100.,100.]]))

class TestP7(unittest.TestCase):
    def test_gate_missing_masks_and_sum(self):
        h=make_h(); out=ReliabilityGate()(h)
        self.assertTrue(torch.allclose(out.weights.sum(1),torch.ones(h.n),atol=1e-6))
        self.assertTrue(bool((out.weights[~h.m_R,0]==0).all())); self.assertTrue(bool((out.weights[~h.m_V,1]==0).all()))

    def test_shared_query_one_shape_for_all_types(self):
        h=make_h(); q=TypedSharedQuery()(h)
        self.assertEqual(tuple(q.shape),(h.n,128)); self.assertEqual(set(h.hypothesis_type.tolist()),{HYP_RV,HYP_R,HYP_V})

    def test_decoder_padding_and_backward(self):
        h=make_h(); q=TypedSharedQuery()(h); dec=FusionTransformerDecoder(dropout=0.)
        r2=torch.randn(3,128,requires_grad=True); rb=torch.tensor([0,0,1]); v2=torch.randn(2,384,2,3,requires_grad=True)
        out,aux=dec(q,h.batch_index,r2,rb,v2,return_aux=True)
        self.assertEqual(tuple(out.shape),(h.n,128)); self.assertEqual(aux.memory_key_padding_mask.ndim,2)
        out.sum().backward(); self.assertTrue(torch.isfinite(r2.grad).all()); self.assertTrue(torch.isfinite(v2.grad).all())

    def test_decoder_masks_spatial_image_padding_and_preserves_batch_isolation(self):
        torch.manual_seed(7); dec=FusionTransformerDecoder(dropout=0.).eval()
        query=torch.randn(2,128); qbatch=torch.tensor([0,1]); v2=torch.randn(2,384,2,3)
        image_mask=torch.zeros(2,4,6,dtype=torch.bool); image_mask[1,:,3:]=True
        both,aux=dec(query,qbatch,torch.empty(0,128),torch.empty(0,dtype=torch.long),v2,
            sample_m_R=torch.tensor([0,0],dtype=torch.bool),sample_m_V=torch.tensor([1,1],dtype=torch.bool),
            vision_padding_mask=image_mask,return_aux=True)
        self.assertEqual(aux.memory_key_padding_mask[0].sum().item(),0)
        self.assertEqual(aux.memory_key_padding_mask[1].sum().item(),2)
        single,_=dec(query[:1],torch.tensor([0]),torch.empty(0,128),torch.empty(0,dtype=torch.long),v2[:1],
            sample_m_R=torch.tensor([0],dtype=torch.bool),sample_m_V=torch.tensor([1],dtype=torch.bool),
            vision_padding_mask=image_mask[:1])
        self.assertTrue(torch.allclose(both[:1],single,atol=1e-6,rtol=1e-6))


    def test_missing_required_gt_is_not_negative(self):
        h=make_h(); gate=ReliabilityGate(); go=gate(h)
        # Remove all 2D GT: V-only/RV classification has no valid quality target and must be ignored.
        targets=FusionTargets(torch.zeros(2,4),torch.tensor([0,0],dtype=torch.bool),torch.tensor([[1.,2.,3.],[4.,5.,6.]]),torch.tensor([1,1],dtype=torch.bool))
        pred_box=torch.zeros(h.n,4); pred_xyz=h.radar_xyz.clone(); c2=torch.zeros(h.n); c3=torch.zeros(h.n)
        losses=FusionLoss()(fused_score=go.fused_score,pred_box=pred_box,pred_xyz=pred_xyz,c2d_logit=c2,c3d_logit=c3,hypotheses=h,targets=targets,source_image_size_wh=torch.tensor([[1280.,960.],[1280.,960.]]))
        self.assertTrue(torch.isfinite(losses['loss']))

    def test_rv_with_only_3d_gt_keeps_xyz_regression_without_joint_positive(self):
        h=make_h(); gate=ReliabilityGate(); go=gate(h)
        targets=FusionTargets(torch.zeros(2,4),torch.tensor([0,0],dtype=torch.bool),
            torch.tensor([[1.,2.,3.],[4.,5.,6.]]),torch.tensor([1,0],dtype=torch.bool))
        heads=FusionPredictionHeads(); decoded=torch.randn(h.n,128,requires_grad=True)
        pred=heads(decoded,h,torch.zeros(h.n,2),torch.tensor([[1280.,960.],[1280.,960.]]))
        losses=FusionLoss()(fused_score=go.fused_score,pred_box=pred.box_xyxy_px,pred_xyz=pred.xyz,
            c2d_logit=pred.c2d_logit,c3d_logit=pred.c3d_logit,hypotheses=h,targets=targets,
            source_image_size_wh=torch.tensor([[1280.,960.],[1280.,960.]]))
        self.assertGreater(float(losses['loss_3d']),0.)
        self.assertEqual(int(losses['num_positive']),0)
        self.assertEqual(int(losses['num_reg3d']),1)
        losses['loss_3d'].backward()
        self.assertGreater(float(heads.xyz_head[-1].weight.grad.abs().sum()),0.)

    def test_validity_mask_not_presence(self):
        h=make_h(); gate=ReliabilityGate(); q=TypedSharedQuery()(h); go=gate(h)
        dec=FusionTransformerDecoder(dropout=0.); r2=torch.randn(3,128); rb=torch.tensor([0,0,1]); v2=torch.randn(2,384,2,3)
        z,_=dec(q,h.batch_index,r2,rb,v2); heads=FusionPredictionHeads(xyz_mean=(10,0,0),xyz_std=(5,5,5))
        proj=torch.zeros(h.n,2); proj[h.m_R]=torch.tensor([15.,15.])
        pred=heads(z,h,proj,torch.tensor([[1280.,960.],[1280.,960.]]))
        # Both samples have both GT dimensions. Therefore V-only gets c3D supervision and R-only gets c2D supervision.
        targets=FusionTargets(torch.tensor([[10.,10.,20.,20.],[30.,30.,40.,40.]]),torch.tensor([1,1],dtype=torch.bool),torch.tensor([[1.,2.,3.],[4.,5.,6.]]),torch.tensor([1,1],dtype=torch.bool))
        losses=FusionLoss()(fused_score=go.fused_score,pred_box=pred.box_xyxy_px,pred_xyz=pred.xyz,c2d_logit=pred.c2d_logit,c3d_logit=pred.c3d_logit,hypotheses=h,targets=targets,source_image_size_wh=torch.tensor([[1280.,960.],[1280.,960.]]))
        self.assertTrue(torch.isfinite(losses['loss_c2d'])); self.assertTrue(torch.isfinite(losses['loss_c3d']))
        losses['loss'].backward()
        grads=[p.grad for p in list(gate.parameters())+list(dec.parameters())+list(heads.parameters()) if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

if __name__=='__main__': unittest.main()
