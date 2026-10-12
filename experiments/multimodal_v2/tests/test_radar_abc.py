"""Behavioral checks for A/B/C, including two real optimizer updates."""
import copy
from pathlib import Path
import importlib.util

import pytest
import torch
import yaml

from rdq_uav.multimodal_v2.radar_loss import CandidateLoss
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.radar_experiment_loss import quality_targets, negative_weights, range_weights
from rdq_uav.multimodal_v2.radar_experiment_init import load_experiment_warmstart

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('suite',ROOT/'experiments/multimodal_v2/abc/run_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)

@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(name):
    base=yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/radar_a2_learned_sbe32.yaml').read_text())
    return suite.ablation_config(base,name)


def batch():
    torch.manual_seed(17)
    # Two scenes with distant distractors, and >1 point per voxel.
    points=torch.cat([torch.rand(9,3)*.1+torch.tensor(c) for c in ([0.,0.,0.],[5.,5.,5.],[40.,0.,0.],[60.,10.,0.])])
    return dict(points=points,point_batch_index=torch.tensor([0]*18+[1]*18),
                sensor_id=torch.arange(36)%2,delta_t=torch.linspace(-.9,0,36),num_samples=2,
                target_xyz=torch.tensor([[.05,.05,.05],[40.05,.05,.05]]),target_valid=torch.tensor([True,True]))


def test_quality_targets_batch_index_and_detach():
    b=batch();xyz=torch.tensor([[0.,0.,0.],[40.,0.,0.]],requires_grad=True)
    out=dict(pred_xyz=xyz,batch_index=torch.tensor([0,1]))
    q=quality_targets(out,b,1.)
    torch.testing.assert_close(q[0],q[1],atol=1e-6,rtol=0)
    assert not q.requires_grad


def test_quality_ranks_before_preselection_and_nms():
    o=dict(logits=torch.tensor([4.,2.]),quality_logits=torch.tensor([-5.,5.]),
           pred_xyz=torch.tensor([[0.,0.,0.],[.1,0.,0.]]),fine_features=torch.zeros(2,128),
           source_token_id=torch.arange(2),batch_index=torch.zeros(2,dtype=torch.long))
    selector=CandidateSelector(dict(raw_topk=1,pre_nms_topk=1,nms_radius_m=1.,final_topk=1))
    result=selector(o)[0]
    assert result['nms']['source_token_id'].tolist()==[1]
    torch.testing.assert_close(result['nms']['score'], (o['logits'].sigmoid()*o['quality_logits'].sigmoid())[1:])


def test_negative_only_and_missing_gt():
    cfg=config('control');model=LiDARUAVDetector(cfg);b=batch();b['target_xyz']+=1000
    out=model(b);out['logits'].retain_grad()
    result=CandidateLoss(cfg)(out,b)
    assert result['num_no_current_support']==2 and result['num_supervised_samples']==2
    result['loss'].backward();assert torch.all(out['logits'].grad>0)
    b['target_valid'][:]=False
    result=CandidateLoss(cfg)(model(b),b)
    assert result['num_supervised_samples']==0 and result['loss'].item()==0


def test_ohem_per_query_and_range():
    z=torch.tensor([-10.,-11.,-12.]);n=torch.ones(3,dtype=torch.bool)
    w,count=negative_weights(z,n,dict(enabled=True,topk=1,easy_weight=.25,hard_weight=2.))
    assert count==1 and w.tolist()==[2.,.25,.25]
    assert range_weights(torch.tensor([[10.,0.,0.],[40.,0.,0.],[80.,0.,0.]]),{}).tolist()==[1.,1.5,2.5]


@pytest.mark.parametrize('name',suite.VARIANTS)
def test_warmstart_updates_and_reload(name,tmp_path):
    b=batch();torch.manual_seed(42)
    baseline=LiDARUAVDetector(config('baseline'))
    path=tmp_path/'base.ckpt'
    torch.save({'state_dict':{'network.lidar.detector.'+k:v for k,v in baseline.state_dict().items()}},path)
    cfg=config(name);model=LiDARUAVDetector(cfg)
    load_experiment_warmstart(model,path)
    model.eval()
    if name!='c_query':
        a,c=baseline.eval()(b),model(b)
        for key in ('logits','pred_xyz','fine_features'):
            torch.testing.assert_close(a[key],c[key],rtol=0,atol=0)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    for _ in range(2):
        opt.zero_grad();out=model(b);loss=CandidateLoss(cfg)(out,b)
        assert loss['num_supervised_samples']==2 and torch.isfinite(loss['loss'])
        loss['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
    if name=='a_head': assert not torch.equal(initial['quality_head.3.weight'],model.quality_head[3].weight)
    if name=='c_query':
        assert loss['num_pos']==2 and loss['num_neg']==62
        assert out['pred_xyz'].shape==(64,3)
        assert model.query_decoder.queries.weight.grad.abs().sum()>0
        assert model.voxel_embed.proj.weight.grad is not None
        assert model.voxel_embed.proj.weight.grad.abs().sum() > 0
    new=LiDARUAVDetector(cfg);new.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(model.eval()(b)['pred_xyz'],new.eval()(b)['pred_xyz'])


def test_query_isolation_and_no_nms():
    b=batch();model=LiDARUAVDetector(config('c_query')).eval()
    with torch.no_grad():
        a=model(b);other=copy.deepcopy(b);other['points'][18:]+=100
        c=model(other)
    torch.testing.assert_close(a['pred_xyz'][:32],c['pred_xyz'][:32],rtol=0,atol=0)
    a['pred_xyz'][:]=0
    selected=CandidateSelector(dict(raw_topk=10,pre_nms_topk=1,nms_radius_m=1.,final_topk=10))(a)
    assert all(len(s['nms']['score'])==10 for s in selected)


def test_soft_quality_no_coordinate_label_gradient():
    cfg=config('a_soft');b=batch();out=LiDARUAVDetector(cfg)(b)
    out['pred_xyz']=out['pred_xyz'].detach().requires_grad_()
    result=CandidateLoss(cfg)(out,b)
    assert torch.autograd.grad(result['loss_cls'],out['pred_xyz'],allow_unused=True)[0] is None
