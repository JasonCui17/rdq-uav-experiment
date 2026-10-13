"""Fixed-voxel gated MTCE: physics, identity, gradients and public contracts."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from rdq_uav.multimodal_v2.radar_geometry import HierarchyBuilder
from rdq_uav.multimodal_v2.radar_temporal_centroid import GatedTemporalMotion
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_loss import CandidateLoss
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.candidate_builders import RadarCandidateBuilder
from rdq_uav.multimodal_v2.radar_experiment_init import load_experiment_warmstart
from rdq_uav.multimodal_v2.radar_lora import set_efficient_train_mode

ROOT = Path(__file__).resolve().parents[3]


def batch(speed=.2, missing=(), sensors=False):
    xyz, times, ids, sensors_out = [], [], [], []
    for b, v in enumerate((speed, 0.)):
        for k, t in enumerate((-.9, -.7, -.5, -.3, -.1)):
            if k in missing:
                continue
            for j in range(4):
                xyz.append([.25+v*t, .15+.005*j, .1])
                times.append(t); ids.append(b)
                sensors_out.append(k % 2 if sensors else j % 2)
    return dict(points=torch.tensor(xyz).reshape(-1, 3), delta_t=torch.tensor(times),
                point_batch_index=torch.tensor(ids, dtype=torch.long),
                sensor_id=torch.tensor(sensors_out, dtype=torch.long), num_samples=2,
                target_xyz=torch.tensor([[.3, .16, .1], [.25, .16, .1]]),
                target_valid=torch.ones(2, dtype=torch.bool))


def hierarchy(b):
    return HierarchyBuilder()(b['points'], b['point_batch_index'])


def config(gate=True, velocity=True, enabled=True, lora=False):
    c = yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/radar_a2_learned_sbe32.yaml').read_text())
    c['model']['voxel']['motion_centroid'] = dict(enabled=enabled, mode='gated_velocity',
                                                feature_gate=gate, velocity_guidance=velocity)
    if lora:
        c['model']['lora'] = dict(enabled=True, rank=8, alpha=16.)
    return c


def test_signed_velocity_real_time_and_isolation():
    b=batch(); contexts,_=GatedTemporalMotion({})(b,hierarchy(b))
    for x in contexts:
        assert x.descriptor.shape == (2, 11)
        torch.testing.assert_close(x.velocity_xyz[0],torch.tensor([.2,0.,0.]),atol=1e-6,rtol=0)
        assert x.velocity_xyz[1].norm() < 1e-6
        assert x.valid.all() and x.dynamic.tolist() == [True,False]
        torch.testing.assert_close(x.reference_dt,torch.tensor([-.5,-.5]),atol=1e-6,rtol=0)
    neg=batch(-.2); x=GatedTemporalMotion({})(neg,hierarchy(neg))[0][0]
    assert x.velocity_xyz[0,0] < 0


@pytest.mark.parametrize('missing',[(1,3),(0,1,2),(2,)])
def test_do_not_bridge_missing_bins(missing):
    b=batch(missing=missing); contexts,_=GatedTemporalMotion({})(b,hierarchy(b))
    assert all(not x.valid.any() for x in contexts)


def test_bad_sensor_composition_rejects_fit():
    b=batch(sensors=True); contexts,_=GatedTemporalMotion({})(b,hierarchy(b))
    assert all(not x.valid.any() for x in contexts)


def test_static_unknown_and_displacement_bound():
    b=batch(); module=GatedTemporalMotion(dict(max_offset_m=.02))
    x=module(b,hierarchy(b))[0][0]; f=torch.randn(2,128)
    with torch.no_grad(): module.offset_gate[-1].bias.fill_(3.)
    offset=module.offset(f,x)
    assert offset[0,0] > 0 and offset[0].norm() <= .020001
    assert torch.equal(offset[1],torch.zeros(3))
    unknown=batch(missing=(1,3)); x=module(unknown,hierarchy(unknown))[0][0]
    assert torch.count_nonzero(module.offset(f,x)) == 0
    # Feature conditioning is also strict identity for unsupported observations.
    with torch.no_grad(): module.conditioners[0][-1].bias.fill_(1.)
    torch.testing.assert_close(module.condition(f,x,0),f,rtol=0,atol=0)


def test_no_gt_dependence_and_invalid_times():
    b=batch(); m=GatedTemporalMotion({}); first=m(b,hierarchy(b))[0]
    b['target_xyz'].fill_(999.)
    second=m(b,hierarchy(b))[0]
    for x,y in zip(first,second):
        torch.testing.assert_close(x.descriptor,y.descriptor,rtol=0,atol=0)
    b['delta_t'][0]=.1
    with pytest.raises(ValueError):m(b,hierarchy(b))


def test_empty_and_outlier_fit():
    b=batch()
    for key in ('points','delta_t','sensor_id','point_batch_index'): b[key]=b[key][:0]
    contexts,_=GatedTemporalMotion({})(b,hierarchy(b))
    assert all(x.descriptor.shape == (0,11) for x in contexts)
    b=batch(); b['points'][8:12,1] += .25
    contexts,_=GatedTemporalMotion(dict(max_fit_error_m=.01))(b,hierarchy(b))
    assert not contexts[0].valid[0]


@pytest.mark.parametrize('gate,velocity',[(True,False),(False,True),(True,True)])
def test_a2_identity_contract_and_checkpoint(tmp_path,gate,velocity):
    base=LiDARUAVDetector(config(enabled=False)).eval()
    path=tmp_path/'a2.ckpt'
    torch.save(dict(state_dict={'network.lidar.detector.'+k:v for k,v in base.state_dict().items()}),path)
    model=LiDARUAVDetector(config(gate,velocity)).eval()
    report=load_experiment_warmstart(model,path)
    assert all(k.startswith('temporal_motion.') for k in report['new_keys'])
    b=batch(); before,after=base(b),model(b)
    for name in ('logits','pred_xyz','residual_xyz','fine_features','voxel_centers','batch_index','source_token_id'):
        torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)
    restored=LiDARUAVDetector(config(gate,velocity)).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored(b)['pred_xyz'],after['pred_xyz'],rtol=0,atol=0)
    build=RadarCandidateBuilder(CandidateSelector(dict(raw_topk=50,pre_nms_topk=50,
        nms_radius_m=1.,final_topk=50),raw_only=True),use_raw=True)
    a,c=build(before),build(after)
    assert c.feature.shape == (c.n,128) and c.xyz_m.shape == (c.n,3)
    torch.testing.assert_close(a.feature,c.feature,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['heads','lora'])
def test_updates_gradients_freezing_and_total_residual(tmp_path,mode):
    c=config(lora=mode=='lora'); model=LiDARUAVDetector(c)
    set_efficient_train_mode(model,mode)
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    assert all(p.requires_grad for p in model.temporal_motion.parameters())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    b=batch()
    for _ in range(3):
        opt.zero_grad(); out=model(b)
        torch.testing.assert_close(out['pred_xyz'],out['voxel_centers']+out['residual_xyz'])
        CandidateLoss(c)(out,b)['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None and torch.equal(p,before[name])
    assert model.temporal_motion.conditioners[0][-1].weight.grad.abs().sum() > 0
    assert model.temporal_motion.offset_gate[-1].weight.grad.abs().sum() > 0
    out=model(b); d=out['motion_diagnostics']
    assert torch.count_nonzero(d['offset_xyz'][~d['dynamic']]) == 0
    assert d['offset_xyz'][d['dynamic']].norm() > 0
    restored=LiDARUAVDetector(c); restored.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(b)['pred_xyz'],model.eval()(b)['pred_xyz'],rtol=0,atol=0)


def test_warmstart_rejects_missing_old_parameter(tmp_path):
    base=LiDARUAVDetector(config(enabled=False))
    state={'network.lidar.detector.'+k:v for k,v in base.state_dict().items()}
    del state['network.lidar.detector.head.cls.0.weight']
    path=tmp_path/'bad.ckpt'; torch.save(dict(state_dict=state),path)
    with pytest.raises(ValueError):load_experiment_warmstart(LiDARUAVDetector(config()),path)


def test_query_decoder_combination_rejected():
    c=config(); c['model']['query_decoder']=dict(enabled=True)
    with pytest.raises(ValueError):LiDARUAVDetector(c)


def test_noise_threshold_and_stale_motion():
    b=batch(.05); ctx=GatedTemporalMotion({})(b,hierarchy(b))[0][0]
    assert ctx.valid.all() and not ctx.dynamic.any()
    b=batch(missing=(3,4)); ctx=GatedTemporalMotion({})(b,hierarchy(b))[0][0]
    assert not ctx.valid.any()  # Three old bins are not current motion evidence.


def test_bfloat16_model_update():
    model=LiDARUAVDetector(config()); b=batch()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=model(b); loss=CandidateLoss(config())(out,b)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert out['pred_xyz'].shape == (2,3) and out['fine_features'].shape == (2,128)


@pytest.mark.parametrize('regime',['full','heads','lora'])
def test_runtime_freeze_and_optimizer(tmp_path,regime):
    from rdq_uav.multimodal_v2.training import build_runtime,trainable_parameter_groups
    original=LiDARUAVDetector(config(enabled=False))
    ckpt=tmp_path/'a2.ckpt'
    torch.save(dict(state_dict={'network.lidar.detector.'+k:v for k,v in original.state_dict().items()}),ckpt)
    rcfg=config(lora=regime=='lora'); rp=tmp_path/'radar.yaml';rp.write_text(yaml.safe_dump(rcfg))
    cfg=yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/b0_a2_learned_sbe32.yaml').read_text())
    cfg['initialization'].update(lidar_config=str(rp),abc_warmstart_checkpoint=str(ckpt))
    cfg['training']['radar_train_mode']=regime
    runtime=build_runtime(cfg,ROOT,torch.device('cpu'))
    groups=trainable_parameter_groups(runtime,.0001,.0001)
    params={id(p) for group in groups for p in group['params']}
    assert all(id(p) in params for p in runtime.lidar_detector.temporal_motion.parameters())
    assert not any(p.requires_grad for p in runtime.model.interaction.parameters())
    cfg['initialization']['abc_warmstart_checkpoint']='missing_initialization_file.ckpt'
    rebuilt=build_runtime(cfg,ROOT,torch.device('cpu'),initialize_from_baselines=False)
    rebuilt.model.load_state_dict(runtime.model.state_dict(),strict=True)


def test_b4_accepts_gated_candidates_without_shape_changes():
    from dataclasses import replace
    from rdq_uav.multimodal_v2.candidate_builders import RGBCandidateBuilder
    from rdq_uav.multimodal_v2.pair_fusion import CandidatePairFusion
    model=LiDARUAVDetector(config()).eval()
    with torch.no_grad():
        model.temporal_motion.offset_gate[-1].bias.fill_(1.)
    raw=model(batch())
    r=RadarCandidateBuilder(CandidateSelector(dict(raw_topk=50,pre_nms_topk=50,
        nms_radius_m=1.,final_topk=50),raw_only=True),use_raw=True)(raw)
    r=r.with_projection(torch.full((r.n,2),50.),torch.ones(r.n,dtype=torch.bool))
    visual=dict(pred_logits=torch.zeros(2,3,1),pred_boxes=torch.tensor([.5,.5,.2,.2]).expand(2,3,4),
                decoder_query_features=torch.randn(2,3,256))
    v=RGBCandidateBuilder(pre_topk=50,final_topk=50,apply_nms=False)(visual,torch.tensor([[100,100],[100,100]]))
    out=CandidatePairFusion()(r,v,num_samples=2)
    for item in out.diagnostics['pair_alignment']:
        assert item['similarity'].shape == (1,3)
    for b,indices in enumerate(out.top3d_indices(2)):
        torch.testing.assert_close(out.xyz_m[indices[0]],r.xyz_m[r.batch_index==b][0])
    assert out.radar_candidates.feature.shape == (2,128)
