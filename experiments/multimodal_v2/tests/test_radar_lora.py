"""LoRA correctness, frozen-prefix behavior, reload, and actual runtime setup."""
import copy
from pathlib import Path
import pytest
import torch
import yaml
from test_radar_abc import config, batch
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_loss import CandidateLoss
from rdq_uav.multimodal_v2.radar_lora import LoRALinear, inject_lora, set_efficient_train_mode, parameter_report
from rdq_uav.multimodal_v2.radar_experiment_init import load_experiment_warmstart

ROOT=Path(__file__).resolve().parents[3]

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def lora_config(variant='control'):
    cfg=config(variant)
    cfg['model']['lora']={'enabled':True,'rank':8,'alpha':16.,'dropout':0.}
    return cfg


def base_checkpoint(tmp_path):
    torch.manual_seed(42);model=LiDARUAVDetector(config('baseline'))
    path=tmp_path/'a2.ckpt'
    torch.save({'state_dict':{'network.lidar.detector.'+k:v for k,v in model.state_dict().items()}},path)
    return model,path


def test_zero_identity_strict_warmstart_and_targets(tmp_path):
    base,path=base_checkpoint(tmp_path);model=LiDARUAVDetector(lora_config())
    report=load_experiment_warmstart(model,path)
    assert len(model.lora_targets)==8 and len(report['new_keys'])==16
    for key in ('logits','pred_xyz','fine_features'):
        torch.testing.assert_close(base.eval()(batch())[key],model.eval()(batch())[key],rtol=0,atol=0)
    set_efficient_train_mode(model,'lora')
    report=parameter_report(model)
    assert report['adapter_parameters']==24576
    assert not any('.weight' in n and n.startswith('encoder') for n in report['trainable_names'])


@pytest.mark.parametrize('variant',['control','a_head'])
@pytest.mark.parametrize('mode',['heads','lora'])
def test_updates_only_intended_parameters_and_reload(tmp_path,variant,mode):
    _,path=base_checkpoint(tmp_path)
    cfg=lora_config(variant) if mode=='lora' else config(variant)
    model=LiDARUAVDetector(cfg);load_experiment_warmstart(model,path);set_efficient_train_mode(model,mode)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    trace={}
    handles=[getattr(model,name).register_forward_hook(lambda m,i,o,n=name:trace.update({n:o.requires_grad})) for name in ('encoder0','encoder1','up10')]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad();result=CandidateLoss(cfg)(model(batch()),batch());result['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    for handle in handles:handle.remove()
    for n,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
            assert torch.equal(before[n],p)
    assert not trace['encoder0']
    assert trace['encoder1']==(mode=='lora') and trace['up10']==(mode=='lora')
    assert not torch.equal(before['head.cls.2.weight'],model.head.cls[2].weight)
    if variant=='a_head':assert not torch.equal(before['quality_head.3.weight'],model.quality_head[3].weight)
    if mode=='lora':
        layer=model.encoder1.blocks[0].qkv
        assert layer.lora_A.grad.abs().sum()>0 and layer.lora_B.grad.abs().sum()>0
    restored=LiDARUAVDetector(cfg);restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored.eval()(batch())['pred_xyz'],model.eval()(batch())['pred_xyz'],rtol=0,atol=0)


def test_bad_targets_and_modes_fail():
    model=LiDARUAVDetector(config('control'))
    with pytest.raises(ValueError,match='matched no'):inject_lora(model,{'targets':['missing']})
    with pytest.raises(ValueError):set_efficient_train_mode(model,'lora')
    model=LiDARUAVDetector(lora_config())
    with pytest.raises(ValueError):set_efficient_train_mode(model,'heads')


def test_runtime_integration(tmp_path):
    from rdq_uav.multimodal_v2.training import build_runtime,trainable_parameter_groups
    _,ckpt=base_checkpoint(tmp_path)
    rcfg=lora_config('a_head');rp=tmp_path/'radar.yaml';rp.write_text(yaml.safe_dump(rcfg))
    cfg=yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/b0_a2_learned_sbe32.yaml').read_text())
    cfg['initialization']['lidar_config']=str(rp)
    cfg['initialization']['abc_warmstart_checkpoint']=str(ckpt)
    cfg['training']['radar_train_mode']='lora'
    runtime=build_runtime(cfg,ROOT,torch.device('cpu'))
    groups=trainable_parameter_groups(runtime,.0001,.0001)
    assert sum(p.numel() for g in groups for p in g['params'])==parameter_report(runtime.lidar_detector)['trainable_parameters']
    assert not any(p.requires_grad for p in runtime.model.interaction.parameters())
    # Complete checkpoint restores without access to original A2 warmstart.
    cfg['initialization']['abc_warmstart_checkpoint']='does_not_exist.ckpt'
    rebuilt=build_runtime(cfg,ROOT,torch.device('cpu'),initialize_from_baselines=False)
    rebuilt.model.load_state_dict(runtime.model.state_dict(),strict=True)
