"""Preserve 01f2ac5's trained-motion contract for the new gated frontend."""
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from rdq_uav.multimodal_v2.radar_architecture import validate_b4_radar_config, audit_b4_radar_detector
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_loss import CandidateLoss
from rdq_uav.multimodal_v2.radar_lora import set_efficient_train_mode
from test_radar_gated_motion import config,batch

ROOT=Path(__file__).resolve().parents[3]


def trained():
    torch.manual_seed(42)
    c=config();model=LiDARUAVDetector(c)
    set_efficient_train_mode(model,'heads')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    for _ in range(3):
        opt.zero_grad();CandidateLoss(c)(model(batch()),batch())['loss'].backward();opt.step()
    return model.eval()


def test_config_accepts_joint_and_gate_but_not_velocity_only():
    assert validate_b4_radar_config(config(),1.)['motion_mode']=='gated_velocity'
    validate_b4_radar_config(config(velocity=False),1.)
    with pytest.raises(ValueError,match='three-scale'):validate_b4_radar_config(config(gate=False),1.)


def test_zero_initialized_or_untrained_stage_cannot_be_frozen():
    with pytest.raises(ValueError,match='all zero'):audit_b4_radar_detector(LiDARUAVDetector(config()))
    model=trained();report=audit_b4_radar_detector(model)
    assert report['motion_mode']=='gated_velocity' and len(report['motion_projection_norms'])==4
    with torch.no_grad():
        model.temporal_motion.conditioners[1][-1].weight.zero_()
        model.temporal_motion.conditioners[1][-1].bias.zero_()
    with pytest.raises(ValueError,match='all zero'):audit_b4_radar_detector(model)


def test_migration_strictly_loads_trained_gated_checkpoint(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_gated_b4',ROOT/'experiments/multimodal_v2/diagnostics/prepare_b4_learnsbe_mtce.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    model=trained();radar=tmp_path/'radar.yaml';radar.write_text(yaml.safe_dump(config()))
    ckpt=tmp_path/'b0.ckpt'
    torch.save(dict(state_dict={'network.lidar.detector.'+k:v for k,v in model.state_dict().items()},
                    hyper_parameters={'config':{'experiment':{'stage':'B0'},'initialization':{'lidar_config':str(radar)}}}),ckpt)
    cfg=yaml.safe_load((ROOT/'experiments/multimodal_v2/configs/b4_learnsbe_mtce.yaml').read_text())
    b1=tmp_path/'b1.ckpt';b1.touch();cfg['initialization']['b1_checkpoint']=str(b1)
    result,report=tool.prepare(cfg,ckpt,tmp_path)
    assert report['motion_mode']=='gated_velocity'
    assert result['initialization']['b0_checkpoint']=='b0.ckpt'
    assert all(value>0 for value in report['motion_projection_norms'].values())
