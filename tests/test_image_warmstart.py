from types import SimpleNamespace

import pytest
import torch

from rdq_uav.multimodal_v3.training.train_image import build_image_runtime
from rdq_uav.multimodal_v3.modeling import training


def setup_runtime(monkeypatch):
    detector = torch.nn.Linear(2, 1)
    projection = torch.nn.Linear(2, 1)
    runtime = SimpleNamespace(dino_detector=detector,
        model=SimpleNamespace(vision=SimpleNamespace(builder=SimpleNamespace(feature_proj=projection))))
    calls = []
    def builder(config, root, device, *, initialize_from_baselines):
        assert 'dino_checkpoint' not in config['initialization']
        calls.append(initialize_from_baselines)
        return runtime
    monkeypatch.setattr(training, 'build_runtime', builder)
    return runtime, calls


def config():
    return {'experiment': {'stage': 'B1'}, 'initialization': {'dino_checkpoint': 'unneeded_COCO.pth'}}


def test_warmstart_loads_trained_classifier_and_projection_strictly(tmp_path, monkeypatch):
    runtime, calls = setup_runtime(monkeypatch)
    state = {}
    for prefix, module, value in [('network.vision.dino.detector.', runtime.dino_detector, 3.),
                                  ('network.vision.builder.feature_proj.', runtime.model.vision.builder.feature_proj, 5.)]:
        state.update({prefix + k: torch.full_like(v, value) for k, v in module.state_dict().items()})
    path = tmp_path / 'b4.ckpt'; torch.save({'state_dict': state}, path)
    cfg = config()
    build_image_runtime(cfg, tmp_path, torch.device('cpu'), warmstart=path)
    assert calls == [False]
    assert torch.all(runtime.dino_detector.weight == 3.)
    assert torch.all(runtime.model.vision.builder.feature_proj.weight == 5.)
    assert cfg['initialization']['image_warmstart']['strict']
    assert not cfg['initialization']['image_warmstart']['optimizer_and_epoch_restored']


def test_missing_trained_branch_fails_instead_of_random_training(tmp_path, monkeypatch):
    setup_runtime(monkeypatch)
    path = tmp_path / 'bad.ckpt'; torch.save({'state_dict': {}}, path)
    with pytest.raises(RuntimeError, match='has no'):
        build_image_runtime(config(), tmp_path, torch.device('cpu'), warmstart=path)


def test_new_run_requires_explicit_warmstart(tmp_path):
    with pytest.raises(ValueError, match='warmstart'):
        build_image_runtime(config(), tmp_path, torch.device('cpu'))


def test_resume_leaves_full_state_restore_to_lightning(tmp_path, monkeypatch):
    runtime, calls = setup_runtime(monkeypatch)
    before = runtime.dino_detector.weight.detach().clone()
    build_image_runtime(config(), tmp_path, torch.device('cpu'), initialize_from_baselines=False)
    assert calls == [False]
    torch.testing.assert_close(runtime.dino_detector.weight, before)


def test_frozen_backbone_stays_eval_after_parent_train(tmp_path, monkeypatch):
    runtime, _ = setup_runtime(monkeypatch)
    runtime.dino_detector.backbone = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
    class Swin(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            return self.backbone(x)
    swin = Swin(runtime.dino_detector.backbone)
    runtime.model.vision.dino = SimpleNamespace(swin=swin)
    cfg = config()
    build_image_runtime(cfg, tmp_path, torch.device('cpu'), initialize_from_baselines=False,
                        freeze_backbone=True)
    assert not any(p.requires_grad for p in runtime.dino_detector.backbone.parameters())
    assert runtime.dino_detector.weight.requires_grad
    swin.train()
    x = torch.ones(1, 2)
    torch.testing.assert_close(swin(x), swin(x))
    assert not runtime.dino_detector.backbone.training
    assert cfg['training']['image_freeze_backbone']
