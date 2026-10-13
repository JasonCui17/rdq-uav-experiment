"""Regression checks for scaled overflow, accumulation and B4 precision boundaries."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rdq_uav.multimodal_v2.candidate_builders import RGBCandidateBuilder
from rdq_uav.multimodal_v2.pair_fusion import CandidatePairFusion
from rdq_uav.multimodal_v2.pair_loss import JointPairLoss
from rdq_uav.multimodal_v2.training import MultimodalV2LightningModule
from test_v2_pair_fusion import candidate, targets


def hooked_module(scaler=None):
    network = nn.Linear(1, 1, bias=False)
    network.weight.data.fill_(1.)
    module = MultimodalV2LightningModule(
        SimpleNamespace(model=network, initialization={"stage": "B4"}), {})
    module._trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))
    logged = {}
    module.log = lambda name, value, **kwargs: logged.update({name: float(value)})
    return module, logged


def cpu_scaler():
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        pytest.skip("CPU GradScaler requires recent PyTorch; CUDA training uses its native scaler")
    return torch.amp.GradScaler("cpu", init_scale=1e38)


def test_scaled_overflow_skips_update_lowers_scale_and_recovers():
    scaler = cpu_scaler()
    module, logged = hooked_module(scaler)
    optimizer = torch.optim.SGD(module.network.parameters(), lr=.1)
    before = module.network.weight.detach().clone()
    # Two backward calls exercise the accumulation boundary: never unscale
    # partial gradients or interrupt GradScaler inside on_after_backward.
    for _ in range(2):
        loss = module.network(torch.tensor([[10.]])).square().sum() / 2
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        module.on_after_backward()
    assert not torch.isfinite(module.network.weight.grad).all()
    scaler.unscale_(optimizer)
    module.on_before_optimizer_step(optimizer)
    assert logged["train/amp_nonfinite_step"] == 1
    scaler.step(optimizer)
    scaler.update()
    assert torch.equal(before, module.network.weight)
    assert scaler.get_scale() < 1e38

    optimizer.zero_grad()
    loss = module.network(torch.tensor([[.1]])).square().sum()
    scaler.scale(loss).backward()
    module.on_after_backward()
    scaler.unscale_(optimizer)
    module.on_before_optimizer_step(optimizer)
    assert logged["train/amp_nonfinite_step"] == 0
    assert logged["train/has_nonzero_gradient"] == 1
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(module.network.weight).all()
    assert not torch.equal(before, module.network.weight)


def test_unscaled_nonfinite_gradient_still_fails_with_parameter_name():
    module, _ = hooked_module()
    module.network.weight.grad = torch.full_like(module.network.weight, float("nan"))
    with pytest.raises(FloatingPointError, match="unscaled.*weight"):
        module.on_after_backward()


def test_lightning_amp_skips_overflow_and_resumes_checkpoint(tmp_path):
    import lightning.pytorch as L
    from lightning.pytorch.plugins.precision.amp import MixedPrecision
    from torch.utils.data import DataLoader

    class ToyModule(MultimodalV2LightningModule):
        def __init__(self):
            network = nn.Linear(1, 1, bias=False)
            network.weight.data.fill_(1.)
            super().__init__(SimpleNamespace(model=network, initialization={"stage": "B4"}), {})
        def on_fit_start(self): pass
        def on_train_epoch_start(self): pass
        def training_step(self, batch, batch_idx):
            with torch.autocast("cpu", enabled=False):
                return self.network(batch).square().sum()
        def configure_optimizers(self):
            return torch.optim.SGD(self.network.parameters(), lr=.1)

    scaler = cpu_scaler()
    module = ToyModule()
    trainer = L.Trainer(accelerator="cpu", max_steps=2, logger=False,
                        plugins=[MixedPrecision("16-mixed", "cpu", scaler)],
                        gradient_clip_val=1., enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, train_dataloaders=DataLoader(
        [torch.tensor([[10.]]), torch.tensor([[.1]])], batch_size=None))
    assert trainer.global_step == 2
    assert float(trainer.callback_metrics["train/amp_nonfinite_step_epoch"]) == .5
    assert .99 < float(module.network.weight) < 1.
    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    recovered = ToyModule()
    restored_scaler = cpu_scaler()
    resumed = L.Trainer(accelerator="cpu", max_steps=3, logger=False,
                        plugins=[MixedPrecision("16-mixed", "cpu", restored_scaler)],
                        gradient_clip_val=1., enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    resumed.fit(recovered, train_dataloaders=DataLoader(
        [torch.tensor([[.1]])], batch_size=None), ckpt_path=checkpoint)
    assert resumed.global_step == 3
    assert restored_scaler.get_scale() == scaler.get_scale()
    assert torch.isfinite(recovered.network.weight).all()
    assert recovered.network.weight < module.network.weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_b4_trainable_path_stays_fp32_under_autocast(dtype):
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if dtype == torch.float16 and version < (2, 3):
        pytest.skip("Older PyTorch CPU autocast supports BF16 only; CUDA FP16 is unaffected")
    builder = RGBCandidateBuilder(pre_topk=2, final_topk=2, apply_nms=False)
    builder.projection_fp32 = True
    fusion = CandidatePairFusion()
    r = replace(candidate("R"), feature=torch.randn(2, 128) * 1e6)
    dino = dict(pred_logits=torch.ones(1, 2, 1),
                pred_boxes=torch.tensor([[[.5, .5, .2, .2]] * 2]),
                decoder_query_features=torch.randn(1, 2, 256) * 1e6)
    with torch.autocast("cpu", dtype=dtype):
        visual = builder(dino, torch.tensor([[100., 100.]]))
        output = fusion(r, visual, num_samples=1)
        values = JointPairLoss()(output, targets())
    assert visual.feature.dtype == torch.float32
    assert output.joint_logit.dtype == torch.float32
    assert output.diagnostics["pair_alignment"][0]["similarity"].dtype == torch.float32
    assert torch.isfinite(values["loss"])
    values["loss"].backward()
    for model in (builder, fusion):
        grads = [p.grad for p in model.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
        assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_projection_precision_switch_preserves_checkpoint_keys():
    original, repaired = RGBCandidateBuilder(), RGBCandidateBuilder()
    repaired.projection_fp32 = True
    repaired.load_state_dict(original.state_dict(), strict=True)
    assert original.state_dict().keys() == repaired.state_dict().keys()
