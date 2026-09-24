from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import TensorDataset

from rdq_uav.multimodal_v1.lightning_system import (
    LexicographicBestCheckpoint,
    MultimodalV1DataModule,
    MultimodalV1LightningModule,
    validate_lightning_config,
    warmup_cosine_factor,
)
from tools.train_multimodal_v1_full import schedule_lr
from tools.train_multimodal_v1_lightning import build_precision_plugin


def config() -> dict:
    return {
        "experiment": {"variant": "E5", "seed": 42},
        "training": {
            "epochs": 3, "batch_size": 1, "accumulate": 1,
            "warmup_fraction": .2, "final_lr_ratio": .01,
            "weight_decay": 1e-4, "grad_clip_norm": 1.0,
            "learning_rates": {
                "new_modules": 1e-3, "radar": 2e-4,
                "dino_head": 1e-4, "swin_last_two": 1e-5,
            },
            "stages": [
                {"name": "T1", "start_epoch": 1, "end_epoch": 1},
                {"name": "T2", "start_epoch": 2, "end_epoch": 2},
                {"name": "T3", "start_epoch": 3, "end_epoch": 3},
            ],
        },
        "validation": {"precision": "fp32"},
        "loss": {"lambda_R": 1., "lambda_V": 1., "lambda_F": 1.},
    }


def test_config_contract_rejects_unverified_variant_and_large_batch() -> None:
    cfg = config()
    validate_lightning_config(cfg)
    cfg["experiment"]["variant"] = "E4"
    with pytest.raises(ValueError, match="verified implementation for E5"):
        validate_lightning_config(cfg)
    cfg = config(); cfg["training"]["batch_size"] = 9
    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        validate_lightning_config(cfg)


def test_precision_plugin_only_overrides_cuda_fp16_scaler() -> None:
    assert build_precision_plugin("32-true", "gpu", {}) == "32-true"
    assert build_precision_plugin("16-mixed", "cpu", {}) == "16-mixed"
    with pytest.raises(ValueError, match="amp_initial_scale must be positive"):
        build_precision_plugin("16-mixed", "gpu", {"amp_initial_scale": 0})


def test_lightning_entrypoint_direct_execution_can_import_reference_tool() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/train_multimodal_v1_lightning.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--fast-dev-run" in result.stdout


def test_warmup_cosine_is_reference_schedule_equivalent() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "base_lr": 2e-4, "lr": 2e-4, "stage_active": True}])
    reference = {"training": {"warmup_fraction": .2, "final_lr_ratio": .01}}
    for step in range(1, 101):
        expected = schedule_lr(optimizer, step, 100, reference)
        actual = warmup_cosine_factor(step, 100, .2, .01)
        assert actual == pytest.approx(expected, abs=1e-15)


def test_datamodule_loader_and_rng_state_roundtrip() -> None:
    dataset = TensorDataset(torch.arange(9))
    collate = lambda samples: torch.stack([sample[0] for sample in samples])
    dm = MultimodalV1DataModule(
        train_dataset=dataset, val_dataset=dataset, train_collate=collate,
        val_collate=collate, batch_size=2, num_workers=0, prefetch_factor=2,
        seed=42, pin_memory=False,
    )
    state = dm.state_dict()
    expected = torch.cat(list(dm.train_dataloader()))
    dm.load_state_dict(state)
    actual = torch.cat(list(dm.train_dataloader()))
    assert torch.equal(expected, actual)


class ToyLightningModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        x = batch.float().reshape(-1, 1)
        loss = self.layer(x).square().mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.float().reshape(-1, 1)
        self.log("val/loss", self.layer(x).square().mean())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


def _toy_datamodule() -> MultimodalV1DataModule:
    dataset = TensorDataset(torch.arange(4, dtype=torch.float32))
    collate = lambda samples: torch.stack([sample[0] for sample in samples])
    return MultimodalV1DataModule(
        train_dataset=dataset, val_dataset=dataset, train_collate=collate,
        val_collate=collate, batch_size=2, num_workers=0, prefetch_factor=2,
        seed=42, pin_memory=False,
    )


def test_lightning_fast_dev_run() -> None:
    trainer = L.Trainer(
        accelerator="cpu", devices=1, fast_dev_run=True, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(ToyLightningModule(), datamodule=_toy_datamodule())
    assert trainer.global_step == 1


def test_lightning_native_checkpoint_resume(tmp_path) -> None:
    dm = _toy_datamodule()
    first = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        default_root_dir=tmp_path, enable_progress_bar=False,
    )
    first.fit(ToyLightningModule(), datamodule=dm)
    first_steps = first.global_step
    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)
    second = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=2, logger=False,
        default_root_dir=tmp_path, enable_progress_bar=False,
    )
    second.fit(ToyLightningModule(), datamodule=_toy_datamodule(), ckpt_path=checkpoint)
    assert second.global_step == first_steps * 2


class ToyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Linear(1, 1)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(4)])
        self.norm2 = torch.nn.LayerNorm(1)
        self.norm3 = torch.nn.LayerNorm(1)


class ToyDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ToyBackbone()
        self.head = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        pixel_mean = torch.zeros((1, 1))
        pixel_std = torch.ones((1, 1))
        self.normalizer = lambda value: (value - pixel_mean) / pixel_std


class ToyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lidar = torch.nn.Linear(1, 1)
        self.dino = ToyDino()
        self.new = torch.nn.Linear(1, 1)


class ExercisedE5Module(MultimodalV1LightningModule):
    def _refresh_projection_context(self) -> None:
        pass

    def _compute(self, batch, *, return_aux=True):
        value = sum(parameter.sum() for parameter in self.network.parameters())
        base = value.square() * .001
        radar = {"loss": base, "loss_cls": base * .4, "loss_reg": base * .3}
        output = SimpleNamespace(
            losses={"loss": base * .2},
            aux={"num_both_modalities_missing": 0},
            batch_index=torch.tensor([0], device=self.device),
            fused_score=torch.tensor([1.], device=self.device),
            xyz=torch.zeros((1, 3), device=self.device),
            box_xyxy_px=torch.tensor([[0., 0., 1., 1.]], device=self.device),
        )
        targets = SimpleNamespace(
            gt_3d_valid=torch.tensor([True], device=self.device),
            gt_xyz=torch.zeros((1, 3), device=self.device),
            gt_2d_valid=torch.tensor([True], device=self.device),
            gt_box_xyxy_px=torch.tensor([[0., 0., 1., 1.]], device=self.device),
        )
        return base * 1.3, radar, base * .1, {}, 1, output, targets


def _exercised_module() -> ExercisedE5Module:
    network = ToyNetwork()
    runtime = SimpleNamespace(
        model=network,
        radar_criterion=torch.nn.Identity(),
        lidar_detector=network.lidar,
        dino_detector=network.dino,
        new_modules=(network.new,),
        initialization={"kind": "toy"},
    )
    return ExercisedE5Module(runtime, config())


def _e5_datamodule() -> MultimodalV1DataModule:
    dataset = [{"sample_id": ["toy"]}]

    def collate(_samples):
        return {
            "sample_id": ["toy"], "skip_training_batch": False,
            "both_modalities_missing": 0,
        }

    return MultimodalV1DataModule(
        train_dataset=dataset, val_dataset=dataset,
        train_collate=collate, val_collate=collate,
        batch_size=1, num_workers=0, prefetch_factor=2, seed=42,
        pin_memory=False,
    )


def test_actual_e5_lightning_lifecycle_fast_dev_run() -> None:
    trainer = L.Trainer(
        accelerator="cpu", devices=1, fast_dev_run=True, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    module = _exercised_module()
    trainer.fit(module, datamodule=_e5_datamodule())
    assert trainer.global_step == 1
    assert module._stage == "T1"
    assert trainer.callback_metrics["val/final_3d_success_1m"] == 1


def test_actual_e5_resume_restores_stage_contract(tmp_path) -> None:
    dataset = [{"sample_id": [f"toy-{index}"]} for index in range(4)]

    def collate(_samples):
        return {
            "sample_id": ["toy"], "skip_training_batch": False,
            "both_modalities_missing": 0,
        }

    def datamodule():
        return MultimodalV1DataModule(
            train_dataset=dataset, val_dataset=dataset,
            train_collate=collate, val_collate=collate,
            batch_size=1, num_workers=0, prefetch_factor=2, seed=42,
            pin_memory=False,
        )

    first = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=3, max_steps=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, limit_val_batches=0,
    )
    first.fit(_exercised_module(), datamodule=datamodule())
    checkpoint = tmp_path / "mid_epoch.ckpt"
    first.save_checkpoint(checkpoint)

    resumed_module = _exercised_module()
    resumed = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=3, max_steps=2,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, limit_val_batches=0,
    )
    resumed.fit(resumed_module, datamodule=datamodule(), ckpt_path=checkpoint)
    assert resumed.global_step == 2
    assert resumed_module._stage == "T2"
    assert all(
        parameter.requires_grad
        for parameter in resumed_module.runtime.lidar_detector.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in resumed_module.runtime.dino_detector.head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in resumed_module.runtime.dino_detector.backbone.parameters()
    )


def test_checkpoint_load_restores_saved_stage_before_first_resumed_batch() -> None:
    module = _exercised_module()
    assert module._stage == ""
    module.on_load_checkpoint({"e5_metadata": {"stage": "T1"}})
    assert module._stage == "T1"


def test_checkpoint_load_accepts_legacy_empty_stage_metadata() -> None:
    module = _exercised_module()
    module.on_load_checkpoint({"e5_metadata": {"stage": ""}})
    assert module._stage == ""


def test_missing_modalities_and_normal_batches_share_logging_contract() -> None:
    dataset = [{"skip": False}, {"skip": True}]

    def collate(samples):
        skip = bool(samples[0]["skip"])
        return {
            "sample_id": ["toy"], "skip_training_batch": skip,
            "both_modalities_missing": int(skip),
        }

    datamodule = MultimodalV1DataModule(
        train_dataset=dataset, val_dataset=dataset,
        train_collate=collate, val_collate=collate,
        batch_size=1, num_workers=0, prefetch_factor=2, seed=42,
        pin_memory=False,
    )
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, limit_val_batches=0,
    )
    trainer.fit(_exercised_module(), datamodule=datamodule)
    assert trainer.callback_metrics["train/both_modalities_missing"] == 1


def test_actual_e5_stage_switch_reaches_t3() -> None:
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_epochs=3, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    module = _exercised_module()
    trainer.fit(module, datamodule=_e5_datamodule())
    assert module._stage == "T3"
    assert all(parameter.requires_grad for parameter in module.runtime.lidar_detector.parameters())
    assert all(
        parameter.requires_grad
        for index in (2, 3)
        for parameter in module.runtime.dino_detector.backbone.layers[index].parameters()
    )
    assert all(
        not parameter.requires_grad
        for index in (0, 1)
        for parameter in module.runtime.dino_detector.backbone.layers[index].parameters()
    )


def test_legacy_weight_loading_is_strict(tmp_path) -> None:
    module = _exercised_module()
    expected = {name: tensor.clone() for name, tensor in module.network.state_dict().items()}
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": expected, "epoch": 3}, path)
    with torch.no_grad():
        for parameter in module.network.parameters():
            parameter.zero_()
    payload = module.load_legacy_weights(path)
    assert payload["epoch"] == 3
    for name, tensor in module.network.state_dict().items():
        assert torch.equal(tensor, expected[name])


def test_external_dino_device_and_normalizer_are_synchronized() -> None:
    module = _exercised_module()
    old_normalizer = module.runtime.dino_detector.normalizer
    module._synchronize_external_runtime_device()
    detector = module.runtime.dino_detector
    assert detector.device == module.device
    assert detector.normalizer is not old_normalizer
    value = torch.ones((1, 1), device=module.device)
    assert detector.normalizer(value).device == module.device


def test_setup_defers_external_device_sync_until_strategy_moves_parameters() -> None:
    module = _exercised_module()
    old_normalizer = module.runtime.dino_detector.normalizer
    module.setup("fit")
    assert module.runtime.dino_detector.normalizer is old_normalizer


def test_lexicographic_best_checkpoint_uses_existing_policy(tmp_path) -> None:
    callback = LexicographicBestCheckpoint(tmp_path / "best.ckpt")
    saved = []
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        callback_metrics={
            "val/final_3d_success_1m": torch.tensor(.5),
            "val/final_3d_median_error": torch.tensor(2.),
        },
        save_checkpoint=lambda path: saved.append(path),
    )
    callback.on_validation_end(trainer, None)
    trainer.current_epoch = 1
    trainer.callback_metrics["val/final_3d_median_error"] = torch.tensor(1.)
    callback.on_validation_end(trainer, None)
    trainer.current_epoch = 2
    trainer.callback_metrics["val/final_3d_success_1m"] = torch.tensor(.4)
    callback.on_validation_end(trainer, None)
    assert len(saved) == 2
    assert callback.best_epoch == 2
