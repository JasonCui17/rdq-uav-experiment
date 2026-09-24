"""PyTorch Lightning orchestration for the frozen Multimodal V1 E5 model.

The scientific model, target construction, and losses remain implemented by
the existing E5 runtime.  This module owns only training infrastructure:
optimizer/scheduler configuration, stage transitions, logging, validation
aggregation, and checkpoint metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset, Sampler

from .training import (
    optimizer_parameter_names,
    stage_for_epoch,
    summarize_validation_outcomes,
    validate_optimizer_parameter_names,
)


def validate_lightning_config(cfg: Mapping[str, Any]) -> None:
    """Fail fast on training settings that would change the frozen protocol."""

    variant = str(cfg.get("experiment", {}).get("variant", "E5")).upper()
    if variant != "E5":
        raise ValueError(
            f"Lightning currently has a verified implementation for E5, got {variant!r}; "
            "register and equivalence-test E0-E4 before selecting them"
        )
    training = cfg["training"]
    batch_size = int(training["batch_size"])
    accumulate = int(training["accumulate"])
    if not 1 <= batch_size <= 8:
        raise ValueError("training.batch_size must be in [1, 8]")
    if accumulate < 1:
        raise ValueError("training.accumulate must be positive")
    stages = training["stages"]
    epochs = int(training["epochs"])
    for epoch in range(1, epochs + 1):
        stage_for_epoch(stages, epoch)
    if str(cfg["validation"]["precision"]).lower() != "fp32":
        raise ValueError("the verified E5 validation contract requires FP32")


def warmup_cosine_factor(step: int, total: int, warmup_fraction: float, final_ratio: float) -> float:
    """The exact factor used by the reference PyTorch E5 loop.

    ``step`` is one-based and denotes the optimizer update about to execute.
    """

    if total <= 0 or step <= 0:
        raise ValueError("step and total must be positive")
    warmup = max(1, round(total * float(warmup_fraction)))
    if step <= warmup:
        return step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return float(final_ratio) + (1.0 - float(final_ratio)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


class GateSampler(Sampler[int]):
    """Deterministically put verified 2D samples first for short real-data gates."""

    def __init__(self, dataset: Dataset, seed: int) -> None:
        import random

        labeled = list(getattr(dataset, "labeled_indices"))
        labeled_set = set(labeled)
        remaining = [index for index in range(len(dataset)) if index not in labeled_set]
        generator = random.Random(int(seed))
        generator.shuffle(labeled)
        generator.shuffle(remaining)
        self.indices = labeled + remaining

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class MultimodalV1DataModule(L.LightningDataModule):
    """Reusable DataLoader boundary around the unchanged E5 query datasets."""

    def __init__(
        self,
        *,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_collate: Callable[[Sequence[Any]], Any],
        val_collate: Callable[[Sequence[Any]], Any],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
        pin_memory: bool,
        gate_order: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_collate = train_collate
        self.val_collate = val_collate
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        self.gate_order = bool(gate_order)
        self.train_generator = torch.Generator().manual_seed(self.seed)
        self.val_generator = torch.Generator().manual_seed(self.seed + 1)

    def setup(self, stage: str | None = None) -> None:
        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            raise RuntimeError("E5 train and validation datasets must be non-empty")

    def _loader_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            # Recreating workers preserves the reference script's epoch-boundary
            # RNG behavior and avoids stale large image buffers under WSL2.
            "persistent_workers": False,
        }
        if self.num_workers > 0:
            options["prefetch_factor"] = self.prefetch_factor
        return options

    def train_dataloader(self) -> DataLoader:
        sampler = GateSampler(self.train_dataset, self.seed) if self.gate_order else None
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=self.train_collate,
            generator=self.train_generator,
            **self._loader_options(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.val_collate,
            generator=self.val_generator,
            **self._loader_options(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "train_generator_state": self.train_generator.get_state(),
            "val_generator_state": self.val_generator.get_state(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.train_generator.set_state(state_dict["train_generator_state"])
        self.val_generator.set_state(state_dict["val_generator_state"])


class MultimodalV1LightningModule(L.LightningModule):
    """Lightning wrapper whose forward/loss path is the reference E5 runtime."""

    def __init__(self, runtime: Any, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.runtime = runtime
        self.network = runtime.model
        self.radar_criterion = runtime.radar_criterion
        self.config = dict(config)
        validate_lightning_config(self.config)
        self.save_hyperparameters({"config": self.config})
        self._stage = ""
        self._validation_xyz: list[float | None] = []
        self._validation_iou: list[float | None] = []
        self._validation_vision_labels = 0
        self._validation_both_missing = 0
        pixel_mean, pixel_std = self._extract_dino_normalizer_tensors(
            runtime.dino_detector.normalizer
        )
        self.register_buffer(
            "_dino_pixel_mean", pixel_mean.detach().clone(), persistent=False
        )
        self.register_buffer(
            "_dino_pixel_std", pixel_std.detach().clone(), persistent=False
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.network(*args, **kwargs)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # The existing prepare_batch() is the single owner of device transfer.
        # Keeping image sizes on CPU also avoids a GPU synchronization per sample.
        return batch

    def _refresh_projection_context(self) -> None:
        from rdq_uav.multimodal_v1 import load_left_projection_context

        self.runtime.projection_base = load_left_projection_context(
            self.runtime.camera_config,
            self.runtime.geometry_calibration,
            image_scale_xy=torch.ones((1, 2), dtype=torch.float32, device=self.device),
            device=self.device,
        )

    @staticmethod
    def _extract_dino_normalizer_tensors(normalizer: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """Read detrex DINO's non-buffer normalization closure once."""

        closure = getattr(normalizer, "__closure__", None)
        freevars = getattr(getattr(normalizer, "__code__", None), "co_freevars", ())
        captured = {
            name: cell.cell_contents for name, cell in zip(freevars, closure or ())
        }
        pixel_mean = captured.get("pixel_mean")
        pixel_std = captured.get("pixel_std")
        if not torch.is_tensor(pixel_mean) or not torch.is_tensor(pixel_std):
            raise RuntimeError(
                "unsupported detrex DINO normalizer: expected pixel_mean/pixel_std closure tensors"
            )
        return pixel_mean, pixel_std

    def _synchronize_external_runtime_device(self) -> None:
        """Synchronize detrex state that ``nn.Module.to`` cannot move.

        DINO stores both ``device`` and its normalization tensors in ordinary
        Python attributes/closure cells. Lightning moves parameters and
        buffers, but these values otherwise remain on the CPU used during
        ``build_runtime``.
        """

        detector = self.runtime.dino_detector
        detector.device = self.device
        pixel_mean = self._dino_pixel_mean
        pixel_std = self._dino_pixel_std
        detector.normalizer = (
            lambda value, mean=pixel_mean, std=pixel_std: (value - mean) / std
        )
        parameter = next(detector.parameters())
        if parameter.device != self.device:
            raise RuntimeError(
                f"DINO parameters are on {parameter.device}, expected {self.device}"
            )
        if pixel_mean.device != self.device or pixel_std.device != self.device:
            raise RuntimeError("DINO normalization tensors did not follow the Lightning device")

    def setup(self, stage: str) -> None:
        # Lightning invokes ``setup`` before ``strategy.setup`` moves module
        # parameters.  External detrex device state must therefore be updated
        # in ``on_fit_start`` rather than here.
        pass

    def _compute(self, batch: Mapping[str, Any], *, return_aux: bool = True):
        # Importing here avoids making the stable scientific implementation
        # depend on Lightning. It is also the explicit old/new equivalence seam.
        from tools.train_multimodal_v1_full import forward_losses

        return forward_losses(
            self.runtime, batch, self.device, self.config, return_aux=return_aux
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        both_missing = int(batch.pop("both_modalities_missing", 0))
        batch_size = len(batch.get("sample_id", ())) or 1
        if bool(batch.pop("skip_training_batch", False)):
            self.log(
                "train/both_modalities_missing", float(both_missing),
                on_step=False, on_epoch=True, reduce_fx="sum", batch_size=batch_size,
            )
            return None
        total, radar, vision, _, labeled, output, _ = self._compute(batch)
        if not bool(torch.isfinite(total.float())):
            raise FloatingPointError(f"non-finite E5 loss at training batch {batch_idx}")
        metrics = {
            "train/loss": total,
            "train/loss_R": radar["loss"],
            "train/loss_V": vision,
            "train/loss_F": output.losses["loss"],
            "train/loss_cls_R": radar["loss_cls"],
            "train/loss_reg_R": radar["loss_reg"],
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        # Keep persistent logger names stable while restoring the concise live
        # status line used by the original training loop. These detached
        # scalars are consumed by Lightning's progress bar without changing
        # the loss graph or optimizer behavior.
        live_lr = max(
            float(group["lr"])
            for group in self.trainer.optimizers[0].param_groups
        )
        progress = {
            "loss": total.detach(),
            "R": radar["loss"].detach(),
            "V": vision.detach(),
            "F": output.losses["loss"].detach(),
            # RichProgressBar rounds ordinary floats to three decimals. Show
            # micro-units so the E5 learning rates do not appear as 0.000.
            "lr_e6": live_lr * 1e6,
        }
        if self.device.type == "cuda":
            progress["mem_GiB"] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        self.log_dict(
            progress,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            batch_size=batch_size,
        )
        self.log("train/vision_supervised", float(labeled), on_step=False, on_epoch=True, reduce_fx="sum", batch_size=batch_size)
        self.log("train/both_modalities_missing", float(both_missing), on_step=False, on_epoch=True, reduce_fx="sum", batch_size=batch_size)
        self.log("train/stage_index", float({"T1": 1, "T2": 2, "T3": 3}[self._stage]), on_step=False, on_epoch=True, batch_size=batch_size)
        return total

    def on_validation_epoch_start(self) -> None:
        self._validation_xyz.clear()
        self._validation_iou.clear()
        self._validation_vision_labels = 0
        self._validation_both_missing = 0

    def on_validation_start(self) -> None:
        # ``Trainer.validate`` does not call ``on_fit_start``. Keep standalone
        # evaluation on the same detrex device/normalizer contract as fitting.
        self._synchronize_external_runtime_device()
        self._refresh_projection_context()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        # Formal validation remains FP32 even when training uses 16-mixed.
        with torch.autocast(device_type=self.device.type, enabled=False):
            total, radar, vision, _, labeled, output, targets = self._compute(batch)
        batch_size = len(batch["sample_id"])
        self.log_dict(
            {
                "val/loss": total.float(),
                "val/loss_R": radar["loss"].float(),
                "val/loss_V": vision.float(),
                "val/loss_F": output.losses["loss"].float(),
            },
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=False,
        )
        self._validation_vision_labels += int(labeled)
        self._validation_both_missing += int(output.aux["num_both_modalities_missing"])
        for index in range(batch_size):
            mask = output.batch_index == index
            top = None
            if bool(mask.any()):
                local = torch.nonzero(mask, as_tuple=False).flatten()
                top = local[torch.argmax(output.fused_score[local].float())]
            if bool(targets.gt_3d_valid[index]):
                self._validation_xyz.append(
                    None if top is None else float(
                        torch.linalg.vector_norm(
                            output.xyz[top].float() - targets.gt_xyz[index].float()
                        )
                    )
                )
            if bool(targets.gt_2d_valid[index]):
                if top is None:
                    self._validation_iou.append(None)
                else:
                    predicted = output.box_xyxy_px[top].float()
                    target = targets.gt_box_xyxy_px[index].float()
                    lt, rb = torch.maximum(predicted[:2], target[:2]), torch.minimum(predicted[2:], target[2:])
                    intersection = torch.prod((rb - lt).clamp_min(0))
                    union = (
                        torch.prod((predicted[2:] - predicted[:2]).clamp_min(0))
                        + torch.prod((target[2:] - target[:2]).clamp_min(0))
                        - intersection
                    )
                    self._validation_iou.append(float(intersection / union) if float(union) > 0 else 0.0)

    def on_validation_epoch_end(self) -> None:
        metrics = summarize_validation_outcomes(self._validation_xyz, self._validation_iou)
        metrics.update(
            vision_supervised=self._validation_vision_labels,
            both_modalities_missing=self._validation_both_missing,
            final_3d_count=len(self._validation_xyz),
        )
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f"val/{name}", float(value), on_step=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        from tools.train_multimodal_v1_full import optimizer_groups

        training = self.config["training"]
        optimizer = torch.optim.AdamW(
            optimizer_groups(self.runtime, self.config),
            weight_decay=float(training["weight_decay"]),
        )
        total = int(self.trainer.estimated_stepping_batches)
        warmup_fraction = float(training["warmup_fraction"])
        final_ratio = float(training["final_lr_ratio"])
        lambdas = []
        for group in optimizer.param_groups:
            lambdas.append(
                lambda scheduler_step, group=group: (
                    warmup_cosine_factor(
                        int(scheduler_step) + 1, total, warmup_fraction, final_ratio
                    )
                    if group.get("stage_active", False)
                    else 0.0
                )
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _activate_current_training_stage(self) -> float:
        """Restore stage-dependent trainability and LR at any fit boundary.

        Lightning does not call ``on_train_epoch_start`` when resuming in the
        middle of an epoch.  ``requires_grad`` is not checkpoint state, so a
        freshly constructed module must re-apply the frozen T1/T2/T3 contract
        before the first resumed batch.
        """

        from tools.train_multimodal_v1_full import activate_stage

        optimizer = self.optimizers(use_pl_optimizer=False)
        epoch = int(self.current_epoch) + 1
        self._stage = activate_stage(self.runtime, optimizer, self.config, epoch)
        total = int(self.trainer.estimated_stepping_batches)
        factor = warmup_cosine_factor(
            int(self.global_step) + 1,
            total,
            float(self.config["training"]["warmup_fraction"]),
            float(self.config["training"]["final_lr_ratio"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor if group.get("stage_active", False) else 0.0
        return factor

    def on_train_epoch_start(self) -> None:
        factor = self._activate_current_training_stage()
        self.log("train/lr_factor", factor, on_step=False, on_epoch=True)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        optimizer_names = None
        if self.trainer is not None and self.trainer.optimizers:
            optimizer_names = optimizer_parameter_names(self.trainer.optimizers[0])
        checkpoint["e5_metadata"] = {
            "effective_config": self.config,
            "initialization": self.runtime.initialization,
            "stage": self._stage,
            "optimizer_param_names": optimizer_names,
        }

    def on_fit_start(self) -> None:
        self._synchronize_external_runtime_device()
        self._refresh_projection_context()
        metadata = getattr(self, "_loaded_e5_metadata", None)
        if metadata:
            saved_config = metadata.get("effective_config")
            if saved_config is not None:
                critical = ("data", "initialization", "loss", "training", "validation")
                drift = [key for key in critical if saved_config.get(key) != self.config.get(key)]
                if drift:
                    raise ValueError(
                        f"native resume configuration drift in {drift}; use a new run or "
                        "load the old checkpoint as weight-only initialization"
                    )
            if metadata.get("optimizer_param_names") is not None:
                validate_optimizer_parameter_names(
                    self.trainer.optimizers[0], metadata["optimizer_param_names"]
                )
        # Also runs for fresh fits. It is essential for an intra-epoch resume,
        # where Lightning deliberately skips ``on_train_epoch_start``.
        self._activate_current_training_stage()

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self._loaded_e5_metadata = checkpoint.get("e5_metadata")
        if self._loaded_e5_metadata:
            saved_stage = self._loaded_e5_metadata.get("stage")
            if saved_stage in {"T1", "T2", "T3"}:
                self._stage = str(saved_stage)
            elif saved_stage not in {None, ""}:
                raise ValueError(f"checkpoint has invalid E5 stage {saved_stage!r}")

    def load_legacy_weights(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Strictly load network weights from the original pure-PyTorch format."""

        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, Mapping) or "model_state" not in payload:
            raise ValueError("legacy E5 checkpoint must contain model_state")
        self.network.load_state_dict(payload["model_state"], strict=True)
        return dict(payload)


class LexicographicBestCheckpoint(Callback):
    """Preserve the existing success@1m, then median-error comparator."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = str(path)
        self.best: tuple[float, float] | None = None
        self.best_epoch: int | None = None

    @property
    def state_key(self) -> str:
        return f"{self.__class__.__qualname__}:{self.path}"

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        value = state_dict.get("best")
        self.best = None if value is None else tuple(map(float, value))
        self.best_epoch = state_dict.get("best_epoch")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        success = metrics.get("val/final_3d_success_1m")
        median = metrics.get("val/final_3d_median_error")
        if success is None or median is None:
            return
        median_value = float(median)
        key = (float(success), -median_value if np.isfinite(median_value) else -math.inf)
        if self.best is None or key > self.best:
            self.best = key
            self.best_epoch = int(trainer.current_epoch) + 1
            trainer.save_checkpoint(self.path)


class PeakMemoryMonitor(Callback):
    """Low-overhead epoch peak memory logging for the 8 GB deployment target."""

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if pl_module.device.type == "cuda":
            gib = 1024 ** 3
            pl_module.log("system/peak_allocated_gib", torch.cuda.max_memory_allocated(pl_module.device) / gib)
            pl_module.log("system/peak_reserved_gib", torch.cuda.max_memory_reserved(pl_module.device) / gib)


class StopAfterEpoch(Callback):
    """Test-only clean stop while retaining the original total-step schedule."""

    def __init__(self, epoch: int) -> None:
        super().__init__()
        if epoch < 1:
            raise ValueError("stop-after epoch must be positive")
        self.epoch = int(epoch)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if int(trainer.current_epoch) + 1 >= self.epoch:
            trainer.should_stop = True
