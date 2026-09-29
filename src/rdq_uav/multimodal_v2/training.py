"""Runtime construction, freezing, optimization, and validation for V2."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import yaml

try:
    import lightning as L
except ImportError:  # Core model remains importable without training extras.
    L = None
from torch.utils.data import DataLoader

from rdq_uav.lidar_v2.loss import CandidateLoss
from rdq_uav.lidar_v2.model import LiDARUAVDetector
from rdq_uav.lidar_v2.selector import CandidateSelector
from rdq_uav.multimodal_v1 import load_left_projection_context
from rdq_uav.multimodal_v1.training import supervised_dino_loss
from rdq_uav.multimodal_v1.vision.uav_dino import adapt_dino_class_head_to_single_uav
from rdq_uav.runtime_paths import ensure_detrex_config_link

from .data import prepare_model_batch
from .interaction import CandidateCrossAttention
from .lidar import LiDARCandidateModel, load_lidar_weights
from .loss import CandidateRankingLoss
from .model import MultimodalV2
from .scoring import CandidateScoring
from .vision import VisionCandidateModel


@dataclass
class V2Runtime:
    model: MultimodalV2
    lidar_detector: nn.Module
    dino_detector: nn.Module
    ranking_loss: CandidateRankingLoss
    lidar_loss: CandidateLoss
    projection_base: Any
    camera_config: Path
    geometry_calibration: Path
    initialization: dict[str, Any]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_e5_dino(vision: VisionCandidateModel, checkpoint: Path) -> None:
    vision.load_e5_weights(str(checkpoint))


def build_runtime(config: Mapping[str, Any], root: Path, device: torch.device) -> V2Runtime:
    init, model_cfg = config["initialization"], config["model"]
    detrex = _resolve(root, init["dino_root"])
    ensure_detrex_config_link(detrex)
    for path in (root / "src", detrex / "detectron2", detrex):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate

    dino_cfg = LazyConfig.load(str(_resolve(root, init["dino_config"])))
    dino_cfg.model.device = str(device)
    dino_detector = instantiate(dino_cfg.model).to(device)
    DetectionCheckpointer(dino_detector).load(str(_resolve(root, init["dino_checkpoint"])))
    adapt_dino_class_head_to_single_uav(dino_detector)
    candidate_cfg = yaml.safe_load(_resolve(root, init["p6_config"]).read_text())
    vision = VisionCandidateModel(
        dino_detector,
        pre_topk=int(candidate_cfg["candidate"]["rgb"]["pre_topk"]),
        final_topk=int(candidate_cfg["candidate"]["rgb"]["final_topk"]),
        nms_iou=float(candidate_cfg["candidate"]["rgb"]["nms_iou"]),
    )
    e5_visual = init.get("e5_visual_checkpoint")
    if e5_visual:
        _load_e5_dino(vision, _resolve(root, e5_visual))

    lidar_cfg = yaml.safe_load(_resolve(root, init["lidar_config"]).read_text())
    lidar_detector = LiDARUAVDetector(lidar_cfg).to(device)
    lidar_checkpoint = _resolve(root, init["lidar_checkpoint"])
    load_lidar_weights(lidar_detector, str(lidar_checkpoint))
    lidar = LiDARCandidateModel(lidar_detector, CandidateSelector(candidate_cfg["candidate"]["radar"]))

    interaction = CandidateCrossAttention(
        visual_radius=int(model_cfg["visual_radius_cells"]),
        lidar_neighbors=int(model_cfg["lidar_neighbors"]),
        box_margin_px=float(model_cfg["box_margin_px"]),
    )
    scoring = CandidateScoring(
        geometry_gate_px=float(candidate_cfg["association"]["geometry_gate_px"]),
        max_abs_delta_logit=float(model_cfg["max_abs_delta_logit"]),
    )
    model = MultimodalV2(
        lidar, vision, interaction, scoring,
        interaction_enabled=bool(model_cfg["interaction_enabled"]),
        vision_reads_radar=bool(model_cfg["vision_reads_radar"]),
        vision_scoring_enabled=bool(model_cfg["vision_scoring_enabled"]),
    ).to(device)
    loss_cfg = config["loss"]
    ranking = CandidateRankingLoss(
        focal_alpha=float(loss_cfg["focal_alpha"]), focal_gamma=float(loss_cfg["focal_gamma"]),
        xyz_positive_m=float(loss_cfg["xyz_positive_m"]), xyz_ignore_m=float(loss_cfg["xyz_ignore_m"]),
        box_positive_iou=float(loss_cfg["box_positive_iou"]), box_ignore_iou=float(loss_cfg["box_ignore_iou"]),
        lambda_3d=float(loss_cfg["lambda_rank_3d"]), lambda_2d=float(loss_cfg["lambda_rank_2d"]),
        negative_only_weight=float(loss_cfg.get("negative_only_weight", 0.25)),
    )
    data_cfg = config["data"]
    camera_path = _resolve(root, data_cfg["camera_config"])
    geometry_path = _resolve(root, data_cfg["geometry_calibration"])
    projection = load_left_projection_context(
        camera_path, geometry_path,
        image_scale_xy=torch.ones((1, 2), device=device), device=device,
    )
    runtime = V2Runtime(
        model, lidar_detector, dino_detector, ranking, CandidateLoss(lidar_cfg),
        projection, camera_path, geometry_path,
        {"lidar_checkpoint": str(lidar_checkpoint), "dino_checkpoint": str(_resolve(root, init["dino_checkpoint"])),
         "e5_visual_checkpoint": None if not e5_visual else str(_resolve(root, e5_visual))},
    )
    freeze_for_stage(runtime, str(config["experiment"]["stage"]))
    return runtime


def freeze_for_stage(runtime: V2Runtime, stage: str) -> None:
    if stage not in {"B0", "B1", "B2", "B3"}:
        raise ValueError(f"unknown V2 stage {stage}")
    runtime.model.requires_grad_(False)
    if stage in {"B2", "B3"}:
        runtime.model.interaction.radar_reads_vision.requires_grad_(True)
        runtime.model.interaction.visual_proj.requires_grad_(True)
        runtime.model.scoring.score_head.radar.requires_grad_(True)
    if stage == "B3":
        runtime.model.interaction.vision_reads_radar.requires_grad_(True)
        runtime.model.interaction.depth_embed.requires_grad_(True)
        runtime.model.scoring.score_head.vision.requires_grad_(True)


def trainable_parameter_groups(runtime: V2Runtime, learning_rate: float,
                               weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay, names = [], [], []
    for name, parameter in runtime.model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    if not names:
        raise RuntimeError("selected stage has no trainable parameters")
    return [
        {"params": decay, "lr": learning_rate, "weight_decay": weight_decay, "name": "v2_decay"},
        {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0, "name": "v2_no_decay"},
    ]


def synchronize_dino_device(runtime: V2Runtime, device: torch.device) -> None:
    detector = runtime.dino_detector
    detector.device = device
    # Detrex captures its normalizer tensors in a Python closure, outside the
    # nn.Module buffer tree. Rebuild it on the actual trainer device.
    closure = getattr(detector.normalizer, "__closure__", None)
    freevars = getattr(getattr(detector.normalizer, "__code__", None), "co_freevars", ())
    captured = {name: cell.cell_contents for name, cell in zip(freevars, closure or ())}
    mean, std = captured.get("pixel_mean"), captured.get("pixel_std")
    if torch.is_tensor(mean) and torch.is_tensor(std):
        mean, std = mean.to(device), std.to(device)
        detector.normalizer = lambda value, mean=mean, std=std: (value - mean) / std


def forward_step(runtime: V2Runtime, batch: Mapping[str, Any], device: torch.device,
                 *, compute_frozen_losses: bool = True) -> dict[str, Any]:
    lidar_batch, images, masks, context, targets, transforms = prepare_model_batch(
        batch, runtime.dino_detector, runtime.projection_base, device,
    )
    output = runtime.model(lidar_batch, images, masks, context)
    rank = runtime.ranking_loss(output, targets)
    result = {"output": output, "targets": targets, **rank}
    if compute_frozen_losses:
        lidar_raw = output.diagnostics["lidar_raw"]
        result["loss_lidar_frozen"] = runtime.lidar_loss(lidar_raw, lidar_batch)["loss"]
        vision, _, count = supervised_dino_loss(
            runtime.dino_detector, output.diagnostics["vision_raw"],
            gt_box_xyxy_source=targets.box_xyxy_px,
            gt_2d_valid=targets.has_box, transforms=transforms,
        )
        result["loss_vision_frozen"] = vision
        result["vision_supervised"] = count
    return result


def summarize_3d(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    top = np.asarray([row["top1_error"] if row["top1_error"] is not None else np.inf for row in rows])
    finite = top[np.isfinite(top)]
    result: dict[str, float | int] = {"samples": count, "outputs": int(len(finite)),
                                     "coverage": float(len(finite) / count) if count else 0.0}
    for radius in (0.5, 1.0, 2.0):
        result[f"success_{radius:g}m"] = float(np.mean(top <= radius)) if count else 0.0
    for k in (1, 5, 10, 20):
        for radius in (0.5, 1.0, 2.0):
            result[f"recall_at_{k}_{radius:g}m"] = float(np.mean([
                bool(row["distances"][:k]) and min(row["distances"][:k]) <= radius for row in rows
            ])) if count else 0.0
    for name, function in (("mean", np.mean), ("median", np.median)):
        result[f"error_{name}_m"] = float(function(finite)) if len(finite) else float("inf")
    result["correct_candidate_top1_rate"] = result["success_1m"]
    return result


@torch.no_grad()
def validation_rows(output: Any, targets: Any, sample_ids: list[str],
                    sequence_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for batch in range(len(targets.has_xyz)):
        if not bool(targets.has_xyz[batch]):
            continue
        ids = output.top3d_indices(len(targets.has_xyz))[batch]
        if len(ids):
            distances = torch.linalg.vector_norm(output.xyz_m[ids].float() - targets.xyz_m[batch].float(), dim=1).cpu().tolist()
            top = float(distances[0])
        else:
            distances, top = [], None
        rows.append({"sample_id": sample_ids[batch], "sequence_id": sequence_ids[batch],
                     "top1_error": top, "distances": distances,
                     "has_output": bool(ids.numel()),
                     "hypothesis_types": output.hypothesis_type[ids].cpu().tolist() if len(ids) else []})
    return rows


if L is not None:
    class MultimodalV2DataModule(L.LightningDataModule):
        def __init__(self, train_dataset, val_dataset, *, batch_size: int,
                     num_workers: int, prefetch_factor: int, seed: int) -> None:
            super().__init__()
            self.train_dataset, self.val_dataset = train_dataset, val_dataset
            self.batch_size, self.num_workers = int(batch_size), int(num_workers)
            self.prefetch_factor, self.seed = int(prefetch_factor), int(seed)
            self.train_generator = torch.Generator().manual_seed(self.seed)
            self.val_generator = torch.Generator().manual_seed(self.seed + 1)

        def _options(self):
            options = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                           pin_memory=torch.cuda.is_available(), persistent_workers=False)
            if self.num_workers:
                options["prefetch_factor"] = self.prefetch_factor
            return options

        def train_dataloader(self):
            from .data import collate_multimodal_v2
            return DataLoader(self.train_dataset, shuffle=True, generator=self.train_generator,
                              collate_fn=collate_multimodal_v2, **self._options())

        def val_dataloader(self):
            from .data import collate_multimodal_v2
            return DataLoader(self.val_dataset, shuffle=False, generator=self.val_generator,
                              collate_fn=collate_multimodal_v2, **self._options())

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {"train_generator": self.train_generator.get_state(),
                    "val_generator": self.val_generator.get_state()}

        def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
            self.train_generator.set_state(state_dict["train_generator"])
            self.val_generator.set_state(state_dict["val_generator"])


    class MultimodalV2LightningModule(L.LightningModule):
        def __init__(self, runtime: V2Runtime, config: Mapping[str, Any]) -> None:
            super().__init__()
            self.runtime = runtime
            self.network = runtime.model
            self.config = dict(config)
            self.save_hyperparameters({"config": self.config})
            self._val_rows: list[dict[str, Any]] = []

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch

        def _refresh_external_device(self) -> None:
            synchronize_dino_device(self.runtime, self.device)
            self.runtime.projection_base = load_left_projection_context(
                self.runtime.camera_config, self.runtime.geometry_calibration,
                image_scale_xy=torch.ones((1, 2), device=self.device), device=self.device,
            )

        def on_fit_start(self) -> None:
            self._refresh_external_device()

        def on_train_epoch_start(self) -> None:
            # B2/B3 keep both pretrained candidate generators frozen and in
            # deterministic eval mode; only candidate interaction/scoring train.
            self.runtime.lidar_detector.eval()
            self.runtime.dino_detector.eval()

        def on_validation_start(self) -> None:
            self._refresh_external_device()

        def training_step(self, batch, batch_idx):
            if not bool(batch["m_R"].any() | batch["m_V"].any()):
                self.log("train/both_modalities_missing", float(len(batch["m_R"])),
                         on_step=False, on_epoch=True, reduce_fx="sum")
            values = forward_step(self.runtime, batch, self.device, compute_frozen_losses=True)
            loss = values["loss"]
            batch_size = len(batch["m_R"])
            statistic_names = (
                "n_gt3d", "n_with_3d_candidate", "n_with_positive_3d",
                "n_negative_only_3d", "n_no_3d_candidate", "n_3d_loss_queries",
                "n_gt2d", "n_with_2d_candidate", "n_with_positive_2d",
                "n_negative_only_2d", "n_no_2d_candidate", "n_2d_loss_queries",
            )
            self.log_dict(
                {f"train/{name}": float(values[name]) for name in statistic_names},
                on_step=False, on_epoch=True, batch_size=batch_size, reduce_fx="sum",
            )
            if not values["has_trainable_loss"]:
                # A batch can legitimately contain only missing GT or no
                # task-valid candidates. Lightning accepts None as an explicit
                # skipped optimization batch; it must not abort the epoch.
                self.log("train/skipped_no_loss_batch", 1.0, on_step=False,
                         on_epoch=True, batch_size=batch_size, reduce_fx="sum")
                return None
            if not loss.requires_grad:
                raise RuntimeError("reported trainable V2 loss has no gradient path")
            if not bool(torch.isfinite(loss.float())):
                raise FloatingPointError(f"non-finite V2 loss at batch {batch_idx}")
            self.log_dict({
                "train/loss": loss, "train/rank_3d": values["loss_rank_3d"],
                "train/rank_2d": values["loss_rank_2d"],
                "train/lidar_frozen": values["loss_lidar_frozen"],
                "train/vision_frozen": values["loss_vision_frozen"],
            }, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss", loss.detach(), prog_bar=True, logger=False)
            self.log("rank3d", values["loss_rank_3d"].detach(), prog_bar=True, logger=False)
            if self.device.type == "cuda":
                self.log("mem_GiB", torch.cuda.max_memory_reserved() / 1024 ** 3,
                         prog_bar=True, logger=False)
            return loss

        def on_after_backward(self) -> None:
            finite = True
            nonzero = False
            for parameter in self.network.parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                finite = finite and bool(torch.isfinite(gradient).all())
                nonzero = nonzero or bool(torch.count_nonzero(gradient))
            if not finite:
                raise FloatingPointError("non-finite gradient in Multimodal V2 trainable path")
            self.log("train/has_nonzero_gradient", float(nonzero), on_step=True,
                     on_epoch=False, logger=True)

        def on_validation_epoch_start(self) -> None:
            self._val_rows.clear()

        def validation_step(self, batch, batch_idx):
            with torch.autocast(device_type=self.device.type, enabled=False):
                values = forward_step(self.runtime, batch, self.device, compute_frozen_losses=False)
            self._val_rows.extend(validation_rows(
                values["output"], values["targets"], batch["sample_id"], batch["sequence_id"]
            ))
            self.log("val/loss", values["loss"].float(), on_epoch=True,
                     batch_size=len(batch["m_R"]))

        def on_validation_epoch_end(self) -> None:
            metrics = summarize_3d(self._val_rows)
            self.log("val/success_1m", float(metrics["success_1m"]), prog_bar=True)
            self.log("val/median_error_m", float(metrics["error_median_m"]), prog_bar=True)
            self.log("val/coverage", float(metrics["coverage"]))

        def configure_optimizers(self):
            cfg = self.config["training"]
            groups = trainable_parameter_groups(
                self.runtime, float(cfg["learning_rate"]), float(cfg["weight_decay"])
            )
            optimizer = torch.optim.AdamW(
                groups, betas=tuple(cfg["betas"]), eps=float(cfg["eps"])
            )
            total = max(1, int(self.trainer.estimated_stepping_batches))
            warmup = max(1, round(total * float(cfg["warmup_fraction"])))
            final_ratio = float(cfg["final_lr_ratio"])
            def factor(step: int) -> float:
                update = step + 1
                if update <= warmup:
                    return update / warmup
                progress = min(1.0, (update - warmup) / max(1, total - warmup))
                return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
