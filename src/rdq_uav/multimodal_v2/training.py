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

from .radar_loss import CandidateLoss
from .radar_model import LiDARUAVDetector
from .radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.geometry import load_left_projection_context
from .dino_supervision import build_dino_targets, supervised_dino_loss
from .paper_metrics import localization_metrics, coco_bbox_metrics
from .monitoring import RunningLocalization, dino_loss_components, visual_metric_rows, localization_display
from .uav_dino import adapt_dino_class_head_to_single_uav, load_coco_pretrained_dino
from rdq_uav.runtime_paths import ensure_detrex_config_link

from .data import prepare_model_batch
from .interaction import CandidateCrossAttention
from .lidar import LiDARCandidateModel
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


def _load_trained_branch(module: nn.Module, checkpoint: Path, prefix: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise RuntimeError(f"{checkpoint} has no {prefix} weights")
    module.load_state_dict(selected, strict=True)


def build_runtime(config: Mapping[str, Any], root: Path, device: torch.device) -> V2Runtime:
    init, model_cfg = config["initialization"], config["model"]
    stage = str(config["experiment"]["stage"])
    if stage not in {"B0", "B1", "B2", "B3"}:
        raise ValueError(f"unknown V2 stage {stage}")
    candidate_cfg = yaml.safe_load(_resolve(root, init["candidate_config"]).read_text())
    lidar_cfg = yaml.safe_load(_resolve(root, init["lidar_config"]).read_text())
    lidar_detector: nn.Module = nn.Identity()
    lidar: nn.Module = nn.Identity()
    if stage != "B1":
        lidar_detector = LiDARUAVDetector(lidar_cfg).to(device)
        lidar = LiDARCandidateModel(lidar_detector, CandidateSelector(candidate_cfg["candidate"]["radar"]))
        if stage in {"B2", "B3"}:
            _load_trained_branch(lidar_detector, _resolve(root, init["b0_checkpoint"]), "network.lidar.detector.")

    dino_initialization = None
    dino_detector: nn.Module = nn.Identity()
    vision: nn.Module = nn.Identity()
    if stage != "B0":
        detrex = _resolve(root, init["dino_root"])
        ensure_detrex_config_link(detrex)
        for path in (root / "src", detrex / "detectron2", detrex):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from detectron2.config import LazyConfig, instantiate
        dino_cfg = LazyConfig.load(str(_resolve(root, init["dino_config"])))
        dino_cfg.model.device = str(device)
        dino_detector = instantiate(dino_cfg.model).to(device)
        if stage == "B1" and init.get("dino_checkpoint"):
            dino_initialization = load_coco_pretrained_dino(
                dino_detector, _resolve(root, init["dino_checkpoint"])
            )
        adapt_dino_class_head_to_single_uav(dino_detector)
        vision = VisionCandidateModel(
            dino_detector,
            pre_topk=int(candidate_cfg["candidate"]["rgb"]["pre_topk"]),
            final_topk=int(candidate_cfg["candidate"]["rgb"]["final_topk"]),
            nms_iou=float(candidate_cfg["candidate"]["rgb"]["nms_iou"]),
        )
        if stage in {"B2", "B3"}:
            checkpoint = _resolve(root, init["b1_checkpoint"])
            _load_trained_branch(dino_detector, checkpoint, "network.vision.dino.detector.")
            _load_trained_branch(vision.builder.feature_proj, checkpoint, "network.vision.builder.feature_proj.")

    interaction = CandidateCrossAttention(
        visual_radius=int(model_cfg["visual_radius_cells"]),
        lidar_neighbors=int(model_cfg["lidar_neighbors"]),
        geometry_gate=model_cfg["geometry_gate"],
    )
    scoring = CandidateScoring(
        geometry_gate=model_cfg["geometry_gate"],
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
        {"stage": stage, "b0_checkpoint": init.get("b0_checkpoint"), "b1_checkpoint": init.get("b1_checkpoint"),
         "dino_pretrained": dino_initialization},
    )
    freeze_for_stage(runtime, str(config["experiment"]["stage"]))
    return runtime


def freeze_for_stage(runtime: V2Runtime, stage: str) -> None:
    if stage not in {"B0", "B1", "B2", "B3"}:
        raise ValueError(f"unknown V2 stage {stage}")
    runtime.model.requires_grad_(False)
    if stage == "B0":
        runtime.model.lidar.requires_grad_(True)
    if stage == "B1":
        runtime.model.vision.requires_grad_(True)
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
    if isinstance(detector, nn.Identity):
        return
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


def _stage_batch(batch: Mapping[str, Any], stage: str) -> dict[str, Any]:
    selected = dict(batch)
    if stage == "B0":
        selected["m_V"] = torch.zeros_like(batch["m_V"])
        selected["gt_2d_valid"] = torch.zeros_like(batch["gt_2d_valid"])
        selected["vision_batch_index"] = batch["vision_batch_index"][:0]
        selected["image_uint8"] = None
        for key in ("image_source_wh", "image_view_wh", "image_scale_xy", "vision_delta_t"):
            selected[key] = batch[key][:0]
    elif stage == "B1":
        selected["m_R"] = torch.zeros_like(batch["m_R"])
        selected["radar_batch_index"] = batch["radar_batch_index"][:0]
        for key in ("points", "delta_t", "sensor_id", "point_counts", "point_batch_index"):
            selected[key] = batch[key][:0]
    return selected


def forward_step(runtime: V2Runtime, batch: Mapping[str, Any], device: torch.device,
                 *, compute_frozen_losses: bool = True) -> dict[str, Any]:
    stage = runtime.initialization["stage"]
    batch = _stage_batch(batch, stage)
    lidar_batch, images, masks, projection, targets, transforms = prepare_model_batch(
        batch, runtime.dino_detector, runtime.projection_base, device,
    )
    model_kwargs = {}
    if stage == "B1" and runtime.model.training and images is not None:
        ids = lidar_batch["vision_batch_index"]
        model_kwargs["vision_targets"] = build_dino_targets(
            targets.box_xyxy_px[ids], targets.has_box[ids], transforms,
        )
    output = runtime.model(lidar_batch, images, masks, projection, **model_kwargs)
    rank = runtime.ranking_loss(output, targets)
    result = {"output": output, "targets": targets, **rank}
    if stage == "B0":
        raw = output.diagnostics["lidar_raw"]
        radar_loss = runtime.lidar_loss(raw, output.diagnostics["lidar_batch"]) if raw is not None else None
        result["loss"] = radar_loss["loss"] if radar_loss is not None else rank["loss"]
        result["has_trainable_loss"] = radar_loss is not None and radar_loss["num_supervised_samples"] > 0
        result["radar_supervised"] = 0 if radar_loss is None else radar_loss["num_supervised_samples"]
        result["loss_components"] = {} if radar_loss is None else {
            "cls": radar_loss["loss_cls"].detach(),
            "reg": (runtime.lidar_loss.reg_weight * radar_loss["loss_reg"]).detach(),
        }
    elif stage == "B1":
        ids = lidar_batch["vision_batch_index"]
        raw = output.diagnostics["vision_raw"]
        vision_loss, weighted, count = supervised_dino_loss(
            runtime.dino_detector, raw, gt_box_xyxy_source=targets.box_xyxy_px[ids],
            gt_2d_valid=targets.has_box[ids], transforms=transforms,
        ) if raw is not None else (rank["loss"], {}, 0)
        result["loss"] = vision_loss
        result["has_trainable_loss"] = count > 0
        result["vision_supervised"] = count
        result["loss_components"] = dino_loss_components(weighted)
    if compute_frozen_losses:
        zero = targets.xyz_m.new_zeros(())
        lidar_raw = output.diagnostics["lidar_raw"]
        result["loss_lidar_frozen"] = (
            runtime.lidar_loss(lidar_raw, output.diagnostics["lidar_batch"])["loss"]
            if stage in {"B2", "B3"} and lidar_raw is not None else zero
        )
        vision_raw = output.diagnostics["vision_raw"]
        vision, count = zero, 0
        if stage in {"B2", "B3"} and vision_raw is not None:
            ids = lidar_batch["vision_batch_index"]
            vision, _, count = supervised_dino_loss(
                runtime.dino_detector, vision_raw,
                gt_box_xyxy_source=targets.box_xyxy_px[ids],
                gt_2d_valid=targets.has_box[ids], transforms=transforms,
            )
        result["loss_vision_frozen"] = vision
        if stage in {"B2", "B3"}:
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
                     "has_gt3d": True, "gt_xyz": targets.xyz_m[batch].float().cpu().tolist(),
                     "pred_xyz": output.xyz_m[ids[0]].float().cpu().tolist() if len(ids) else None,
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
            self._val_2d: list[float] = []
            self._val_visual: list[dict[str, Any]] = []
            self._train_position = RunningLocalization()
            self._val_position = RunningLocalization()
            self.live_progress = {}
            self._last_visual = {}
            self._train_loss_sum = 0.0
            self._train_loss_samples = 0
            self._in_validation = False
            self._train_progress_snapshot = {}
            self._live_every = int(self.config.get("logging", {}).get("ap_every_val_batches", 50))
            if self._live_every < 1:
                raise ValueError("ap_every_val_batches must be positive")

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
            self._train_position = RunningLocalization()
            self._train_loss_sum = 0.0
            self._train_loss_samples = 0
            self.live_progress = dict(self._last_visual)
            self._in_validation = False
            # B2/B3 keep both pretrained candidate generators frozen and in
            # deterministic eval mode; only candidate interaction/scoring train.
            if self.runtime.initialization["stage"] in {"B2", "B3"}:
                self.runtime.lidar_detector.eval()
                self.runtime.dino_detector.eval()
            else:
                self.network.train()

        def on_validation_start(self) -> None:
            self._refresh_external_device()

        def training_step(self, batch, batch_idx):
            if self._in_validation:
                self.live_progress = dict(self._train_progress_snapshot)
                self.live_progress.update(self._last_visual)
                self._in_validation = False
            if not bool(batch["m_R"].any() | batch["m_V"].any()):
                self.log("train/both_modalities_missing", float(len(batch["m_R"])),
                         on_step=False, on_epoch=True, reduce_fx="sum")
            values = forward_step(self.runtime, batch, self.device, compute_frozen_losses=True)
            loss = values["loss"]
            batch_size = len(batch["m_R"])
            if self.runtime.initialization["stage"] != "B1":
                self._train_position.update(validation_rows(
                    values["output"], values["targets"], batch["sample_id"], batch["sequence_id"]))
                position = self._train_position.compute()
                self.live_progress.update(localization_display(position, "T"))
                self.log_dict({f"train/live_{k}": float(v) for k, v in position.items() if v is not None},
                              on_step=True, on_epoch=False, batch_size=batch_size)
            self.live_progress["loss"] = float(loss.detach())
            component_names = (("cls", "reg") if self.runtime.initialization["stage"] == "B0"
                               else ("cls", "bbox", "giou", "dn", "aux", "enc", "other")
                               if self.runtime.initialization["stage"] == "B1" else ())
            for key in component_names:
                value = values.get("loss_components", {}).get(key)
                self.live_progress[key] = float(value) if value is not None else 0.0
            if self.runtime.initialization["stage"] in {"B2", "B3"}:
                self.live_progress.update(rank3d=float(values["loss_rank_3d"].detach()),
                                          rank2d=float(values["loss_rank_2d"].detach()))
            if self.device.type == "cuda":
                self.live_progress["mem"] = torch.cuda.max_memory_reserved()/1024**3
            if self._trainer is not None and self.trainer.optimizers:
                self.live_progress["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
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
                self.live_progress["skip"] = 1
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
            self.live_progress["skip"] = 0
            self._train_loss_sum += float(loss.detach())*batch_size
            self._train_loss_samples += batch_size
            self.live_progress["loss_avg"] = self._train_loss_sum/self._train_loss_samples
            for key, value in values.get("loss_components", {}).items():
                self.log(f"train/native_{key}", value, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log_dict({
                "train/loss": loss, "train/rank_3d": values["loss_rank_3d"],
                "train/rank_2d": values["loss_rank_2d"],
                "train/lidar_frozen": values.get("loss_lidar_frozen", loss.new_zeros(())),
                "train/vision_frozen": values.get("loss_vision_frozen", loss.new_zeros(())),
            }, on_step=True, on_epoch=True, batch_size=batch_size)
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
            self._train_progress_snapshot = dict(self.live_progress)
            self._in_validation = True
            self._val_rows.clear()
            self._val_2d.clear()
            self._val_visual.clear()
            self._val_position = RunningLocalization()
            self.live_progress = {}
            if self.runtime.initialization["stage"] != "B0":
                self.live_progress.update({f"V/{key}": "--" for key in ("AP", "AP50", "AP75", "AP_small", "AR100")})

        def _refresh_visual_metrics(self, final=False):
            metrics = coco_bbox_metrics(self._val_visual)
            prefix = "V" if final else "V~"
            for key in ("AP", "AP50", "AP75", "AP_small", "AR100"):
                self.live_progress.pop(f"V/{key}", None)
                self.live_progress.pop(f"V~/{key}", None)
                value = metrics[key]
                self.live_progress[f"{prefix}/{key}"] = "--" if value is None else value
                if final and value is not None:
                    self.log(f"val/{key}", value)
            self.live_progress[f"{prefix}/images"] = metrics["unique_labeled_images"]
            if final:
                self.live_progress.pop("V~/images", None)
                if not self.trainer.sanity_checking:
                    self._last_visual = {f"lastV/{key}": self.live_progress[f"V/{key}"]
                                         for key in ("AP", "AP50", "AP75", "AP_small", "AR100")}
                self.log("val/coco_images", float(metrics["unique_labeled_images"]))

        def validation_step(self, batch, batch_idx):
            with torch.autocast(device_type=self.device.type, enabled=False):
                values = forward_step(self.runtime, batch, self.device, compute_frozen_losses=False)
            rows = validation_rows(
                values["output"], values["targets"], batch["sample_id"], batch["sequence_id"]
            )
            self._val_rows.extend(rows)
            if self.runtime.initialization["stage"] != "B1":
                self._val_position.update(rows)
                self.live_progress.update(localization_display(self._val_position.compute(), "V~"))
            if self.runtime.initialization["stage"] != "B0":
                self._val_visual.extend(visual_metric_rows(values["output"], values["targets"], batch))
                if (batch_idx+1) % self._live_every == 0:
                    self._refresh_visual_metrics()
            self.live_progress["val_loss"] = float(values["loss"].detach())
            from .loss import box_iou_aligned
            for index in range(len(values["targets"].has_box)):
                if not bool(values["targets"].has_box[index]):
                    continue
                ids = values["output"].top2d_indices(len(values["targets"].has_box))[index]
                iou = box_iou_aligned(values["output"].box_xyxy_px[ids[:1]].float(),
                                      values["targets"].box_xyxy_px[index].float()) if len(ids) else []
                self._val_2d.append(float(iou[0]) if len(iou) else 0.0)
            self.log("val/loss", values["loss"].float(), on_epoch=True,
                     batch_size=len(batch["m_R"]))

        def on_validation_epoch_end(self) -> None:
            metrics = summarize_3d(self._val_rows)
            if self.runtime.initialization["stage"] != "B1":
                position = localization_metrics(self._val_rows)
                for key in ("rmse_x_m", "rmse_y_m", "rmse_z_m", "rmse_3d_m"):
                    if position[key] is not None:
                        self.log(f"val/{key}", position[key])
                self.log("val/rmse_output_count", float(position["predicted_queries"]))
                for key in list(self.live_progress):
                    if key.startswith("V~/") and key.split("/")[1] in {"rx", "ry", "rz", "r3", "Cov", "S1", "N"}:
                        self.live_progress.pop(key)
                self.live_progress.update(localization_display(position, "V"))
                self.log("val/success_1m", float(metrics["success_1m"]))
                self.log("val/median_error_m", float(metrics["error_median_m"]))
                self.log("val/coverage", float(metrics["coverage"]))
            if self.runtime.initialization["stage"] != "B0":
                self._refresh_visual_metrics(final=True)
            self.log("val/2d_iou50", sum(value >= 0.5 for value in self._val_2d) / len(self._val_2d)
                     if self._val_2d else 0.0)
            if not self.trainer.sanity_checking:
                from tqdm import tqdm
                text = " ".join(f"{key}={value:.4g}" if isinstance(value, (float, int)) else f"{key}={value}"
                                for key, value in self.live_progress.items())
                tqdm.write(f"Validation epoch {self.current_epoch}: {text}")

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
