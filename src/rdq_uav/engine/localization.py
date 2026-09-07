from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torchvision.ops import generalized_box_iou_loss


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half_size = boxes[..., 2:] / 2.0
    return torch.cat((center - half_size, center + half_size), dim=-1)


def aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise-aligned IoU for normalized XYXY boxes."""
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0).prod(dim=-1)
    return intersection / (area1 + area2 - intersection).clamp(min=1e-7)


class LocalizationLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.bbox_l1_weight = float(config["bbox_l1_weight"])
        self.giou_weight = float(config["giou_weight"])
        self.position_weight = float(config["position_weight"])
        projection = config.get("projection_consistency", {"enabled": False})
        if bool(projection.get("enabled", False)):
            raise RuntimeError(
                "Projection consistency is disabled until coordinate frame, units, time convention "
                "and fisheye calibration are jointly verified"
            )

    def forward(
        self,
        pred_box: torch.Tensor,
        gt_box: torch.Tensor,
        pred_position_normalized: torch.Tensor,
        gt_position_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bbox_l1 = nn.functional.smooth_l1_loss(pred_box, gt_box)
        bbox_center_l1 = nn.functional.smooth_l1_loss(pred_box[..., :2], gt_box[..., :2])
        bbox_size_l1 = nn.functional.smooth_l1_loss(pred_box[..., 2:], gt_box[..., 2:])
        giou = generalized_box_iou_loss(
            cxcywh_to_xyxy(pred_box), cxcywh_to_xyxy(gt_box), reduction="mean"
        )
        position = nn.functional.smooth_l1_loss(
            pred_position_normalized, gt_position_normalized
        )
        total = (
            self.bbox_l1_weight * bbox_l1
            + self.giou_weight * giou
            + self.position_weight * position
        )
        return {
            "total_loss": total,
            "bbox_l1_loss": bbox_l1,
            "bbox_center_l1_loss": bbox_center_l1,
            "bbox_size_l1_loss": bbox_size_l1,
            "giou_loss": giou,
            "position_loss": position,
        }


class LocalizationMetrics:
    def __init__(self, processed_height: int, processed_stitched_width: int) -> None:
        self.height = int(processed_height)
        self.width = int(processed_stitched_width)
        self.ious: list[torch.Tensor] = []
        self.center_error_px: list[torch.Tensor] = []
        self.center_error_normalized: list[torch.Tensor] = []
        self.center_absolute_error: list[torch.Tensor] = []
        self.size_absolute_error: list[torch.Tensor] = []
        self.predicted_size: list[torch.Tensor] = []
        self.target_size: list[torch.Tensor] = []
        self.position_error: list[torch.Tensor] = []
        self.axis_absolute_error: list[torch.Tensor] = []
        self.range_error: list[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        pred_box: torch.Tensor,
        gt_box: torch.Tensor,
        pred_position_m: torch.Tensor,
        gt_position_m: torch.Tensor,
    ) -> None:
        pred_box = pred_box.detach().cpu().float()
        gt_box = gt_box.detach().cpu().float()
        pred_position_m = pred_position_m.detach().cpu().float()
        gt_position_m = gt_position_m.detach().cpu().float()
        delta_center = pred_box[:, :2] - gt_box[:, :2]
        pixel_scale = torch.tensor([self.width, self.height], dtype=torch.float32)
        self.ious.append(aligned_iou(cxcywh_to_xyxy(pred_box), cxcywh_to_xyxy(gt_box)))
        self.center_error_normalized.append(delta_center.norm(dim=1))
        self.center_error_px.append((delta_center * pixel_scale).norm(dim=1))
        self.center_absolute_error.append(delta_center.abs())
        self.size_absolute_error.append((pred_box[:, 2:] - gt_box[:, 2:]).abs())
        self.predicted_size.append(pred_box[:, 2:])
        self.target_size.append(gt_box[:, 2:])
        delta_position = pred_position_m - gt_position_m
        self.position_error.append(delta_position.norm(dim=1))
        self.axis_absolute_error.append(delta_position.abs())
        self.range_error.append(
            (pred_position_m.norm(dim=1) - gt_position_m.norm(dim=1)).abs()
        )

    def compute(self) -> dict[str, float | int]:
        if not self.ious:
            raise RuntimeError("No localization samples were accumulated")
        iou = torch.cat(self.ious).double()
        center_px = torch.cat(self.center_error_px).double()
        center_norm = torch.cat(self.center_error_normalized).double()
        center_absolute = torch.cat(self.center_absolute_error).double()
        size_absolute = torch.cat(self.size_absolute_error).double()
        predicted_size = torch.cat(self.predicted_size).double()
        target_size = torch.cat(self.target_size).double()
        position = torch.cat(self.position_error).double()
        axis = torch.cat(self.axis_absolute_error).double()
        range_error = torch.cat(self.range_error).double()
        return {
            "mean_iou": float(iou.mean()),
            "median_iou": float(iou.quantile(0.5)),
            "recall_iou_0.5": float((iou >= 0.5).double().mean()),
            "bbox_center_error_px_mean": float(center_px.mean()),
            "bbox_center_error_px_median": float(center_px.quantile(0.5)),
            "center_error_px_mean": float(center_px.mean()),
            "center_error_px_median": float(center_px.quantile(0.5)),
            "normalized_center_error": float(center_norm.mean()),
            "center_abs_error_x": float(center_absolute[:, 0].mean()),
            "center_abs_error_y": float(center_absolute[:, 1].mean()),
            "width_abs_error_mean": float(size_absolute[:, 0].mean()),
            "width_abs_error_median": float(size_absolute[:, 0].quantile(0.5)),
            "height_abs_error_mean": float(size_absolute[:, 1].mean()),
            "height_abs_error_median": float(size_absolute[:, 1].quantile(0.5)),
            "pred_width_mean": float(predicted_size[:, 0].mean()),
            "pred_width_median": float(predicted_size[:, 0].quantile(0.5)),
            "gt_width_mean": float(target_size[:, 0].mean()),
            "gt_width_median": float(target_size[:, 0].quantile(0.5)),
            "pred_height_mean": float(predicted_size[:, 1].mean()),
            "pred_height_median": float(predicted_size[:, 1].quantile(0.5)),
            "gt_height_mean": float(target_size[:, 1].mean()),
            "gt_height_median": float(target_size[:, 1].quantile(0.5)),
            "position_error_mean_m": float(position.mean()),
            "position_error_median_m": float(position.quantile(0.5)),
            "mae_x_m": float(axis[:, 0].mean()),
            "mae_y_m": float(axis[:, 1].mean()),
            "mae_z_m": float(axis[:, 2].mean()),
            "range_error_mean_m": float(range_error.mean()),
            "range_error_median_m": float(range_error.quantile(0.5)),
            "samples": int(len(iou)),
        }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_localization_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: LocalizationLoss,
    position_mean: torch.Tensor,
    position_std: torch.Tensor,
    processed_height: int,
    processed_stitched_width: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    log_interval: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    meter = LocalizationMetrics(processed_height, processed_stitched_width)
    totals: defaultdict[str, float] = defaultdict(float)
    predictions: list[dict[str, Any]] = []
    sample_count = 0
    start = time.perf_counter()
    position_mean = position_mean.to(device)
    position_std = position_std.to(device)

    for step, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        batch_size = int(batch["bbox"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.cuda.amp.autocast(enabled=amp) if device.type == "cuda" else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast_context:
            outputs = model(batch["image"], batch["radar"], batch["radar_mask"])
            losses = criterion(
                outputs["box"],
                batch["bbox"],
                outputs["position"],
                batch["position_normalized"],
            )

        if training:
            loss = losses["total_loss"]
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        pred_position_m = outputs["position"] * position_std + position_mean
        meter.update(outputs["box"], batch["bbox"], pred_position_m, batch["position"])
        for key, value in losses.items():
            totals[key] += float(value.detach()) * batch_size
        sample_count += batch_size

        pred_box_cpu = outputs["box"].detach().cpu()
        gt_box_cpu = raw_batch["bbox"].cpu()
        pred_position_cpu = pred_position_m.detach().cpu()
        gt_position_cpu = raw_batch["position"].cpu()
        ious = aligned_iou(cxcywh_to_xyxy(pred_box_cpu), cxcywh_to_xyxy(gt_box_cpu))
        position_errors = (pred_position_cpu - gt_position_cpu).norm(dim=1)
        for index in range(batch_size):
            predictions.append(
                {
                    "sample_id": raw_batch["sample_id"][index],
                    "pred_bbox": pred_box_cpu[index].tolist(),
                    "gt_bbox": gt_box_cpu[index].tolist(),
                    "iou": float(ious[index]),
                    "pred_xyz": pred_position_cpu[index].tolist(),
                    "gt_xyz": gt_position_cpu[index].tolist(),
                    "position_error_m": float(position_errors[index]),
                    "distance_m": float(raw_batch["distance"][index]),
                    "sequence_id": raw_batch["sequence_id"][index],
                    "temporal_block": int(raw_batch["temporal_block"][index]),
                    "gt_time": float(raw_batch["gt_time"][index]),
                }
            )
        if training and log_interval > 0 and step % log_interval == 0:
            print(
                f"step={step}/{len(loader)} samples={sample_count} "
                f"total={totals['total_loss']/sample_count:.4f} "
                f"bbox_l1={totals['bbox_l1_loss']/sample_count:.4f} "
                f"center_l1={totals['bbox_center_l1_loss']/sample_count:.4f} "
                f"size_l1={totals['bbox_size_l1_loss']/sample_count:.4f} "
                f"giou={totals['giou_loss']/sample_count:.4f} "
                f"xyz={totals['position_loss']/sample_count:.4f}",
                flush=True,
            )

    result = meter.compute()
    divisor = max(sample_count, 1)
    result.update({key: value / divisor for key, value in totals.items()})
    result["seconds"] = time.perf_counter() - start
    return result, predictions
