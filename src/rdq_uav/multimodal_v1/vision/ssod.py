"""Geometry-guided semi-supervised utilities for the single-class UAV DINO path.

The 3D target projection is a weak localization cue only. It is never converted
into a synthetic bounding box. High-quality teacher boxes supervise class + box;
medium-quality teacher boxes supervise classification only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import torch


class PseudoQuality(IntEnum):
    IGNORE = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class ViewTransform:
    """Invertible resize + optional horizontal flip from source camera pixels."""

    source_wh: tuple[int, int]
    view_wh: tuple[int, int]
    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        sw, sh = self.source_wh
        vw, vh = self.view_wh
        if min(sw, sh, vw, vh) <= 0:
            raise ValueError("source/view dimensions must be positive")

    @property
    def scale_xy(self) -> tuple[float, float]:
        sw, sh = self.source_wh
        vw, vh = self.view_wh
        return vw / sw, vh / sh

    def source_points_to_view(self, points_xy: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points_xy)
        if points.shape[-1] != 2:
            raise ValueError("points must end in XY")
        sx, sy = self.scale_xy
        out = points * points.new_tensor((sx, sy))
        if self.horizontal_flip:
            out = torch.stack((out.new_tensor(float(self.view_wh[0])) - out[..., 0], out[..., 1]), dim=-1)
        return out

    def source_boxes_to_view(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        boxes = torch.as_tensor(boxes_xyxy)
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must end in XYXY")
        sx, sy = self.scale_xy
        x1, y1, x2, y2 = (boxes * boxes.new_tensor((sx, sy, sx, sy))).unbind(-1)
        if self.horizontal_flip:
            width = boxes.new_tensor(float(self.view_wh[0]))
            x1, x2 = width - x2, width - x1
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def view_boxes_to_source(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        boxes = torch.as_tensor(boxes_xyxy)
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must end in XYXY")
        x1, y1, x2, y2 = boxes.unbind(-1)
        if self.horizontal_flip:
            width = boxes.new_tensor(float(self.view_wh[0]))
            x1, x2 = width - x2, width - x1
        sx, sy = self.scale_xy
        return torch.stack((x1 / sx, y1 / sy, x2 / sx, y2 / sy), dim=-1)


@dataclass(frozen=True)
class PseudoLabelPolicy:
    tau_high: float
    tau_mid: float
    high_geometry_px: float = 16.0
    medium_geometry_px: float = 32.0
    high_stability_iou: float = 0.60
    medium_stability_iou: float = 0.40

    def __post_init__(self) -> None:
        if not 0 <= self.tau_mid <= self.tau_high <= 1:
            raise ValueError("require 0 <= tau_mid <= tau_high <= 1")
        if self.high_geometry_px > self.medium_geometry_px:
            raise ValueError("high geometry gate cannot be wider than medium gate")
        if self.high_stability_iou < self.medium_stability_iou:
            raise ValueError("high stability requirement cannot be weaker than medium")


@dataclass(frozen=True)
class TeacherCandidate:
    box_xyxy_source: torch.Tensor
    score: float
    geometry_px: float
    query_index: int


@dataclass(frozen=True)
class PseudoLabel:
    quality: PseudoQuality
    box_xyxy_source: torch.Tensor
    score: float
    geometry_px: float
    stability_iou: float

    @property
    def valid(self) -> bool:
        return self.quality != PseudoQuality.IGNORE


@dataclass(frozen=True)
class CalibrationObservation:
    score: float
    geometry_px: float
    stability_iou: float
    iou_to_gt: float


def box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    a = torch.as_tensor(box_a)
    b = torch.as_tensor(box_b, device=a.device, dtype=a.dtype)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("box IoU expects XYXY")
    left_top = torch.maximum(a[..., :2], b[..., :2])
    right_bottom = torch.minimum(a[..., 2:], b[..., 2:])
    inter_wh = (right_bottom - left_top).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = ((a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0))
    area_b = ((b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0))
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def point_to_box_distance(point_xy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    point = torch.as_tensor(point_xy)
    boxes = torch.as_tensor(boxes_xyxy, device=point.device, dtype=point.dtype)
    if point.shape != (2,) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("expected point [2] and boxes [N,4]")
    px, py = point
    dx = torch.maximum(torch.maximum(boxes[:, 0] - px, px - boxes[:, 2]), torch.zeros_like(boxes[:, 0]))
    dy = torch.maximum(torch.maximum(boxes[:, 1] - py, py - boxes[:, 3]), torch.zeros_like(boxes[:, 1]))
    return torch.sqrt(dx.square() + dy.square())


def _normalized_cxcywh_to_view_xyxy(boxes: torch.Tensor, view_wh: tuple[int, int]) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("DINO boxes must be [Q,4] normalized cxcywh")
    width, height = view_wh
    cx, cy, bw, bh = boxes.unbind(1)
    scale = boxes.new_tensor((width, height, width, height))
    return torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), dim=1) * scale


def source_xyxy_to_normalized_cxcywh(box_xyxy_source: torch.Tensor, transform: ViewTransform) -> torch.Tensor:
    box = transform.source_boxes_to_view(torch.as_tensor(box_xyxy_source))
    width, height = transform.view_wh
    x1, y1, x2, y2 = box.unbind(-1)
    out = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
    return out / out.new_tensor((width, height, width, height))


def select_single_uav_candidate(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    *,
    transform: ViewTransform,
    projected_point_source: torch.Tensor,
    max_geometry_px: float,
) -> TeacherCandidate | None:
    """Select the highest-score one-class DINO query inside a geometry envelope."""
    logits = torch.as_tensor(pred_logits)
    boxes = torch.as_tensor(pred_boxes, device=logits.device, dtype=logits.dtype)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or boxes.shape != (len(logits), 4):
        raise ValueError("expected one-class logits [Q] / boxes [Q,4]")
    if len(logits) == 0:
        return None
    source_boxes = transform.view_boxes_to_source(_normalized_cxcywh_to_view_xyxy(boxes, transform.view_wh))
    score = logits.sigmoid()
    finite = torch.isfinite(score) & torch.isfinite(source_boxes).all(1)
    positive_size = (source_boxes[:, 2] > source_boxes[:, 0]) & (source_boxes[:, 3] > source_boxes[:, 1])
    distance = point_to_box_distance(projected_point_source.to(logits), source_boxes)
    feasible = finite & positive_size & (distance <= float(max_geometry_px))
    if not bool(feasible.any()):
        return None
    candidates = torch.nonzero(feasible, as_tuple=False).flatten()
    candidate_scores = score[candidates]
    best_local = torch.argmax(candidate_scores)
    index = int(candidates[best_local].item())
    return TeacherCandidate(
        box_xyxy_source=source_boxes[index],
        score=float(score[index].detach().item()),
        geometry_px=float(distance[index].detach().item()),
        query_index=index,
    )


def mine_geometry_guided_pseudo(
    weak1: Mapping[str, torch.Tensor],
    weak2: Mapping[str, torch.Tensor],
    *,
    transform1: ViewTransform,
    transform2: ViewTransform,
    projected_point_source: torch.Tensor,
    policy: PseudoLabelPolicy,
) -> PseudoLabel:
    """Mine one pseudo UAV box from two teacher weak views."""
    c1 = select_single_uav_candidate(
        weak1["pred_logits"], weak1["pred_boxes"], transform=transform1,
        projected_point_source=projected_point_source, max_geometry_px=policy.medium_geometry_px,
    )
    c2 = select_single_uav_candidate(
        weak2["pred_logits"], weak2["pred_boxes"], transform=transform2,
        projected_point_source=projected_point_source, max_geometry_px=policy.medium_geometry_px,
    )
    if c1 is None or c2 is None:
        dtype = weak1["pred_boxes"].dtype
        device = weak1["pred_boxes"].device
        return PseudoLabel(PseudoQuality.IGNORE, torch.zeros(4, dtype=dtype, device=device), 0.0, float("inf"), 0.0)

    stability = float(box_iou_xyxy(c1.box_xyxy_source, c2.box_xyxy_source).detach().item())
    score = min(c1.score, c2.score)
    geometry = max(c1.geometry_px, c2.geometry_px)
    chosen = c1 if c1.score >= c2.score else c2
    if score >= policy.tau_high and geometry <= policy.high_geometry_px and stability >= policy.high_stability_iou:
        quality = PseudoQuality.HIGH
    elif score >= policy.tau_mid and geometry <= policy.medium_geometry_px and stability >= policy.medium_stability_iou:
        quality = PseudoQuality.MEDIUM
    else:
        quality = PseudoQuality.IGNORE
    return PseudoLabel(quality, chosen.box_xyxy_source, score, geometry, stability)


def calibrate_score_threshold(
    observations: Sequence[CalibrationObservation],
    *,
    geometry_limit_px: float,
    stability_min_iou: float,
    success_iou_threshold: float,
    minimum_precision: float,
    min_count: int = 5,
) -> float:
    """Lowest teacher score threshold meeting precision on labeled calibration data."""
    eligible = [
        obs for obs in observations
        if obs.geometry_px <= geometry_limit_px and obs.stability_iou >= stability_min_iou
    ]
    if len(eligible) < min_count:
        raise RuntimeError(f"only {len(eligible)} eligible calibration observations; need {min_count}")
    thresholds = sorted({float(obs.score) for obs in eligible})
    for threshold in thresholds:
        selected = [obs for obs in eligible if obs.score >= threshold]
        if len(selected) < min_count:
            continue
        precision = sum(obs.iou_to_gt >= success_iou_threshold for obs in selected) / len(selected)
        if precision >= minimum_precision:
            return threshold
    raise RuntimeError(
        f"no score threshold reaches precision={minimum_precision:.3f} "
        f"for IoU>={success_iou_threshold:.3f}"
    )


@torch.no_grad()
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, decay: float = 0.999) -> None:
    """EMA parameters; copy non-floating buffers exactly from student."""
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if teacher_state.keys() != student_state.keys():
        raise ValueError("teacher/student state_dict keys differ")
    for key, teacher_value in teacher_state.items():
        student_value = student_state[key].to(device=teacher_value.device)
        if teacher_value.dtype.is_floating_point:
            teacher_value.mul_(decay).add_(student_value.to(dtype=teacher_value.dtype), alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


def linear_unsup_weight(step: int, total_steps: int, *, target: float = 1.0, ramp_fraction: float = 0.10) -> float:
    if total_steps <= 0 or step < 0:
        raise ValueError("invalid SSOD step/total_steps")
    if not 0 <= ramp_fraction <= 1:
        raise ValueError("ramp_fraction must be in [0,1]")
    ramp_steps = max(1, int(round(total_steps * ramp_fraction))) if ramp_fraction > 0 else 0
    if ramp_steps == 0:
        return float(target)
    return float(target) * min(1.0, float(step + 1) / float(ramp_steps))


def _distributed_num_boxes(targets: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> float:
    count = float(sum(int(len(target["labels"])) for target in targets))
    value = torch.tensor([count], dtype=torch.float32, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value)
        value /= torch.distributed.get_world_size()
    return float(value.clamp(min=1).item())


def classification_only_pseudo_loss(
    criterion: Any,
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """DINO matcher + focal classification only; deliberately excludes box/GIoU."""
    if not hasattr(criterion, "matcher") or not hasattr(criterion, "loss_labels"):
        raise TypeError("criterion must expose matcher and loss_labels")
    main = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
    num_boxes = _distributed_num_boxes(targets, main["pred_logits"].device)
    result: dict[str, torch.Tensor] = {}

    def add_level(level_outputs: Mapping[str, torch.Tensor], suffix: str = "") -> None:
        indices = criterion.matcher(level_outputs, targets)
        raw = criterion.loss_labels(level_outputs, targets, indices, num_boxes)
        for key, value in raw.items():
            weighted_key = key + suffix
            weight = criterion.weight_dict.get(weighted_key, criterion.weight_dict.get(key, 1.0))
            result[weighted_key] = value * float(weight)

    add_level(main)
    for index, aux in enumerate(outputs.get("aux_outputs", ())):
        add_level(aux, f"_{index}")
    return result


def weighted_loss_sum(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = list(losses.values())
    if not values:
        raise ValueError("loss dictionary is empty")
    return torch.stack([value if value.ndim == 0 else value.sum() for value in values]).sum()
