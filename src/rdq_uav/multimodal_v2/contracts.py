"""Tensor contracts for candidate-level Multimodal V2.

Coordinates are explicit: ``xyz_m`` is in the LiDAR/GT metric frame and
``box_xyxy_px``/``projected_xy_px`` are calibrated left-camera source pixels.
Invalid values always have a boolean mask; zero-filled storage is never used
to infer validity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

import torch

CandidateSource = Literal["R", "V"]


@dataclass(frozen=True)
class CandidateBatch:
    feature: torch.Tensor
    score: torch.Tensor
    xyz_m: torch.Tensor
    has_xyz: torch.Tensor
    box_xyxy_px: torch.Tensor
    has_box: torch.Tensor
    batch_index: torch.Tensor
    source_index: torch.Tensor
    source: CandidateSource
    projected_xy_px: torch.Tensor
    projection_valid: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.score.shape[0])
        shapes = {
            "feature": (n, 128), "score": (n,), "xyz_m": (n, 3),
            "has_xyz": (n,), "box_xyxy_px": (n, 4), "has_box": (n,),
            "batch_index": (n,), "source_index": (n,),
            "projected_xy_px": (n, 2), "projection_valid": (n,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(value.shape)}")
        if self.has_xyz.dtype != torch.bool or self.has_box.dtype != torch.bool:
            raise TypeError("has_xyz and has_box must be bool")
        if self.batch_index.dtype != torch.long or self.source_index.dtype != torch.long:
            raise TypeError("candidate indices must be torch.long")
        if self.projection_valid.dtype != torch.bool:
            raise TypeError("projection_valid must be bool")
        if self.source == "R":
            if n and (not bool(self.has_xyz.all()) or bool(self.has_box.any())):
                raise ValueError("R candidates require XYZ and cannot claim a 2D box")
        elif self.source == "V":
            if n and (bool(self.has_xyz.any()) or not bool(self.has_box.all())):
                raise ValueError("V candidates require a 2D box and cannot claim XYZ")
            boxes = self.box_xyxy_px.float()
            if n and (
                not bool(torch.isfinite(boxes).all())
                or bool((boxes[:, 2] <= boxes[:, 0]).any())
                or bool((boxes[:, 3] <= boxes[:, 1]).any())
            ):
                raise ValueError("V candidate boxes must be finite, non-degenerate xyxy")
        else:
            raise ValueError(f"unknown candidate source {self.source!r}")
        for name in ("feature", "score"):
            if n and not bool(torch.isfinite(getattr(self, name).float()).all()):
                raise ValueError(f"{name} must be finite")

    @property
    def n(self) -> int:
        return int(self.score.numel())

    def index_select(self, index: torch.Tensor) -> "CandidateBatch":
        index = torch.as_tensor(index, device=self.score.device)
        if index.dtype == torch.bool:
            if tuple(index.shape) != (self.n,):
                raise ValueError("candidate boolean mask has wrong shape")
            index = torch.nonzero(index, as_tuple=False).flatten()
        if index.dtype != torch.long or index.ndim != 1:
            raise TypeError("candidate selection must be bool [N] or long [K]")
        return replace(
            self,
            feature=self.feature[index], score=self.score[index],
            xyz_m=self.xyz_m[index], has_xyz=self.has_xyz[index],
            box_xyxy_px=self.box_xyxy_px[index], has_box=self.has_box[index],
            batch_index=self.batch_index[index], source_index=self.source_index[index],
            projected_xy_px=self.projected_xy_px[index],
            projection_valid=self.projection_valid[index],
        )

    def with_projection(self, xy_px: torch.Tensor, valid: torch.Tensor) -> "CandidateBatch":
        if self.source != "R":
            raise ValueError("only R candidates can carry calibrated XYZ projection")
        return replace(self, projected_xy_px=xy_px, projection_valid=valid)

    @classmethod
    def empty(cls, source: CandidateSource, device: torch.device | str,
              dtype: torch.dtype = torch.float32) -> "CandidateBatch":
        return cls(
            torch.empty((0, 128), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
            torch.empty((0, 3), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool),
            torch.empty((0, 4), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long), source,
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool),
        )


@dataclass(frozen=True)
class CrossModalEvidence:
    feature: torch.Tensor
    valid: torch.Tensor
    token_count: torch.Tensor
    gate_weight: torch.Tensor
    projected_xy_px: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.valid.numel())
        expected = {
            "feature": (n, 128), "valid": (n,), "token_count": (n,),
            "gate_weight": (n,), "projected_xy_px": (n, 2),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if self.valid.dtype != torch.bool or self.token_count.dtype != torch.long:
            raise TypeError("evidence valid/count dtypes are bool/long")
        if n and not bool((self.feature[~self.valid] == 0).all()):
            raise ValueError("invalid evidence feature must be exactly zero")


@dataclass(frozen=True)
class MultimodalOutput:
    score_3d_before: torch.Tensor
    score_3d_after: torch.Tensor
    delta_3d: torch.Tensor
    score_2d_before: torch.Tensor
    score_2d_after: torch.Tensor
    delta_2d: torch.Tensor
    xyz_m: torch.Tensor
    has_xyz: torch.Tensor
    box_xyxy_px: torch.Tensor
    has_box: torch.Tensor
    batch_index: torch.Tensor
    hypothesis_type: torch.Tensor  # 0=RV, 1=R, 2=V
    radar_source_index: torch.Tensor
    vision_source_index: torch.Tensor
    evidence_valid_3d: torch.Tensor
    gate_3d: torch.Tensor
    token_count_3d: torch.Tensor
    evidence_valid_2d: torch.Tensor
    gate_2d: torch.Tensor
    token_count_2d: torch.Tensor
    association_valid: torch.Tensor
    association_d_box_px: torch.Tensor
    association_d_center_normalized: torch.Tensor
    radar_candidates: CandidateBatch
    vision_candidates: CandidateBatch
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        n = int(self.score_3d_after.numel())
        shapes = {
            "score_3d_before": (n,), "score_3d_after": (n,), "delta_3d": (n,),
            "score_2d_before": (n,), "score_2d_after": (n,), "delta_2d": (n,),
            "xyz_m": (n, 3), "has_xyz": (n,), "box_xyxy_px": (n, 4),
            "has_box": (n,), "batch_index": (n,), "hypothesis_type": (n,),
            "radar_source_index": (n,), "vision_source_index": (n,),
            "evidence_valid_3d": (n,), "gate_3d": (n,), "token_count_3d": (n,),
            "evidence_valid_2d": (n,), "gate_2d": (n,), "token_count_2d": (n,),
            "association_valid": (n,), "association_d_box_px": (n,),
            "association_d_center_normalized": (n,),
        }
        for name, shape in shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if n:
            v_only = self.hypothesis_type == 2
            if bool(self.has_xyz[v_only].any()):
                raise ValueError("V-only hypotheses may not expose absolute XYZ")
            r_only = self.hypothesis_type == 1
            if bool(self.has_box[r_only].any()):
                raise ValueError("R-only hypotheses may not expose a 2D box")
            if bool((self.score_3d_before[~self.has_xyz] != 0).any()):
                raise ValueError("invalid 3D score slots must be zero")
            if bool((self.score_2d_before[~self.has_box] != 0).any()):
                raise ValueError("invalid 2D score slots must be zero")

    @staticmethod
    def _rank_task(
        indices: torch.Tensor, score: torch.Tensor, source_index: torch.Tensor
    ) -> torch.Tensor:
        # Establish the explicit source-id tie order first, then rely on the
        # stable score sort. Association row construction cannot change ties.
        indices = indices[
            torch.argsort(source_index[indices], descending=False, stable=True)
        ]
        return indices[
            torch.argsort(score[indices].float(), descending=True, stable=True)
        ]

    def top3d_indices(
        self, num_samples: int, *, before_interaction: bool = False
    ) -> list[torch.Tensor]:
        scores = self.score_3d_before if before_interaction else self.score_3d_after
        result = []
        for batch_idx in range(num_samples):
            valid = self.has_xyz & (self.batch_index == batch_idx)
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if len(indices):
                indices = self._rank_task(indices, scores, self.radar_source_index)
            result.append(indices)
        return result

    def top2d_indices(
        self, num_samples: int, *, before_interaction: bool = False
    ) -> list[torch.Tensor]:
        scores = self.score_2d_before if before_interaction else self.score_2d_after
        result = []
        for batch_idx in range(num_samples):
            valid = self.has_box & (self.batch_index == batch_idx)
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if len(indices):
                indices = self._rank_task(indices, scores, self.vision_source_index)
            result.append(indices)
        return result


def validate_batch(batch: Mapping[str, Any]) -> int:
    """Validate compact modality batches and their maps into the full Batch."""
    required = ("points", "delta_t", "sensor_id", "point_counts", "point_batch_index", "num_samples",
                "target_xyz", "target_valid", "gt_box_xyxy_px", "gt_2d_valid", "m_R", "m_V",
                "image_uint8", "image_source_wh", "image_view_wh", "image_scale_xy", "vision_delta_t",
                "radar_batch_index", "vision_batch_index")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Multimodal V2 batch missing {missing}")
    count = int(batch["num_samples"])
    if count <= 0:
        raise ValueError("num_samples must be positive")
    for key in ("target_valid", "gt_2d_valid", "m_R", "m_V"):
        if tuple(batch[key].shape) != (count,) or batch[key].dtype != torch.bool:
            raise ValueError(f"{key} must be bool [{count}]")
    for mask, key in (("m_R", "radar_batch_index"), ("m_V", "vision_batch_index")):
        ids = batch[key]
        expected = torch.nonzero(batch[mask], as_tuple=False).flatten()
        if ids.dtype != torch.long or ids.ndim != 1 or not torch.equal(ids, expected):
            raise ValueError(f"{key} must exactly map the valid {mask} samples in order")
    br, bv = len(batch["radar_batch_index"]), len(batch["vision_batch_index"])
    counts = batch["point_counts"]
    if counts.shape != (br,) or counts.dtype != torch.long or bool((counts <= 0).any()):
        raise ValueError("point_counts must be positive long [Br]")
    n = int(counts.sum())
    if batch["points"].shape != (n, 3) or not torch.isfinite(batch["points"]).all():
        raise ValueError("points must be finite [sum(Nr),3]")
    for key in ("delta_t", "sensor_id", "point_batch_index"):
        if batch[key].shape != (n,):
            raise ValueError(f"{key} must be [sum(Nr)]")
    expected = torch.repeat_interleave(torch.arange(br, device=counts.device), counts)
    if batch["point_batch_index"].dtype != torch.long or not torch.equal(batch["point_batch_index"], expected):
        raise ValueError("point_batch_index must use compact radar-local indices")
    if batch["sensor_id"].dtype != torch.long or bool(((batch["sensor_id"] < 0) | (batch["sensor_id"] > 1)).any()):
        raise ValueError("invalid radar sensor ids")
    if not torch.isfinite(batch["delta_t"]).all() or bool((batch["delta_t"] > 0).any()):
        raise ValueError("radar relative time must be finite and causal")
    for key in ("image_source_wh", "image_view_wh", "image_scale_xy"):
        if batch[key].shape != (bv, 2) or not torch.isfinite(batch[key]).all() or bool((batch[key] <= 0).any()):
            raise ValueError(f"{key} must be finite positive [Bv,2]")
    images = batch["image_uint8"]
    if bv:
        if images is None or images.ndim != 4 or images.shape[:2] != (bv, 3) or images.dtype != torch.uint8:
            raise ValueError("images must be uint8 [Bv,3,H,W]")
        wh = batch["image_view_wh"]
        if bool((wh != wh.new_tensor((images.shape[3], images.shape[2]))).any()):
            raise ValueError("image view size mismatch")
    elif images is not None:
        raise ValueError("zero visual samples must have image_uint8=None")
    if batch["vision_delta_t"].shape != (bv,) or not torch.isfinite(batch["vision_delta_t"]).all() or bool((batch["vision_delta_t"] > 0).any()):
        raise ValueError("vision_delta_t must be finite causal [Bv]")
    if batch["target_xyz"].shape != (count, 3) or batch["gt_box_xyxy_px"].shape != (count, 4):
        raise ValueError("targets must preserve full Sample batch dimensions")
    if bool((batch["gt_2d_valid"] & ~batch["m_V"]).any()):
        raise ValueError("valid box supervision requires a valid image")
    return count
