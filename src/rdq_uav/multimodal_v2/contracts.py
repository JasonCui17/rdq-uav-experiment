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

    def __post_init__(self) -> None:
        n = int(self.score.shape[0])
        shapes = {
            "feature": (n, 128), "score": (n,), "xyz_m": (n, 3),
            "has_xyz": (n,), "box_xyxy_px": (n, 4), "has_box": (n,),
            "batch_index": (n,), "source_index": (n,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(value.shape)}")
        if self.has_xyz.dtype != torch.bool or self.has_box.dtype != torch.bool:
            raise TypeError("has_xyz and has_box must be bool")
        if self.batch_index.dtype != torch.long or self.source_index.dtype != torch.long:
            raise TypeError("candidate indices must be torch.long")
        if self.source == "R":
            if n and (not bool(self.has_xyz.all()) or bool(self.has_box.any())):
                raise ValueError("R candidates require XYZ and cannot claim a 2D box")
        elif self.source == "V":
            if n and (bool(self.has_xyz.any()) or not bool(self.has_box.all())):
                raise ValueError("V candidates require a 2D box and cannot claim XYZ")
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
        )

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
    score_before: torch.Tensor
    score_after: torch.Tensor
    score_delta_logit: torch.Tensor
    xyz_m: torch.Tensor
    has_xyz: torch.Tensor
    box_xyxy_px: torch.Tensor
    has_box: torch.Tensor
    batch_index: torch.Tensor
    hypothesis_type: torch.Tensor  # 0=RV, 1=R, 2=V
    radar_source_index: torch.Tensor
    vision_source_index: torch.Tensor
    attention_token_count: torch.Tensor
    evidence_gate_weight: torch.Tensor
    radar_candidates: CandidateBatch
    vision_candidates: CandidateBatch
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        n = int(self.score_after.numel())
        shapes = {
            "score_before": (n,), "score_delta_logit": (n,),
            "xyz_m": (n, 3), "has_xyz": (n,), "box_xyxy_px": (n, 4),
            "has_box": (n,), "batch_index": (n,), "hypothesis_type": (n,),
            "radar_source_index": (n,), "vision_source_index": (n,),
            "attention_token_count": (n,), "evidence_gate_weight": (n,),
        }
        for name, shape in shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if n:
            v_only = self.hypothesis_type == 2
            if bool(self.has_xyz[v_only].any()):
                raise ValueError("V-only hypotheses may not expose absolute XYZ")

    def top3d_indices(self, num_samples: int) -> list[torch.Tensor]:
        result = []
        for batch_idx in range(num_samples):
            valid = self.has_xyz & (self.batch_index == batch_idx)
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if len(indices):
                indices = indices[torch.argsort(self.score_after[indices].float(), descending=True, stable=True)]
            result.append(indices)
        return result


def validate_batch(batch: Mapping[str, Any]) -> int:
    required = ("points", "point_batch_index", "num_samples", "target_xyz", "target_valid",
                "m_R", "m_V", "image_uint8", "image_source_wh", "image_scale_xy")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Multimodal V2 batch missing {missing}")
    count = int(batch["num_samples"])
    for key in ("target_valid", "m_R", "m_V"):
        if tuple(batch[key].shape) != (count,) or batch[key].dtype != torch.bool:
            raise ValueError(f"{key} must be bool [{count}]")
    return count
