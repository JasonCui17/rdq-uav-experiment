"""Typed contracts shared by Multimodal V1 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rdq_uav.lidar_v2.geometry import SparseHierarchy, SparseLevel


@dataclass(frozen=True)
class LiDARPyramidContext:
    """One immutable sparse layout shared by every LiDAR pyramid stage.

    ``prepare`` is the sole hierarchy construction boundary. Stage methods
    consume this object and never voxelize the batch again.
    """

    batch: Mapping[str, Any]
    hierarchy: SparseHierarchy
    level0: SparseLevel
    level1: SparseLevel
    level2: SparseLevel
    r0_pre: torch.Tensor

    def assert_hierarchy(self, hierarchy: SparseHierarchy) -> None:
        if hierarchy is not self.hierarchy:
            raise AssertionError("LiDAR pyramid stage received a different hierarchy")


@dataclass(frozen=True)
class ProjectionContext:
    """Tensor-only left-camera geometry consumed by P5.

    The current MMAUD coordinate audit establishes that released LiDAR XYZ and
    the GT reference frame are the same reference frame. Therefore the fitted
    camera-from-GT transform is packed here as camera-from-radar without any
    extra frame conversion.

    Intrinsic order: [xi, fu, fv, pu, pv].
    Distortion order: [k1, k2, p1, p2].
    image_size_wh is the calibrated source image size [width, height].
    image_scale_xy maps calibrated source pixels to the resized DINO tensor.
    """

    rotation_camera_from_radar: torch.Tensor
    translation_camera_from_radar_m: torch.Tensor
    intrinsics: torch.Tensor
    distortion: torch.Tensor
    image_size_wh: torch.Tensor
    image_scale_xy: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.rotation_camera_from_radar.shape[0])

    def validate(self, batch_size: int) -> None:
        expected = {
            "rotation_camera_from_radar": (batch_size, 3, 3),
            "translation_camera_from_radar_m": (batch_size, 3),
            "intrinsics": (batch_size, 5),
            "distortion": (batch_size, 4),
            "image_size_wh": (batch_size, 2),
            "image_scale_xy": (batch_size, 2),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if not bool(torch.isfinite(value.float()).all()):
                raise ValueError(f"{name} must be finite")
        if not bool((self.image_size_wh > 0).all()):
            raise ValueError("image_size_wh must be positive")
        if not bool((self.image_scale_xy > 0).all()):
            raise ValueError("image_scale_xy must be positive")


@dataclass(frozen=True)
class InteractionContext:
    """Batch-level context required by Geometry Bi-HCI.

    File IO is deliberately outside the HCI forward path. calibration_handle
    is provenance only; projection carries the parsed tensor geometry.
    """

    calibration_handle: tuple[str, ...]
    m_R: torch.Tensor
    m_V: torch.Tensor
    projection: ProjectionContext

    @property
    def image_scale_xy(self) -> torch.Tensor:
        return self.projection.image_scale_xy

    def validate(self, batch_size: int) -> None:
        if len(self.calibration_handle) != batch_size:
            raise ValueError("calibration_handle length must equal batch size")
        if self.m_R.shape != (batch_size,):
            raise ValueError(
                f"m_R must have shape ({batch_size},), got {tuple(self.m_R.shape)}"
            )
        if self.m_V.shape != (batch_size,):
            raise ValueError(
                f"m_V must have shape ({batch_size},), got {tuple(self.m_V.shape)}"
            )
        if self.m_R.dtype != torch.bool or self.m_V.dtype != torch.bool:
            raise TypeError("m_R and m_V must be bool tensors")
        self.projection.validate(batch_size)
