"""Typed contracts shared by Multimodal V1 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rdq_uav.lidar_v2.geometry import SparseHierarchy, SparseLevel


@dataclass(frozen=True)
class LiDARPyramidContext:
    """One immutable sparse layout shared by every LiDAR pyramid stage.

    ``prepare`` is the sole hierarchy construction boundary.  Stage methods
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
class InteractionContext:
    """Batch-level context required by Geometry Bi-HCI.

    GeometryLocal receives geometry/calibration through this contract and must
    not hard-code dataset paths or calibration constants.

    ``image_scale_xy`` maps calibrated left-camera pixel coordinates into the
    actual tensor coordinate system consumed by the shared Swin backbone.
    """

    calibration_handle: tuple[str, ...]
    m_R: torch.Tensor
    m_V: torch.Tensor
    image_scale_xy: torch.Tensor

    def validate(self, batch_size: int) -> None:
        if len(self.calibration_handle) != batch_size:
            raise ValueError(
                "calibration_handle length must equal batch size"
            )

        if self.m_R.shape != (batch_size,):
            raise ValueError(
                f"m_R must have shape ({batch_size},), got {tuple(self.m_R.shape)}"
            )

        if self.m_V.shape != (batch_size,):
            raise ValueError(
                f"m_V must have shape ({batch_size},), got {tuple(self.m_V.shape)}"
            )

        if self.image_scale_xy.shape != (batch_size, 2):
            raise ValueError(
                "image_scale_xy must have shape "
                f"({batch_size}, 2), got {tuple(self.image_scale_xy.shape)}"
            )
