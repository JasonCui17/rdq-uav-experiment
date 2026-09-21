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
