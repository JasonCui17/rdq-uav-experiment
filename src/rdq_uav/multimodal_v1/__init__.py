"""Multimodal V1 components built around frozen, audited backbones."""

from .contracts import LiDARPyramidContext
from .data import (
    LeftImageIndex,
    LeftImageMatch,
    MultimodalQueryDataset,
    collate_multimodal_queries,
    deduplicate_multimodal_queries,
)
from .registry import COMPONENTS, Registry
from .radar import LiDARV2PyramidAdapter
from .vision import DINOAdapter, SwinPyramidAdapter

__all__ = [
    "COMPONENTS",
    "LiDARPyramidContext",
    "LiDARV2PyramidAdapter",
    "DINOAdapter",
    "LeftImageIndex",
    "LeftImageMatch",
    "MultimodalQueryDataset",
    "Registry",
    "SwinPyramidAdapter",
    "collate_multimodal_queries",
    "deduplicate_multimodal_queries",
]
