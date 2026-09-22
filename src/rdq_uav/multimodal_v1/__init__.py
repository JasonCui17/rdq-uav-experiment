"""Multimodal V1 components built around frozen, audited backbones."""

from .contracts import InteractionContext, LiDARPyramidContext, ProjectionContext
from .data import (
    LeftImageIndex,
    LeftImageMatch,
    MultimodalQueryDataset,
    collate_multimodal_queries,
    deduplicate_multimodal_queries,
)
from .interaction import (
    GeometryBiHCI,
    GeometryBiHCIStack,
    GeometryEdges,
    GeometryLocal,
    HCIOutput,
    IdentityInteraction,
)
from .model import P5BackboneOutput, P5MultimodalBackbone
from .projection import load_left_projection_context, make_interaction_context
from .registry import COMPONENTS, Registry
from .radar import LiDARV2PyramidAdapter
from .vision import DINOAdapter, SwinPyramidAdapter

__all__ = [
    "COMPONENTS",
    "DINOAdapter",
    "GeometryBiHCI",
    "GeometryBiHCIStack",
    "GeometryEdges",
    "GeometryLocal",
    "HCIOutput",
    "IdentityInteraction",
    "InteractionContext",
    "LeftImageIndex",
    "LeftImageMatch",
    "LiDARPyramidContext",
    "LiDARV2PyramidAdapter",
    "MultimodalQueryDataset",
    "P5BackboneOutput",
    "P5MultimodalBackbone",
    "ProjectionContext",
    "Registry",
    "SwinPyramidAdapter",
    "collate_multimodal_queries",
    "deduplicate_multimodal_queries",
    "load_left_projection_context",
    "make_interaction_context",
]
