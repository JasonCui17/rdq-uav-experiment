"""Geometry-guided bidirectional local cross-attention for P5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..contracts import InteractionContext
from ..registry import COMPONENTS
from .geometry_local import GeometryEdges, GeometryLocal


@dataclass(frozen=True)
class HCIOutput:
    radar_features: torch.Tensor
    vision_tokens: torch.Tensor
    aux: dict[str, Any] | None = None


class LocalCrossAttention(nn.Module):
    """One-query grouped MHA with a learned 3x3 relative-position bias."""

    def __init__(self, dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 9))

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        valid_mask: torch.Tensor,
        relative_index: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[1] != self.dim:
            raise ValueError(f"query must be [M,{self.dim}]")
        if (
            key_value.ndim != 3
            or key_value.shape[0] != len(query)
            or key_value.shape[2] != self.dim
        ):
            raise ValueError(f"key_value must be [M,K,{self.dim}]")
        if valid_mask.shape != key_value.shape[:2]:
            raise ValueError("valid_mask must be [M,K]")
        if relative_index.shape != valid_mask.shape:
            raise ValueError("relative_index must be [M,K]")
        if len(query) == 0:
            return query.new_empty((0, self.dim))
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("each attention group must contain at least one valid key")
        if int(relative_index.min()) < 0 or int(relative_index.max()) > 8:
            raise ValueError("relative_index must lie in [0,8]")

        groups, keys = valid_mask.shape
        q = self.q_proj(query).view(groups, self.num_heads, self.head_dim)
        k = (
            self.k_proj(key_value)
            .view(groups, keys, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(key_value)
            .view(groups, keys, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        logits = torch.einsum("mhd,mhkd->mhk", q, k) * self.scale
        bias = self.relative_bias.unsqueeze(0).expand(groups, -1, -1).gather(
            2, relative_index[:, None, :].expand(-1, self.num_heads, -1)
        )
        logits = logits + bias
        logits = logits.masked_fill(
            ~valid_mask[:, None, :], torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=-1)
        output = torch.einsum("mhk,mhkd->mhd", weights, v).reshape(groups, self.dim)
        return self.out_proj(output)


def _pad_groups(
    group_index: torch.Tensor,
    member_index: torch.Tensor,
    relative_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a sparse edge list into padded groups without dropping members."""

    if not (
        group_index.ndim == member_index.ndim == relative_index.ndim == 1
        and len(group_index) == len(member_index) == len(relative_index)
    ):
        raise ValueError(
            "group_index, member_index and relative_index must be equal 1D tensors"
        )
    if len(group_index) == 0:
        device = group_index.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return (
            empty,
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0, 0), dtype=torch.bool, device=device),
            torch.empty((0, 0), dtype=torch.long, device=device),
        )

    unique, inverse, counts = torch.unique(
        group_index, sorted=True, return_inverse=True, return_counts=True
    )
    order = torch.argsort(inverse)
    sorted_group = inverse[order]
    sorted_member = member_index[order]
    sorted_relative = relative_index[order]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(len(order), device=order.device) - torch.repeat_interleave(
        starts, counts
    )
    max_count = int(counts.max().item())

    member_padded = torch.zeros(
        (len(unique), max_count), dtype=torch.long, device=group_index.device
    )
    relative_padded = torch.zeros_like(member_padded)
    valid = torch.zeros(
        (len(unique), max_count), dtype=torch.bool, device=group_index.device
    )
    member_padded[sorted_group, positions] = sorted_member
    relative_padded[sorted_group, positions] = sorted_relative
    valid[sorted_group, positions] = True
    return unique, member_padded, valid, relative_padded


class GeometryBiHCI(nn.Module):
    """One frozen-spec Pre-Stage Geometry Bi-HCI block."""

    def __init__(
        self,
        *,
        vision_dim: int,
        feature_stride: int,
        radar_dim: int = 128,
        interaction_dim: int = 128,
        num_heads: int = 4,
        gate_bias_init: float = -4.6,
    ) -> None:
        super().__init__()
        if radar_dim != interaction_dim:
            raise ValueError("P5 requires native radar_dim == interaction_dim")
        self.radar_dim = int(radar_dim)
        self.vision_dim = int(vision_dim)
        self.interaction_dim = int(interaction_dim)
        self.feature_stride = int(feature_stride)

        self.geometry = GeometryLocal(feature_stride)
        self.vision_to_interaction = nn.Linear(vision_dim, interaction_dim)
        self.vision_from_interaction = nn.Linear(interaction_dim, vision_dim)
        self.v_to_r = LocalCrossAttention(interaction_dim, num_heads)
        self.r_to_v = LocalCrossAttention(interaction_dim, num_heads)
        self.radar_gate = nn.Linear(radar_dim * 2, 1)
        self.vision_gate = nn.Linear(vision_dim * 2, 1)
        nn.init.zeros_(self.radar_gate.weight)
        nn.init.zeros_(self.vision_gate.weight)
        nn.init.constant_(self.radar_gate.bias, gate_bias_init)
        nn.init.constant_(self.vision_gate.bias, gate_bias_init)

    def forward(
        self,
        radar_features: torch.Tensor,
        radar_centers: torch.Tensor,
        radar_batch_index: torch.Tensor,
        vision_tokens: torch.Tensor,
        *,
        height: int,
        width: int,
        context: InteractionContext,
        return_aux: bool = False,
    ) -> HCIOutput:
        if radar_features.ndim != 2 or radar_features.shape[1] != self.radar_dim:
            raise ValueError(f"radar_features must be [N,{self.radar_dim}]")
        if radar_centers.shape != (len(radar_features), 3):
            raise ValueError("radar_centers must be [N,3]")
        if radar_batch_index.shape != (len(radar_features),):
            raise ValueError("radar_batch_index must be [N]")
        if vision_tokens.ndim != 3 or vision_tokens.shape[2] != self.vision_dim:
            raise ValueError(f"vision_tokens must be [B,HW,{self.vision_dim}]")
        if vision_tokens.shape[1] != height * width:
            raise ValueError("vision token count must equal height * width")
        if (
            radar_features.device != vision_tokens.device
            or radar_centers.device != radar_features.device
        ):
            raise ValueError("radar and vision features must share a device")

        batch_size, vision_count, _ = vision_tokens.shape
        edges = self.geometry(
            radar_centers,
            radar_batch_index,
            batch_size=batch_size,
            feature_height=height,
            feature_width=width,
            context=context,
        )
        if edges.edge_count == 0:
            aux = None
            if return_aux:
                aux = {
                    "valid_projection_mask": edges.valid_projection_mask,
                    "active_radar_mask": edges.active_radar_mask,
                    "edge_count": 0,
                    "radar_neighbor_count": torch.zeros(
                        len(radar_features), dtype=torch.long, device=radar_features.device
                    ),
                    "vision_support_count": torch.zeros(
                        (batch_size, vision_count),
                        dtype=torch.long,
                        device=vision_tokens.device,
                    ),
                    "gate_R": radar_features.new_zeros(len(radar_features)),
                    "gate_V": vision_tokens.new_zeros((batch_size, vision_count)),
                }
            return HCIOutput(radar_features, vision_tokens, aux)

        vision_interaction = self.vision_to_interaction(vision_tokens)
        vision_interaction_flat = vision_interaction.reshape(
            batch_size * vision_count, self.interaction_dim
        )
        vision_native_flat = vision_tokens.reshape(
            batch_size * vision_count, self.vision_dim
        )

        # Both directions read the same original R/V tensors.
        radar_ids, vision_members, vr_valid, vr_relative = _pad_groups(
            edges.radar_index,
            edges.vision_index,
            edges.relative_index_v_to_r,
        )
        radar_delta_active = self.v_to_r(
            radar_features[radar_ids],
            vision_interaction_flat[vision_members],
            vr_valid,
            vr_relative,
        )
        radar_gate_active = torch.sigmoid(
            self.radar_gate(
                torch.cat((radar_features[radar_ids], radar_delta_active), dim=-1)
            )
        )
        radar_update = radar_gate_active * radar_delta_active
        radar_delta_full = torch.zeros_like(radar_features).index_add(
            0, radar_ids, radar_update.to(radar_features.dtype)
        )
        radar_out = radar_features + radar_delta_full

        vision_ids, radar_members, rv_valid, rv_relative = _pad_groups(
            edges.vision_index,
            edges.radar_index,
            edges.relative_index_r_to_v,
        )
        vision_delta_interaction = self.r_to_v(
            vision_interaction_flat[vision_ids],
            radar_features[radar_members],
            rv_valid,
            rv_relative,
        )
        vision_delta_native = self.vision_from_interaction(vision_delta_interaction)
        vision_gate_active = torch.sigmoid(
            self.vision_gate(
                torch.cat((vision_native_flat[vision_ids], vision_delta_native), dim=-1)
            )
        )
        vision_update = vision_gate_active * vision_delta_native
        vision_delta_full = torch.zeros_like(vision_native_flat).index_add(
            0, vision_ids, vision_update.to(vision_native_flat.dtype)
        )
        vision_out = (vision_native_flat + vision_delta_full).view_as(vision_tokens)

        aux = None
        if return_aux:
            radar_neighbor_count = torch.bincount(
                edges.radar_index, minlength=len(radar_features)
            )
            vision_support_count = torch.bincount(
                edges.vision_index, minlength=batch_size * vision_count
            ).view(batch_size, vision_count)
            gate_r_full = radar_features.new_zeros(len(radar_features)).index_add(
                0, radar_ids, radar_gate_active.squeeze(-1).to(radar_features.dtype)
            )
            gate_v_flat = vision_tokens.new_zeros(
                batch_size * vision_count
            ).index_add(0, vision_ids, vision_gate_active.squeeze(-1).to(vision_tokens.dtype))
            aux = {
                "valid_projection_mask": edges.valid_projection_mask,
                "active_radar_mask": edges.active_radar_mask,
                "edge_count": edges.edge_count,
                "radar_neighbor_count": radar_neighbor_count,
                "vision_support_count": vision_support_count,
                "gate_R": gate_r_full,
                "gate_V": gate_v_flat.view(batch_size, vision_count),
            }

        return HCIOutput(radar_out, vision_out, aux)


@COMPONENTS.register("geometry_bi_hci")
class GeometryBiHCIStack(nn.Module):
    """Three independent same-level HCI blocks: R0/V0, R1/V1, R2/V2."""

    def __init__(
        self,
        vision_dims: tuple[int, int, int] = (96, 192, 384),
        feature_strides: tuple[int, int, int] = (4, 8, 16),
        radar_dim: int = 128,
        interaction_dim: int = 128,
        num_heads: int = 4,
        gate_bias_init: float = -4.6,
    ) -> None:
        super().__init__()
        if len(vision_dims) != 3 or len(feature_strides) != 3:
            raise ValueError("P5 requires exactly three same-level HCI stages")
        self.stages = nn.ModuleList(
            [
                GeometryBiHCI(
                    vision_dim=vision_dim,
                    feature_stride=stride,
                    radar_dim=radar_dim,
                    interaction_dim=interaction_dim,
                    num_heads=num_heads,
                    gate_bias_init=gate_bias_init,
                )
                for vision_dim, stride in zip(vision_dims, feature_strides, strict=True)
            ]
        )

    def forward_stage(self, index: int, *args: Any, **kwargs: Any) -> HCIOutput:
        if not 0 <= index < 3:
            raise IndexError(f"invalid HCI stage {index}")
        return self.stages[index](*args, **kwargs)
