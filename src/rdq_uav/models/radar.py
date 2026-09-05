from __future__ import annotations

import torch
from torch import nn


class MaskedPointMLP(nn.Module):
    """Point-wise MLP followed by padding-safe masked max pooling."""

    def __init__(self, input_dim: int, hidden_dims: list[int], embed_dim: int, dropout: float) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims, embed_dim]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(dimensions) - 2:
                layers.extend((nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)))
        self.point_mlp = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or mask.ndim != 2 or points.shape[:2] != mask.shape:
            raise ValueError(
                f"Expected points [B,N,D] and mask [B,N], got {tuple(points.shape)}, {tuple(mask.shape)}"
            )
        point_features = self.point_mlp(points)
        masked = point_features.masked_fill(~mask[..., None], torch.finfo(point_features.dtype).min)
        pooled = masked.amax(dim=1)
        empty = ~mask.any(dim=1)
        pooled = torch.where(empty[:, None], torch.zeros_like(pooled), pooled)
        return self.output_norm(pooled), point_features
