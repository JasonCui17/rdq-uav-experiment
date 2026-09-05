from __future__ import annotations

import math

import torch
from torch import nn


class SinePositionEncoding2D(nn.Module):
    """DETR-style 2D sine/cosine positional encoding.

    Design source: facebookresearch/detr ``PositionEmbeddingSine`` (Apache-2.0).
    This implementation is adapted for dense, padding-free feature maps and
    returns flattened tokens for two camera views.
    """

    def __init__(self, embed_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_pos_feats = self.embed_dim // 2
        y = torch.arange(1, height + 1, device=device, dtype=torch.float32)
        x = torch.arange(1, width + 1, device=device, dtype=torch.float32)
        y = y / (height + 1e-6) * (2 * math.pi)
        x = x / (width + 1e-6) * (2 * math.pi)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        dim_t = torch.arange(num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        pos_x = grid_x[..., None] / dim_t
        pos_y = grid_y[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1).reshape(height * width, self.embed_dim).to(dtype=dtype)
