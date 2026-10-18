"""Expose the four feature stages of the unchanged detrex Swin backbone."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SwinPyramidOutput:
    features: tuple[torch.Tensor, ...]
    dino_features: dict[str, torch.Tensor]


class SwinPyramidAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, *, register_backbone: bool = True) -> None:
        super().__init__()
        if register_backbone:
            self.backbone = backbone
        else:
            object.__setattr__(self, "backbone", backbone)
        required = ("patch_embed", "pos_drop", "layers", "out_indices", "num_features")
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise TypeError(f"unsupported Swin backbone; missing {missing}")
        if len(backbone.layers) != 4:
            raise ValueError("SwinPyramidAdapter requires the four-stage Swin hierarchy")

    def forward(self, images: torch.Tensor) -> SwinPyramidOutput:
        backbone = self.backbone
        x = backbone.patch_embed(images)
        height, width = int(x.shape[2]), int(x.shape[3])
        if backbone.ape:
            position = F.interpolate(backbone.absolute_pos_embed, size=(height, width), mode="bicubic")
            x = (x + position).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        tokens = backbone.pos_drop(x)
        features = []
        for index, layer in enumerate(backbone.layers):
            raw, current_h, current_w, tokens, next_h, next_w = layer(tokens, height, width)
            if (current_h, current_w) != (height, width):
                raise RuntimeError("detrex Swin stage changed its declared current resolution")
            norm = getattr(backbone, f"norm{index}", None)
            exposed = norm(raw) if norm is not None else raw
            channels = int(backbone.num_features[index])
            features.append(exposed.view(-1, height, width, channels).permute(0, 3, 1, 2).contiguous())
            if index < 3:
                height, width = int(next_h), int(next_w)
        return SwinPyramidOutput(tuple(features), {f"p{i}": features[i] for i in backbone.out_indices})
