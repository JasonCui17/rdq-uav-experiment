"""Writable stage seams over the official detrex Swin backbone.

The adapter owns no second backbone.  It calls the exact patch embedding,
BasicLayer instances, and output normalizations owned by the supplied detrex
SwinTransformer.  A caller may replace ``SwinStageInput.tokens`` before any
stage and then continue through the original layers; this is the seam reserved
for Pre-Stage HCI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import COMPONENTS


@dataclass(frozen=True)
class SwinStageInput:
    """Token grid immediately before one original Swin ``BasicLayer``."""

    index: int
    tokens: torch.Tensor
    height: int
    width: int


@dataclass(frozen=True)
class SwinStageOutput:
    """Current-stage feature plus the downsampled input to the next stage."""

    index: int
    pre: SwinStageInput
    raw_tokens: torch.Tensor
    feature: torch.Tensor
    next_input: SwinStageInput | None


@dataclass(frozen=True)
class SwinPyramidOutput:
    """One shared Swin execution exposed at all four true stage boundaries."""

    stages: tuple[SwinStageOutput, ...]
    dino_features: dict[str, torch.Tensor]

    @property
    def features(self) -> tuple[torch.Tensor, ...]:
        return tuple(stage.feature for stage in self.stages)


PreStageTransform = Callable[[SwinStageInput], torch.Tensor | SwinStageInput]


@COMPONENTS.register("swin_pyramid")
class SwinPyramidAdapter(nn.Module):
    """Execute an existing detrex Swin backbone one writable stage at a time."""

    def __init__(self, backbone: nn.Module, *, register_backbone: bool = True) -> None:
        super().__init__()
        if register_backbone:
            self.backbone = backbone
        else:
            # DINOAdapter already owns this exact module through detector. Keep
            # a plain reference so state_dict has one canonical backbone path.
            object.__setattr__(self, "backbone", backbone)
        required = ("patch_embed", "pos_drop", "layers", "out_indices", "num_features")
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise TypeError(f"unsupported Swin backbone; missing {missing}")
        if len(backbone.layers) != 4:
            raise ValueError("SwinPyramidAdapter requires the four-stage Swin hierarchy")

    def prepare(self, images: torch.Tensor) -> SwinStageInput:
        """Run the unchanged patch embedding and return the real V0 input."""

        x = self.backbone.patch_embed(images)
        height, width = int(x.shape[2]), int(x.shape[3])
        if self.backbone.ape:
            position = F.interpolate(
                self.backbone.absolute_pos_embed,
                size=(height, width),
                mode="bicubic",
            )
            x = (x + position).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        return SwinStageInput(0, self.backbone.pos_drop(x), height, width)

    def run_stage(self, stage_input: SwinStageInput) -> SwinStageOutput:
        """Run exactly one original Swin stage from caller-supplied tokens."""

        index = stage_input.index
        if not 0 <= index < len(self.backbone.layers):
            raise IndexError(f"invalid Swin stage index {index}")
        tokens = stage_input.tokens
        expected = stage_input.height * stage_input.width
        if tokens.ndim != 3 or tokens.shape[1] != expected:
            raise ValueError(
                f"stage {index} expects [B,{expected},C] tokens, got {tuple(tokens.shape)}"
            )
        raw, height, width, next_tokens, next_height, next_width = self.backbone.layers[index](
            tokens, stage_input.height, stage_input.width
        )
        if (height, width) != (stage_input.height, stage_input.width):
            raise RuntimeError("detrex Swin stage changed its declared current resolution")

        # The official DINO checkpoint only creates norm1/norm2/norm3 because
        # out_indices=(1,2,3). V0 is therefore the exact raw stage-0 output.
        norm = getattr(self.backbone, f"norm{index}", None)
        exposed = norm(raw) if norm is not None else raw
        channels = int(self.backbone.num_features[index])
        feature = (
            exposed.view(-1, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        following = None
        if index + 1 < len(self.backbone.layers):
            following = SwinStageInput(
                index + 1, next_tokens, int(next_height), int(next_width)
            )
        return SwinStageOutput(index, stage_input, raw, feature, following)

    def forward(
        self,
        images: torch.Tensor,
        pre_stage_transform: PreStageTransform | None = None,
    ) -> SwinPyramidOutput:
        """Return V0--V3 and the exact p1--p3 mapping consumed by DINO."""

        current = self.prepare(images)
        outputs: list[SwinStageOutput] = []
        for index in range(4):
            if pre_stage_transform is not None:
                replacement = pre_stage_transform(current)
                if isinstance(replacement, SwinStageInput):
                    if replacement.index != index:
                        raise ValueError("pre-stage replacement changed the stage identity")
                    current = replacement
                else:
                    current = SwinStageInput(
                        index, replacement, current.height, current.width
                    )
            result = self.run_stage(current)
            outputs.append(result)
            if index < 3:
                if result.next_input is None:
                    raise RuntimeError(f"Swin stage {index} did not produce the next input")
                current = result.next_input

        dino_features = {
            f"p{index}": outputs[index].feature
            for index in self.backbone.out_indices
        }
        return SwinPyramidOutput(tuple(outputs), dino_features)
