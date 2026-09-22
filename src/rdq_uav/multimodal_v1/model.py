"""P5 stage-wise integration of frozen LiDAR V2 and shared Swin backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import InteractionContext
from .interaction import GeometryBiHCIStack
from .radar import LiDARV2PyramidAdapter
from .registry import COMPONENTS
from .vision import SwinPyramidAdapter
from .vision.swin_adapter import SwinPyramidOutput, SwinStageInput


@dataclass(frozen=True)
class P5BackboneOutput:
    """Outputs after three Pre-Stage HCI blocks and both original backbones."""

    radar: dict[str, Any]
    vision: SwinPyramidOutput
    hci_aux: tuple[dict[str, Any] | None, ...]


@COMPONENTS.register("p5_multimodal_backbone")
class P5MultimodalBackbone(nn.Module):
    """Interleave HCI0--2 with the original Radar/Swin stages.

    Ordering is fixed:
      HCI0 -> Radar stage0 / Swin stage0
      merge01 -> HCI1 -> Radar stage1 / Swin stage1
      merge12 -> HCI2 -> Radar stage2 / Swin stage2
      Swin stage3 unchanged

    No P6/P7 candidate association or fused decoder logic is included here.
    """

    def __init__(
        self,
        radar: LiDARV2PyramidAdapter,
        vision: SwinPyramidAdapter,
        interaction: GeometryBiHCIStack,
    ) -> None:
        super().__init__()
        self.radar = radar
        self.vision = vision
        self.interaction = interaction

    def forward(
        self,
        lidar_batch: Mapping[str, Any],
        images: torch.Tensor,
        context: InteractionContext,
        *,
        return_aux: bool = False,
    ) -> P5BackboneOutput:
        if images.ndim != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")
        batch_size = int(images.shape[0])
        if int(lidar_batch["num_samples"]) != batch_size:
            raise ValueError("LiDAR and vision batch sizes must match")
        context.validate(batch_size)

        radar_ctx = self.radar.prepare(lidar_batch)
        current = self.vision.prepare(images)

        levels = (radar_ctx.level0, radar_ctx.level1, radar_ctx.level2)
        radar_pre = radar_ctx.r0_pre
        radar_posts: list[torch.Tensor] = []
        vision_outputs = []
        hci_aux: list[dict[str, Any] | None] = []

        for stage_index in range(3):
            if current.index != stage_index:
                raise RuntimeError(
                    f"expected Swin pre-stage {stage_index}, got {current.index}"
                )
            level = levels[stage_index]
            interaction = self.interaction.forward_stage(
                stage_index,
                radar_pre,
                level.centers,
                level.batch_index,
                current.tokens,
                height=current.height,
                width=current.width,
                context=context,
                return_aux=return_aux,
            )
            hci_aux.append(interaction.aux)

            visual_stage = self.vision.run_stage(
                SwinStageInput(
                    stage_index,
                    interaction.vision_tokens,
                    current.height,
                    current.width,
                )
            )
            vision_outputs.append(visual_stage)

            if stage_index == 0:
                radar_post = self.radar.run_stage0(
                    interaction.radar_features, radar_ctx
                )
                radar_pre = self.radar.merge01(radar_post, radar_ctx)
            elif stage_index == 1:
                radar_post = self.radar.run_stage1(
                    interaction.radar_features, radar_ctx
                )
                radar_pre = self.radar.merge12(radar_post, radar_ctx)
            else:
                radar_post = self.radar.run_stage2(
                    interaction.radar_features, radar_ctx
                )

            radar_posts.append(radar_post)
            if visual_stage.next_input is None:
                raise RuntimeError(f"Swin stage {stage_index} did not expose next_input")
            current = visual_stage.next_input

        # V3 is intentionally not part of the frozen HCI design.
        if current.index != 3:
            raise RuntimeError(f"expected Swin pre-stage 3, got {current.index}")
        stage3 = self.vision.run_stage(current)
        vision_outputs.append(stage3)

        dino_features = {
            f"p{index}": vision_outputs[index].feature
            for index in self.vision.backbone.out_indices
        }
        vision_pyramid = SwinPyramidOutput(
            tuple(vision_outputs),
            dino_features,
        )

        fine = self.radar.decode_to_fine(
            radar_posts[0],
            radar_posts[1],
            radar_posts[2],
            radar_ctx,
        )
        radar_output = self.radar.candidate_head(fine, radar_ctx)
        return P5BackboneOutput(
            radar=radar_output,
            vision=vision_pyramid,
            hci_aux=tuple(hci_aux),
        )
