"""DINO continuation from a writable, shared Swin visual pyramid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import COMPONENTS
from .swin_adapter import PreStageTransform, SwinPyramidAdapter, SwinPyramidOutput


def _inverse_sigmoid(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    value = value.clamp(min=0, max=1)
    return torch.log(value.clamp(min=eps) / (1 - value).clamp(min=eps))


@COMPONENTS.register("dino_swin_t")
class DINOAdapter(nn.Module):
    """Run one shared Swin and continue through the unchanged detrex DINO head.

    The returned ``decoder_query_features`` are the final DINO decoder hidden
    states.  They are the only feature source intended for the later RGB
    CandidateSet; pyramid/ROI features are deliberately not substituted.
    """

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector
        self.swin = SwinPyramidAdapter(detector.backbone, register_backbone=False)

    def forward_from_pyramid(
        self,
        pyramid: SwinPyramidOutput,
        image_masks: torch.Tensor,
    ) -> dict[str, Any]:
        """Continue the official inference graph from p1--p3 features."""

        if self.detector.training:
            raise RuntimeError(
                "P2 DINOAdapter exposes the identity inference path only; "
                "training integration belongs to P8"
            )
        multi_level_features = self.detector.neck(pyramid.dino_features)
        masks = [
            F.interpolate(image_masks[None], size=feature.shape[-2:])
            .to(torch.bool)
            .squeeze(0)
            for feature in multi_level_features
        ]
        positions = [self.detector.position_embedding(mask) for mask in masks]
        (
            decoder_states,
            initial_reference,
            intermediate_references,
            encoder_state,
            encoder_reference,
        ) = self.detector.transformer(
            multi_level_features,
            masks,
            positions,
            (None, None),
            attn_masks=[None, None],
        )
        decoder_states[0] += self.detector.label_enc.weight[0, 0] * 0.0

        classes: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []
        for level in range(decoder_states.shape[0]):
            reference = (
                initial_reference
                if level == 0
                else intermediate_references[level - 1]
            )
            reference = _inverse_sigmoid(reference)
            logits = self.detector.class_embed[level](decoder_states[level])
            residual = self.detector.bbox_embed[level](decoder_states[level])
            if reference.shape[-1] == 4:
                residual = residual + reference
            else:
                if reference.shape[-1] != 2:
                    raise RuntimeError("DINO reference points must have dimension 2 or 4")
                residual = torch.cat(
                    (residual[..., :2] + reference, residual[..., 2:]), dim=-1
                )
            classes.append(logits)
            boxes.append(residual.sigmoid())

        stacked_classes = torch.stack(classes)
        stacked_boxes = torch.stack(boxes)
        encoder_logits = self.detector.transformer.decoder.class_embed[-1](encoder_state)
        return {
            "pred_logits": stacked_classes[-1],
            "pred_boxes": stacked_boxes[-1],
            "decoder_query_features": decoder_states[-1],
            "decoder_features_all_layers": decoder_states,
            "aux_outputs": self.detector._set_aux_loss(stacked_classes, stacked_boxes),
            "enc_outputs": {
                "pred_logits": encoder_logits,
                "pred_boxes": encoder_reference,
            },
            "pyramid": pyramid,
            "multi_level_features": tuple(multi_level_features),
        }

    def forward(
        self,
        batched_inputs: Sequence[Mapping[str, Any]],
        pre_stage_transform: PreStageTransform | None = None,
    ) -> dict[str, Any]:
        images = self.detector.preprocess_image(list(batched_inputs))
        batch_size, _, height, width = images.tensor.shape
        image_masks = images.tensor.new_zeros(batch_size, height, width)
        pyramid = self.swin(images.tensor, pre_stage_transform=pre_stage_transform)
        output = self.forward_from_pyramid(pyramid, image_masks)
        output["image_sizes"] = tuple(images.image_sizes)
        return output
