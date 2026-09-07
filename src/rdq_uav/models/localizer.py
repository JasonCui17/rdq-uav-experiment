from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rdq_uav.models.model import MultiModalClassifier


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class MultiModalLocalizer(MultiModalClassifier):
    """Minimal single-UAV 2D/3D localizer sharing the Stage-3 fusion encoder."""

    def __init__(self, config: dict[str, Any]) -> None:
        classifier_config = dict(config)
        classifier_config["num_classes"] = int(config.get("num_classes", 5))
        classifier_config["auxiliary_position"] = {"enabled": False, "loss_weight": 0.0}
        super().__init__(classifier_config)

        # Remove task-specific Stage-3 heads while keeping exactly the same
        # visual, radar and fusion modules.
        self.classifier = nn.Identity()
        self.position_head = RegressionHead(
            self.fused_dim, self.embed_dim, 3, float(config["dropout"])
        )
        self.box_head = RegressionHead(
            self.fused_dim, self.embed_dim, 4, float(config["dropout"])
        )
        self.bbox_parameterization = str(
            config.get("bbox_parameterization", "sigmoid_cxcywh")
        )
        if self.bbox_parameterization not in {
            "sigmoid_cxcywh",
            "sigmoid_center_log_size",
        }:
            raise ValueError(
                f"Unsupported bbox parameterization: {self.bbox_parameterization}"
            )
        reference = config.get("bbox_reference_wh")
        if self.bbox_parameterization == "sigmoid_center_log_size":
            if reference is None or len(reference) != 2:
                raise ValueError(
                    "sigmoid_center_log_size requires train-derived bbox_reference_wh"
                )
            reference_tensor = torch.tensor(reference, dtype=torch.float32)
            if not bool(torch.all(reference_tensor > 0)):
                raise ValueError("bbox_reference_wh must be positive")
            self.register_buffer("bbox_reference_wh", reference_tensor)
            final_layer = self.box_head.layers[-1]
            if not isinstance(final_layer, nn.Linear):
                raise TypeError("RegressionHead must end with nn.Linear")
            with torch.no_grad():
                # delta_w=delta_h=0 gives the train-median reference size.
                final_layer.bias[2:].zero_()
        else:
            self.register_buffer("bbox_reference_wh", torch.empty(0))

    def _decode_box(self, raw_box: torch.Tensor) -> torch.Tensor:
        if self.bbox_parameterization == "sigmoid_cxcywh":
            return raw_box.sigmoid()
        center = raw_box[..., :2].sigmoid()
        delta_size = raw_box[..., 2:].clamp(-4.0, 4.0)
        size = self.bbox_reference_wh.to(raw_box).mul(delta_size.exp())
        return torch.cat((center, size), dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        radar: torch.Tensor,
        radar_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        encoded = self.forward_features(image, radar, radar_mask, return_attention)
        fused = encoded["features"]
        assert isinstance(fused, torch.Tensor)
        return {
            "box": self._decode_box(self.box_head(fused)),
            "position": self.position_head(fused),
            "attention": encoded["attention"],
            "features": fused,
            "visual_grid": encoded["visual_grid"],
        }
