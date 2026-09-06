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
            "box": self.box_head(fused).sigmoid(),
            "position": self.position_head(fused),
            "attention": encoded["attention"],
            "features": fused,
            "visual_grid": encoded["visual_grid"],
        }
