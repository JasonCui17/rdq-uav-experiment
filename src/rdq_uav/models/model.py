from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rdq_uav.models.backbones import DualViewTokenizer, build_backbone
from rdq_uav.models.fusion import ClassificationHead, CrossAttentionBlock
from rdq_uav.models.position import SinePositionEncoding2D
from rdq_uav.models.radar import MaskedPointMLP


SUPPORTED_VARIANTS = {"rgb", "radar", "concat", "learned_query", "rdq"}


class MultiModalClassifier(nn.Module):
    """One interface for all controlled RGB/radar fusion variants."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.variant = str(config["variant"])
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unknown model variant: {self.variant}")
        self.embed_dim = int(config["embed_dim"])
        self.radar_skip = bool(config.get("radar_skip", True))
        dropout = float(config["dropout"])

        self.visual_tokenizer: DualViewTokenizer | None = None
        self.position_encoding: SinePositionEncoding2D | None = None
        if self.variant != "radar":
            backbone = build_backbone(config["backbone"])
            self.visual_tokenizer = DualViewTokenizer(backbone, self.embed_dim)
            self.position_encoding = SinePositionEncoding2D(self.embed_dim)

        self.radar_encoder: MaskedPointMLP | None = None
        if self.variant != "rgb":
            radar_cfg = config["radar_encoder"]
            self.radar_encoder = MaskedPointMLP(
                input_dim=int(radar_cfg["input_dim"]),
                hidden_dims=[int(value) for value in radar_cfg["hidden_dims"]],
                embed_dim=self.embed_dim,
                dropout=dropout,
            )

        self.cross_attention: CrossAttentionBlock | None = None
        self.query_projection: nn.Module | None = None
        self.learned_query: nn.Parameter | None = None
        if self.variant in {"learned_query", "rdq"}:
            attention_cfg = config["attention"]
            self.cross_attention = CrossAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=int(attention_cfg["num_heads"]),
                dropout=dropout,
                ffn_ratio=int(attention_cfg["ffn_ratio"]),
            )
            if self.variant == "learned_query":
                self.learned_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                nn.init.normal_(self.learned_query, std=0.02)
            else:
                self.query_projection = nn.Sequential(
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )

        if self.variant in {"rgb", "radar"}:
            classifier_input = self.embed_dim
        elif self.variant == "concat":
            classifier_input = 2 * self.embed_dim
        else:
            classifier_input = self.embed_dim * (2 if self.radar_skip else 1)
        self.classifier = ClassificationHead(
            classifier_input,
            self.embed_dim,
            int(config["num_classes"]),
            dropout,
        )
        auxiliary_cfg = config["auxiliary_position"]
        self.auxiliary_position_enabled = bool(auxiliary_cfg["enabled"])
        self.position_head = nn.Linear(classifier_input, 3) if self.auxiliary_position_enabled else None

    def _visual_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.visual_tokenizer is None or self.position_encoding is None:
            raise RuntimeError("Visual branch is disabled")
        tokens, (height, width) = self.visual_tokenizer(images)
        position = self.position_encoding(height, width, tokens.device, tokens.dtype)
        views = images.shape[1]
        position = position.repeat(views, 1)[None, :, :]
        return tokens + position, (height, width)

    def forward(
        self,
        image: torch.Tensor,
        radar: torch.Tensor,
        radar_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        visual_grid = None
        if self.variant != "radar":
            visual_tokens, visual_grid = self._visual_tokens(image)
        else:
            visual_tokens = None
        radar_token = None
        if self.radar_encoder is not None:
            radar_token, _ = self.radar_encoder(radar, radar_mask)

        attention = None
        if self.variant == "rgb":
            assert visual_tokens is not None
            fused = visual_tokens.mean(dim=1)
        elif self.variant == "radar":
            assert radar_token is not None
            fused = radar_token
        elif self.variant == "concat":
            assert visual_tokens is not None and radar_token is not None
            fused = torch.cat((visual_tokens.mean(dim=1), radar_token), dim=-1)
        else:
            assert visual_tokens is not None and radar_token is not None
            assert self.cross_attention is not None
            if self.variant == "learned_query":
                assert self.learned_query is not None
                query = self.learned_query.expand(image.shape[0], -1, -1)
            else:
                assert self.query_projection is not None
                query = self.query_projection(radar_token).unsqueeze(1)
            attended, attention = self.cross_attention(
                query, visual_tokens, need_weights=return_attention
            )
            fused = attended[:, 0]
            if self.radar_skip:
                fused = torch.cat((fused, radar_token), dim=-1)

        logits = self.classifier(fused)
        position = self.position_head(fused) if self.position_head is not None else None
        return {
            "logits": logits,
            "position": position,
            "attention": attention,
            "features": fused,
            "visual_grid": visual_grid,
        }
