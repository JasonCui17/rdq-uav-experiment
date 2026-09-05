from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Dependency-light backbone for smoke tests, not for reported results."""

    out_channels = 128

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, self.out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class TimmBackbone(nn.Module):
    """Thin adapter around timm's maintained ``features_only`` API."""

    def __init__(self, name: str, pretrained: bool, out_index: int) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm or use provider=builtin/name=tiny_cnn") from exc
        try:
            self.model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(int(out_index),),
            )
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not create pretrained timm backbone '{name}'. "
                    "Check network/cache, or set model.backbone.pretrained=false."
                ) from exc
            raise
        self.out_channels = int(self.model.feature_info.channels()[0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if not isinstance(features, (list, tuple)) or len(features) != 1:
            raise RuntimeError("Expected one feature map from timm features_only backbone")
        return features[0]


def build_backbone(config: dict) -> nn.Module:
    provider = str(config["provider"])
    name = str(config["name"])
    if provider == "builtin" and name == "tiny_cnn":
        backbone: nn.Module = TinyCNN()
    elif provider == "timm":
        backbone = TimmBackbone(
            name=name,
            pretrained=bool(config["pretrained"]),
            out_index=int(config["out_index"]),
        )
    else:
        raise ValueError(f"Unsupported backbone: provider={provider}, name={name}")
    if not bool(config.get("trainable", True)):
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    return backbone


class DualViewTokenizer(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int) -> None:
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise TypeError("Backbone adapter must expose out_channels")
        self.backbone = backbone
        self.projection = nn.Conv2d(int(backbone.out_channels), embed_dim, kernel_size=1)
        self.camera_embedding = nn.Embedding(2, embed_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 5 or images.shape[1] not in {1, 2} or images.shape[2] != 3:
            raise ValueError(f"Expected image [B,V,3,H,W] with V=1 or 2, got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch * views, channels, height, width))
        features = self.projection(features)
        _, embed_dim, feature_h, feature_w = features.shape
        features = features.reshape(batch, views, embed_dim, feature_h, feature_w)
        tokens = features.flatten(3).permute(0, 1, 3, 2)
        camera_ids = torch.arange(views, device=images.device)
        tokens = tokens + self.camera_embedding(camera_ids)[None, :, None, :]
        return tokens.reshape(batch, views * feature_h * feature_w, embed_dim), (
            feature_h,
            feature_w,
        )
