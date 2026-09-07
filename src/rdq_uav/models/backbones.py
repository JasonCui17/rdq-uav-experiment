from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class MinimalTopDownFusion(nn.Module):
    """Minimal stride16-to-stride8 additive fusion without an FPN neck."""

    def __init__(self, stride8_channels: int, stride16_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.stride8_projection = nn.Conv2d(stride8_channels, embed_dim, kernel_size=1)
        self.stride16_projection = nn.Conv2d(stride16_channels, embed_dim, kernel_size=1)

    def forward(self, stride8: torch.Tensor, stride16: torch.Tensor) -> torch.Tensor:
        shallow = self.stride8_projection(stride8)
        deep = self.stride16_projection(stride16)
        deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        return shallow + deep


class TimmMinimalFPNBackbone(nn.Module):
    """ResNet stride8 + stride16 top-down fusion exposed as one stride8 feature map."""

    def __init__(self, name: str, pretrained: bool, embed_dim: int) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use minimal_fpn") from exc
        try:
            self.model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(2, 3),
            )
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not create pretrained timm backbone '{name}' for minimal_fpn"
                ) from exc
            raise
        channels = self.model.feature_info.channels()
        if len(channels) != 2:
            raise RuntimeError(f"Expected two feature levels, got channels={channels}")
        self.fusion = MinimalTopDownFusion(int(channels[0]), int(channels[1]), embed_dim)
        self.out_channels = int(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise RuntimeError("minimal_fpn expected stride8 and stride16 feature maps")
        return self.fusion(features[0], features[1])


def build_backbone(config: dict) -> nn.Module:
    provider = str(config["provider"])
    name = str(config["name"])
    if provider == "builtin" and name == "tiny_cnn":
        backbone: nn.Module = TinyCNN()
    elif provider == "timm":
        fusion = str(config.get("fusion", "none"))
        if fusion == "none":
            backbone = TimmBackbone(
                name=name,
                pretrained=bool(config["pretrained"]),
                out_index=int(config["out_index"]),
            )
        elif fusion == "minimal_fpn":
            backbone = TimmMinimalFPNBackbone(
                name=name,
                pretrained=bool(config["pretrained"]),
                embed_dim=int(config["embed_dim"]),
            )
        else:
            raise ValueError(f"Unsupported backbone fusion: {fusion}")
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
