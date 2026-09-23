"""P2 writable Swin stage seam tests independent of optional detrex binaries."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rdq_uav.multimodal_v1 import COMPONENTS, DINOAdapter, SwinPyramidAdapter


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 4, 4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class FakeStage(nn.Module):
    def __init__(self, channels: int, downsample: bool) -> None:
        super().__init__()
        self.channels = channels
        self.downsample = downsample
        self.transform = nn.Linear(channels, channels)
        if downsample:
            self.merge = nn.Linear(channels * 4, channels * 2)

    def forward(self, tokens: torch.Tensor, height: int, width: int):
        output = self.transform(tokens)
        if not self.downsample:
            return output, height, width, output, height, width
        grid = output.view(output.shape[0], height, width, self.channels)
        merged = torch.cat(
            (
                grid[:, 0::2, 0::2],
                grid[:, 1::2, 0::2],
                grid[:, 0::2, 1::2],
                grid[:, 1::2, 1::2],
            ),
            dim=-1,
        )
        next_tokens = self.merge(merged).flatten(1, 2)
        return output, height, width, next_tokens, height // 2, width // 2


class FakeSwin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.ape = False
        self.pos_drop = nn.Dropout(0.0)
        self.layers = nn.ModuleList(
            [FakeStage(4, True), FakeStage(8, True), FakeStage(16, True), FakeStage(32, False)]
        )
        self.out_indices = (1, 2, 3)
        self.num_features = (4, 8, 16, 32)
        self.norm1 = nn.LayerNorm(8)
        self.norm2 = nn.LayerNorm(16)
        self.norm3 = nn.LayerNorm(32)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        current = SwinPyramidAdapter(self).prepare(images)
        output = {}
        for index, layer in enumerate(self.layers):
            raw, height, width, tokens, next_height, next_width = layer(
                current.tokens, current.height, current.width
            )
            if index in self.out_indices:
                normalized = getattr(self, f"norm{index}")(raw)
                output[f"p{index}"] = normalized.view(
                    -1, height, width, self.num_features[index]
                ).permute(0, 3, 1, 2).contiguous()
            if index < 3:
                current = type(current)(index + 1, tokens, next_height, next_width)
        return output


class SwinPyramidAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.backbone = FakeSwin().eval()
        self.adapter = SwinPyramidAdapter(self.backbone).eval()
        self.images = torch.randn(2, 3, 32, 32)

    def test_identity_and_shape_contract(self) -> None:
        with torch.no_grad():
            reference = self.backbone(self.images)
            result = self.adapter(self.images)
        self.assertEqual(
            [tuple(value.shape) for value in result.features],
            [(2, 4, 8, 8), (2, 8, 4, 4), (2, 16, 2, 2), (2, 32, 1, 1)],
        )
        for key in ("p1", "p2", "p3"):
            self.assertTrue(torch.equal(reference[key], result.dino_features[key]))

    def test_identity_writeback_is_exact(self) -> None:
        with torch.no_grad():
            plain = self.adapter(self.images)
            intercepted = self.adapter(
                self.images, pre_stage_transform=lambda stage: stage.tokens
            )
        for expected, actual in zip(plain.features, intercepted.features, strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_modified_pre_stage_tokens_continue_through_original_stages(self) -> None:
        def replace(stage):
            return stage.tokens + 0.25 if stage.index == 2 else stage.tokens

        with torch.no_grad():
            plain = self.adapter(self.images)
            modified = self.adapter(self.images, pre_stage_transform=replace)
        self.assertFalse(torch.equal(plain.features[2], modified.features[2]))
        self.assertTrue(torch.isfinite(modified.features[3]).all())

    def test_registry_builds_shared_adapter(self) -> None:
        built = COMPONENTS.build("swin_pyramid", backbone=self.backbone)
        self.assertIs(built.backbone, self.backbone)

    def test_image_list_padding_mask_uses_each_valid_image_size(self) -> None:
        images=SimpleNamespace(tensor=torch.zeros(2,3,8,12),image_sizes=[(8,12),(4,6)])
        mask=DINOAdapter._image_masks(images).to(torch.bool)
        self.assertFalse(bool(mask[0].any()))
        self.assertFalse(bool(mask[1,:4,:6].any()))
        self.assertTrue(bool(mask[1,4:,:].all()))
        self.assertTrue(bool(mask[1,:,:][...,6:].all()))


if __name__ == "__main__":
    unittest.main()
