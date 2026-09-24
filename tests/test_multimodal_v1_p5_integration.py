"""P5 stage integration tests over real LiDAR V2 code and a small Swin seam."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from rdq_uav.lidar_v2 import LiDARUAVDetector
from rdq_uav.multimodal_v1 import (
    GeometryBiHCIStack,
    InteractionContext,
    LiDARV2PyramidAdapter,
    P5MultimodalBackbone,
    ProjectionContext,
    SwinPyramidAdapter,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/lidar_uav_v2.yaml").read_text())


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
            [
                FakeStage(4, True),
                FakeStage(8, True),
                FakeStage(16, True),
                FakeStage(32, False),
            ]
        )
        self.out_indices = (1, 2, 3)
        self.num_features = (4, 8, 16, 32)
        self.norm1 = nn.LayerNorm(8)
        self.norm2 = nn.LayerNorm(16)
        self.norm3 = nn.LayerNorm(32)


def synthetic_batch() -> dict[str, object]:
    points = torch.tensor(
        [
            [-1.0, -1.0, 5.0],
            [-0.4, -0.2, 5.0],
            [0.2, 0.1, 5.0],
            [0.8, 0.7, 5.0],
            [1.1, -0.8, 5.0],
            [-1.2, 0.9, 5.0],
        ],
        dtype=torch.float32,
    )
    return {
        "points": points,
        "point_batch_index": torch.zeros(len(points), dtype=torch.long),
        "sensor_id": torch.tensor([0, 1, 0, 1, 0, 1]),
        "delta_t": torch.linspace(-0.5, -0.1, len(points)),
        "num_samples": 1,
    }


def context(m_v: bool) -> InteractionContext:
    return InteractionContext(
        calibration_handle=("synthetic",),
        m_R=torch.tensor([True]),
        m_V=torch.tensor([m_v]),
        projection=ProjectionContext(
            rotation_camera_from_radar=torch.eye(3).unsqueeze(0),
            translation_camera_from_radar_m=torch.zeros(1, 3),
            intrinsics=torch.tensor([[0.0, 16.0, 16.0, 16.0, 16.0]]),
            distortion=torch.zeros(1, 4),
            image_size_wh=torch.tensor([[32.0, 32.0]]),
            image_scale_xy=torch.ones(1, 2),
        ),
    )


class P5IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        detector = LiDARUAVDetector(CFG).eval()
        self.radar = LiDARV2PyramidAdapter(detector=detector).eval()
        self.swin = SwinPyramidAdapter(FakeSwin().eval()).eval()
        self.hci = GeometryBiHCIStack(
            vision_dims=(4, 8, 16),
            feature_strides=(4, 8, 16),
        ).eval()
        self.model = P5MultimodalBackbone(
            radar=self.radar,
            vision=self.swin,
            interaction=self.hci,
        ).eval()
        self.batch = synthetic_batch()
        self.images = torch.randn(1, 3, 32, 32)

    def test_missing_vision_is_exact_identity_for_both_backbones(self):
        with torch.no_grad():
            reference_radar = self.radar(self.batch)
            reference_vision = self.swin(self.images)
            actual = self.model(
                self.batch,
                self.images,
                context(False),
                return_aux=True,
            )

        for field in (
            "logits",
            "residual_xyz",
            "pred_xyz",
            "fine_features",
            "voxel_centers",
            "batch_index",
            "source_token_id",
        ):
            self.assertTrue(torch.equal(reference_radar[field], actual.radar[field]))
        for key in ("p1", "p2", "p3"):
            self.assertTrue(
                torch.equal(
                    reference_vision.dino_features[key],
                    actual.vision.dino_features[key],
                )
            )
        self.assertEqual([aux["edge_count"] for aux in actual.hci_aux], [0, 0, 0])

    def test_present_modalities_run_all_three_hci_stages(self):
        with torch.no_grad():
            output = self.model(
                self.batch,
                self.images,
                context(True),
                return_aux=True,
            )
        self.assertEqual(len(output.hci_aux), 3)
        self.assertTrue(all(aux["edge_count"] > 0 for aux in output.hci_aux))
        self.assertTrue(torch.isfinite(output.radar["fine_features"]).all())
        for feature in output.vision.features:
            self.assertTrue(torch.isfinite(feature).all())

    def test_disabling_diagnostics_preserves_backbone_outputs(self):
        with torch.no_grad():
            diagnostic = self.model(self.batch, self.images, context(True), return_aux=True)
            lean = self.model(self.batch, self.images, context(True), return_aux=False)
        for field in ("logits", "residual_xyz", "pred_xyz", "fine_features"):
            self.assertTrue(torch.equal(diagnostic.radar[field], lean.radar[field]))
        for expected, actual in zip(diagnostic.vision.features, lean.vision.features):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(all(value is None for value in lean.hci_aux))

    def test_padded_visual_cells_are_not_used(self):
        # Calibrated image is only 28px wide but the Swin input is 32px wide.
        # A projection near the valid right edge may not expand into padded cells.
        local_context = context(True)
        local_context = InteractionContext(
            calibration_handle=local_context.calibration_handle,
            m_R=local_context.m_R,
            m_V=local_context.m_V,
            projection=ProjectionContext(
                rotation_camera_from_radar=local_context.projection.rotation_camera_from_radar,
                translation_camera_from_radar_m=local_context.projection.translation_camera_from_radar_m,
                intrinsics=local_context.projection.intrinsics,
                distortion=local_context.projection.distortion,
                image_size_wh=torch.tensor([[28.0, 32.0]]),
                image_scale_xy=torch.ones(1, 2),
            ),
        )
        stage0 = self.hci.stages[0].geometry(
            torch.tensor([[3.4, 0.0, 5.0]]),
            torch.tensor([0], dtype=torch.long),
            batch_size=1,
            feature_height=8,
            feature_width=8,
            context=local_context,
        )
        self.assertTrue(stage0.edge_count > 0)
        local_indices = stage0.vision_index % 64
        xs = local_indices % 8
        self.assertTrue(bool((xs < 7).all()))


if __name__ == "__main__":
    unittest.main()
