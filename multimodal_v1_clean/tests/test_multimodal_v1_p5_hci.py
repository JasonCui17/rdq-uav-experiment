"""P5 Geometry Bi-HCI geometry, masking, grouping and gradient contracts."""

from __future__ import annotations

import unittest

import torch

from rdq_uav.multimodal_v1.contracts import InteractionContext, ProjectionContext
from rdq_uav.multimodal_v1.interaction import GeometryBiHCI, GeometryBiHCIStack, GeometryLocal


def synthetic_context(
    batch_size: int,
    *,
    m_r: list[bool] | None = None,
    m_v: list[bool] | None = None,
) -> InteractionContext:
    rotation = torch.eye(3).repeat(batch_size, 1, 1)
    translation = torch.zeros(batch_size, 3)
    intrinsics = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]]).repeat(batch_size, 1)
    distortion = torch.zeros(batch_size, 4)
    image_size = torch.tensor([[8.0, 8.0]]).repeat(batch_size, 1)
    scale = torch.ones(batch_size, 2)
    return InteractionContext(
        calibration_handle=tuple("synthetic" for _ in range(batch_size)),
        m_R=torch.tensor(m_r if m_r is not None else [True] * batch_size),
        m_V=torch.tensor(m_v if m_v is not None else [True] * batch_size),
        projection=ProjectionContext(
            rotation_camera_from_radar=rotation,
            translation_camera_from_radar_m=translation,
            intrinsics=intrinsics,
            distortion=distortion,
            image_size_wh=image_size,
            image_scale_xy=scale,
        ),
    )


class GeometryLocalTests(unittest.TestCase):
    def test_projection_and_exact_3x3_indices(self):
        geometry = GeometryLocal(feature_stride=1)
        centers = torch.tensor([[3.2, 4.2, 1.0]])
        edges = geometry(
            centers,
            torch.tensor([0], dtype=torch.long),
            batch_size=1,
            feature_height=8,
            feature_width=8,
            context=synthetic_context(1),
        )
        self.assertEqual(edges.valid_projection_mask.tolist(), [True])
        self.assertEqual(edges.anchor_xy.tolist(), [[3, 4]])
        self.assertEqual(edges.radar_index.tolist(), [0] * 9)
        self.assertEqual(edges.relative_index_v_to_r.tolist(), list(range(9)))
        self.assertEqual(edges.relative_index_r_to_v.tolist(), list(range(8, -1, -1)))
        self.assertEqual(
            edges.vision_index.tolist(),
            [26, 27, 28, 34, 35, 36, 42, 43, 44],
        )

    def test_batch_indices_prevent_cross_sample_edges(self):
        geometry = GeometryLocal(feature_stride=1)
        centers = torch.tensor([[3.2, 3.2, 1.0], [3.2, 3.2, 1.0]])
        edges = geometry(
            centers,
            torch.tensor([0, 1], dtype=torch.long),
            batch_size=2,
            feature_height=8,
            feature_width=8,
            context=synthetic_context(2),
        )
        first = edges.vision_index[edges.radar_index == 0]
        second = edges.vision_index[edges.radar_index == 1]
        self.assertTrue(bool((first < 64).all()))
        self.assertTrue(bool((second >= 64).all()))

    def test_boundary_masks_invalid_neighbors_without_clamping(self):
        geometry = GeometryLocal(feature_stride=1)
        edges = geometry(
            torch.tensor([[0.1, 0.1, 1.0]]),
            torch.tensor([0], dtype=torch.long),
            batch_size=1,
            feature_height=8,
            feature_width=8,
            context=synthetic_context(1),
        )
        self.assertEqual(edges.edge_count, 4)
        self.assertEqual(sorted(edges.vision_index.tolist()), [0, 1, 8, 9])
        self.assertEqual(len(set(edges.vision_index.tolist())), 4)

    def test_missing_modality_removes_all_cross_modal_edges(self):
        geometry = GeometryLocal(feature_stride=1)
        edges = geometry(
            torch.tensor([[3.0, 3.0, 1.0]]),
            torch.tensor([0], dtype=torch.long),
            batch_size=1,
            feature_height=8,
            feature_width=8,
            context=synthetic_context(1, m_v=[False]),
        )
        self.assertTrue(edges.valid_projection_mask.item())
        self.assertFalse(edges.active_radar_mask.item())
        self.assertEqual(edges.edge_count, 0)


class GeometryBiHCITests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def module(self):
        return GeometryBiHCI(
            vision_dim=96,
            feature_stride=1,
            radar_dim=128,
            interaction_dim=128,
            num_heads=4,
        )

    def test_many_to_one_radar_support_is_grouped_for_vision(self):
        module = self.module()
        radar = torch.randn(2, 128)
        centers = torch.tensor([[3.2, 3.2, 1.0], [4.2, 3.2, 1.0]])
        vision = torch.randn(1, 64, 96)
        output = module(
            radar,
            centers,
            torch.tensor([0, 0], dtype=torch.long),
            vision,
            height=8,
            width=8,
            context=synthetic_context(1),
            return_aux=True,
        )
        support = output.aux["vision_support_count"]
        self.assertTrue(bool((support == 2).any()))
        self.assertEqual(output.radar_features.shape, radar.shape)
        self.assertEqual(output.vision_tokens.shape, vision.shape)

    def test_missing_modality_is_exact_identity(self):
        module = self.module()
        radar = torch.randn(2, 128)
        centers = torch.tensor([[3.2, 3.2, 1.0], [4.2, 3.2, 1.0]])
        vision = torch.randn(1, 64, 96)
        output = module(
            radar,
            centers,
            torch.tensor([0, 0], dtype=torch.long),
            vision,
            height=8,
            width=8,
            context=synthetic_context(1, m_v=[False]),
            return_aux=True,
        )
        self.assertTrue(torch.equal(output.radar_features, radar))
        self.assertTrue(torch.equal(output.vision_tokens, vision))
        self.assertEqual(output.aux["edge_count"], 0)

    def test_near_zero_gate_preserves_backbone_behavior(self):
        module = self.module()
        radar = torch.randn(3, 128)
        centers = torch.tensor(
            [[2.2, 2.2, 1.0], [3.2, 3.2, 1.0], [4.2, 4.2, 1.0]]
        )
        vision = torch.randn(1, 64, 96)
        output = module(
            radar,
            centers,
            torch.zeros(3, dtype=torch.long),
            vision,
            height=8,
            width=8,
            context=synthetic_context(1),
            return_aux=True,
        )
        self.assertLess(float(output.aux["gate_R"].max().detach()), 0.011)
        self.assertLess(float(output.aux["gate_V"].max().detach()), 0.011)
        radar_relative = torch.linalg.vector_norm(output.radar_features - radar) / (
            torch.linalg.vector_norm(radar) + 1e-8
        )
        vision_relative = torch.linalg.vector_norm(output.vision_tokens - vision) / (
            torch.linalg.vector_norm(vision) + 1e-8
        )
        self.assertLess(float(radar_relative.detach()), 0.02)
        self.assertLess(float(vision_relative.detach()), 0.02)

    def test_backward_has_finite_input_and_parameter_gradients(self):
        module = self.module()
        radar = torch.randn(2, 128, requires_grad=True)
        vision = torch.randn(1, 64, 96, requires_grad=True)
        centers = torch.tensor([[3.2, 3.2, 1.0], [4.2, 3.2, 1.0]])
        output = module(
            radar,
            centers,
            torch.tensor([0, 0], dtype=torch.long),
            vision,
            height=8,
            width=8,
            context=synthetic_context(1),
        )
        loss = output.radar_features.square().mean() + output.vision_tokens.square().mean()
        loss.backward()
        self.assertIsNotNone(radar.grad)
        self.assertIsNotNone(vision.grad)
        self.assertTrue(bool(torch.isfinite(radar.grad).all()))
        self.assertTrue(bool(torch.isfinite(vision.grad).all()))
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_three_stage_stack_uses_independent_parameters(self):
        stack = GeometryBiHCIStack()
        self.assertEqual([stage.vision_dim for stage in stack.stages], [96, 192, 384])
        self.assertEqual([stage.feature_stride for stage in stack.stages], [4, 8, 16])
        self.assertIsNot(
            stack.stages[0].v_to_r.q_proj.weight,
            stack.stages[1].v_to_r.q_proj.weight,
        )


if __name__ == "__main__":
    unittest.main()
