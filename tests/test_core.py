from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from rdq_uav.data.dataset import RadarProcessor
from rdq_uav.data.transforms import DualFisheyeTransform
from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory
from rdq_uav.calibration.omni import transform_points
from rdq_uav.calibration.spatiotemporal import (
    CenterObservation,
    fit_spatiotemporal_calibration,
)
from rdq_uav.engine.metrics import ClassificationMetrics
from rdq_uav.engine.sequence import aggregate_temporal_blocks
from rdq_uav.models.model import MultiModalClassifier
from rdq_uav.models.radar import MaskedPointMLP


def model_config(variant: str) -> dict:
    return {
        "variant": variant,
        "num_classes": 5,
        "embed_dim": 32,
        "dropout": 0.0,
        "backbone": {
            "name": "tiny_cnn",
            "provider": "builtin",
            "pretrained": False,
            "out_index": 0,
            "trainable": True,
        },
        "radar_encoder": {"input_dim": 3, "hidden_dims": [16, 24]},
        "attention": {"num_heads": 4, "ffn_ratio": 2},
        "radar_skip": True,
        "auxiliary_position": {"enabled": False, "loss_weight": 0.1},
    }


class CoreTests(unittest.TestCase):
    def test_oracle_left_transform_returns_one_view_and_masks_center(self) -> None:
        image = Image.fromarray(np.full((100, 200, 3), 255, dtype=np.uint8))
        transform = DualFisheyeTransform(
            [32, 32], False, image_mode="oracle_left", center_mask_fraction=0.5
        )
        output = transform(
            image,
            {
                "roi_x1": "10",
                "roi_y1": "10",
                "roi_x2": "90",
                "roi_y2": "90",
                "roi_center_u": "50",
                "roi_center_v": "50",
            },
        )
        self.assertEqual(tuple(output.shape), (1, 3, 32, 32))
        self.assertLess(float(output[0, :, 16, 16].abs().max()), 0.03)

    def test_official_bbox_counterfactual_modes_are_complementary(self) -> None:
        pixels = np.full((100, 200, 3), 255, dtype=np.uint8)
        pixels[40:60, 40:60] = (255, 0, 0)
        image = Image.fromarray(pixels)
        row = {
            "roi_x1": "0",
            "roi_y1": "0",
            "roi_x2": "100",
            "roi_y2": "100",
            "official_bbox_x1": "40",
            "official_bbox_y1": "40",
            "official_bbox_x2": "60",
            "official_bbox_y2": "60",
        }
        erased = DualFisheyeTransform(
            [100, 100], False, image_mode="oracle_left", bbox_mode="erase"
        )(image, row)
        foreground = DualFisheyeTransform(
            [100, 100], False, image_mode="oracle_left", bbox_mode="foreground_only"
        )(image, row)
        self.assertLess(float(erased[0, :, 50, 50].abs().max()), 0.03)
        self.assertGreater(float(erased[0, :, 10, 10].abs().max()), 1.0)
        self.assertGreater(float(foreground[0, :, 50, 50].abs().max()), 1.0)
        self.assertLess(float(foreground[0, :, 10, 10].abs().max()), 0.03)

    def test_omni_projection_center_and_scale_invariance(self) -> None:
        camera = OmniRadtanCamera(2.0, 400.0, 410.0, 320.0, 240.0, 0, 0, 0, 0, 640, 480)
        points = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [2.0, 0.0, 4.0]])
        pixels, valid = camera.project(points)
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(pixels[0], [320.0, 240.0]))
        self.assertTrue(np.allclose(pixels[1], pixels[2]))

    def test_position_trajectory_interpolates_at_image_time(self) -> None:
        trajectory = PositionTrajectory(
            np.asarray([1000.0, 1000.2, 1000.4]),
            np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        )
        position, valid = trajectory.evaluate(1000.1)
        self.assertTrue(bool(valid))
        self.assertTrue(np.allclose(position, [0.5, 1.0, 1.5]))
        _, outside = trajectory.evaluate(999.0)
        self.assertFalse(bool(outside))

    def test_spatiotemporal_fit_recovers_synthetic_solution(self) -> None:
        timestamps = np.linspace(1000.0, 1006.0, 121)
        relative = timestamps - timestamps[0]
        positions = np.column_stack(
            (
                3.0 * np.sin(0.7 * relative) + 2.0,
                2.0 * np.cos(0.4 * relative) + 1.0,
                8.0 + 0.5 * relative + 0.3 * np.sin(relative),
            )
        )
        trajectory = PositionTrajectory(timestamps, positions)
        camera = OmniRadtanCamera(
            2.9, 1800.0, 1800.0, 640.0, 480.0, -0.2, 0.2, 0, 0, 1280, 960
        )
        true_rotation = Rotation.from_euler("xyz", [0.1, -0.15, 0.4]).as_matrix()
        true_translation = np.asarray([0.2, -0.1, 0.05])
        true_offset = 0.035
        observations = []
        for image_time in np.linspace(1000.3, 1005.7, 30):
            point, _ = trajectory.evaluate(image_time + true_offset)
            pixel, valid = camera.project(
                transform_points(point, true_rotation, true_translation)
            )
            self.assertTrue(bool(valid))
            observations.append(CenterObservation("sequence", "camera", image_time, *pixel))
        initial_rotation = Rotation.from_euler("xyz", [0.08, -0.12, 0.35]).as_matrix()
        solution = fit_spatiotemporal_calibration(
            observations,
            {"sequence": trajectory},
            {"camera": camera},
            {"camera": initial_rotation},
            {"camera": np.zeros(3)},
            max_time_offset_s=0.1,
        )
        rotation_error = Rotation.from_matrix(
            solution.rotation_camera_from_gt["camera"] @ true_rotation.T
        ).magnitude()
        self.assertLess(rotation_error, 1e-5)
        self.assertLess(
            np.linalg.norm(solution.translation_camera_from_gt["camera"] - true_translation),
            1e-5,
        )
        self.assertAlmostEqual(solution.time_offset_s, true_offset, places=5)

    def test_masked_pool_ignores_padding(self) -> None:
        encoder = MaskedPointMLP(3, [8], 16, 0.0).eval()
        valid = torch.randn(2, 3)
        points_a = torch.zeros(1, 5, 3)
        points_b = torch.full((1, 5, 3), 999.0)
        points_a[0, :2] = valid
        points_b[0, :2] = valid
        mask = torch.tensor([[True, True, False, False, False]])
        token_a, _ = encoder(points_a, mask)
        token_b, _ = encoder(points_b, mask)
        self.assertTrue(torch.allclose(token_a, token_b))

    def test_empty_radar_is_finite(self) -> None:
        encoder = MaskedPointMLP(3, [8], 16, 0.0).eval()
        token, _ = encoder(torch.zeros(2, 5, 3), torch.zeros(2, 5, dtype=torch.bool))
        self.assertTrue(torch.isfinite(token).all())

    def test_radar_processor_filters_range_and_is_deterministic(self) -> None:
        processor = RadarProcessor(2, 50.0, [0, 0, 0], [1, 1, 1], False, True)
        array = np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 0, 100]], np.float32)
        first, first_mask = processor(array, "sample")
        second, second_mask = processor(array, "sample")
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_mask, second_mask))
        self.assertEqual(int(first_mask.sum()), 2)
        self.assertLess(float(first.abs().max()), 50.0)

    def test_all_variants_forward(self) -> None:
        radar = torch.randn(2, 12, 3)
        mask = torch.ones(2, 12, dtype=torch.bool)
        for views in (1, 2):
            images = torch.randn(2, views, 3, 64, 96)
            for variant in ("rgb", "radar", "concat", "learned_query", "rdq"):
                config = copy.deepcopy(model_config(variant))
                model = MultiModalClassifier(config)
                output = model(images, radar, mask, return_attention=True)
                self.assertEqual(tuple(output["logits"].shape), (2, 5))
                self.assertTrue(torch.isfinite(output["logits"]).all())
                if variant in {"learned_query", "rdq"}:
                    self.assertIsNotNone(output["attention"])

    def test_metrics(self) -> None:
        meter = ClassificationMetrics(3)
        logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 5.0, 0]])
        meter.update(logits, torch.tensor([0, 1, 2]))
        result = meter.compute()
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["support"], [1, 1, 1])

    def test_temporal_block_soft_vote_and_keyframe_gap(self) -> None:
        rows = [
            {
                "sample_id": f"s{index}",
                "sequence_id": "seq",
                "temporal_block": 3,
                "target": 1,
                "gt_time": float(index),
                "bbox_area_px": float(10 - index),
                "probabilities": probability,
            }
            for index, probability in enumerate(
                ([0.6, 0.4], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9])
            )
        ]
        metrics, aggregated = aggregate_temporal_blocks(
            rows, 2, top_k=2, min_gap_seconds=1.5
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(aggregated[0]["selected_sample_ids"], ["s0", "s2"])


if __name__ == "__main__":
    unittest.main()
