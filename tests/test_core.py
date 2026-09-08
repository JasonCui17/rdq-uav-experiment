from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from rdq_uav.data.dataset import RadarProcessor
from rdq_uav.data.localization import MMAUDLocalizationDataset
from rdq_uav.data.transforms import DualFisheyeTransform
from rdq_uav.calibration import OmniRadtanCamera, PositionTrajectory
from rdq_uav.calibration.omni import transform_points
from rdq_uav.calibration.spatiotemporal import (
    CenterObservation,
    fit_spatiotemporal_calibration,
)
from rdq_uav.engine.metrics import ClassificationMetrics
from rdq_uav.engine.localization import (
    LocalizationLoss,
    LocalizationMetrics,
    aligned_iou,
)
from rdq_uav.engine.attention_audit import (
    attention_center_diagnostics,
    stitched_token_centers,
)
from rdq_uav.engine.sequence import aggregate_temporal_blocks
from rdq_uav.models.model import MultiModalClassifier
from rdq_uav.models.localizer import MultiModalLocalizer
from rdq_uav.models.backbones import MinimalTopDownFusion
from rdq_uav.models.fusion import CrossAttentionBlock
from rdq_uav.models.radar import MaskedPointMLP
from tools.radar_image_geometry_audit import (
    bbox_distance,
    deterministic_shuffle,
    project_raw_radar,
    score_pixels,
)
from tools.calibration.resolve_radar_coordinate_frame import (
    kabsch,
    proper_axis_rotations as radar_axis_rotations,
)
from tools.radar_target_association_audit import (
    candidate_rules,
    deterministic_shuffle_indices as association_shuffle_indices,
    score_candidates,
)
from tools.visualize_nearest_radar_projection import (
    nearest_row as nearest_projection_row,
    project_chain as nearest_projection_chain,
)


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
    def test_nearest_projection_row_is_deterministic(self) -> None:
        rows = [{"radar_time": "1.0"}, {"radar_time": "1.2"}, {"radar_time": "1.4"}]
        self.assertEqual(nearest_projection_row(rows, 1.31, "radar_time")["radar_time"], "1.4")
        self.assertEqual(nearest_projection_row(rows, 1.09, "radar_time")["radar_time"], "1.0")

    def test_nearest_projection_transform_chain(self) -> None:
        camera = OmniRadtanCamera(
            xi=0.0, fu=100.0, fv=100.0, pu=50.0, pv=50.0,
            k1=0.0, k2=0.0, p1=0.0, p2=0.0, width=100, height=100,
        )
        radar = np.asarray([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
        gt, camera_points, pixels, valid = nearest_projection_chain(
            radar, camera, np.eye(3), np.asarray([0.0, 0.0, 5.0]),
            np.eye(3), np.zeros(3),
        )
        self.assertTrue(np.allclose(gt[0], [0.0, 0.0, 10.0]))
        self.assertTrue(np.allclose(camera_points, gt))
        self.assertTrue(np.allclose(pixels[0], [50.0, 50.0]))
        self.assertEqual(valid.tolist(), [True, False])

    def test_target_association_rules_use_confirmed_xyz_and_range(self) -> None:
        points = np.asarray([[3.0, 4.0, 0.0], [30.0, 40.0, 1.0], [60.0, 0.0, 0.0]])
        rules = candidate_rules(max_range_m=50.0, gt_range_gate_m=0.5)
        self.assertEqual(
            set(rules), {"released_xyz_all", "finite_range_le_50m", "oracle_gt_range_gate"}
        )
        self.assertEqual(rules["released_xyz_all"](points, None).tolist(), [True, True, True])
        self.assertEqual(rules["finite_range_le_50m"](points, None).tolist(), [True, False, False])
        target = np.asarray([0.0, 5.2, 0.0])
        self.assertEqual(rules["oracle_gt_range_gate"](points, target).tolist(), [True, False, False])

    def test_target_association_shuffle_stays_within_sequence(self) -> None:
        rows = [
            {"sequence_id": "a"}, {"sequence_id": "b"}, {"sequence_id": "a"},
            {"sequence_id": "b"}, {"sequence_id": "a"},
        ]
        indices = association_shuffle_indices(rows)
        self.assertEqual(len(indices), len(rows))
        for index, shuffled_index in enumerate(indices):
            self.assertEqual(rows[index]["sequence_id"], rows[shuffled_index]["sequence_id"])

    def test_target_association_scoring_identity_projection(self) -> None:
        camera = OmniRadtanCamera(
            xi=0.0, fu=100.0, fv=100.0, pu=50.0, pv=50.0,
            k1=0.0, k2=0.0, p1=0.0, p2=0.0, width=100, height=100,
        )
        points = np.asarray([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]])
        result = score_candidates(
            points, np.asarray([0.0, 0.0, 10.0]), np.asarray([50.0, 50.0]),
            np.eye(3), np.zeros(3), camera, np.eye(3), np.zeros(3),
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertAlmostEqual(result["nearest_candidate_gt_3d_m"], 0.0)
        self.assertAlmostEqual(result["nearest_candidate_gt_2d_px"], 0.0)
        self.assertEqual(result["coverage_8px"], 1)

    def test_radar_frame_resolution_axis_set_and_kabsch(self) -> None:
        rotations = radar_axis_rotations()
        self.assertEqual(len(rotations), 24)
        self.assertTrue(all(np.isclose(np.linalg.det(rotation), 1.0) for rotation in rotations))
        source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected_rotation = Rotation.from_euler("z", 35, degrees=True).as_matrix()
        expected_translation = np.asarray([0.2, -0.3, 0.5])
        target = source @ expected_rotation.T + expected_translation
        rotation, translation = kabsch(source, target)
        self.assertTrue(np.allclose(rotation, expected_rotation))
        self.assertTrue(np.allclose(translation, expected_translation))

    def test_geometry_audit_raw_path_and_scoring(self) -> None:
        camera = OmniRadtanCamera(
            xi=1.0, fu=100.0, fv=100.0, pu=50.0, pv=50.0,
            k1=0.0, k2=0.0, p1=0.0, p2=0.0, width=100, height=100,
        )
        points = np.asarray([[0.0, 0.0, 2.0], [100.0, 0.0, 1.0]])
        pixels = project_raw_radar(points, camera, np.eye(3), np.zeros(3))
        self.assertEqual(pixels.shape, (1, 2))
        score = score_pixels(pixels, np.asarray([45.0, 45.0, 55.0, 55.0]))
        self.assertEqual(score["valid_projected_point_count"], 1)
        self.assertEqual(score["points_inside_bbox"], 1)
        self.assertEqual(score["coverage_8px"], 1)
        distances = bbox_distance(
            np.asarray([[50.0, 50.0], [60.0, 60.0]]),
            np.asarray([45.0, 45.0, 55.0, 55.0]),
        )
        self.assertTrue(np.allclose(distances, [0.0, np.sqrt(50.0)]))

    def test_geometry_audit_shuffle_stays_within_sequence(self) -> None:
        rows = [
            {"sequence_id": "a", "sample_id": "a0"},
            {"sequence_id": "a", "sample_id": "a1"},
            {"sequence_id": "b", "sample_id": "b0"},
            {"sequence_id": "b", "sample_id": "b1"},
        ]
        shuffled = deterministic_shuffle(rows)
        self.assertEqual([row["sequence_id"] for row in shuffled], ["a", "a", "b", "b"])
        self.assertEqual([row["sample_id"] for row in shuffled], ["a1", "a0", "b1", "b0"])

    def test_aligned_iou_promotes_cpu_half_for_amp_bookkeeping(self) -> None:
        box = torch.tensor([[0.1, 0.2, 0.4, 0.5]], dtype=torch.float16)
        iou = aligned_iou(box, box)
        self.assertEqual(iou.dtype, torch.float32)
        self.assertTrue(torch.allclose(iou, torch.ones_like(iou)))

    def test_cross_attention_attended_only_removes_query_residual_and_ffn(self) -> None:
        block = CrossAttentionBlock(16, 4, 0.0, 2).eval()
        query = torch.randn(2, 1, 16)
        memory = torch.randn(2, 7, 16)
        expected, expected_weights = block.attention(
            block.query_norm(query),
            block.memory_norm(memory),
            block.memory_norm(memory),
            need_weights=True,
            average_attn_weights=False,
        )
        actual, weights = block(
            query, memory, need_weights=True, output_mode="attended_only"
        )
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(weights, expected_weights))

    def test_attention_audit_uses_dual_view_tokenizer_order(self) -> None:
        centers = stitched_token_centers(
            2, 3, 2, 20, 30, device=torch.device("cpu"), dtype=torch.float32
        )
        expected = torch.tensor(
            [
                [5, 5], [15, 5], [25, 5], [5, 15], [15, 15], [25, 15],
                [35, 5], [45, 5], [55, 5], [35, 15], [45, 15], [55, 15],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(centers, expected))
        weights = torch.zeros(1, 2, 1, 12)
        weights[:, :, :, 7] = 1.0
        diagnostics = attention_center_diagnostics(
            weights,
            torch.tensor([[45 / 60, 5 / 20]], dtype=torch.float32),
            centers,
            stitched_width=60,
            image_height=20,
            cell_width=10,
            cell_height=10,
        )
        self.assertAlmostEqual(float(diagnostics["mean_attention_error_px"][0]), 0.0)
        self.assertAlmostEqual(float(diagnostics["peak_attention_error_px"][0]), 0.0)
        self.assertAlmostEqual(float(diagnostics["best_head_error_px"][0]), 0.0)
        self.assertAlmostEqual(float(diagnostics["grid_oracle_error_px"][0]), 0.0)
        self.assertAlmostEqual(float(diagnostics["gt_mass_1cell"][0]), 1.0)
        self.assertAlmostEqual(float(diagnostics["attention_entropy_normalized"][0]), 0.0)

    def test_minimal_top_down_fusion_shape_and_gradient(self) -> None:
        fusion = MinimalTopDownFusion(16, 32, 24)
        stride8 = torch.randn(2, 16, 8, 12, requires_grad=True)
        stride16 = torch.randn(2, 32, 4, 6, requires_grad=True)
        output = fusion(stride8, stride16)
        self.assertEqual(tuple(output.shape), (2, 24, 8, 12))
        output.mean().backward()
        self.assertTrue(torch.isfinite(stride8.grad).all())
        self.assertTrue(torch.isfinite(stride16.grad).all())
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
                    self.assertEqual(
                        tuple(output["attention"].shape),
                        (2, 4, 1, views * 8 * 12),
                    )

    def test_localizer_all_variants_forward_and_backward(self) -> None:
        images = torch.randn(2, 2, 3, 64, 96)
        radar = torch.randn(2, 12, 3)
        mask = torch.ones(2, 12, dtype=torch.bool)
        target_box = torch.tensor([[0.25, 0.5, 0.1, 0.2]]).repeat(2, 1)
        target_position = torch.randn(2, 3)
        criterion = LocalizationLoss(
            {
                "bbox_l1_weight": 5.0,
                "giou_weight": 2.0,
                "position_weight": 1.0,
                "projection_consistency": {"enabled": False},
            }
        )
        for variant in ("rgb", "radar", "concat", "learned_query", "rdq"):
            model = MultiModalLocalizer(copy.deepcopy(model_config(variant)))
            output = model(images, radar, mask, return_attention=True)
            self.assertEqual(tuple(output["box"].shape), (2, 4))
            self.assertEqual(tuple(output["position"].shape), (2, 3))
            self.assertTrue(bool(((output["box"] >= 0) & (output["box"] <= 1)).all()))
            losses = criterion(output["box"], target_box, output["position"], target_position)
            self.assertTrue(
                torch.allclose(
                    losses["bbox_l1_loss"],
                    (losses["bbox_center_l1_loss"] + losses["bbox_size_l1_loss"]) / 2,
                )
            )
            self.assertTrue(
                torch.allclose(
                    losses["total_loss"],
                    5.0 * losses["bbox_l1_loss"]
                    + 2.0 * losses["giou_loss"]
                    + losses["position_loss"],
                )
            )
            losses["total_loss"].backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_log_size_parameterization_uses_reference_and_zero_size_bias(self) -> None:
        config = copy.deepcopy(model_config("rdq"))
        config["bbox_parameterization"] = "sigmoid_center_log_size"
        config["bbox_reference_wh"] = [0.025, 0.06]
        model = MultiModalLocalizer(config)
        final_layer = model.box_head.layers[-1]
        self.assertTrue(torch.equal(final_layer.bias[2:], torch.zeros(2)))
        decoded = model._decode_box(torch.zeros(2, 4))
        self.assertTrue(torch.allclose(decoded[:, :2], torch.full((2, 2), 0.5)))
        self.assertTrue(torch.allclose(decoded[:, 2:], torch.tensor([[0.025, 0.06]]).repeat(2, 1)))

    def test_localization_l1_regression_mode_preserves_total_loss_formula(self) -> None:
        criterion = LocalizationLoss(
            {
                "bbox_regression": "l1",
                "bbox_l1_weight": 5.0,
                "giou_weight": 0.0,
                "position_weight": 1.0,
                "projection_consistency": {"enabled": False},
            }
        )
        pred_box = torch.tensor([[0.50, 0.50, 0.02, 0.04]], requires_grad=True)
        gt_box = torch.tensor([[0.51, 0.48, 0.01, 0.03]])
        pred_position = torch.zeros(1, 3, requires_grad=True)
        gt_position = torch.ones(1, 3)
        losses = criterion(pred_box, gt_box, pred_position, gt_position)
        expected = torch.nn.functional.l1_loss(pred_box, gt_box)
        self.assertTrue(torch.allclose(losses["bbox_regression_loss"], expected))
        self.assertTrue(torch.allclose(losses["bbox_l1_loss"], expected))
        self.assertTrue(
            torch.allclose(
                losses["total_loss"],
                5.0 * expected + losses["position_loss"],
            )
        )
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(pred_box.grad).all())

    def test_localization_bbox_is_normalized_on_stitched_full_panorama(self) -> None:
        dataset = object.__new__(MMAUDLocalizationDataset)
        dataset.panorama_width = 2560
        dataset.panorama_height = 960
        bbox = dataset._normalized_bbox(
            {
                "official_bbox_x1": "620",
                "official_bbox_y1": "460",
                "official_bbox_x2": "676",
                "official_bbox_y2": "520",
            }
        )
        expected = torch.tensor([648 / 2560, 490 / 960, 56 / 2560, 60 / 960])
        self.assertTrue(torch.allclose(bbox, expected))

    def test_localization_metrics_perfect_prediction(self) -> None:
        meter = LocalizationMetrics(processed_height=288, processed_stitched_width=768)
        boxes = torch.tensor([[0.25, 0.5, 0.1, 0.2], [0.3, 0.4, 0.2, 0.1]])
        positions = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        meter.update(boxes, boxes, positions, positions)
        result = meter.compute()
        self.assertAlmostEqual(result["mean_iou"], 1.0)
        self.assertAlmostEqual(result["bbox_center_error_px_mean"], 0.0)
        self.assertAlmostEqual(result["center_error_lt_2px"], 1.0)
        self.assertAlmostEqual(result["center_error_lt_4px"], 1.0)
        self.assertAlmostEqual(result["center_abs_error_x"], 0.0)
        self.assertAlmostEqual(result["center_abs_error_y"], 0.0)
        self.assertAlmostEqual(result["width_abs_error_mean"], 0.0)
        self.assertAlmostEqual(result["height_abs_error_mean"], 0.0)
        self.assertAlmostEqual(result["pred_width_mean"], result["gt_width_mean"])
        self.assertAlmostEqual(result["pred_height_mean"], result["gt_height_mean"])
        self.assertAlmostEqual(result["position_error_mean_m"], 0.0)

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
