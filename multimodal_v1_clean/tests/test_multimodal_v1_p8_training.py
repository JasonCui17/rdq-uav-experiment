from __future__ import annotations

import unittest

import torch

from rdq_uav.multimodal_v1.training import combine_e5_losses, stage_for_epoch, supervised_dino_loss
from rdq_uav.multimodal_v1.vision.ssod import ViewTransform


class E5TrainingContractTests(unittest.TestCase):
    def test_frozen_loss_formula(self):
        values = [torch.tensor(value) for value in (1.0, 2.0, 3.0)]
        result = combine_e5_losses(*values, lambda_r=1.0, lambda_v=1.0, lambda_f=1.0)
        self.assertEqual(float(result), 6.0)

    def test_stage_ranges_are_explicit(self):
        stages = [
            {"name": "T1", "start_epoch": 1, "end_epoch": 1},
            {"name": "T2", "start_epoch": 2, "end_epoch": 8},
            {"name": "T3", "start_epoch": 9, "end_epoch": 12},
        ]
        self.assertEqual(stage_for_epoch(stages, 1), "T1")
        self.assertEqual(stage_for_epoch(stages, 6), "T2")
        self.assertEqual(stage_for_epoch(stages, 12), "T3")
        with self.assertRaises(ValueError):
            stage_for_epoch(stages, 13)

    def test_missing_2d_gt_is_removed_from_dino_criterion(self):
        class Criterion:
            weight_dict = {"loss_ce": 1.0}

            def __call__(self, output, targets, _):
                self.batch = output["pred_logits"].shape[0]
                self.targets = targets
                return {"loss_ce": output["pred_logits"].sum()}

        detector = type("Detector", (), {})()
        detector.criterion = Criterion()
        output = {
            "pred_logits": torch.ones(2, 3, 1, requires_grad=True),
            "pred_boxes": torch.full((2, 3, 4), 0.5),
        }
        transforms = [ViewTransform((100, 100), (200, 200)) for _ in range(2)]
        loss, _, count = supervised_dino_loss(
            detector, output,
            gt_box_xyxy_source=torch.tensor([[0., 0., 0., 0.], [10., 20., 30., 40.]]),
            gt_2d_valid=torch.tensor([False, True]), transforms=transforms,
        )
        self.assertEqual(count, 1)
        self.assertEqual(detector.criterion.batch, 1)
        self.assertEqual(len(detector.criterion.targets), 1)
        self.assertTrue(loss.requires_grad)


if __name__ == "__main__":
    unittest.main()
