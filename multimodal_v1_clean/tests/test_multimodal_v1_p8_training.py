from __future__ import annotations

import unittest

import torch

from rdq_uav.multimodal_v1.training import (
    combine_e5_losses, optimizer_parameter_names, ordered_optimizer_groups,
    stage_for_epoch, summarize_validation_outcomes, supervised_dino_loss,
    usable_multimodal_training_samples, validate_optimizer_parameter_names,
)
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

    def test_validation_no_output_remains_in_success_denominator(self):
        metrics=summarize_validation_outcomes([0.5,2.0,None],[0.8,None])
        self.assertAlmostEqual(metrics["final_3d_success_1m"],1/3)
        self.assertEqual(metrics["n_gt3d"],3)
        self.assertEqual(metrics["n_output_3d"],2)
        self.assertEqual(metrics["n_no_output_3d"],1)
        self.assertAlmostEqual(metrics["output_coverage_3d"],2/3)
        self.assertAlmostEqual(metrics["vision_top1_iou_mean"],0.4)

    def test_both_modalities_missing_training_sample_is_skipped_and_counted(self):
        samples=[{"m_R":True,"m_V":True},{"m_R":True,"m_V":False},
                 {"m_R":False,"m_V":True},{"m_R":False,"m_V":False}]
        usable,missing=usable_multimodal_training_samples(samples)
        self.assertEqual(len(usable),3);self.assertEqual(missing,1)
        self.assertTrue(all(item["m_R"] or item["m_V"] for item in usable))

    def test_optimizer_parameter_names_are_stable_and_checked(self):
        def build():
            model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,1))
            first={id(p) for p in model[0].parameters()}; second={id(p) for p in model[1].parameters()}
            groups=ordered_optimizer_groups(model,(("first",first,1e-3),("second",second,1e-4)))
            return torch.optim.AdamW(groups)
        left,right=build(),build()
        names=optimizer_parameter_names(left)
        self.assertEqual(names,optimizer_parameter_names(right))
        validate_optimizer_parameter_names(right,names)
        right.load_state_dict(left.state_dict())
        self.assertEqual(names,optimizer_parameter_names(right))
        corrupted=[dict(names[0]),dict(names[1])]
        corrupted[0]["param_names"]=list(reversed(corrupted[0]["param_names"]))
        with self.assertRaises(ValueError):
            validate_optimizer_parameter_names(right,corrupted)
        with self.assertRaises(ValueError):
            validate_optimizer_parameter_names(right,None)


if __name__ == "__main__":
    unittest.main()
