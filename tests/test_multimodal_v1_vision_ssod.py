from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ssod = load_module("ssod_under_test", "src/rdq_uav/multimodal_v1/vision/ssod.py")
ssod_data = load_module("ssod_data_under_test", "src/rdq_uav/multimodal_v1/vision/ssod_data.py")


class SSODCoreTests(unittest.TestCase):
    def test_view_transform_round_trip_with_flip(self):
        transform = ssod.ViewTransform((1280, 960), (1067, 800), True)
        box = torch.tensor([100.0, 200.0, 240.0, 330.0])
        restored = transform.view_boxes_to_source(transform.source_boxes_to_view(box))
        self.assertTrue(torch.allclose(box, restored, atol=1e-4))

    def test_high_medium_ignore_policy(self):
        # One query in each view, same source box after inverse transform.
        transform = ssod.ViewTransform((100, 100), (100, 100), False)
        box = torch.tensor([[0.50, 0.50, 0.20, 0.20]])
        point = torch.tensor([50.0, 50.0])
        high = ssod.mine_geometry_guided_pseudo(
            {"pred_logits": torch.tensor([[4.0]]), "pred_boxes": box},
            {"pred_logits": torch.tensor([[4.0]]), "pred_boxes": box},
            transform1=transform, transform2=transform, projected_point_source=point,
            policy=ssod.PseudoLabelPolicy(tau_high=.9, tau_mid=.5),
        )
        self.assertEqual(high.quality, ssod.PseudoQuality.HIGH)
        medium = ssod.mine_geometry_guided_pseudo(
            {"pred_logits": torch.tensor([[1.0]]), "pred_boxes": box},
            {"pred_logits": torch.tensor([[1.0]]), "pred_boxes": box},
            transform1=transform, transform2=transform, projected_point_source=point,
            policy=ssod.PseudoLabelPolicy(tau_high=.9, tau_mid=.5),
        )
        self.assertEqual(medium.quality, ssod.PseudoQuality.MEDIUM)
        ignored = ssod.mine_geometry_guided_pseudo(
            {"pred_logits": torch.tensor([[-2.0]]), "pred_boxes": box},
            {"pred_logits": torch.tensor([[-2.0]]), "pred_boxes": box},
            transform1=transform, transform2=transform, projected_point_source=point,
            policy=ssod.PseudoLabelPolicy(tau_high=.9, tau_mid=.5),
        )
        self.assertEqual(ignored.quality, ssod.PseudoQuality.IGNORE)

    def test_geometry_rejects_high_score_wrong_region(self):
        transform = ssod.ViewTransform((100, 100), (100, 100), False)
        wrong_box = torch.tensor([[0.10, 0.10, 0.10, 0.10]])
        candidate = ssod.select_single_uav_candidate(
            torch.tensor([[10.0]]), wrong_box, transform=transform,
            projected_point_source=torch.tensor([90.0, 90.0]), max_geometry_px=16.0,
        )
        self.assertIsNone(candidate)

    def test_calibration_selects_lowest_threshold_meeting_precision(self):
        obs = [
            ssod.CalibrationObservation(.2, 0, .9, .0),
            ssod.CalibrationObservation(.4, 0, .9, .6),
            ssod.CalibrationObservation(.6, 0, .9, .7),
            ssod.CalibrationObservation(.8, 0, .9, .8),
            ssod.CalibrationObservation(.9, 0, .9, .9),
        ]
        threshold = ssod.calibrate_score_threshold(
            obs, geometry_limit_px=16, stability_min_iou=.6,
            success_iou_threshold=.5, minimum_precision=1.0, min_count=2,
        )
        self.assertAlmostEqual(threshold, .4)

    def test_ema_updates_parameters_and_copies_integer_buffer(self):
        class Tiny(torch.nn.Module):
            def __init__(self, value):
                super().__init__(); self.w=torch.nn.Parameter(torch.tensor([value])); self.register_buffer("n", torch.tensor([int(value)]))
        teacher=Tiny(0.0); student=Tiny(2.0)
        ssod.ema_update(teacher,student,decay=.75)
        self.assertTrue(torch.allclose(teacher.w,torch.tensor([.5])))
        self.assertEqual(int(teacher.n.item()),2)

    def test_medium_pseudo_loss_is_classification_only(self):
        class Matcher:
            def __call__(self, outputs, targets):
                device=outputs["pred_logits"].device
                return [(torch.tensor([0],device=device),torch.tensor([0],device=device))]
        class Criterion:
            def __init__(self):
                self.matcher=Matcher(); self.weight_dict={"loss_class":2.0,"loss_bbox":99.0,"loss_giou":99.0}
            def loss_labels(self, outputs, targets, indices, num_boxes):
                return {"loss_class": outputs["pred_logits"][0,0,0].square()+1}
        outputs={"pred_logits":torch.tensor([[[.2],[.1]]]),"pred_boxes":torch.rand(1,2,4),"aux_outputs":[]}
        targets=[{"labels":torch.zeros(1,dtype=torch.long),"boxes":torch.rand(1,4)}]
        losses=ssod.classification_only_pseudo_loss(Criterion(),outputs,targets)
        self.assertEqual(set(losses),{"loss_class"})
        self.assertNotIn("loss_bbox",losses); self.assertNotIn("loss_giou",losses)

    def test_unsup_ramp(self):
        self.assertAlmostEqual(ssod.linear_unsup_weight(0,100,target=1,ramp_fraction=.1), .1)
        self.assertAlmostEqual(ssod.linear_unsup_weight(9,100,target=1,ramp_fraction=.1), 1.0)
        self.assertAlmostEqual(ssod.linear_unsup_weight(99,100,target=1,ramp_fraction=.1), 1.0)


class ManifestTests(unittest.TestCase):
    def test_stratified_roles_are_deterministic_disjoint_and_train_only(self):
        records=[]
        for seq in ("seqA","seqB"):
            for i in range(100):
                records.append({"sequence_id":seq,"query_uid":i,"query_time":float(i),"range_m":10.0 + (i%5)*20})
        a=ssod_data.assign_stratified_roles(records, seed=7)
        b=ssod_data.assign_stratified_roles(records, seed=7)
        self.assertEqual(a,b)
        self.assertEqual(len(a),10)  # 5% of 200
        self.assertEqual(set(a.values()),{"labeled_train","labeled_calibration"})
        self.assertEqual(sum(v=="labeled_calibration" for v in a.values()),2)

    def test_manifest_requires_verified_box(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"manifest.jsonl"
            path.write_text('{"sequence_id":"s","query_uid":1,"query_time":1.0,"image_path":"x.png","role":"labeled_train","gt_xyz_m":[1,2,3],"range_m":3.7,"box_xyxy_px":null,"gt_2d_valid":false}\n')
            with self.assertRaises(ValueError):
                ssod_data.load_label_manifest(path,require_boxes=True)


if __name__ == "__main__":
    unittest.main()
