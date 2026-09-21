"""P4 unit/contract tests; no formal dataset audit is performed here."""

import unittest

from rdq_uav.multimodal_v1.geometry_audit import (
    evaluate_geometry_gate,
    summarize_geometry_measurements,
    validate_threshold_contract,
)


class GeometryGateContractTests(unittest.TestCase):
    def records(self):
        result = []
        for pairing, distances in (
            ("real", (2.0, 4.0)),
            ("same_sequence_shuffled", (40.0, 80.0)),
        ):
            for index, distance in enumerate(distances):
                result.append(
                    {
                        "pairing": pairing,
                        "nearest_3d_distance_m": distance / 10,
                        "image_reprojection_distance_px": distance,
                        "nearest_center_distance_px": distance + 1,
                        "nearest_bbox_distance_px": max(0, distance - 1),
                        "feature_neighborhood_hit_stride_4": index == 0,
                        "feature_neighborhood_hit_stride_8": True,
                        "feature_neighborhood_hit_stride_16": True,
                    }
                )
        return result

    def test_units_and_coverage_are_separate(self):
        summary = summarize_geometry_measurements(self.records())
        real = summary["groups"]["real"]
        self.assertEqual(real["nearest_3d_distance_m"]["unit"], "m")
        self.assertEqual(real["image_reprojection_distance_px"]["unit"], "px")
        self.assertEqual(real["pixel_coverage"]["coverage_at_8px"], 1.0)
        self.assertEqual(
            real["feature_neighborhood_coverage"]["stride_4_3x3"], 0.5
        )

    def test_unfrozen_or_placeholder_thresholds_refuse_formal_gate(self):
        with self.assertRaises(ValueError):
            validate_threshold_contract(
                {"thresholds_frozen": False, "thresholds": {}}
            )
        with self.assertRaises(ValueError):
            validate_threshold_contract(
                {
                    "thresholds_frozen": True,
                    "threshold_commit": "abcdef0",
                    "thresholds": {"metric": {"op": "le", "value": None}},
                }
            )

    def test_frozen_thresholds_produce_deterministic_pass_and_fail(self):
        summary = summarize_geometry_measurements(self.records())
        metric = "groups.real.image_reprojection_distance_px.p95"
        base = {
            "thresholds_frozen": True,
            "threshold_commit": "abcdef0",
        }
        passed = evaluate_geometry_gate(
            summary, {**base, "thresholds": {metric: {"op": "le", "value": 4.0}}}
        )
        failed = evaluate_geometry_gate(
            summary, {**base, "thresholds": {metric: {"op": "le", "value": 2.0}}}
        )
        self.assertEqual(passed["status"], "PASS")
        self.assertEqual(failed["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
