"""P1: LiDAR V2 stage adapter must be an exact spatial identity seam."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from rdq_uav.lidar_v2 import LiDARUAVDetector
from rdq_uav.multimodal_v1 import COMPONENTS, LiDARV2PyramidAdapter

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/lidar_uav_v2.yaml").read_text())


def packed_batch(seed: int, counts: tuple[int, ...]) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    points = []
    point_batch_index = []
    for sample_index, count in enumerate(counts):
        cloud = torch.randn(count, 3, generator=generator) * (0.4 + sample_index)
        cloud += torch.tensor([sample_index * 5.0 - 2.5, -1.25, 0.75])
        points.append(cloud)
        point_batch_index.append(torch.full((count,), sample_index, dtype=torch.long))
    packed = torch.cat(points)
    return {
        "points": packed,
        "point_batch_index": torch.cat(point_batch_index),
        "sensor_id": torch.randint(0, 2, (len(packed),), generator=generator),
        "delta_t": -torch.rand(len(packed), generator=generator),
        "num_samples": len(counts),
    }


class LiDARV2PyramidAdapterIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.detector = LiDARUAVDetector(CFG).eval()
        self.adapter = LiDARV2PyramidAdapter(detector=self.detector).eval()

    def assert_output_identity(self, batch: dict[str, object]) -> None:
        with torch.no_grad():
            reference = self.detector.spatial_forward(batch)
            actual = self.adapter(batch)
        tensor_fields = (
            "logits",
            "residual_xyz",
            "pred_xyz",
            "fine_features",
            "voxel_centers",
            "batch_index",
            "source_token_id",
        )
        for field in tensor_fields:
            self.assertTrue(
                torch.equal(reference[field], actual[field]),
                f"{field} max diff: "
                f"{(reference[field] - actual[field]).abs().max().item()}",
            )
        self.assertEqual(reference["aux_stats"], actual["aux_stats"])
        for ref_level, actual_level in zip(
            reference["layouts"].levels, actual["layouts"].levels, strict=True
        ):
            self.assertTrue(torch.equal(ref_level.coords, actual_level.coords))
            self.assertTrue(
                torch.equal(ref_level.batch_index, actual_level.batch_index)
            )

    def test_random_sparse_batches_are_exact(self) -> None:
        for seed, counts in ((1, (13,)), (2, (27, 19)), (3, (64, 7, 31))):
            with self.subTest(seed=seed, counts=counts):
                self.assert_output_identity(packed_batch(seed, counts))

    def test_one_prepare_context_is_used_by_all_stages(self) -> None:
        batch = packed_batch(9, (21, 18))
        ctx = self.adapter.prepare(batch)
        hierarchy_id = id(ctx.hierarchy)
        r0 = self.adapter.run_stage0(ctx.r0_pre, ctx)
        r1 = self.adapter.run_stage1(self.adapter.merge01(r0, ctx), ctx)
        r2 = self.adapter.run_stage2(self.adapter.merge12(r1, ctx), ctx)
        fine = self.adapter.decode_to_fine(r0, r1, r2, ctx)
        output = self.adapter.candidate_head(fine, ctx)
        self.assertEqual(id(ctx.hierarchy), hierarchy_id)
        self.assertIs(output["layouts"], ctx.hierarchy)
        self.assertEqual(output["aux_stats"]["token_counts"], [
            len(ctx.level0.coords), len(ctx.level1.coords), len(ctx.level2.coords)
        ])

    def test_registry_builds_adapter(self) -> None:
        adapter = COMPONENTS.build(
            {"name": "lidar_v2_pyramid"}, detector=self.detector
        )
        self.assertIsInstance(adapter, LiDARV2PyramidAdapter)
        self.assertIs(adapter.detector, self.detector)


if __name__ == "__main__":
    unittest.main()
