"""P3 multimodal query identity, clock and missing-modality contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rdq_uav.multimodal_v1.data import (
    LeftImageIndex,
    MultimodalQueryDataset,
    collate_multimodal_queries,
    deduplicate_multimodal_queries,
)


def lidar_query(sequence: str, uid: int, query_time: float) -> dict:
    event_time = query_time - 0.1
    return {
        "sequence_id": sequence,
        "query_uid": uid,
        "sample_id": f"{sequence}_{uid}",
        "query_time": query_time,
        "points": torch.tensor([[1.0, 2.0, 3.0]]),
        "sensor_id": torch.tensor([0]),
        "delta_t": torch.tensor([-0.1]),
        "supervision_recent_mask": torch.tensor([True]),
        "event_count": 1,
        "event_timestamps": [event_time],
        "event_sequence_ids": [sequence],
        "target_valid": True,
        "target_timestamp": query_time,
        "target_xyz": torch.tensor([1.0, 2.0, 3.0]),
    }


class Records(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class MultimodalDataContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def image(self, sequence: str, timestamp: float) -> Path:
        path = self.root / sequence / "Image" / f"{timestamp:.6f}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_clock_offset_formula_selects_correct_image(self):
        offset = -0.125
        correct = self.image("seqA", 10.125)
        self.image("seqA", 10.000)
        match = LeftImageIndex(self.root, time_offset_s=offset).match("seqA", 10.0)
        self.assertEqual(match.path, correct)
        self.assertAlmostEqual(match.gap_s, 0.0)

    def test_missing_image_has_explicit_mask_and_no_fake_path(self):
        wrapped = MultimodalQueryDataset(
            Records([lidar_query("seqA", 1, 10.0)]),
            LeftImageIndex(self.root, time_offset_s=0.0),
            calibration_handle="calibration.json",
        )
        sample = wrapped[0]
        self.assertFalse(sample["m_V"])
        self.assertIsNone(sample["left_image_path"])
        self.assertTrue(sample["m_R"])

    def test_composite_key_prevents_cross_sequence_deduplication(self):
        a = lidar_query("seqA", 7, 10.0)
        b = lidar_query("seqB", 7, 10.0)
        for record in (a, b):
            record["left_image_path"] = None
        unique, inverse = deduplicate_multimodal_queries([a, a, b, b])
        self.assertEqual(len(unique), 2)
        self.assertEqual(inverse.tolist(), [0, 0, 1, 1])

    def test_collate_retains_causal_lidar_and_modality_contract(self):
        image = self.image("seqA", 10.125)
        dataset = MultimodalQueryDataset(
            Records([lidar_query("seqA", 1, 10.0)]),
            LeftImageIndex(self.root, time_offset_s=-0.125),
            calibration_handle="calibration.json",
        )
        batch = collate_multimodal_queries([dataset[0]])
        self.assertEqual(batch["left_image_path"], [str(image)])
        self.assertEqual(batch["m_R"].tolist(), [True])
        self.assertEqual(batch["m_V"].tolist(), [True])
        self.assertLessEqual(float(batch["delta_t"].max()), 0.0)

    def test_future_lidar_is_rejected_before_image_binding(self):
        record = lidar_query("seqA", 1, 10.0)
        record["event_timestamps"] = [10.1]
        record["delta_t"] = torch.tensor([0.1])
        dataset = MultimodalQueryDataset(
            Records([record]),
            LeftImageIndex(self.root, time_offset_s=0.0),
            calibration_handle="calibration.json",
        )
        with self.assertRaises(AssertionError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
