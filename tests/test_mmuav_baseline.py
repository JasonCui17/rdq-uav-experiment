from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rdq_uav.baselines.mmuav_preprocess import (
    _accumulate_lidar_360_blocks,
    extract_feature_set_predict,
    farthest_point_sample,
)
from tools.run_mmuav_candidate_baseline import (
    SensorFrame,
    build_processing_units,
    build_unique_sensor_frame_index,
    select_processing_units,
)


class MMUAVBaselineTests(unittest.TestCase):
    def test_adapter_keeps_all_native_rate_frames_inside_manifest_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lidar_directory = root / "Seq" / "lidar_360"
            avia_directory = root / "Seq" / "livox_avia"
            lidar_directory.mkdir(parents=True)
            avia_directory.mkdir(parents=True)
            for timestamp in (0.9, 1.0, 1.05, 1.1, 1.2, 2.0, 2.05, 2.1):
                np.save(lidar_directory / f"{timestamp}.npy", np.ones((2, 3)))
            for timestamp in (0.95, 1.01, 1.03, 1.09, 1.11, 2.01, 2.02, 2.09):
                np.save(avia_directory / f"{timestamp}.npy", np.ones((3, 3)))
            rows = [
                {
                    "sequence_id": "Seq", "gt_time": "1.0",
                    "split": "train", "temporal_block": "0",
                },
                {
                    "sequence_id": "Seq", "gt_time": "1.1",
                    "split": "train", "temporal_block": "0",
                },
                {
                    "sequence_id": "Seq", "gt_time": "2.0",
                    "split": "val", "temporal_block": "1",
                },
                {
                    "sequence_id": "Seq", "gt_time": "2.1",
                    "split": "val", "temporal_block": "1",
                },
            ]
            index, audit, time_ranges = build_unique_sensor_frame_index(rows, root)
            self.assertEqual(
                [frame.timestamp for frame in index[("Seq", "lidar_360")]],
                ["1.0", "1.05", "1.1", "2.0", "2.05", "2.1"],
            )
            self.assertEqual(
                [frame.timestamp for frame in index[("Seq", "livox_avia")]],
                ["1.01", "1.03", "1.09", "2.01", "2.02", "2.09"],
            )
            lidar_audit = audit["by_sequence_sensor"]["Seq/lidar_360"]
            self.assertEqual(lidar_audit["source_directory_frames"], 8)
            self.assertEqual(lidar_audit["unique_sensor_frames_by_split"], {"train": 3, "val": 3})
            units, unit_audit = build_processing_units(
                index, time_ranges, max_frames_per_unit=20
            )
            self.assertEqual(len(units), 2)
            self.assertEqual(unit_audit["processing_unit_count"], 2)
            self.assertFalse(unit_audit["physical_frames_processed_more_than_once"])
            selected = select_processing_units(units, "Seq/train_block00_chunk000")
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].split, "train")

    def test_mid360_cap_defines_common_time_chunks_for_native_rate_avia(self) -> None:
        key = ("Seq", "train", 0)
        index = {
            ("Seq", "lidar_360"): [
                SensorFrame(
                    "Seq", "lidar_360", str(float(timestamp)), Path(f"m{timestamp}.npy"),
                    splits={"train"}, manifest_units={("train", 0)},
                )
                for timestamp in range(25)
            ],
            ("Seq", "livox_avia"): [
                SensorFrame(
                    "Seq", "livox_avia", str(timestamp), Path(f"a{timestamp}.npy"),
                    splits={"train"}, manifest_units={("train", 0)},
                )
                for timestamp in np.arange(0.25, 25.0, 0.5)
            ],
        }
        time_ranges = {
            key: {
                "sequence_id": "Seq", "split": "train", "temporal_block": 0,
                "t_start": 0.0, "t_end": 24.9, "manifest_rows": 2,
            }
        }
        units, audit = build_processing_units(index, time_ranges, max_frames_per_unit=20)
        self.assertEqual(len(units), 2)
        self.assertEqual([len(unit.sensor_frames["lidar_360"]) for unit in units], [20, 5])
        self.assertEqual([len(unit.sensor_frames["livox_avia"]) for unit in units], [39, 11])
        self.assertEqual(units[0].time_end, 19.5)
        self.assertEqual(units[1].time_start, 19.5)
        self.assertEqual(audit["source_groups"][0]["generated_chunks"], 2)
        self.assertFalse(audit["frames_are_sampled_or_discarded"])

    def test_source_fps_is_reproducible_when_entry_seed_is_fixed(self) -> None:
        points = np.arange(90, dtype=np.float64).reshape(30, 3)
        np.random.seed(0)
        first = farthest_point_sample(points, 10)
        np.random.seed(0)
        second = farthest_point_sample(points, 10)
        self.assertEqual(first.shape, (10, 3))
        np.testing.assert_array_equal(first, second)

    def test_feature_extraction_is_20_by_9_and_zero_fills_missing_frame(self) -> None:
        data = np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        labels = np.asarray([0, 0])
        time_ind = np.asarray([1.0, 3.0])
        features, cluster_labels = extract_feature_set_predict(data, labels, time_ind)
        self.assertEqual(features.shape, (1, 20, 9))
        self.assertEqual(cluster_labels.shape, (1, 1))
        np.testing.assert_array_equal(features[0, 1], np.zeros(9))
        np.testing.assert_array_equal(features[0, 0, :3], data[0])

    def test_mid360_blocks_are_not_sliding_and_keep_final_20_logic(self) -> None:
        frames = OrderedDict(
            (str(index), np.asarray([[float(index), 1.0, 2.0]]))
            for index in range(1, 26)
        )
        blocks, timestamps = _accumulate_lidar_360_blocks(frames)
        self.assertEqual(list(blocks), ["20", "25"])
        self.assertEqual(timestamps["20"], [str(index) for index in range(1, 21)])
        self.assertEqual(timestamps["25"], [str(index) for index in range(6, 26)])
        self.assertEqual(blocks["20"].shape, (20, 4))
        self.assertEqual(blocks["25"].shape, (20, 4))


if __name__ == "__main__":
    unittest.main()
