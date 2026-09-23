from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rdq_uav.baselines.mmuav_preprocess import (
    MyLSTMClassifier,
    _accumulate_lidar_360_blocks,
    extract_feature_set_predict,
    farthest_point_sample,
)
from tools.run_mmuav_candidate_baseline import (
    SensorFrame,
    build_eps_comparison_diagnostics,
    build_pre_lstm_cluster_diagnostics,
    build_processing_units,
    build_unique_sensor_frame_index,
    save_pre_lstm_cluster_diagnostics,
    save_eps_comparison_diagnostics,
    select_processing_units,
)
from tools.audit_mmuav_positive_clusters_geometry import nearest_path, summarize_cluster
from tools.run_mmuav_diagnostics_batch import (
    build_diagnostic_windows,
    diagnostic_complete,
)


class MMUAVBaselineTests(unittest.TestCase):
    def test_batch_diagnostics_preserve_source_final_20_window(self) -> None:
        frames = []
        for frame_index in range(45):
            frame = SensorFrame(
                sequence_id="Seq", sensor_type="lidar_360",
                timestamp=f"{frame_index / 10:.1f}", path=Path(f"{frame_index}.npy"),
            )
            frame.splits.add("train")
            frame.manifest_units.add(("train", 3))
            frames.append(frame)
        windows, audit = build_diagnostic_windows({("Seq", "lidar_360"): frames})
        self.assertEqual([window.name for window in windows], [
            "train_block03_chunk000", "train_block03_chunk001",
            "train_block03_final20",
        ])
        self.assertEqual([frame.timestamp for frame in windows[0].frames], [
            f"{index / 10:.1f}" for index in range(20)
        ])
        self.assertEqual([frame.timestamp for frame in windows[-1].frames], [
            f"{index / 10:.1f}" for index in range(25, 45)
        ])
        self.assertEqual(audit[0]["remainder_frames"], 5)
        self.assertTrue(audit[0]["source_final_20_window"])

    def test_batch_diagnostic_complete_requires_both_eps_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            unit_dir = Path(temporary_directory)
            required = [
                unit_dir / "diagnostics/pre_lstm_clusters/cluster_diagnostics.json",
                unit_dir / "diagnostics/pre_lstm_clusters/cluster_features.npz",
                unit_dir / "diagnostics/eps1_cluster_features.npz",
                unit_dir / "diagnostics/eps_comparison.json",
                unit_dir / "unit_metadata.json",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertTrue(diagnostic_complete(unit_dir))
            required[-2].unlink()
            self.assertFalse(diagnostic_complete(unit_dir))

    def test_weak_geometry_audit_keeps_missing_bbox_inconclusive(self) -> None:
        times = np.asarray([1.0, 2.0, 3.0])
        paths = [Path("1.npy"), Path("2.npy"), Path("3.npy")]
        selected_time, selected_path = nearest_path(times, paths, 1.6)
        self.assertEqual(selected_time, 2.0)
        self.assertEqual(selected_path, Path("2.npy"))
        rows = [
            {
                "cluster_id": 12,
                "distance_center_to_gt_m": distance,
                "min_point_to_gt_m": distance / 2,
                "projection_valid": True,
                "official_bbox_available": False,
            }
            for distance in (1.0, 2.0)
        ]
        summary = summarize_cluster(rows, 0.98)
        self.assertEqual(summary["verdict"], "INCONCLUSIVE")
        self.assertEqual(summary["geometry_confidence"], "LOW")
        self.assertIsNone(summary["inside_gt_bbox_frames"])

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

    def test_pre_lstm_diagnostic_shapes_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "lstm_model.pth"
            torch.manual_seed(0)
            model = MyLSTMClassifier(9, 64, 1, 2)
            torch.save(model.state_dict(), checkpoint)
            frames = OrderedDict()
            for frame_index in range(20):
                offsets = np.linspace(-0.05, 0.05, 12)
                first = np.column_stack((
                    offsets,
                    np.zeros_like(offsets),
                    np.full_like(offsets, 1.0 + frame_index * 0.001),
                ))
                second = first + np.asarray([10.0, 0.0, 0.0])
                frames[str(float(frame_index))] = np.vstack((first, second))

            diagnostic = build_pre_lstm_cluster_diagnostics(frames, checkpoint)
            cluster_count = diagnostic["summary"]["dbscan_clusters"]
            self.assertEqual(cluster_count, 2)
            self.assertEqual(diagnostic["features"].shape, (2, 20, 9))
            self.assertEqual(diagnostic["logits"].shape, (2, 2))
            self.assertEqual(diagnostic["probabilities"].shape, (2, 2))
            np.testing.assert_allclose(
                diagnostic["probabilities"].sum(axis=1), np.ones(2), atol=1e-6
            )
            self.assertEqual(diagnostic["per_frame_counts"].shape, (2, 20))
            np.testing.assert_array_equal(
                diagnostic["per_frame_counts"], np.full((2, 20), 12)
            )

            output_dir = root / "diagnostics"
            report = save_pre_lstm_cluster_diagnostics(
                diagnostic, output_dir, root / "missing_calibration.json"
            )
            self.assertTrue((output_dir / "cluster_diagnostics.json").is_file())
            self.assertTrue((output_dir / "cluster_features.npz").is_file())
            self.assertEqual(
                report["gt_oracle_status"], "SKIPPED_UNVERIFIED_TRANSFORM"
            )
            arrays = np.load(output_dir / "cluster_features.npz")
            self.assertEqual(arrays["features"].shape, (2, 20, 9))
            self.assertEqual(arrays["logits"].shape, (2, 2))
            self.assertEqual(arrays["probabilities"].shape, (2, 2))

            comparison = build_eps_comparison_diagnostics(frames, checkpoint)
            self.assertEqual(comparison["eps2"]["features"].shape, (2, 20, 9))
            self.assertEqual(comparison["eps1"]["features"].shape, (2, 20, 9))
            self.assertEqual(comparison["eps1"]["logits"].shape, (2, 2))
            self.assertEqual(comparison["eps1"]["probabilities"].shape, (2, 2))
            comparison_report = save_eps_comparison_diagnostics(
                comparison, output_dir
            )
            self.assertFalse(comparison_report["gt_used"])
            self.assertTrue((output_dir / "eps_comparison.json").is_file())
            eps1_arrays = np.load(output_dir / "eps1_cluster_features.npz")
            self.assertEqual(eps1_arrays["features"].shape, (2, 20, 9))
            self.assertEqual(eps1_arrays["logits"].shape, (2, 2))
            self.assertEqual(eps1_arrays["probabilities"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
