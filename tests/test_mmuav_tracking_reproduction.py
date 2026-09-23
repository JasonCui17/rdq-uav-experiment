from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools/run_mmuav_tracking_reproduction.py"
SPEC = importlib.util.spec_from_file_location("mmuav_tracking_reproduction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeState:
    state_vector = np.arange(6, dtype=float).reshape(6, 1)
    timestamp = datetime.fromtimestamp(1234.5)


class ReproductionRunnerTests(unittest.TestCase):
    def test_state_row_maps_xyz_and_velocity_without_shape_leak(self):
        row = MODULE._state_row("track-a", FakeState(), 3, 2)
        self.assertEqual(row["timestamp"], 1234.5)
        self.assertEqual([row[k] for k in ("x", "y", "z")], [0.0, 2.0, 4.0])
        self.assertEqual([row[k] for k in ("vx", "vy", "vz")], [1.0, 3.0, 5.0])
        self.assertEqual(row["track_age"], 3)
        self.assertEqual(row["num_updates"], 2)

    def test_inspect_raw_uses_timestamp_order_and_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for modality in MODULE.RAW_MODALITIES:
                folder = root / modality
                folder.mkdir()
                np.save(folder / "2.0.npy", np.ones((4, 3), dtype=np.float64))
                np.save(folder / "1.0.npy", np.ones((5, 3), dtype=np.float64))
            result = MODULE.inspect_raw(root)
        self.assertEqual(result["lidar_360"]["frames"], 2)
        self.assertEqual(result["lidar_360"]["first_timestamp"], 1.0)
        self.assertEqual(result["livox_avia"]["last_timestamp"], 2.0)
        self.assertEqual(result["livox_avia"]["sample_shape"], [5, 3])

    def test_existing_staging_reuses_outputs_without_raw_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            staging = output / "seq0001_staging"
            for name in ("lidar_360_processed", "livox_avia_processed", "lidar_fusion"):
                (staging / name).mkdir(parents=True)
            resolved = MODULE.existing_staging(Path("/unused/seq0001"), output)
            self.assertEqual(resolved[0], staging)
            self.assertTrue(resolved[1].is_dir())
            self.assertTrue(resolved[2].is_dir())


if __name__ == "__main__":
    unittest.main()
