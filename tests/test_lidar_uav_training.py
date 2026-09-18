from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import sys
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rdq_uav.lidar_v1.data import ValidationReferenceAdapter
from rdq_uav.lidar_v1.runtime import UpdateScheduler


class ValidationAdapterTests(unittest.TestCase):
    def test_exact_schema_and_position_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"reference.csv"
            with path.open("w",newline="") as handle:
                writer=csv.writer(handle);writer.writerow(["Sequence","Timestamp","Position","Classification"])
                writer.writerow(["seq0001","123.5","[1.0, 2.0, 3.0]","0"])
            adapter=ValidationReferenceAdapter(path)
            self.assertEqual(adapter.total_rows,1);self.assertEqual(adapter.records[0]["sequence_id"],"seq0001")
            self.assertEqual(adapter.records[0]["t0"],adapter.records[0]["gt_timestamp"])
            np.testing.assert_allclose(adapter.records[0]["gt_xyz"],[1,2,3])

    def test_ambiguous_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.csv";path.write_text("seq,time,x,y,z\nseq0001,1,2,3,4\n")
            with self.assertRaisesRegex(ValueError,"Ambiguous validation schema"):ValidationReferenceAdapter(path)

    def test_duplicate_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"duplicate.csv"
            path.write_text("Sequence,Timestamp,Position,Classification\nseq0001,1.0,\"[1,2,3]\",0\nseq0001,1.0,\"[1,2,3]\",0\n")
            with self.assertRaisesRegex(ValueError,"Duplicate validation"):ValidationReferenceAdapter(path)


class SchedulerTests(unittest.TestCase):
    def test_resume_preserves_completed_updates(self):
        parameter=torch.nn.Parameter(torch.tensor(1.0));optimizer=torch.optim.AdamW([parameter],lr=2e-4)
        scheduler=UpdateScheduler(optimizer,100,0.05,2e-4,2e-6);scheduler.prepare_first_update()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],4e-5)
        scheduler.step();state=scheduler.state_dict();lr=optimizer.param_groups[0]["lr"]
        second=UpdateScheduler(optimizer,100,0.05,2e-4,2e-6);second.load_state_dict(state)
        self.assertEqual(second.updates,1);self.assertAlmostEqual(optimizer.param_groups[0]["lr"],lr)


if __name__=="__main__":unittest.main()
