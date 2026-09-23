import unittest,tempfile,json,sys
from pathlib import Path
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_mmuav_observation_path_validation import (paired_errors,metric,tail_stats,prepare_output,validate_fixed,score_saved)


class ValidationTest(unittest.TestCase):
    def test_paired_common_only(self):
        a=np.array([[0,0,0],[np.nan]*3,[2,0,0]],float);b=np.array([[1,0,0],[1,0,0],[np.nan]*3],float)
        common,pa,pb=paired_errors(a,b,np.zeros((3,3)))
        self.assertEqual(common.tolist(),[True,False,False]);self.assertEqual(len(pa),1)

    def test_missing_coverage(self):
        m=metric(np.array([[3,0,0],[np.nan]*3]),2)
        self.assertEqual(m['coverage'],.5);self.assertEqual(m['MSE_coord'],3.)
        self.assertEqual(m['missing_prediction_count'],1)

    def test_coverage_denominator(self):
        m=metric(np.array([[0,0,0],[1,0,0],[np.nan]*3]),3)
        self.assertAlmostEqual(m['coverage'],2/3)
        self.assertEqual(m['matched_timestamp_count'],2)

    def test_tail_thresholds(self):
        h=tail_stats(np.array([[x,0,0] for x in [0,2,3,5,6,10,11]],float),np.arange(7))
        self.assertEqual([h[k] for k in ['error_gt_2m','error_gt_5m','error_gt_10m']],[5,3,1])

    def test_gt_after_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            loader=Mock()
            with self.assertRaises(RuntimeError):score_saved(Path(d)/'not_saved.csv',Path(d)/'original.csv',Path(d),loader)
            loader.assert_not_called()

    def test_fixed_config_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out';prepare_output(p);validate_fixed(p)
            data=json.loads((p/'fixed_config.json').read_text());data['lambda_d']=99
            (p/'fixed_config.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):validate_fixed(p)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out';prepare_output(p)
            with self.assertRaises(FileExistsError):prepare_output(p)
