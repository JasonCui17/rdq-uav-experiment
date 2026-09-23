import unittest,tempfile,sys
from pathlib import Path
from unittest.mock import patch
from dataclasses import FrozenInstanceError
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from rdq_uav.mmuav.bounded_recovery import recover_bounded_missing,RecoveryConfig
from rdq_uav.mmuav.observation_path import select_observation_path
from evaluate_mmuav_bounded_recovery_validation import apply_shared_recovery,prepare_output


class BoundedRecoveryTest(unittest.TestCase):
    def row(self,t,x=0):
        return dict(timestamp=t,x=x,y=0.,z=0.,vx=1.,vy=0.,vz=0.,state_type='observed',track_id='a',measurement_update=True)

    def test_same_recovery(self):
        path=[self.row(0),self.row(.2,.2)]
        self.assertEqual(recover_bounded_missing(path,[0,.1,.2]),recover_bounded_missing(path,[0,.1,.2]))

    def test_selector_metadata_isolation(self):
        path=[self.row(0),self.row(.2,.2)]
        altered=[dict(r,track_id='other',path_score=-999,selector='arbitrary') for r in path]
        self.assertEqual(recover_bounded_missing(path,[.1]),recover_bounded_missing(altered,[.1]))

    def test_internal_short(self):
        r=recover_bounded_missing([self.row(0),self.row(.2,.2)],[.1])[0]
        self.assertAlmostEqual(r['x'],.1);self.assertEqual(r['recovery_source'],'internal_interpolation')

    def test_internal_long(self):
        r=recover_bounded_missing([self.row(0),self.row(2,2)],[1])[0]
        self.assertEqual(r['state_type'],'missing')

    def test_tail_bound(self):
        result=recover_bounded_missing([self.row(0)],[.05,.1,.11])
        self.assertEqual([r['state_type'] for r in result],['predicted','predicted','missing'])

    def test_long_tail_blocked(self):
        result=recover_bounded_missing([self.row(0)],[.2,1,5.5])
        self.assertTrue(all(r['state_type']=='missing' for r in result))

    def test_feedback_blocked(self):
        path=[self.row(0),self.row(.2,.2)];pred=recover_bounded_missing(path,[.1])[0]
        with self.assertRaises(ValueError):recover_bounded_missing([pred],[.15])
        r=dict(pred,measurement_update=False,track_id='extra',vx=1.,vy=0.,vz=0.)
        self.assertEqual(select_observation_path(path)['score'],select_observation_path(path+[r])['score'])

    def test_state_labels(self):
        result=recover_bounded_missing([self.row(0),self.row(.2,.2)],[0,.1,.2,.25,5.5])
        self.assertEqual([r['state_type'] for r in result],['observed','predicted','observed','predicted','missing'])

    def test_same_rule_config(self):
        config=RecoveryConfig();control=[self.row(0)];obs=[self.row(.1)]
        with patch('evaluate_mmuav_bounded_recovery_validation.recover_bounded_missing',return_value=[]) as fn:
            apply_shared_recovery(control,obs,[0,.1],config)
            self.assertEqual(fn.call_count,2)
            self.assertIs(fn.call_args_list[0].args[2],fn.call_args_list[1].args[2])
        with self.assertRaises(FrozenInstanceError):config.tail_prediction_limit=99

    def test_no_gt(self):
        a=[self.row(0),self.row(.2,.2)]
        b=[dict(r,gt_x=999,oracle=True,gt_error=0) for r in a]
        self.assertEqual(recover_bounded_missing(a,[.1]),recover_bounded_missing(b,[.1]))

    def test_deterministic(self):
        a=[self.row(0),self.row(.2,.2)]
        self.assertEqual(recover_bounded_missing(a,[.1]),recover_bounded_missing(list(reversed(a)),[.1]))

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'output';prepare_output(out,{})
            with self.assertRaises(FileExistsError):prepare_output(out,{})
