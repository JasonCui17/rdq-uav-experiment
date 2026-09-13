import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rdq_uav.mmuav.track_robustness import select_track_v2


class TrackRobustnessTest(unittest.TestCase):
    def row(self,t,x,tid,update=True):
        return dict(timestamp=t,track_id=tid,x=x,y=0.,z=0.,vx=1.,vy=0.,vz=0.,measurement_update=update)

    def test_supported_stitching(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(1.,9.,'a',False)]
        rows += [self.row(t,t,'b') for t in [.3,.4]]
        result=select_track_v2(rows)
        self.assertEqual([r['track_id'] for r in result],['a']*3+['b']*2)
        self.assertTrue(all(r['measurement_update'] for r in result))

    def test_gate_and_empty(self):
        self.assertEqual(select_track_v2([]),[])
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(t,100.,'b') for t in [.3,.4]]
        self.assertEqual(len(select_track_v2(rows)),3)

    def test_no_gt_inputs(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]
        self.assertEqual(len(select_track_v2(rows)),3)
