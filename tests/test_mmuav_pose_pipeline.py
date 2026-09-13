import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from rdq_uav.mmuav.pose_trajectory import track_candidates,select_track,ar_complete,resample
from rdq_uav.mmuav.center_regressor import CenterRegressor
from run_mmuav_pose_pipeline import infer_candidates,score


class PipelineTests(unittest.TestCase):
    def test_all_candidates_no_gt(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'lidar_fusion').mkdir()
            np.save(folder/'lidar_fusion/1.npy',[[1.,2.,3.],[20.,21.,22.]])
            frames,audit=infer_candidates(folder,CenterRegressor())
            self.assertEqual(len(audit),2)
            self.assertEqual(frames[0][1].shape,(2,3))
            self.assertTrue(np.isfinite(frames[0][1]).all())

    def test_tracker_selection_format(self):
        frames=[(float(i)/10,np.array([[i/10,0.,1.]])) for i in range(10)]
        rows=track_candidates(frames);selected=select_track(rows)
        self.assertGreater(len(selected),1)
        self.assertTrue(all(k in rows[0] for k in ['timestamp','track_id','x','y','z','vx','vy','vz']))
        self.assertEqual(select_track([]),[])

    def test_ar3_synthetic_no_fit(self):
        t=np.array([0.,.1,.2,.5,.6]);p=np.column_stack([t,t*2,t*0])
        times,points,coeff=ar_complete(t,p,np.arange(0,.61,.1),fit=False)
        self.assertEqual(coeff.shape,(3,4))
        self.assertTrue(np.isfinite(points).all())
        self.assertGreater(len(times),len(t))

    def test_spline_finite_bounded(self):
        t=np.arange(10)*.1;p=np.column_stack([t,t**2,t*0])
        self.assertTrue(np.isfinite(resample(t,p,t,True)).all())
        self.assertTrue(np.isnan(resample(t,p,np.array([-1.,2.]),True)).all())

    def test_missing_not_hidden(self):
        result=score(np.array([[1.,2.,3.],[np.nan]*3]),np.zeros((2,3)))
        self.assertEqual(result['missing_prediction_count'],1)
        self.assertEqual(result['coverage'],.5)

    def test_source_empty_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'lidar_fusion').mkdir()
            np.save(folder/'lidar_fusion/1.npy',np.array([]))
            frames,audit=infer_candidates(folder,None)
            self.assertEqual(audit,[])
            self.assertEqual(frames[0][1].shape,(0,3))

    def test_gt_only_after_predictions(self):
        source=(ROOT/'tools/run_mmuav_pose_pipeline.py').read_text()
        self.assertLess(source.index("trajectory(out/'smoothed_final_trajectory.csv'"),source.index('gt_t,gt=load_gt(raw)'))
        import inspect
        import ast
        tree=ast.parse(inspect.getsource(infer_candidates))
        identifiers={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
        self.assertFalse(identifiers & {'gt','load_gt','accepted_rows'})
        self.assertNotIn('gt',inspect.signature(track_candidates).parameters)


if __name__=='__main__':unittest.main()
