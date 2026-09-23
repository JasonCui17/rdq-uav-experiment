import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE=Path(__file__).resolve().parents[1]/'tools/gt_bbox_annotation_server.py'
spec=importlib.util.spec_from_file_location('gt_bbox_annotation_server',SOURCE)
mod=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.seq=self.root/'seq0001'
        for name in ('Image','ground_truth','2d_detect'):(self.seq/name).mkdir(parents=True)
        # Synthetic valid left camera: pinhole-like omni with xi=0, no distortion.
        self.cal={'R':np.eye(3),'t':np.zeros(3),'intr':np.array([0.,100.,100.,50.,50.]),
                  'dist':np.zeros(4),'wh':(100,100),'dt':0.}
        for i in range(6):
            Image.new('RGB',(200,100)).save(self.seq/'Image'/f'{i:.3f}.png')
            np.save(self.seq/'ground_truth'/f'{i:.3f}.npy',np.array([i*.08,0.,1.]))
        # Anchor at frame 2 -> reference box centered at 40,50, projection x=66.
        (self.seq/'2d_detect'/'2.000.txt').write_text('0 0.400 0.500 0.100 0.100\n')
        self.store=mod.SequenceStore(self.root,'seq0001',self.cal,max_gt_gap_s=.05,discrepancy_px=32)

    def test_initial_backward_and_forward_and_existing_preserved(self):
        s=self.store
        self.assertEqual(s.by_name['2.000.png'].status,'confirmed')
        self.assertEqual(s.by_name['1.000.png'].status,'draft')
        self.assertEqual(s.by_name['5.000.png'].status,'draft')
        # Projection at t=2 x=66, at t=4 x=82 => shift +16 from anchor.
        b=s.by_name['4.000.png'].boxes[0]['box']
        self.assertAlmostEqual((b[0]+b[2])/2,56,places=4)
        self.assertEqual((self.seq/'2d_detect'/'2.000.txt').read_text(),'0 0.400 0.500 0.100 0.100\n')
        self.assertFalse((self.seq/'2d_detect'/'4.000.txt').exists())

    def test_new_confirmed_anchor_updates_only_following_drafts(self):
        s=self.store
        previous=s.by_name['1.000.png'].boxes[0]['box'].copy()
        s.save('3.000.png',[{'class_id':0,'box':[40,40,50,50]}])
        self.assertEqual(s.by_name['3.000.png'].status,'confirmed')
        self.assertEqual(s.by_name['1.000.png'].boxes[0]['box'],previous)
        self.assertEqual(s.by_name['4.000.png'].candidate_anchor,'3.000.png')
        self.assertTrue((self.seq/'2d_detect'/'3.000.txt').exists())
        self.assertEqual(s.by_name['2.000.png'].source,'existing_yolo')
        self.assertEqual(s.by_name['2.000.png'].status,'confirmed')

    def test_negative_barrier_and_bulk_delete_with_original_backup(self):
        s=self.store
        s.bulk(['2.000.png','3.000.png'],'delete')
        self.assertEqual((self.seq/'2d_detect'/'2.000.txt').read_text(),'')
        self.assertTrue((self.seq/'2d_detect'/'_annotation_backup'/'2.000.txt').exists())
        self.assertEqual(s.by_name['4.000.png'].status,'unlabeled')
        self.assertEqual(s.by_name['2.000.png'].status,'confirmed_negative')

    def test_confirmation_and_resume(self):
        s=self.store
        f=s.by_name['4.000.png']
        s.save('4.000.png',f.boxes)
        self.assertEqual(s.by_name['4.000.png'].status,'confirmed')
        self.assertEqual(len((self.seq/'2d_detect'/'4.000.txt').read_text().split()),5)
        s2=mod.SequenceStore(self.root,'seq0001',self.cal,max_gt_gap_s=.05,discrepancy_px=32)
        self.assertEqual(s2.by_name['4.000.png'].status,'confirmed')

    def test_large_projection_discrepancy_pauses_anchor(self):
        # With strict 5px threshold, existing bbox center x40 at GT x66 becomes invalid anchor.
        s=mod.SequenceStore(self.root,'seq0001',self.cal,max_gt_gap_s=.05,discrepancy_px=5)
        self.assertFalse(s.by_name['2.000.png'].force_anchor)
        self.assertFalse(s._usable_anchor(s.by_name['2.000.png']))
        self.assertEqual(s.by_name['3.000.png'].status,'unlabeled')
        s.save('2.000.png',s.by_name['2.000.png'].boxes,force_anchor=True)
        self.assertEqual(s.by_name['3.000.png'].status,'draft')

    def test_nearest_gt_too_far_and_invalid_projection(self):
        (self.seq/'ground_truth'/'5.000.npy').unlink()
        s=mod.SequenceStore(self.root,'seq0001',self.cal,max_gt_gap_s=.05,discrepancy_px=32)
        self.assertEqual(s.by_name['5.000.png'].status,'invalid')
        self.assertIsNone(s.by_name['5.000.png'].projection)

    def test_yolo_roundtrip_and_bad_boxes(self):
        box={'class_id':0,'box':[10.,11.,35.,46.]}
        result=mod.yolo_to_xyxy(mod.xyxy_to_yolo(box,(100,100)),(100,100))
        for actual, expected in zip(result['box'],box['box']):
            self.assertAlmostEqual(actual,expected,places=6)
        with self.assertRaises(mod.UserError):mod.xyxy_to_yolo({'class_id':0,'box':[-1,0,50,50]},(100,100))
        with self.assertRaises(mod.UserError):mod.yolo_to_xyxy('0 0.5 0.5 0.5 2',(100,100))

    def test_uncertain_does_not_create_negative_txt(self):
        s=self.store
        s.save('0.000.png',[],uncertain=True)
        self.assertEqual(s.by_name['0.000.png'].status,'uncertain')
        self.assertFalse((self.seq/'2d_detect'/'0.000.txt').exists())
        with self.assertRaises(mod.UserError):s.save('2.000.png',[],uncertain=True)

    def test_batch_confirm_saves_clean_drafts_and_reports_skips(self):
        s=self.store
        original=(self.seq/'2d_detect'/'2.000.txt').read_text()
        s.by_name['4.000.png'].warning='synthetic warning'
        s.by_name['5.000.png'].boxes=[]

        result=s.bulk(['2.000.png','3.000.png','4.000.png','5.000.png'],'confirm')

        self.assertEqual(result['bulk_result']['action'],'confirm')
        self.assertEqual(result['bulk_result']['selected'],4)
        self.assertEqual(result['bulk_result']['saved'],1)
        self.assertEqual(
            result['bulk_result']['skipped'],
            [
                {'name':'2.000.png','reason':'already_confirmed'},
                {'name':'4.000.png','reason':'synthetic warning'},
                {'name':'5.000.png','reason':'no_bbox'},
            ],
        )
        self.assertEqual(s.by_name['3.000.png'].status,'confirmed')
        self.assertTrue((self.seq/'2d_detect'/'3.000.txt').exists())
        self.assertEqual((self.seq/'2d_detect'/'2.000.txt').read_text(),original)
        self.assertFalse((self.seq/'2d_detect'/'4.000.txt').exists())
        self.assertFalse((self.seq/'2d_detect'/'5.000.txt').exists())

    def test_calibration_file_equivalent_projection(self):
        camera=self.root/'camera.yaml';geometry=self.root/'geometry.json'
        camera.write_text('cameras:\n  left:\n    model: omni\n    distortion_model: radtan\n    intrinsics: [0, 100, 100, 50, 50]\n    distortion_coeffs: [0, 0, 0, 0]\n    resolution: [100, 100]\n')
        geometry.write_text(json.dumps({'time_convention':'gt_query_time = image_time + time_offset_s','time_offset_s':0.0,'cameras':{'left':{'rotation_camera_from_gt':np.eye(3).tolist(),'translation_camera_from_gt_m':[0,0,0]}}}))
        cal=mod.load_calibration(camera,geometry)
        self.assertEqual(mod.project_omni([.1,0,1],cal),[60.,50.])

if __name__=='__main__':unittest.main()
