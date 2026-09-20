"""Current spatial-only correctness contracts retained after protocol cleanup."""
import copy,sys,unittest
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import CandidateSelector,LiDARUAVDetector
from rdq_uav.lidar_v2.contracts import effective_config,validate_frozen_v2_config
from rdq_uav.lidar_v2.geometry import decode_residual,encode_residual
from rdq_uav.lidar_v2.training import better_spatial
CFG=yaml.safe_load((ROOT/'configs/lidar_uav_v2.yaml').read_text())

class CorrectnessGateTests(unittest.TestCase):
    def test_selector_ranks_raw_fp32_logits(self):
        selector=CandidateSelector(CFG)
        for values,expected in (([7,10],[1,0]),([-10,-8],[1,0])):
            logits=torch.tensor(values,dtype=torch.bfloat16)
            output=dict(logits=logits,pred_xyz=torch.tensor([[0.,0,0],[3.,0,0]]),
                fine_features=torch.zeros(2,128),source_token_id=torch.tensor([0,1]),
                batch_index=torch.zeros(2,dtype=torch.long))
            self.assertEqual(selector(output)[0]['raw']['source_token_id'].tolist(),expected)
    def test_config_is_full_query_and_fail_fast(self):
        validate_frozen_v2_config(CFG)
        effective,_,_=effective_config(CFG,torch.device('cpu'))
        self.assertEqual(effective['train']['protocol'],'full_query_v1')
        for path,value in ((('model','transformer','l2_global'),False),(('model','head','residual_scale_m'),2.0)):
            changed=copy.deepcopy(CFG);changed[path[0]][path[1]][path[2]]=value
            with self.assertRaises(ValueError):validate_frozen_v2_config(changed)
    def test_residual_codec_round_trip(self):
        torch.manual_seed(4);center=torch.randn(32,3);target=torch.randn(32,3)
        decoded=decode_residual(encode_residual(target,center,1.),center,1.)
        self.assertLessEqual(float((decoded-target).abs().max()),2e-7)
    def test_spatial_checkpoint_comparator(self):
        best=dict(epoch=2,nms_recall_at_10_1m=.8,nms_top1_success_1m=.7,nms_top1_error_median=.4)
        self.assertTrue(better_spatial(dict(best,epoch=3,nms_recall_at_10_1m=.81),best))
        self.assertTrue(better_spatial(dict(best,epoch=3,nms_top1_success_1m=.71),best))
        self.assertTrue(better_spatial(dict(best,epoch=3,nms_top1_error_median=.3),best))
        self.assertFalse(better_spatial(dict(best,epoch=3),best))
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in LiDARUAVDetector(CFG).parameters()),1045352)

if __name__=='__main__':unittest.main()
