import sys
import tempfile
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from rdq_uav.mmuav.center_regressor import CenterRegressor,predict_delta
from train_mmuav_center_regressor import ClusterDataset


class AblationTests(unittest.TestCase):
    def test_variants_backward_and_shapes(self):
        torch.set_num_threads(1)
        for variant in ('full','points_only','center_only'):
            model=CenterRegressor(variant)
            local=torch.randn(2,64,3);center=torch.randn(2,3)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            for _ in range(2):
                optimizer.zero_grad()
                pred=predict_delta(model,local,center)
                self.assertEqual(pred.shape,(2,3))
                pred.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()

    def test_points_reject_absolute_feature(self):
        m=CenterRegressor('points_only')
        with self.assertRaises(ValueError): m(torch.randn(2,64,3),torch.randn(2,3))

    def test_center_never_reads_shard(self):
        row=dict(shard_path='/nonexistent',shard_index=0,
                 **{f'geometric_{a}':1 for a in 'xyz'},**{f'gt_{a}':2 for a in 'xyz'})
        dataset=ClusterDataset([row],64,False,'center_only')
        local,center,gt=dataset[0]
        self.assertEqual(local.shape,(0,3))
        self.assertEqual(dataset.cache,{})
        self.assertFalse(hasattr(CenterRegressor('center_only'),'point_mlp'))

    def test_full_old_checkpoint(self):
        path=ROOT/'outputs/mmuav_paper_reproduction/center_regression/pointnet_m2/best_val_loss.pth'
        if not path.exists(): self.skipTest('Local frozen checkpoint absent')
        model=CenterRegressor()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.eval()
        x=torch.randn(2,64,3);c=torch.randn(2,3)
        expected=model.head(torch.cat([model.point_mlp(x).amax(1),c],-1))
        torch.testing.assert_close(model(x,c),expected,rtol=0,atol=0)

    def test_tiny_one_epoch_trainers(self):
        from unittest.mock import patch
        import train_mmuav_center_regressor as trainer
        from test_mmuav_center_regression import test_synthetic_trainer
        original=trainer.main
        for variant in ('points_only','center_only'):
            def one_epoch():
                argv=list(sys.argv)
                argv[argv.index('--epochs')+1]='1'
                argv+=['--variant',variant]
                with patch.object(sys,'argv',argv): original()
            with tempfile.TemporaryDirectory() as tmp, patch.object(trainer,'main',one_epoch):
                test_synthetic_trainer(Path(tmp))


if __name__=='__main__': unittest.main()
