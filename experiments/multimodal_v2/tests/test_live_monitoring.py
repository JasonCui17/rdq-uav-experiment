import math
from types import SimpleNamespace
import torch
import pytest
from rdq_uav.multimodal_v2.monitoring import (
    RunningLocalization, dino_loss_components, visual_metric_rows, StageProgressBar,
)
from rdq_uav.multimodal_v2.paper_metrics import localization_metrics, coco_bbox_metrics
from rdq_uav.multimodal_v2.training import MultimodalV2LightningModule


def test_running_metrics_match_pooled_metrics_not_mean_batch_rmse():
    rows = [{'has_gt3d': True, 'gt_xyz': [0, 0, 0], 'pred_xyz': p}
            for p in ([1, 0, 0], [3, 0, 0], [3, 0, 0], None)]
    state = RunningLocalization()
    state.update(rows[:1]); assert state.compute()['rmse_3d_m'] == 1
    state.update(rows[1:]); m = state.compute()
    assert m['rmse_3d_m'] == pytest.approx(math.sqrt(19/3))
    expected = localization_metrics(rows)
    for key in ('rmse_x_m', 'rmse_y_m', 'rmse_z_m', 'rmse_3d_m', 'coverage', 'success_1m'):
        assert m[key] == pytest.approx(expected[key])
    assert m['coverage'] == .75 and m['gt_queries'] == 4
    assert RunningLocalization().compute()['rmse_3d_m'] is None


def test_native_loss_parts_are_disjoint_detached_and_preserve_total():
    weighted = {k: torch.tensor(float(i), requires_grad=True) for i, k in enumerate([
        'loss_class', 'loss_bbox', 'loss_giou', 'loss_class_0', 'loss_bbox_enc',
        'loss_giou_dn_0', 'loss_class_dn', 'custom'], 1)}
    parts = dino_loss_components(weighted)
    assert set(parts) == {'cls', 'bbox', 'giou', 'aux', 'enc', 'dn', 'other'}
    assert float(sum(parts.values())) == float(sum(weighted.values()).detach())
    assert not any(v.requires_grad for v in parts.values())
    assert float(parts['dn']) == 13


def inputs():
    target = SimpleNamespace(has_xyz=torch.tensor([True]), xyz_m=torch.zeros(1,3),
                             has_box=torch.tensor([True]), box_xyxy_px=torch.tensor([[10.,10.,20.,20.]]))
    output = SimpleNamespace(xyz_m=torch.tensor([[1.,0.,0.]]),
                             box_xyxy_px=target.box_xyxy_px.clone(), score_2d_after=torch.tensor([.9]),
                             hypothesis_type=torch.tensor([2]),
                             top3d_indices=lambda n: [torch.empty(0,dtype=torch.long)],
                             top2d_indices=lambda n: [torch.tensor([0])])
    batch = dict(m_R=torch.tensor([False]), m_V=torch.tensor([True]), sample_id=['a'],
                 sequence_id=['seq0001'], query_time=torch.tensor([10.],dtype=torch.float64),
                 image_time=torch.tensor([10.],dtype=torch.float64),
                 left_image_path=['seq0001/Image/10.png'], image_source_wh=torch.tensor([[100,100]]),
                 vision_batch_index=torch.tensor([0]))
    return target, output, batch


def test_visual_rows_use_compact_image_dimensions_and_original_indices():
    target, output, batch = inputs()
    target.has_box = torch.tensor([False, True])
    target.box_xyxy_px = target.box_xyxy_px.repeat(2,1)
    output.top2d_indices = lambda n: [torch.empty(0,dtype=torch.long), torch.tensor([0])]
    batch.update(vision_batch_index=torch.tensor([1]), sample_id=['absent','a'],
                 sequence_id=['seq0001','seq0001'], query_time=torch.tensor([9.,10.]),
                 image_time=torch.tensor([float('nan'),10.]), left_image_path=[None,'seq0001/Image/10.png'])
    rows = visual_metric_rows(output,target,batch)
    assert rows[0]['sample_id'] == 'a' and rows[0]['image_source_wh'] == [100,100]
    assert coco_bbox_metrics(rows)['AP'] == pytest.approx(1)


@pytest.mark.parametrize('stage', ['B0', 'B1'])
def test_partial_final_reset_and_real_lightning_checkpoint(tmp_path, monkeypatch, stage):
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    import rdq_uav.multimodal_v2.training as training
    target, output, batch = inputs()
    if stage == 'B0':
        batch['m_R'] = torch.tensor([True])
        target.has_box = torch.tensor([False])
        output.hypothesis_type = torch.tensor([1])
        output.top3d_indices = lambda n: [torch.tensor([0])]
    network = torch.nn.Linear(1,1,bias=False)
    network.weight.data.fill_(1)
    runtime = SimpleNamespace(model=network, initialization={'stage':stage})
    config = {'logging': {'ap_every_val_batches': 1}}
    class ToyModule(MultimodalV2LightningModule):
        def _refresh_external_device(self): pass
        def configure_optimizers(self): return torch.optim.SGD(self.network.parameters(),lr=.01)
        def on_train_epoch_start(self):
            super().on_train_epoch_start()
            assert not self._last_visual  # A limited sanity pass is not full validation.
    module = ToyModule(runtime,config)
    def forward(*args, **kwargs):
        loss = network.weight.square().sum()
        return dict(output=output,targets=target,loss=loss,has_trainable_loss=True,
                    loss_rank_3d=loss*0,loss_rank_2d=loss*0,
                    loss_components={'cls':loss.detach()},
                    **{k:1 for k in ('n_gt3d','n_with_3d_candidate','n_with_positive_3d',
                        'n_negative_only_3d','n_no_3d_candidate','n_3d_loss_queries','n_gt2d',
                        'n_with_2d_candidate','n_with_positive_2d','n_negative_only_2d',
                        'n_no_2d_candidate','n_2d_loss_queries')})
    monkeypatch.setattr(training,'forward_step',forward)
    monitor = 'val/AP' if stage == 'B1' else 'val/success_1m'
    checkpoint=ModelCheckpoint(dirpath=tmp_path,monitor=monitor,mode='max',filename='best')
    trainer = L.Trainer(accelerator='cpu',max_epochs=1,logger=False,
                        callbacks=[checkpoint,StageProgressBar()],num_sanity_val_steps=1,
                        val_check_interval=.5,enable_model_summary=False)
    # Two full validation passes within one epoch. Prefix rows reset each pass.
    from torch.utils.data import DataLoader
    trainer.fit(module,train_dataloaders=DataLoader([batch,batch],batch_size=None),
                val_dataloaders=DataLoader([batch],batch_size=None))
    assert (tmp_path/'best.ckpt').exists()
    assert float(trainer.callback_metrics[monitor]) == pytest.approx(1)
    if stage == 'B1':
        assert 'val/rmse_3d_m' not in trainer.callback_metrics
        assert len(module._val_visual) == 1
        assert module._last_visual['lastV/AP'] == pytest.approx(1)
    else:
        assert float(trainer.callback_metrics['val/rmse_3d_m']) == pytest.approx(1)
        assert module._train_position.compute()['gt_queries'] == 2
        assert module._val_position.compute()['gt_queries'] == 1
        assert module.live_progress['V/r3'] == pytest.approx(1)
