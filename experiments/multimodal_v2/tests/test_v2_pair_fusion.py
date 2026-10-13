from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rdq_uav.multimodal_v2.contracts import CandidateBatch
from rdq_uav.multimodal_v2.candidate_builders import RadarCandidateBuilder, RGBCandidateBuilder
from rdq_uav.multimodal_v2.model import MultimodalV2
from rdq_uav.multimodal_v2.loss import MultimodalTargets
from rdq_uav.multimodal_v2.pair_fusion import CandidatePairFusion
from rdq_uav.multimodal_v2.pair_loss import JointPairLoss
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.training import freeze_for_stage, trainable_parameter_groups


def candidate(source, n=2, *, batch=0):
    is_r = source == "R"
    return CandidateBatch(
        torch.randn(n, 128), torch.linspace(.2, .8, n),
        torch.tensor([[0., 0., 20.]]*n).reshape(n, 3) if is_r else torch.zeros(n, 3),
        torch.full((n,), is_r, dtype=torch.bool),
        torch.zeros(n, 4) if is_r else torch.tensor([[40., 40., 60., 60.]]*n).reshape(n, 4),
        torch.full((n,), not is_r, dtype=torch.bool),
        torch.full((n,), batch, dtype=torch.long), torch.arange(n), source,
        torch.tensor([[50., 50.]]*n).reshape(n, 2) if is_r else torch.zeros(n, 2),
        torch.full((n,), is_r, dtype=torch.bool))


def targets(*, r=True, v=True):
    return MultimodalTargets(torch.tensor([[0., 0., 20.]]), torch.tensor([r]),
                             torch.tensor([[40., 40., 60., 60.]]), torch.tensor([v]))


def pair_ids(out):
    return list(zip(out.radar_source_index[out.association_valid].tolist(),
                    out.vision_source_index[out.association_valid].tolist()))


def test_50_candidates_produce_50_by_50_similarity_and_50_pairs():
    out = CandidatePairFusion()(candidate("R", 50), candidate("V", 50), num_samples=1)
    assert out.diagnostics["pair_alignment"][0]["similarity"].shape == (50, 50)
    assert out.association_valid.sum() == 50 and len(out.joint_logit) == 50
    assert torch.equal(out.score_3d_after, out.score_2d_after)


def test_semantics_resolve_geometric_ties_and_matching_is_permutation_stable():
    r, v = candidate("R"), candidate("V")
    rf = torch.zeros(2, 128); rf[0, 0] = 1; rf[1, 1] = 1
    r = replace(r, feature=rf)
    v = replace(v, feature=rf.flip(0))
    model = CandidatePairFusion()
    model.radar_adapter = nn.Identity(); model.vision_adapter = nn.Identity()
    out = model(r, v, num_samples=1)
    assert pair_ids(out) == [(0, 1), (1, 0)]
    shuffled = model(r.index_select(torch.tensor([1, 0])),
                     v.index_select(torch.tensor([1, 0])), num_samples=1)
    assert pair_ids(shuffled) == pair_ids(out)
    assert torch.allclose(shuffled.joint_logit, out.joint_logit)


def test_geometry_resolves_semantic_ties_and_full_matching_is_soft_by_default():
    r, v = candidate("R"), candidate("V")
    r = replace(r, feature=torch.ones(2, 128), projected_xy_px=torch.tensor([[50., 50.], [500., 500.]]))
    v = replace(v, feature=torch.ones(2, 128),
                box_xyxy_px=torch.tensor([[490., 490., 510., 510.], [40., 40., 60., 60.]]))
    out = CandidatePairFusion()(r, v, num_samples=1)
    assert pair_ids(out) == [(0, 1), (1, 0)]
    far = replace(v, box_xyxy_px=v.box_xyxy_px+10000)
    assert CandidatePairFusion()(r, far, num_samples=1).association_valid.sum() == 2
    assert not CandidatePairFusion(reject_outside_gate=True)(r, far, num_samples=1).association_valid.any()


def test_original_scores_do_not_enter_assignment_or_joint_head_within_retained_pool():
    r, v = candidate("R"), candidate("V")
    model = CandidatePairFusion()
    before = model(r, v, num_samples=1)
    after = model(replace(r, score=torch.tensor([.999999, .000001])),
                  replace(v, score=torch.tensor([.000001, .999999])), num_samples=1)
    assert pair_ids(before) == pair_ids(after)
    assert torch.equal(before.joint_logit, after.joint_logit)
    assert torch.equal(before.xyz_m, after.xyz_m)
    assert torch.equal(before.box_xyxy_px, after.box_xyxy_px)


def test_ranking_uses_fresh_logits_even_when_sigmoid_saturates():
    out = CandidatePairFusion()(candidate("R"), candidate("V"), num_samples=1)
    out = replace(out, joint_logit=torch.tensor([40., 50.]),
                  score_3d_after=torch.ones(2), score_2d_after=torch.ones(2))
    assert out.top3d_indices(1)[0].tolist() == out.top2d_indices(1)[0].tolist() == [1, 0]


@pytest.mark.parametrize("nr,nv", [(0, 0), (0, 2), (2, 0), (3, 1)])
def test_empty_and_unequal_modalities_keep_truthful_masks(nr, nv):
    r, v = candidate("R", nr), candidate("V", nv)
    out = CandidatePairFusion()(r, v, num_samples=1)
    assert out.has_xyz.sum() == nr and out.has_box.sum() == nv
    assert torch.isfinite(out.joint_logit).all()
    assert len(out.top3d_indices(1)[0]) == nr
    assert len(out.top2d_indices(1)[0]) == nv


def test_no_cross_query_matching_and_invalid_projections_are_unmatched():
    out = CandidatePairFusion()(candidate("R", 1, batch=0), candidate("V", 1, batch=1), num_samples=2)
    assert not out.association_valid.any()
    r = replace(candidate("R", 1), projected_xy_px=torch.tensor([[float("nan"), 0.]]))
    out = CandidatePairFusion()(r, candidate("V", 1), num_samples=1)
    assert not out.association_valid.any() and torch.isfinite(out.joint_logit).all()


def test_joint_positive_hard_negative_and_background_labels():
    r, v = candidate("R", 4), candidate("V", 4)
    r = replace(r, xyz_m=torch.tensor([[0., 0., 20.], [0., 0., 20.], [5., 0., 20.], [5., 0., 20.]]))
    v = replace(v, box_xyxy_px=torch.tensor([[40., 40., 60., 60.], [100., 100., 120., 120.],
                                            [40., 40., 60., 60.], [100., 100., 120., 120.]]))
    rf = torch.eye(128)[:4]; r = replace(r, feature=rf); v = replace(v, feature=rf)
    model = CandidatePairFusion(geometry_weight=0., center_weight=0.)
    model.radar_adapter = nn.Identity(); model.vision_adapter = nn.Identity()
    out = model(r, v, num_samples=1)
    assert pair_ids(out) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    values = JointPairLoss()(out, targets())
    assert values["joint_labels"].tolist() == [1, 0, 0, 0]
    assert values["n_pair_positive"] == 1 and values["n_pair_hard_negative"] == 2
    assert values["n_pair_ordinary_negative"] == 1
    assert values["n_alignment_positive"] == 4  # Multi-positive GT supervision.
    assert values["n_joint_rank_queries"] == 1 and values["has_trainable_loss"]


@pytest.mark.parametrize("has_r,has_v", [(False, False), (True, False), (False, True)])
def test_missing_pair_labels_do_not_become_negatives(has_r, has_v):
    out = CandidatePairFusion()(candidate("R"), candidate("V"), num_samples=1)
    values = JointPairLoss()(out, targets(r=has_r, v=has_v))
    assert values["joint_labels"].tolist() == [-1, -1]
    assert not values["has_trainable_loss"] and float(values["loss"]) == 0


def test_single_modality_labels_train_only_the_available_output():
    out = CandidatePairFusion()(candidate("R"), candidate("V", 0), num_samples=1)
    values = JointPairLoss()(out, targets(r=True, v=False))
    assert values["joint_labels"].tolist() == [1, 1]
    assert values["has_trainable_loss"]
    assert len(out.top2d_indices(1)[0]) == 0


def test_boundary_pair_is_ignored_but_old_1_to_2m_gap_is_supervised():
    r = replace(candidate("R"), xyz_m=torch.tensor([[1., 0., 20.], [1.5, 0., 20.]]))
    values = JointPairLoss()(CandidatePairFusion()(r, candidate("V"), num_samples=1), targets())
    assert sorted(values["joint_labels"].tolist()) == [-1, 0]
    assert values["n_pair_hard_negative"] == 1


def test_background_only_pairs_train_low_confidence_without_alignment_repulsion():
    r = replace(candidate("R"), xyz_m=torch.tensor([[10., 0., 20.]]*2))
    v = replace(candidate("V"), box_xyxy_px=torch.tensor([[100., 100., 120., 120.]]*2))
    values = JointPairLoss()(CandidatePairFusion()(r, v, num_samples=1), targets())
    assert values["has_trainable_loss"] and values["n_pair_ordinary_negative"] == 2
    assert float(values["loss_align"]) == 0 and float(values["loss_joint_rank"]) == 0


def test_alignment_trains_true_pairs_even_when_initial_assignment_misses_them():
    r, v = candidate("R"), candidate("V")
    r = replace(r, xyz_m=torch.tensor([[0., 0., 20.], [10., 0., 20.]]), feature=torch.eye(128)[:2])
    v = replace(v, box_xyxy_px=torch.tensor([[40., 40., 60., 60.], [100., 100., 120., 120.]]),
                feature=torch.eye(128)[:2].flip(0))
    model = CandidatePairFusion(geometry_weight=0., center_weight=0.)
    model.radar_adapter = nn.Identity(); model.vision_adapter = nn.Identity()
    values = JointPairLoss()(model(r, v, num_samples=1), targets())
    assert values["n_pair_positive"] == 0 and values["n_alignment_positive"] == 1
    assert float(values["loss_align"]) > 0


def test_50_raw_candidates_survive_pre_fusion_suppression():
    selector = CandidateSelector(dict(raw_topk=50, pre_nms_topk=50, final_topk=50, nms_radius_m=1.), raw_only=True)
    raw = dict(logits=torch.linspace(10., -10., 60), pred_xyz=torch.zeros(60, 3),
               fine_features=torch.randn(60, 128), source_token_id=torch.arange(60),
               batch_index=torch.zeros(60, dtype=torch.long), aux_stats={"num_samples": 1})
    assert RadarCandidateBuilder(selector, use_raw=True)(raw).n == 50
    builder = RGBCandidateBuilder(pre_topk=50, final_topk=50, apply_nms=False)
    dino = dict(pred_logits=torch.linspace(10., -10., 60).reshape(1, 60, 1),
                pred_boxes=torch.tensor([.5, .5, .2, .2]).expand(1, 60, 4),
                decoder_query_features=torch.randn(1, 60, 256))
    assert builder(dino, torch.tensor([[100., 100.]])).n == 50


@pytest.mark.parametrize("amp", [False, True])
def test_two_updates_reach_both_adapters_visual_projection_and_joint_head(amp):
    torch.manual_seed(42)
    vision = nn.Module(); vision.detector = nn.Linear(4, 4)
    vision.builder = RGBCandidateBuilder(pre_topk=50, final_topk=50, apply_nms=False)
    model = MultimodalV2(nn.Linear(4, 4), vision, nn.Identity(), nn.Identity(),
                        pair_fusion=CandidatePairFusion())
    runtime = SimpleNamespace(model=model)
    freeze_for_stage(runtime, "B4")
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith(("pair_fusion.", "vision.builder.feature_proj.")) for n in names)
    groups = trainable_parameter_groups(runtime, .001, .0001)
    assert {id(p) for g in groups for p in g["params"]} == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups)
    r = replace(candidate("R"), xyz_m=torch.tensor([[0., 0., 20.], [10., 0., 20.]]))
    v = replace(candidate("V"), box_xyxy_px=torch.tensor([[40., 40., 60., 60.], [100., 100., 120., 120.]]))
    query = torch.randn(2, 256)
    for _ in range(2):
        optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            visual = replace(v, feature=vision.builder.feature_proj(query))
            values = JointPairLoss()(model.pair_fusion(r, visual, num_samples=1), targets())
        assert values["has_trainable_loss"] and torch.isfinite(values["loss"])
        values["loss"].backward()
        for prefix in ("pair_fusion.radar_adapter.", "pair_fusion.vision_adapter.",
                       "pair_fusion.joint_head.", "vision.builder.feature_proj."):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
            assert any(g is not None and g.abs().sum() > 0 for g in grads)
            assert all(g is None or torch.isfinite(g).all() for g in grads)
        assert all(p.grad is None for p in model.lidar.parameters())
        assert all(p.grad is None for p in vision.detector.parameters())
        optimizer.step()


def test_joint_state_dict_roundtrip():
    first, second = CandidatePairFusion(), CandidatePairFusion()
    second.load_state_dict(first.state_dict(), strict=True)
    r, v = candidate("R"), candidate("V")
    assert torch.equal(first(r, v, num_samples=1).joint_logit,
                       second(r, v, num_samples=1).joint_logit)


def test_full_forward_step_bypasses_old_interaction_and_never_injects_gt(tmp_path):
    from test_v2_data import release, dataset, projection, FakeDINO, FakeCandidateProducer
    from rdq_uav.multimodal_v2.data import collate_multimodal_v2
    from rdq_uav.multimodal_v2.training import forward_step
    class Forbidden(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("B4 must not execute old attention or residual scoring")
    release(tmp_path, "seq0001", queries=(10., 20.), events=((9.5, 0), (19.5, 0)), images=(9.9,))
    (tmp_path/"seq0001"/"2d_detect"/"9.9.txt").write_text("0 0.5 0.5 0.25 0.5")
    d = dataset(tmp_path)
    batch = collate_multimodal_v2([d[0], d[1]])
    radar, vision = FakeCandidateProducer("R"), FakeCandidateProducer("V")
    model = MultimodalV2(radar, vision, Forbidden(), Forbidden(),
                        pair_fusion=CandidatePairFusion())
    runtime = SimpleNamespace(model=model, dino_detector=FakeDINO(), projection_base=projection(),
                              ranking_loss=JointPairLoss(), initialization={"stage": "B4"})
    result = forward_step(runtime, batch, torch.device("cpu"), compute_frozen_losses=False)
    out = result["output"]
    assert out.joint_logit is not None and out.has_xyz.sum() == 2 and out.has_box.sum() == 1
    assert result["has_trainable_loss"] and result["loss"].requires_grad
    assert vision.vision_targets is None
    result["loss"].backward()
    model.eval()
    without_labels = dict(batch, target_valid=torch.zeros_like(batch["target_valid"]),
                          gt_2d_valid=torch.zeros_like(batch["gt_2d_valid"]))
    with torch.no_grad():
        unlabeled = forward_step(runtime, without_labels, torch.device("cpu"), compute_frozen_losses=False)
    assert torch.equal(out.joint_logit.detach(), unlabeled["output"].joint_logit)


def test_joint_metric_requires_same_correct_pair_to_win_both_tasks():
    from rdq_uav.multimodal_v2.training import joint_validation_rows
    r = replace(candidate("R"), xyz_m=torch.tensor([[0., 0., 20.], [10., 0., 20.]]))
    # Both visual boxes are correct; this must not mask a wrong radar winner.
    out = CandidatePairFusion()(r, candidate("V"), num_samples=1)
    good_row = int(torch.nonzero(out.radar_source_index == 0).flatten()[0])
    logits = torch.zeros(2); logits[good_row] = 10
    assert joint_validation_rows(replace(out, joint_logit=logits), targets()) == [dict(success=True, oracle=True)]
    assert joint_validation_rows(replace(out, joint_logit=-logits), targets()) == [dict(success=False, oracle=True)]
    assert joint_validation_rows(out, targets(v=False)) == []


def test_b4_lightning_optimizes_and_saves_joint_checkpoint(tmp_path, monkeypatch):
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    import rdq_uav.multimodal_v2.training as training
    from test_live_monitoring import inputs
    target, _, batch = inputs()
    batch["m_R"] = torch.tensor([True])
    vision = nn.Module(); vision.builder = RGBCandidateBuilder()
    vision.detector = nn.Linear(1, 1)
    model = MultimodalV2(nn.Linear(1, 1), vision, nn.Identity(), nn.Identity(),
                        pair_fusion=CandidatePairFusion())
    runtime = SimpleNamespace(model=model, lidar_detector=model.lidar, dino_detector=vision.detector,
                              initialization={"stage": "B4"})
    freeze_for_stage(runtime, "B4")
    criterion = JointPairLoss()
    radar = replace(candidate("R", 1), xyz_m=target.xyz_m.clone())
    visual = replace(candidate("V", 1), box_xyxy_px=target.box_xyxy_px.clone())
    query = torch.randn(1, 256)
    def forward(*args, **kwargs):
        output = model.pair_fusion(radar, replace(visual, feature=vision.builder.feature_proj(query)), num_samples=1)
        return dict(output=output, targets=target, **criterion(output, target))
    monkeypatch.setattr(training, "forward_step", forward)
    class ToyModule(training.MultimodalV2LightningModule):
        def _refresh_external_device(self): pass
        def configure_optimizers(self):
            return torch.optim.AdamW(training.trainable_parameter_groups(runtime, .001, .0001))
    module = ToyModule(runtime, {"logging": {"ap_every_val_batches": 1}})
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val/joint_success", mode="max", filename="best_joint")
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, callbacks=[checkpoint],
                        num_sanity_val_steps=0, enable_model_summary=False, enable_progress_bar=False)
    trainer.fit(module, train_dataloaders=DataLoader([batch, batch], batch_size=None),
                val_dataloaders=DataLoader([batch], batch_size=None))
    assert trainer.global_step == 2 and (tmp_path/"best_joint.ckpt").exists()
    assert float(trainer.callback_metrics["val/joint_success"]) == 1
    assert float(trainer.callback_metrics["val/joint_labeled_queries"]) == 1
    assert float(trainer.callback_metrics["val/joint_oracle"]) == 1
    assert not runtime.lidar_detector.training and not runtime.dino_detector.training
