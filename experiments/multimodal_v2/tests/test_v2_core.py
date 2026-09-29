from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rdq_uav.multimodal_v1.contracts import InteractionContext, ProjectionContext
from rdq_uav.multimodal_v2.contracts import CandidateBatch, CrossModalEvidence
from rdq_uav.multimodal_v2.interaction import CandidateCrossAttention
from rdq_uav.multimodal_v2.loss import CandidateRankingLoss, MultimodalTargets
from rdq_uav.multimodal_v2.scoring import CandidateScoring, EvidenceScoreHead
from rdq_uav.multimodal_v2.training import (
    MultimodalV2DataModule,
    freeze_for_stage,
    trainable_parameter_groups,
)
from rdq_uav.multimodal_v2.vision import VisionCandidateModel


def candidate(source: str, *, score=(0.8,), batch=None, projected=None,
              projection_valid=None, boxes=None, xyz=None, source_ids=None,
              feature_dtype=torch.float32, device="cpu"):
    n = len(score)
    batch = [0] * n if batch is None else batch
    source_ids = list(range(n)) if source_ids is None else source_ids
    is_r = source == "R"
    xyz_value = torch.tensor([[0.0, 0.0, 2.0]] * n, device=device) if xyz is None else torch.tensor(xyz, dtype=torch.float32, device=device)
    box_value = torch.tensor([[45.0, 45.0, 55.0, 55.0]] * n, device=device) if boxes is None else torch.tensor(boxes, dtype=torch.float32, device=device)
    projected_value = torch.tensor([[50.0, 50.0]] * n, device=device) if projected is None else torch.tensor(projected, dtype=torch.float32, device=device)
    projection_valid = ([True] * n if is_r else [False] * n) if projection_valid is None else projection_valid
    return CandidateBatch(
        torch.randn(n, 128, dtype=feature_dtype, device=device),
        torch.tensor(score, dtype=feature_dtype, device=device),
        xyz_value if is_r else xyz_value.new_zeros((n, 3)),
        torch.full((n,), is_r, dtype=torch.bool, device=device),
        box_value if not is_r else box_value.new_zeros((n, 4)),
        torch.full((n,), not is_r, dtype=torch.bool, device=device),
        torch.tensor(batch, dtype=torch.long, device=device),
        torch.tensor(source_ids, dtype=torch.long, device=device), source,
        projected_value if is_r else projected_value.new_zeros((n, 2)),
        torch.tensor(projection_valid, dtype=torch.bool, device=device),
    )


def evidence(n: int, *, valid=True, tokens=3, gate=1.0,
             dtype=torch.float32, device="cpu"):
    return CrossModalEvidence(
        torch.randn(n, 128, dtype=dtype, device=device) if valid
        else torch.zeros(n, 128, dtype=dtype, device=device),
        torch.full((n,), valid, dtype=torch.bool, device=device),
        torch.full((n,), tokens if valid else 0, dtype=torch.long, device=device),
        torch.full((n,), gate if valid else 0.0, dtype=dtype, device=device),
        torch.full((n, 2), 50.0, dtype=torch.float32, device=device),
    )


def score(radar, vision, er=None, ev=None, *, vision_scoring=False):
    er = evidence(radar.n, valid=False) if er is None else er
    ev = evidence(vision.n, valid=False) if ev is None else ev
    return CandidateScoring()(radar, vision, er, ev, num_samples=1,
                              enable_vision_scoring=vision_scoring)


def context(m_r=True, m_v=True):
    projection = ProjectionContext(
        torch.eye(3)[None], torch.zeros(1, 3),
        torch.tensor([[0.0, 50.0, 50.0, 50.0, 50.0]]),
        torch.zeros(1, 4), torch.tensor([[100.0, 100.0]]), torch.ones(1, 2),
    )
    return InteractionContext(("cal",), torch.tensor([m_r]), torch.tensor([m_v]), projection)


def test_r_only_and_v_only_have_task_specific_fields_and_rankings():
    radar = candidate("R", projected=((5.0, 5.0),))
    vision = candidate("V", boxes=((80.0, 80.0, 90.0, 90.0),))
    output = score(radar, vision)
    r, v = output.hypothesis_type == 1, output.hypothesis_type == 2
    assert r.sum() == v.sum() == 1
    assert output.has_xyz[r].all() and not output.has_box[r].any()
    assert output.has_box[v].all() and not output.has_xyz[v].any()
    assert output.hypothesis_type[output.top3d_indices(1)[0][0]] == 1
    assert output.hypothesis_type[output.top2d_indices(1)[0][0]] == 2


def test_rv_preserves_independent_base_scores():
    output = score(candidate("R", score=(0.2,)), candidate("V", score=(0.9,)))
    rv = output.hypothesis_type == 0
    assert rv.sum() == 1
    assert torch.equal(output.score_3d_before[rv], torch.tensor([0.2]))
    assert torch.equal(output.score_2d_before[rv], torch.tensor([0.9]))


def test_both_evidence_paths_reach_their_own_score_head():
    radar, vision = candidate("R", score=(0.2,)), candidate("V", score=(0.9,))
    scoring = CandidateScoring()
    nn.init.constant_(scoring.score_head.radar[-1].bias, 0.5)
    nn.init.constant_(scoring.score_head.vision[-1].bias, -0.5)
    output = scoring(radar, vision, evidence(1), evidence(1), num_samples=1,
                     enable_vision_scoring=True)
    rv = output.hypothesis_type == 0
    assert not torch.equal(output.score_3d_after[rv], output.score_3d_before[rv])
    assert not torch.equal(output.score_2d_after[rv], output.score_2d_before[rv])
    assert torch.count_nonzero(output.delta_3d[rv]) == 1
    assert torch.count_nonzero(output.delta_2d[rv]) == 1


class _MixedDtypeProbe(nn.Module):
    """Small scorer used to isolate CandidateScoring buffer assignment."""

    def __init__(self):
        super().__init__()
        self.radar = nn.Parameter(torch.tensor(0.01))
        self.vision = nn.Parameter(torch.tensor(0.01))

    def forward(self, base, own, evidence_value, valid, gate, *, source):
        parameter = self.radar if source == "R" else self.vision
        delta = ((own.float().mean(1) + evidence_value.float().mean(1))
                 * parameter * valid.float() * gate.float())
        return (base.float() + delta).to(base.dtype), delta.to(base.dtype)


def test_mixed_candidate_buffers_preserve_source_dtypes_on_cpu():
    radar = candidate(
        "R", score=(0.7, 0.6), projected=((50, 50), (5, 5)),
        feature_dtype=torch.float16,
    )
    vision = candidate(
        "V", score=(0.8, 0.5),
        boxes=((45, 45, 55, 55), (80, 80, 90, 90)),
        feature_dtype=torch.float32,
    )
    scoring = CandidateScoring()
    scoring.score_head = _MixedDtypeProbe()
    output = scoring(
        radar, vision, evidence(2, dtype=torch.float16),
        evidence(2, dtype=torch.float32), num_samples=1,
        enable_vision_scoring=True,
    )
    assert set(output.hypothesis_type.tolist()) == {0, 1, 2}
    assert output.score_3d_after.dtype == torch.float16
    assert output.score_2d_after.dtype == torch.float32
    loss = output.score_3d_after.float().sum() + output.score_2d_after.sum()
    loss.backward()
    assert torch.isfinite(scoring.score_head.radar.grad)
    assert torch.isfinite(scoring.score_head.vision.grad)
    assert scoring.score_head.radar.grad != 0
    assert scoring.score_head.vision.grad != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FP16 autocast")
def test_mixed_source_dtypes_run_under_cuda_autocast_and_backpropagate():
    device = torch.device("cuda")
    radar = candidate(
        "R", score=(0.7, 0.6), projected=((50, 50), (5, 5)),
        feature_dtype=torch.float16, device=device,
    )
    vision = candidate(
        "V", score=(0.8, 0.5),
        boxes=((45, 45, 55, 55), (80, 80, 90, 90)),
        feature_dtype=torch.float32, device=device,
    )
    radar_evidence = evidence(2, dtype=torch.float16, device=device)
    vision_evidence = evidence(2, dtype=torch.float32, device=device)
    scoring = CandidateScoring().to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = scoring(
            radar, vision, radar_evidence, vision_evidence,
            num_samples=1, enable_vision_scoring=True,
        )
        loss = output.score_3d_after.float().sum() + output.score_2d_after.float().sum()
    assert set(output.hypothesis_type.cpu().tolist()) == {0, 1, 2}
    assert output.score_3d_after.dtype == torch.float16
    assert output.score_2d_after.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    for head in (scoring.score_head.radar, scoring.score_head.vision):
        gradient = head[-1].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient.float()).all()
        assert torch.count_nonzero(gradient) > 0


def test_invalid_evidence_is_exact_identity_even_with_trained_heads():
    radar, vision = candidate("R"), candidate("V")
    scoring = CandidateScoring()
    nn.init.constant_(scoring.score_head.radar[-1].bias, 1.0)
    nn.init.constant_(scoring.score_head.vision[-1].bias, 1.0)
    output = scoring(radar, vision, evidence(1, valid=False), evidence(1, valid=False),
                     num_samples=1, enable_vision_scoring=True)
    assert torch.equal(output.score_3d_after, output.score_3d_before)
    assert torch.equal(output.score_2d_after, output.score_2d_before)


def test_missing_vision_invalid_projection_and_padding_zero_visual_evidence():
    module = CandidateCrossAttention()
    pyramid = (torch.randn(1, 96, 25, 25), torch.randn(1, 192, 13, 13))
    padding = torch.zeros((1, 100, 100), dtype=torch.bool)
    for radar, ctx, mask in (
        (candidate("R"), context(m_v=False), padding),
        (candidate("R", xyz=((0.0, 0.0, -2.0),)), context(), padding),
        (candidate("R"), context(), torch.ones_like(padding)),
    ):
        result = module.read_visual_for_radar(radar, pyramid, mask, ctx)
        assert not result.valid.any()
        assert torch.equal(result.feature, torch.zeros_like(result.feature))


def test_no_nearby_lidar_is_exact_vision_identity():
    radar = candidate("R", projected=((0.0, 0.0),))
    vision = candidate("V", boxes=((80.0, 80.0, 90.0, 90.0),))
    ev = CandidateCrossAttention().read_radar_for_vision(vision, radar, context())
    scoring = CandidateScoring()
    nn.init.constant_(scoring.score_head.vision[-1].bias, 1.0)
    output = scoring(radar, vision, evidence(1, valid=False), ev, num_samples=1,
                     enable_vision_scoring=True)
    assert not ev.valid.any()
    assert torch.equal(output.score_2d_after, output.score_2d_before)


def test_b1_initialization_is_exact_b0_for_values_and_ordering():
    radar = candidate("R", score=(0.2, 0.9), projected=((50.0, 50.0), (90.0, 90.0)))
    vision = candidate("V", score=(0.7, 0.6), boxes=((45, 45, 55, 55), (85, 85, 95, 95)))
    b0 = score(radar, vision)
    b1 = score(radar, vision, evidence(2), evidence(2), vision_scoring=True)
    for name in ("score_3d_after", "score_2d_after", "xyz_m", "box_xyxy_px"):
        assert torch.equal(getattr(b0, name), getattr(b1, name))
    assert torch.equal(b0.top3d_indices(1)[0], b1.top3d_indices(1)[0])
    assert torch.equal(b0.top2d_indices(1)[0], b1.top2d_indices(1)[0])


def test_geometry_match_is_one_to_one_deterministic_and_preserves_unmatched_r():
    radar = candidate("R", score=(0.5, 0.5), projected=((50, 50), (50, 50)), source_ids=(9, 2))
    vision = candidate("V", source_ids=(7,))
    output = score(radar, vision)
    assert (output.hypothesis_type == 0).sum() == 1
    assert (output.hypothesis_type == 1).sum() == 1
    rv = torch.nonzero(output.hypothesis_type == 0).flatten()[0]
    remaining = torch.nonzero(output.hypothesis_type == 1).flatten()[0]
    assert output.radar_source_index[rv] == 2
    assert output.radar_source_index[remaining] == 9
    assert output.diagnostics["association_per_query"][0]["association_conflicts"] == 1
    # Association emits RV before unmatched R, but score ties must retain the
    # explicit source-id order used by the source candidate set.
    assert output.radar_source_index[output.top3d_indices(1)[0][0]] == 2


def test_invalid_projection_rejects_rv_without_deleting_candidates():
    radar = candidate("R", projection_valid=(False,))
    vision = candidate("V")
    output = score(radar, vision)
    assert not (output.hypothesis_type == 0).any()
    assert (output.hypothesis_type == 1).sum() == 1
    assert (output.hypothesis_type == 2).sum() == 1
    audit = output.diagnostics["association_per_query"][0]
    assert audit["invalid_radar_projection"] == 1
    assert audit["feasible_pairs"] == 0


def test_invalid_visual_box_fails_at_candidate_contract_boundary():
    with pytest.raises(ValueError, match="non-degenerate"):
        candidate("V", boxes=((10.0, 10.0, 10.0, 20.0),))


def test_loss_positive_negative_only_no_candidate_and_no_gt_statistics():
    criterion = CandidateRankingLoss(lambda_2d=0.0)
    targets = MultimodalTargets(torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([True]),
                                torch.zeros(1, 4), torch.tensor([False]))
    empty_v = CandidateBatch.empty("V", "cpu")
    positive = criterion(score(candidate("R"), empty_v), targets)
    assert positive["n_with_positive_3d"] == 1 and positive["n_3d_loss_queries"] == 1
    far = criterion(score(candidate("R", xyz=((10.0, 0.0, 2.0),)), empty_v), targets)
    assert far["n_negative_only_3d"] == 1 and far["n_3d_loss_queries"] == 1
    full_weight = CandidateRankingLoss(lambda_2d=0.0, negative_only_weight=1.0)(
        score(candidate("R", xyz=((10.0, 0.0, 2.0),)), empty_v), targets
    )
    assert torch.allclose(far["loss_rank_3d"], full_weight["loss_rank_3d"] * 0.25)
    none = criterion(score(CandidateBatch.empty("R", "cpu"), empty_v), targets)
    assert none["n_no_3d_candidate"] == 1 and not none["has_trainable_loss"]
    missing = MultimodalTargets(torch.zeros(1, 3), torch.tensor([False]),
                                torch.zeros(1, 4), torch.tensor([False]))
    unlabeled = criterion(score(candidate("R"), empty_v), missing)
    assert unlabeled["n_gt3d"] == 0 and not unlabeled["has_trainable_loss"]
    assert torch.isfinite(positive["loss"]) and torch.isfinite(far["loss"])


def test_focal_probability_loss_is_autocast_safe_and_has_finite_gradient():
    criterion = CandidateRankingLoss()
    scores = torch.tensor([0.8, 0.15, 0.6], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([True, False, False])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        terms = criterion._terms(scores, labels)
        loss = terms.sum()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) == scores.numel()


def test_2d_loss_uses_only_valid_box_gt_and_2d_scores():
    radar = candidate("R")
    vision = candidate("V", boxes=((45.0, 45.0, 55.0, 55.0),))
    output = score(radar, vision)
    criterion = CandidateRankingLoss(lambda_3d=0.0, lambda_2d=1.0)
    labeled = MultimodalTargets(torch.zeros(1, 3), torch.tensor([False]),
                                torch.tensor([[45.0, 45.0, 55.0, 55.0]]), torch.tensor([True]))
    values = criterion(output, labeled)
    assert values["n_with_positive_2d"] == 1 and values["n_2d_loss_queries"] == 1
    b2_values = CandidateRankingLoss(lambda_3d=1.0, lambda_2d=0.0)(output, labeled)
    assert b2_values["n_with_positive_2d"] == 1
    assert b2_values["n_2d_loss_queries"] == 0
    unlabeled = MultimodalTargets(torch.zeros(1, 3), torch.tensor([False]),
                                  torch.zeros(1, 4), torch.tensor([False]))
    skipped = criterion(output, unlabeled)
    assert skipped["n_gt2d"] == 0 and skipped["n_2d_loss_queries"] == 0
    assert not skipped["has_trainable_loss"]


def test_zero_initialized_score_head_is_exact_and_trainable():
    head = EvidenceScoreHead()
    base = torch.tensor([0.2, 0.8])
    result, delta = head(base, torch.randn(2, 128), torch.randn(2, 128),
                         torch.ones(2, dtype=torch.bool), torch.ones(2), source="R")
    assert torch.equal(result, base) and torch.equal(delta, torch.zeros_like(delta))
    result.sum().backward()
    assert torch.count_nonzero(head.radar[-1].weight.grad) > 0


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, 1)
        self.pos_drop = nn.Identity()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.out_indices = (1, 2, 3)
        self.num_features = (4, 4, 4, 4)
        self.ape = False


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackbone()
        self.head = nn.Linear(4, 1)


def test_vision_detector_has_single_state_dict_owner_and_strict_roundtrip():
    first = VisionCandidateModel(_FakeDetector())
    keys = tuple(first.state_dict())
    assert any(key.startswith("dino.detector.") for key in keys)
    assert not any(key.startswith("detector.") for key in keys)
    assert len(keys) == len(set(keys))
    second = VisionCandidateModel(_FakeDetector())
    second.load_state_dict(first.state_dict(), strict=True)


class _FakeFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar = nn.Linear(2, 2)
        self.vision = nn.Linear(2, 2)
        self.interaction = CandidateCrossAttention()
        self.scoring = CandidateScoring()


def test_b2_optimizer_contains_only_radar_reads_visual_path():
    model = _FakeFull()
    runtime = SimpleNamespace(model=model)
    freeze_for_stage(runtime, "B2")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all(name.startswith(("interaction.radar_reads_vision.",
                                "interaction.visual_proj.",
                                "scoring.score_head.radar.")) for name in trainable)
    assert not any(name.startswith(("lidar.", "vision.", "interaction.vision_reads_radar.",
                                    "interaction.depth_embed.", "scoring.score_head.vision."))
                   for name in trainable)
    groups = trainable_parameter_groups(runtime, 1e-4, 1e-4)
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    assert grouped == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    model.scoring.score_head.radar[-1].bias.sum().backward()
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name not in trainable)


def test_dataloader_rng_state_roundtrip_for_resume():
    module = MultimodalV2DataModule([0], [0], batch_size=1, num_workers=0,
                                    prefetch_factor=2, seed=42)
    _ = torch.randperm(20, generator=module.train_generator)
    state = module.state_dict()
    expected = torch.randperm(20, generator=module.train_generator)
    restored = MultimodalV2DataModule([0], [0], batch_size=1, num_workers=0,
                                      prefetch_factor=2, seed=999)
    restored.load_state_dict(state)
    assert torch.equal(torch.randperm(20, generator=restored.train_generator), expected)
