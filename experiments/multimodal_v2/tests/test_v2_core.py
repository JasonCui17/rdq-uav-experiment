from __future__ import annotations

import torch

from rdq_uav.multimodal_v1.contracts import InteractionContext, ProjectionContext
from rdq_uav.multimodal_v2.contracts import CandidateBatch, CrossModalEvidence
from rdq_uav.multimodal_v2.interaction import CandidateCrossAttention
from rdq_uav.multimodal_v2.loss import CandidateRankingLoss, MultimodalTargets
from rdq_uav.multimodal_v2.scoring import CandidateScoring, EvidenceScoreHead
from rdq_uav.multimodal_v2.training import MultimodalV2DataModule


def candidate(source: str, *, score=(0.8,), batch=(0,), xy=(50.0, 50.0)):
    n = len(score); device = torch.device("cpu")
    is_r = source == "R"
    xyz = torch.tensor([[0.0, 0.0, 2.0]] * n)
    box = torch.tensor([[xy[0]-5, xy[1]-5, xy[0]+5, xy[1]+5]] * n)
    return CandidateBatch(
        torch.randn(n, 128), torch.tensor(score), xyz if is_r else torch.zeros(n, 3),
        torch.full((n,), is_r), box if not is_r else torch.zeros(n, 4),
        torch.full((n,), not is_r), torch.tensor(batch, dtype=torch.long),
        torch.arange(n), source,
    )


def context(m_r=True, m_v=True):
    # pinhole-equivalent omni projection: x=y=0,z=2 -> (50,50)
    projection = ProjectionContext(
        torch.eye(3)[None], torch.zeros(1, 3),
        torch.tensor([[0.0, 50.0, 50.0, 50.0, 50.0]]),
        torch.zeros(1, 4), torch.tensor([[100.0, 100.0]]), torch.ones(1, 2),
    )
    return InteractionContext(("cal",), torch.tensor([m_r]), torch.tensor([m_v]), projection)


def test_v_only_has_no_xyz_and_is_excluded_from_3d_ranking():
    radar, vision = candidate("R"), candidate("V")
    er = CrossModalEvidence(torch.zeros(1,128), torch.zeros(1,dtype=torch.bool),
                            torch.zeros(1,dtype=torch.long), torch.zeros(1), torch.tensor([[50.,50.]]))
    ev = CrossModalEvidence(torch.zeros(1,128), torch.zeros(1,dtype=torch.bool),
                            torch.zeros(1,dtype=torch.long), torch.zeros(1), torch.tensor([[50.,50.]]))
    output = CandidateScoring()(radar, vision, er, ev, torch.tensor([[50.,50.]]))
    assert output.has_xyz.sum().item() == 1
    assert not output.has_xyz[output.hypothesis_type == 2].any()
    assert output.top3d_indices(1)[0].numel() == 1


def test_zero_initialized_score_head_is_exact_and_trainable():
    head = EvidenceScoreHead()
    score = torch.tensor([0.2, 0.8])
    own, evidence = torch.randn(2,128), torch.randn(2,128)
    result, delta = head(score, own, evidence, torch.ones(2,dtype=torch.bool), torch.ones(2), source="R")
    assert torch.equal(result, score)
    assert torch.equal(delta, torch.zeros_like(delta))
    result.sum().backward()
    assert head.radar[-1].weight.grad is not None
    assert torch.count_nonzero(head.radar[-1].weight.grad) > 0


def test_missing_vision_or_padding_produces_exact_zero_evidence():
    module = CandidateCrossAttention()
    radar = candidate("R")
    pyramid = (torch.randn(1,96,25,25), torch.randn(1,192,13,13))
    padding = torch.zeros((1,100,100), dtype=torch.bool)
    evidence = module.read_visual_for_radar(radar, pyramid, padding, context(m_v=False))
    assert not evidence.valid.any()
    assert torch.equal(evidence.feature, torch.zeros_like(evidence.feature))
    padding[:] = True
    evidence = module.read_visual_for_radar(radar, pyramid, padding, context())
    assert not evidence.valid.any()
    assert torch.equal(evidence.feature, torch.zeros_like(evidence.feature))


def test_radar_xyz_is_unchanged_and_b1_top1_matches_b0():
    radar = candidate("R", score=(0.2, 0.9), batch=(0,0))
    vision = candidate("V", score=(0.99,))
    er = CrossModalEvidence(torch.randn(2,128), torch.ones(2,dtype=torch.bool),
                            torch.full((2,),9,dtype=torch.long), torch.ones(2), torch.tensor([[50.,50.],[50.,50.]]))
    ev = CrossModalEvidence(torch.zeros(1,128), torch.zeros(1,dtype=torch.bool),
                            torch.zeros(1,dtype=torch.long), torch.zeros(1), torch.tensor([[50.,50.]]))
    output = CandidateScoring()(radar, vision, er, ev, torch.tensor([[50.,50.],[50.,50.]]))
    assert torch.equal(output.score_before[output.has_xyz], output.score_after[output.has_xyz])
    for source_id in radar.source_index:
        row = torch.nonzero(output.radar_source_index == source_id).flatten()
        assert len(row) == 1
        assert torch.equal(output.xyz_m[row[0]], radar.xyz_m[source_id])
    assert output.radar_source_index[output.top3d_indices(1)[0][0]].item() == 1


def test_loss_masks_missing_labels_and_v_only_from_3d():
    radar, vision = candidate("R"), candidate("V")
    er = CrossModalEvidence(torch.randn(1,128), torch.ones(1,dtype=torch.bool),
                            torch.ones(1,dtype=torch.long), torch.ones(1), torch.tensor([[50.,50.]]))
    ev = CrossModalEvidence(torch.zeros(1,128), torch.zeros(1,dtype=torch.bool),
                            torch.zeros(1,dtype=torch.long), torch.zeros(1), torch.tensor([[50.,50.]]))
    output = CandidateScoring()(radar, vision, er, ev, torch.tensor([[50.,50.]]))
    targets = MultimodalTargets(torch.tensor([[0.,0.,2.]]), torch.tensor([True]),
                                torch.zeros(1,4), torch.tensor([False]))
    values = CandidateRankingLoss()(output, targets)
    assert values["num_supervised_3d"] == 1
    assert values["num_supervised_2d"] == 0
    assert torch.isfinite(values["loss"])


def test_dataloader_rng_state_roundtrip_for_resume():
    module = MultimodalV2DataModule(
        [0], [0], batch_size=1, num_workers=0, prefetch_factor=2, seed=42,
    )
    _ = torch.randperm(20, generator=module.train_generator)
    state = module.state_dict()
    expected = torch.randperm(20, generator=module.train_generator)
    restored = MultimodalV2DataModule(
        [0], [0], batch_size=1, num_workers=0, prefetch_factor=2, seed=999,
    )
    restored.load_state_dict(state)
    actual = torch.randperm(20, generator=restored.train_generator)
    assert torch.equal(actual, expected)
