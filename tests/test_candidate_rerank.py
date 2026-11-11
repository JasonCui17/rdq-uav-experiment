from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rdq_uav.multimodal_v3 import visual_birth
from rdq_uav.multimodal_v3.candidate_adapter import CandidateFrame
from rdq_uav.multimodal_v3.candidate_rerank.birth import BirthReranker
from rdq_uav.multimodal_v3.candidate_rerank.model import load_bundle
from rdq_uav.multimodal_v3.target_tracker import TargetConfig
from rdq_uav.multimodal_v3.tracking import Observations


def test_visual_birth_rerank_respects_gates_and_keeps_continuation(
    monkeypatch, tmp_path
):
    model = Path(__file__).resolve().parents[1] / "configs/final/candidate_rerank.npz"
    bundle = load_bundle(model)
    frames = [{"query_time": t, "sequence_id": "seq0020"} for t in (1.0, 1.1, 2.0, 3.0)]
    adapter = BirthReranker(
        frames,
        {
            "schema_version": 2,
            "data_root": tmp_path,
            **bundle.cache_identity,
        },
        bundle,
    )
    original_updates = []
    proposed_tokens = [3, 2, 1]
    stream = SimpleNamespace(
        step=lambda t, dense: SimpleNamespace(token_ids=proposed_tokens),
        observe_original=lambda t, *args: original_updates.append(t),
    )
    adapter.streams["seq0020"] = stream
    monkeypatch.setattr(
        visual_birth,
        "project_omni_radtan",
        lambda *args: (
            torch.tensor([[15.0, 15.0], [15.0, 15.0], [15.0, 15.0], [200.0, 200.0]]),
            torch.ones(4, dtype=torch.bool),
        ),
    )
    obs = Observations(
        [[0, 0, 25.0], [0.2, 0, 25.0], [0.3, 0, 25.0], [0.4, 0, 25.0]],
        [0.99, 0.95, 0.5, 0.99],
        [10.0, 9.0, 8.0, 7.0],
        np.arange(4),
    )
    info = dict(
        point_count=1,
        motion_valid=False,
        motion_dynamic=False,
        motion_speed_mps=0.0,
        motion_reliability=0.0,
    )
    frame = CandidateFrame(obs, obs, obs, dict.fromkeys(range(4), info))

    def hint(t):
        return None if t == 3.0 else visual_birth.VisualHint(t, (10, 10, 20, 20), 0.99)

    config = TargetConfig(high_rank_first=True)
    original = visual_birth.VisualBirthTracker(config, None, visual_lookup=hint)
    ranked = visual_birth.VisualBirthTracker(
        config, None, visual_lookup=hint, rerank_birth=adapter
    )
    assert original.step(1.0, frame)["selected_token_id"] == 0
    assert ranked.step(1.0, frame)["selected_token_id"] == 1
    assert ranked.step(1.1, frame)["selected_token_id"] == 0
    proposed_tokens[:] = [3, 2]
    assert ranked.step(2.0, frame)["selected_token_id"] == 0
    assert ranked.step(3.0, frame)["xyz"] is None
    assert original_updates == [1.1, 3.0]
    assert adapter.scored_queries == 2 and adapter.preferred_queries == 1
    np.testing.assert_array_equal(obs.rank_value, [10.0, 9.0, 8.0, 7.0])
