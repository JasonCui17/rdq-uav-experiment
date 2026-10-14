"""Synthetic observations only; no model, checkpoint or real data is loaded."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rdq_uav.multimodal_v3.candidate_adapter import CandidateFrame, from_cache
from rdq_uav.multimodal_v3.stone_tracker import MultiCandidateTracker, MultiFilterConfig, SequenceMultiTrackers
from rdq_uav.multimodal_v3.track_selection import SelectionConfig, TrackSelector
from rdq_uav.multimodal_v3.tracking import Observations


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_tracking.py"


def frame(xyz, scores, tokens=None, *, baseline=0, evidence=None):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    scores = np.asarray(scores, dtype=float)
    tokens = np.arange(len(xyz)) if tokens is None else np.asarray(tokens)
    dense = Observations(xyz, scores, scores, tokens)
    top = dense.take([baseline]) if len(xyz) else Observations.empty()
    return CandidateFrame(dense, dense, top, evidence or {})


def evidence(count, *, dynamic=False, speed=0., reliability=1., valid=True):
    return {"point_count": count, "motion_valid": valid, "motion_dynamic": dynamic,
            "motion_reliability": reliability, "motion_speed_mps": speed}


def tracker(*, select=False, **overrides):
    return MultiCandidateTracker(MultiFilterConfig(**overrides), SelectionConfig(),
                                 select_evidence=select)


def row(t, score, token, *, motion=None, innovation=0.):
    return dict(timestamp=t, score=score, selected_token_id=token,
                output_kind="updated", innovation_m=innovation,
                evidence=motion or {})


def test_background_top1_does_not_block_second_track():
    state = tracker(select=True)
    last = None
    for i in range(4):
        last = state.step(i * .1, frame([[20, 0, 0], [.1 * i, 0, 0]],
                                       [.95, .6], [10, 20], evidence={
                                           10: evidence(120, speed=0.),
                                           20: evidence(2, dynamic=True, speed=2.),
                                       }))
    assert last["active_track_count"] == 2
    assert {history[0]["selected_token_id"] for history in state.history.values()} == {10, 20}
    assert last["selected_token_id"] == 20
    assert last["track_scores"][last["track_id"]]["background_status"] == "unknown"


def test_far_high_score_cannot_displace_low_score_continuation():
    state = tracker(confirm_hits=1)
    for i in range(3):
        state.step(i * .1, frame([[i * .1, 0, 0]], [.6], [7]))
    result = state.step(.3, frame([[50, 0, 0], [.3, 0, 0]], [.999, .001], [99, 7]))
    assert result["selected_token_id"] == 7
    assert result["output_kind"] == "updated"
    assert result["active_track_count"] >= 1


def test_static_background_cannot_win_only_from_lifetime():
    selector = TrackSelector(SelectionConfig(), use_evidence=True)
    background = [row(i * .1, .8, 10, motion=evidence(120, speed=0.)) for i in range(20)]
    target = [row(i * .1, .7, 20, motion=evidence(2, dynamic=True, speed=2.)) for i in range(15, 20)]
    chosen, scores, _ = selector.choose(1.9, {1: background, 2: target})
    assert chosen == 2
    assert scores[1]["window_observations"] <= 11
    assert scores[1]["static_background_hint"] is not None


def test_confirmed_hovering_track_is_not_deleted_for_zero_speed():
    state = tracker(select=True, confirm_hits=1)
    for i in range(4):
        result = state.step(i * .1, frame([[0, 0, 0]], [.8], [1],
                          evidence={1: evidence(1, speed=0.)}))
    assert result["active_track_count"] == 1
    assert result["track_id"] is not None
    assert result["track_scores"][result["track_id"]]["background_status"] == "unknown"


def test_background_hint_alone_cannot_suppress_confirmed_hovering_output():
    selector = TrackSelector(SelectionConfig(), use_evidence=True)
    chosen, scores, _ = selector.choose(0., {
        1: [row(0., .3, 1, motion=evidence(120, speed=0.))]}, {1})
    assert chosen == 1
    assert scores[1]["static_background_hint"] == 1.
    assert scores[1]["total"] >= selector.cfg.min_selection_score


def test_empty_prediction_expiry_and_reacquisition():
    state = tracker(confirm_hits=1, max_prediction_age_s=.5)
    first = state.step(0., frame([[0, 0, 0]], [.8], [1]))
    missing = state.step(.2, frame([], []))
    assert missing["output_kind"] == "predicted"
    assert missing["selected_token_id"] is None
    lost = state.step(.6, frame([], []))
    assert lost["xyz"] is None
    assert any(x["reason"] == "expired" for x in lost["removed_tracks"])
    again = state.step(.7, frame([[30, 0, 0]], [.8], [2]))
    assert again["track_id"] != first["track_id"]
    assert again["reacquisition"]


def test_one_detection_updates_at_most_one_track():
    state = tracker(confirm_hits=1, birth_nms_radius_m=.1, duplicate_radius_m=.1)
    state.step(0, frame([[0, 0, 0], [.6, 0, 0]], [.7, .7], [1, 2]))
    result = state.step(.1, frame([[.3, 0, 0]], [.7], [3]))
    current = [x for x in result["track_states"] if x["timestamp"] == .1 and
               x["output_kind"] == "updated" and x["selected_token_id"] == 3]
    assert len(current) == 1


def test_sequence_isolation_and_order():
    manager = SequenceMultiTrackers(MultiFilterConfig(confirm_hits=1),
                                    SelectionConfig(), select_evidence=False)
    a = manager.step("a", 1., frame([[0, 0, 0]], [.8], [1]))
    b = manager.step("b", 1., frame([[30, 0, 0]], [.8], [1]))
    assert a["xyz"][0] == 0 and b["xyz"][0] == 30
    with pytest.raises(ValueError, match="chronological"):
        manager.step("a", 1., frame([[0, 0, 0]], [.8], [1]))
    with pytest.raises(ValueError, match="chronological"):
        manager.step("b", .9, frame([[30, 0, 0]], [.8], [1]))


def test_deleted_track_keeps_entire_history():
    state = tracker(confirm_hits=1)
    state.step(0, frame([[0, 0, 0]], [.8], [1]))
    state.step(.1, frame([[.1, 0, 0]], [.8], [1]))
    state.step(.2, frame([], []))
    state.step(.8, frame([], []))
    assert len(state.history[1]) == 4
    assert [x["output_kind"] for x in state.history[1]] == ["updated", "updated", "predicted", "predicted"]
    assert state.history[1][-1]["status"] == "deleted"


def _make_cache(root, *, schema=1, mutate_gt=False):
    root.mkdir()
    rows = []
    for i in range(6):
        xyz = np.array([[.1 * i, 0, 0], [20, 0, 0]], dtype=np.float32)
        arrays = dict(dense_xyz=xyz, dense_score=np.array([.7, .9]),
                      dense_rank=np.array([.7, .9]), dense_token_id=np.array([1, 2]),
                      global_xyz=xyz, global_score=np.array([.7, .9]),
                      global_token_id=np.array([1, 2]), baseline_xyz=xyz[1:2],
                      baseline_score=np.array([.9]), baseline_token_id=np.array([2]),
                      gt_xyz=np.array([1000, 1000, 1000]) if mutate_gt else xyz[0],
                      has_gt=np.array(True))
        if schema == 2:
            arrays.update(dense_point_count=np.array([2, 120]),
                          dense_motion_valid=np.array([True, True]),
                          dense_motion_dynamic=np.array([True, False]),
                          dense_motion_reliability=np.array([1., 1.]),
                          dense_motion_speed_mps=np.array([2., 0.]))
        np.savez_compressed(root / f"{i}.npz", **arrays)
        rows.append(dict(sequence_id="synthetic", sample_id=f"q{i}",
                         query_time=.1 * i, file=f"{i}.npz"))
    (root / "manifest.json").write_text(json.dumps(dict(
        complete=True, queries=6, protocol="synthetic", schema_version=schema)))
    (root / "index.jsonl").write_text("\n".join(map(json.dumps, rows)))


def _evaluate(cache, output, methods):
    subprocess.run([sys.executable, str(SCRIPT), "--cache", str(cache),
                    "--output", str(output), "--methods", methods],
                   check=True, capture_output=True)
    return json.loads((output / "summary.json").read_text())


def test_gt_mutation_does_not_change_tracking_or_selection(tmp_path):
    normal, altered = tmp_path / "normal", tmp_path / "altered"
    _make_cache(normal, schema=2)
    _make_cache(altered, schema=2, mutate_gt=True)
    first = _evaluate(normal, tmp_path / "first", "all")
    second = _evaluate(altered, tmp_path / "second", "all")
    assert (tmp_path / "first/tracking_events.jsonl").read_bytes() == (
        tmp_path / "second/tracking_events.jsonl").read_bytes()
    assert (tmp_path / "first/track_scores.jsonl").read_bytes() == (
        tmp_path / "second/track_scores.jsonl").read_bytes()
    assert first["multi_candidate_kf_select"]["mean_error_m"] != second["multi_candidate_kf_select"]["mean_error_m"]
    assert (tmp_path / "first/all_track_history.jsonl").is_file()
    assert (tmp_path / "first/reacquisition_events.csv").is_file()


def test_original_methods_and_schema1_cache_still_work(tmp_path):
    cache = tmp_path / "cache"
    _make_cache(cache, schema=1)
    original = _evaluate(cache, tmp_path / "original", "original")
    all_methods = _evaluate(cache, tmp_path / "all", "all")
    assert [key for key in original if key not in ("runtime", "protocol")] == list((
        "independent_top1", "top1_kf", "candidate_kf"))
    for method in ("independent_top1", "top1_kf", "candidate_kf"):
        for metric in ("queries", "outputs", "coverage", "success_1m", "severe_gt10m"):
            assert original[method][metric] == all_methods[method][metric]
    with np.load(cache / "0.npz", allow_pickle=False) as arrays:
        assert from_cache(arrays, 1).evidence == {}
    assert all_methods["multi_candidate_kf_select"]["queries"] == 6


def test_missing_motion_evidence_remains_unknown():
    selector = TrackSelector(SelectionConfig(), use_evidence=True)
    chosen, scores, _ = selector.choose(0., {1: [row(0., .8, 1, motion=evidence(100, valid=False))]})
    assert chosen == 1
    assert scores[1]["static_background_hint"] is None
    assert scores[1]["background_status"] == "unknown"


def test_schema2_partial_evidence_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    _make_cache(cache, schema=2)
    with np.load(cache / "0.npz", allow_pickle=False) as arrays:
        partial = {name: arrays[name] for name in arrays.files
                   if name != "dense_motion_speed_mps"}
    with pytest.raises(ValueError, match="complete"):
        from_cache(partial, 2)


def test_switch_requires_sustained_advantage():
    cfg = SelectionConfig(switch_margin=.05, switch_consecutive=2)
    selector = TrackSelector(cfg, use_evidence=False)
    first, _, _ = selector.choose(0., {1: [row(0., .8, 1)], 2: [row(0., .6, 2)]})
    assert first == 1
    waiting, _, reason = selector.choose(.1, {
        1: [row(0., .8, 1), row(.1, .7, 1)],
        2: [row(0., .6, 2), row(.1, 1., 2)],
    })
    assert waiting == 1 and reason == "wait_switch"
    switched, _, reason = selector.choose(.2, {
        1: [row(0., .8, 1), row(.1, .7, 1), row(.2, .7, 1)],
        2: [row(0., .6, 2), row(.1, 1., 2), row(.2, 1., 2)],
    })
    assert switched == 2 and reason == "switch_sustained_advantage"


def test_persistent_wrong_lock_requires_consecutive_same_track():
    sys.path.insert(0, str(SCRIPT.parent))
    from evaluate_tracking import persistent_wrong_queries
    rows = []
    for i, (track_id, error) in enumerate(((1, 12.), (1, 13.), (2, 12.),
                                           (2, 0.), (2, 11.), (2, 12.))):
        rows.append({"sequence_id": "a", "has_gt": True,
                     "multi_candidate_kf_status": "confirmed",
                     "multi_candidate_kf_track_id": track_id,
                     "multi_candidate_kf_error_m": error})
    assert persistent_wrong_queries(rows, "multi_candidate_kf") == 4
