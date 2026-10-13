"""Small contract checks for the opt-in trace path; real-data checks use trace_model.py."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments.multimodal_v2.diagnostics.trace_model import (
    ReplayHandler, choose, save_hierarchy, save_selector_table,
)
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.scoring import CandidateScoring
from rdq_uav.multimodal_v2.trace import TraceRecorder

ROOT = Path(__file__).resolve().parents[3]


def _radar():
    cfg = yaml.safe_load((ROOT / "experiments/multimodal_v2/configs/radar_a2_learned_sbe32.yaml").read_text())
    return LiDARUAVDetector(cfg).eval()


def _batch():
    return {"points": torch.tensor([[.01, .02, .03], [.12, .13, .14],
                                     [.3, .3, .3], [.01, .02, .03], [.4, .4, .4]]),
            "sensor_id": torch.tensor([0, 1, 0, 1, 0]),
            "delta_t": torch.tensor([0., -.2, -.4, -.1, -.9]),
            "point_batch_index": torch.tensor([0, 0, 0, 1, 1]),
            "num_samples": 2}


def _wrapper(radar, selector):
    attention = SimpleNamespace(radar_reads_vision=torch.nn.Identity(), vision_reads_radar=torch.nn.Identity())
    return SimpleNamespace(lidar=SimpleNamespace(detector=radar, builder=SimpleNamespace(selector=selector)),
                           interaction=attention, scoring=CandidateScoring())


def test_trace_maps_batches_slots_parents_and_preserves_forward(tmp_path):
    torch.manual_seed(42)
    radar = _radar(); selector = CandidateSelector({"raw_topk": 100, "pre_nms_topk": 50,
                                                    "nms_radius_m": 1., "final_topk": 10})
    model = _wrapper(radar, selector); batch = _batch()
    with torch.no_grad():
        plain = radar(batch); plain_selection = selector(plain)
        with TraceRecorder(model, tmp_path) as recorder:
            traced = radar(batch); selected = selector(traced)
            save_hierarchy(recorder, traced)
            selector_rows = save_selector_table(recorder, traced, None)
        assert torch.equal(plain["pred_xyz"], traced["pred_xyz"])
        for i in range(2):
            assert torch.equal(plain_selection[i]["nms"]["source_token_id"],
                               selected[i]["nms"]["source_token_id"])
    tensors = json.loads((tmp_path / "tensor_index.json").read_text())
    def load(name): return np.load(tmp_path / tensors[name]["file"])
    point_to_l0 = load("mapping.point_to_l0")
    slot = load("frontend.point_to_slot")
    flat = load("frontend.point_to_slot_flat")
    assert np.array_equal(flat, point_to_l0 * 8 + slot)
    assert load("frontend.slot_count").sum() == len(batch["points"])
    assert load("frontend.slot_mean").shape[-1] == 32
    assert np.array_equal(load("frontend.empty_slot_mask"), load("frontend.slot_count") == 0)
    l0_to_l1 = load("mapping.parent_l0_to_l1")
    l1_to_l2 = load("mapping.parent_l1_to_l2")
    assert len(l0_to_l1) == len(load("level0.coords"))
    assert l0_to_l1.max() < len(l1_to_l2)
    assert not np.any(load("level0.batch_index")[point_to_l0] != batch["point_batch_index"].numpy())
    assert load("frontend.l0_tokens").shape[1] == 128
    assert load("radar.merge01.internal.occupancy").shape[1] == 8
    assert load("radar.up10.internal.skip").shape[1] == 128
    assert load("radar.encoder0.block0.window.weights").ndim == 3
    assert [row["source_token_id"] for row in selector_rows if row["reason"] == "selected"] == selected[0]["nms"]["source_token_id"].tolist()
    assert "radar.encoder0.block0.groups.token_ids" in tensors
    assert "frontend.slot_attention.input_1" in tensors
    assert "frontend.slot_attention.output" in tensors
    assert "frontend.slot_attention.weights" not in tensors  # MHA ran with need_weights=False.
    assert selector.trace_observer is None
    assert radar.trace_observer is None
    assert not radar.encoder0.blocks[0]._forward_hooks


def test_trace_cleans_hooks_on_error_and_repeated_use(tmp_path):
    radar = _radar(); selector = CandidateSelector({"raw_topk": 4, "pre_nms_topk": 4,
                                                    "nms_radius_m": 1., "final_topk": 2})
    model = _wrapper(radar, selector)
    for i in range(2):
        with pytest.raises(RuntimeError, match="probe"):
            with TraceRecorder(model, tmp_path / str(i)):
                radar(_batch())
                raise RuntimeError("probe")
        assert radar.trace_observer is None
        assert selector.trace_observer is None
        assert radar.encoder0.blocks[0].trace_observer is None
        assert not radar.encoder0.blocks[0]._forward_hooks


def test_selector_trace_records_real_evaluation_boundary_and_ties():
    selector = CandidateSelector({"raw_topk": 3, "pre_nms_topk": 3,
                                  "nms_radius_m": 1., "final_topk": 1})
    events = []; selector.trace_observer = events.append
    fields = {"pred_xyz": torch.tensor([[0., 0., 0.], [.1, 0., 0.], [3., 0., 0.], [4., 0., 0.]]),
              "logits": torch.ones(4), "fine_features": torch.zeros(4, 128),
              "source_token_id": torch.arange(4), "batch_index": torch.zeros(4, dtype=torch.long)}
    result = selector(fields)[0]
    assert events[0]["order"].tolist() == [0, 1, 2, 3]
    assert events[0]["evaluated_count"] == 1
    assert result["nms"]["source_token_id"].tolist() == [0]


def test_query_alignment_uses_identity_and_tolerance():
    rows = [SimpleNamespace(sequence_id="a", sample_id="a_1", query_time=1.0),
            SimpleNamespace(sequence_id="b", sample_id="b_1", query_time=1.0)]
    data = SimpleNamespace(query_records=rows)
    assert choose(data, "b", 1.000001, None, 1e-5) == (1, rows[1])
    with pytest.raises(ValueError, match="matched 2"):
        choose(data, None, 1.0, None, 1e-5)
    with pytest.raises(ValueError, match="matched 0"):
        choose(data, "a", 1.1, None, 1e-5)


def test_viewer_serves_index_page_and_rejects_missing_tensor(tmp_path):
    query = tmp_path / "s1" / "A2"; query.mkdir(parents=True)
    (query / "manifest.json").write_text(json.dumps({"model": "A2", "sequence_id": "seq01",
                                                      "query_time": 1.0, "split": "validation_sub"}))
    def request(path):
        handler = object.__new__(ReplayHandler)
        handler.directory = str(tmp_path); handler.path = path; handler.wfile = BytesIO()
        handler.send_response = lambda status: setattr(handler, "status", status)
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        handler.do_GET()
        return handler.status, handler.wfile.getvalue()
    status, page = request("/")
    assert status == 200 and b"stageRange" in page
    status, payload = request("/api/index")
    assert status == 200
    assert json.loads(payload)["traces"] == [{"query": "s1", "model": "A2", "sequence": "seq01",
                                                "query_time": 1.0, "split": "validation_sub"}]
    assert request("/api/tensor?query=s1&model=A2&name=made.up")[0] == 404


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_viewer_stage_candidate_and_point_linkage():
    probe = Path(__file__).with_name("trace_viewer_probe.js")
    page = ROOT / "experiments/multimodal_v2/diagnostics/trace_viewer.html"
    result = subprocess.run(["node", str(probe), str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
