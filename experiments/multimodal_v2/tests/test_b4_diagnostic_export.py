"""Evidence export preserves predictions and distinguishes missing labels from failures."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rdq_uav.multimodal_v2.diagnostic_export import candidate_evidence, save_feature_evidence
from rdq_uav.multimodal_v2.pair_fusion import CandidatePairFusion
from rdq_uav.multimodal_v2.pair_loss import JointPairLoss
from test_v2_pair_fusion import candidate, targets

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("collector", ROOT / "experiments/multimodal_v2/diagnostics/collect_b4_diagnostics.py")
collector = importlib.util.module_from_spec(spec); spec.loader.exec_module(collector)


def test_candidate_export_is_read_only_and_source_ids_are_truthful(tmp_path):
    fusion = CandidatePairFusion().eval()
    r, v = candidate("R"), candidate("V")
    r = replace(r, xyz_m=torch.tensor([[0., 0., 20.], [10., 0., 20.]]))
    v = replace(v, box_xyxy_px=torch.tensor([[40., 40., 60., 60.], [100., 100., 120., 120.]]))
    out = fusion(r, v, num_samples=1)
    before = out.joint_logit.detach().clone()
    criterion = JointPairLoss(); loss = criterion(out, targets())
    report = candidate_evidence(out, targets(), loss, criterion)
    assert torch.equal(before, out.joint_logit)
    assert len(report["radar_candidates"]) == len(report["vision_candidates"]) == 2
    assert {row["source_id"] for row in report["radar_candidates"]} == {0, 1}
    assert sorted(row["rank_3d"] for row in report["evidence_pairs"] if row["rank_3d"]) == [1, 2]
    for row in report["evidence_pairs"]:
        assert row["train_label"] == int(loss["joint_labels"][row["output_row"]])
    json.dumps(report, allow_nan=False)
    file = tmp_path / "features.npz"
    save_feature_evidence(file, out, fusion)
    with np.load(file, allow_pickle=False) as arrays:
        assert arrays["radar_features"].shape == (2, 128)
        assert arrays["similarity"].shape == arrays["matching_cost"].shape == (2, 2)
        np.testing.assert_array_equal(arrays["similarity_radar_source_ids"], [0, 1])
    assert torch.equal(before, out.joint_logit)


def test_wrong_winner_is_distinguished_from_no_correct_pair():
    fusion = CandidatePairFusion()
    r = replace(candidate("R"), xyz_m=torch.tensor([[0., 0., 20.], [10., 0., 20.]]))
    out = fusion(r, candidate("V"), num_samples=1)
    logits = torch.where(out.radar_source_index == 0, -10., 10.)
    out = replace(out, joint_logit=logits)
    criterion = JointPairLoss()
    report = candidate_evidence(out, targets(), criterion(out, targets()), criterion)
    assert report["failure_category"] == "wrong_pair_ranked_first"
    assert report["positive_negative_margin"] == -20.
    r = replace(r, xyz_m=r.xyz_m + 100.)
    out = fusion(r, candidate("V"), num_samples=1)
    assert candidate_evidence(out, targets(), criterion(out, targets()), criterion)["failure_category"] == "no_correct_rv_pair"


def test_missing_gt_is_not_exported_as_negative_evidence():
    out = CandidatePairFusion()(candidate("R"), candidate("V"), num_samples=1)
    target = targets(v=False); criterion = JointPairLoss()
    report = candidate_evidence(out, target, criterion(out, target), criterion)
    assert report["failure_category"] == "missing_joint_gt"
    assert all(row["train_label"] == -1 and row["iou"] is None for row in report["evidence_pairs"])


def test_query_subset_is_deterministic_and_spans_train_order():
    ids = collector.choose_indices(5586, 512)
    assert len(ids) == len(set(ids)) == 512
    assert ids[0] == 0 and ids[-1] == 5585
    assert collector.choose_indices(3, 0) == [0, 1, 2]


def test_csv_merge_retains_separate_train_and_validation_loss(tmp_path):
    file = tmp_path / "metrics.csv"
    file.write_text("epoch,train/loss_epoch,val/loss,val/joint_success\n0,1.2,,\n0,,2.3,0.7\n1,1.0,,\n1,,3.4,0.6\n")
    rows = collector.epoch_rows([file])
    assert len(rows) == 2
    assert rows[0]["train/loss_epoch"] == "1.2" and rows[0]["val/loss"] == "2.3"


def test_sample_level_summary_counts_ranking_and_availability_separately(tmp_path):
    rows = [dict(sample_id="q0", dataset_index=0, sequence_id="s0", image_path="a.png", final_error=10.,
                 radar_error=.1, radar_oracle_error=.1, gt_xyz=[0., 0., 20.], has_gt3d=True)]
    evidence = [dict(sample_id="q0", failure_category="wrong_pair_ranked_first", positive_negative_margin=-2.,
                     evidence_pairs=[dict(rank_3d=1, iou=.6, matching_cost=1., eval_joint_correct=False),
                                     dict(rank_3d=2, eval_joint_correct=True)])]
    (tmp_path / "per_query.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (tmp_path / "b4_candidates.jsonl").write_text(json.dumps(evidence[0]) + "\n")
    failures = collector.summarize_results(tmp_path)
    assert failures[0]["correct_rv_pair_count"] == 1
    assert json.loads((tmp_path / "failure_counts.json").read_text())["wrong_pair_ranked_first"] == 1


def test_metadata_collection_cli_does_not_copy_or_modify_model_weights(tmp_path, monkeypatch):
    import sys
    import yaml
    run = tmp_path / "run"
    log = run / "logs/csv/version_0"; log.mkdir(parents=True)
    (log / "metrics.csv").write_text("epoch,val/loss\n0,2.0\n")
    checkpoint = run / "checkpoints/best_joint.ckpt"; checkpoint.parent.mkdir()
    cfg = {"experiment": {"stage": "B4"}, "initialization": {}, "data": {}}
    for section, key in (("initialization", "lidar_config"), ("initialization", "candidate_config"),
                         ("data", "split_file"), ("data", "camera_config"), ("data", "geometry_calibration")):
        ref = tmp_path / f"{key}.yaml"; ref.write_text("{}")
        cfg[section][key] = str(ref)
    config = run / "effective_config.yaml"; config.write_text(yaml.safe_dump(cfg))
    torch.save({"epoch": 0, "global_step": 123, "state_dict": {"test": torch.ones(1)},
                "hyper_parameters": {"config": cfg}}, checkpoint)
    original = checkpoint.read_bytes()
    output = tmp_path / "diagnostics"
    monkeypatch.setattr(sys, "argv", ["collect", "--run", str(run), "--output", str(output), "--metadata-only"])
    collector.main()
    assert checkpoint.read_bytes() == original
    assert json.loads((output / "collection_status.json").read_text())["complete"]
    with tarfile.open(tmp_path / "diagnostics.tar.gz") as archive:
        assert not any(name.endswith(".ckpt") for name in archive.getnames())
        assert any(name.endswith("epoch_metrics.csv") for name in archive.getnames())


def test_sample_manifest_uses_actual_points_labels_and_missing_masks(tmp_path):
    sequence = tmp_path / "seq0001"
    labels = sequence / "2d_detect"; labels.mkdir(parents=True)
    (labels / "0.9.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    gt = sequence / "ground_truth/1.0.npy"; gt.parent.mkdir()
    np.save(gt, np.array([0., 0., 20.], dtype=np.float32))
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.array([[0., 0., 20.], [10., 0., 20.]], dtype=np.float32))
    np.save(b, np.array([[.1, 0., 20.], [20., 0., 20.]], dtype=np.float32))
    events = [SimpleNamespace(file_path=a, sensor_name="Avia"), SimpleNamespace(file_path=b, sensor_name="Mid360")]
    records = [SimpleNamespace(sequence_id="seq0001", query_time=t, sample_id=f"q{t}") for t in (1., 2.)]
    image_index = SimpleNamespace(match=lambda seq, t: SimpleNamespace(
        path=sequence / "Image/0.9.png" if t == 1. else None,
        valid=t == 1., image_time=.9 if t == 1. else None, delta_t=-.1 if t == 1. else None))
    dataset = SimpleNamespace(root=tmp_path, label_directory="2d_detect", camera_wh=(1280, 960),
        query_records=records, target_3d_index={1.: gt}, image_index=image_index,
        select_radar_events=lambda seq, t: events if t == 1. else [])
    file = tmp_path / "manifest.jsonl"
    summary = collector.sample_manifest(dataset, "validation_sub", file)
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    assert rows[0]["point_count"] == 4 and rows[0]["input_points_within_1m"] == 2
    assert rows[0]["sensor_point_counts"] == {"Avia": 2, "Mid360": 2}
    assert rows[0]["gt_box_width_px"] == pytest.approx(12.8, abs=.0002)
    assert not rows[1]["has_gt3d"] and not rows[1]["has_gt2d"]
    assert rows[1]["input_point_min_gt_distance_m"] is None
    assert summary[0]["dual_gt_queries"] == summary[0]["unique_supervised_images"] == 1
