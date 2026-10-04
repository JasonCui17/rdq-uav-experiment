"""Sequence isolation, evaluation selection and audit supervision tests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

from rdq_uav.multimodal_v2.data import (
    MultimodalV2Dataset, build_datasets, build_split_dataset, load_sequence_splits,
    query_records_from_3d_gt, build_3d_target_index,
)
from experiments.multimodal_v2 import evaluate
from experiments.multimodal_v2.diagnostics.audit_dataset import audit_sequence


def split_payload():
    names = [f"seq{i:04d}" for i in range(20)]
    return {"schema_version": 1, "num_sequences": 20,
            "train_sub": names[:14], "validation_sub": names[14:17],
            "heldout_test_sub": names[17:]}


def write_split(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_new_split_validation_rejects_duplicates_overlap_and_wrong_count(tmp_path):
    path = tmp_path / "split.json"
    assert sum(map(len, load_sequence_splits(write_split(path, split_payload())).values())) == 20
    changes = (
        ("duplicate", lambda p: p["train_sub"].__setitem__(1, p["train_sub"][0])),
        ("shared", lambda p: p["heldout_test_sub"].__setitem__(0, p["validation_sub"][0])),
        ("3 sequences", lambda p: p["heldout_test_sub"].pop()),
    )
    for error, change in changes:
        payload = copy.deepcopy(split_payload())
        change(payload)
        with pytest.raises(ValueError, match=error):
            load_sequence_splits(write_split(path, payload))


def test_heldout_cannot_be_training_or_checkpoint_selection(tmp_path):
    config = {"data": {"train_split": "heldout_test_sub", "val_split": "validation_sub"}}
    with pytest.raises(ValueError, match="heldout_test_sub"):
        build_datasets(config, tmp_path)
    config["data"].update(train_split="train_sub", val_split="heldout_test_sub")
    with pytest.raises(ValueError, match="heldout_test_sub"):
        build_datasets(config, tmp_path)


def test_selected_evaluation_dataset_reads_only_selected_sequence(tmp_path, monkeypatch):
    write_split(tmp_path / "split.json", split_payload())
    (tmp_path / "geometry.json").write_text('{"time_offset_s": 0.0}')
    (tmp_path / "camera.yaml").write_text('cameras:\n  left:\n    resolution: [16, 8]\n')
    selected = split_payload()["heldout_test_sub"]
    for index, sequence in enumerate(selected):
        directory = tmp_path / sequence / "ground_truth"
        directory.mkdir(parents=True)
        np.save(directory / f"{100 + index}.npy", np.ones(3))
    config = {"data": dict(root=".", split_file="split.json",
                           geometry_calibration="geometry.json", camera_config="camera.yaml",
                           dino_short_edge=8, dino_max_size=16)}
    import rdq_uav.multimodal_v2.data as data
    original = data.query_records_from_3d_gt
    calls = []
    def checked(root, sequences):
        calls.append(list(sequences))
        assert list(sequences) == selected
        return original(root, sequences)
    monkeypatch.setattr(data, "query_records_from_3d_gt", checked)
    dataset = build_split_dataset(config, tmp_path, "heldout_test_sub")
    assert calls == [selected]
    assert len(dataset) == 3


def test_evaluate_passes_requested_split_without_building_train(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  stage: B0\nmodel: {}\n")
    selections = []
    def selected(config, root, split):
        selections.append(split)
        raise RuntimeError("dataset selected")
    monkeypatch.setattr(evaluate, "build_split_dataset", selected)
    for split in ("validation_sub", "heldout_test_sub"):
        monkeypatch.setattr(evaluate, "parse_args", lambda split=split: SimpleNamespace(
            config=config_path, mode="B0", checkpoint=Path("unused.ckpt"),
            output=tmp_path / "out", device="cpu", split=split,
            indices=None, limit=None, num_workers=0))
        with pytest.raises(RuntimeError, match="dataset selected"):
            evaluate.main()
    assert selections == ["validation_sub", "heldout_test_sub"]


def test_evaluate_cli_defaults_to_validation_and_rejects_train(tmp_path, monkeypatch):
    args = ["evaluate.py", "--config", str(tmp_path / "config.yaml"),
            "--mode", "B0", "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", args)
    assert evaluate.parse_args().split == "validation_sub"
    monkeypatch.setattr(sys, "argv", args + ["--split", "heldout_test_sub"])
    assert evaluate.parse_args().split == "heldout_test_sub"
    monkeypatch.setattr(sys, "argv", args + ["--split", "train_sub"])
    with pytest.raises(SystemExit):
        evaluate.parse_args()


def test_audit_b0_requires_positive_point_and_counts_distinct_images(tmp_path):
    sequence = "seq0001"
    base = tmp_path / sequence
    for directory in ("ground_truth", "livox_avia", "lidar_360", "Image", "2d_detect"):
        (base / directory).mkdir(parents=True)
    for time in (10.0, 10.1):
        np.save(base / "ground_truth" / f"{time}.npy", np.array([1., 2., 3.]))
    np.save(base / "livox_avia" / "10.0.npy", np.array([[10., 10., 10.]], np.float32))
    Image.new("RGB", (16, 8)).save(base / "Image" / "10.0.png")
    (base / "2d_detect" / "10.0.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    dataset = MultimodalV2Dataset(tmp_path, query_records_from_3d_gt(tmp_path, [sequence]),
                                  target_3d_index=build_3d_target_index(tmp_path, [sequence]),
                                  camera_wh=(16, 8), short_edge=8, max_size=16)
    counts = audit_sequence(dataset, dataset.query_records)
    assert counts["queries"] == counts["valid_radar_samples"] == 2
    assert counts["b0_supervised_samples"] == 0
    assert counts["b1_supervised_samples"] == counts["valid_bbox_samples"] == 2
    assert counts["supervised_images"] == 1
