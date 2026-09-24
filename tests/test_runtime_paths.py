from __future__ import annotations

from pathlib import Path

import pytest

from rdq_uav.runtime_paths import (
    apply_runtime_path_overrides,
    ensure_detrex_config_link,
    resolve_project_path,
)


def _config() -> dict:
    return {
        "experiment": {"output_dir": "outputs/run"},
        "data": {
            "root": "data/default",
            "split_file": "splits/default.json",
            "annotation_manifest": "manifests/default.jsonl",
            "camera_config": "configs/camera.yaml",
            "geometry_calibration": "calibration/geometry.json",
        },
        "initialization": {
            "lidar_config": "configs/lidar.yaml",
            "lidar_checkpoint": "checkpoints/lidar.pt",
            "dino_root": "third_party/detrex",
            "dino_config": "configs/dino.py",
            "dino_checkpoint": "checkpoints/dino.pth",
            "p6_config": "configs/p6.yaml",
        },
    }


def test_relative_paths_are_repository_rooted_and_absolute_paths_are_preserved(tmp_path: Path) -> None:
    assert resolve_project_path("data/train", tmp_path) == tmp_path / "data/train"
    absolute = tmp_path / "external/checkpoint.pt"
    assert resolve_project_path(absolute, Path("/unused")) == absolute


def test_runtime_environment_overrides_are_explicit_and_do_not_mutate_source() -> None:
    source = _config()
    effective = apply_runtime_path_overrides(
        source,
        {
            "RDQ_DATA_ROOT": "/datasets/MMAUD/train",
            "RDQ_LIDAR_CHECKPOINT": "/models/lidar.pt",
            "RDQ_OUTPUT_DIR": "/experiments/e5",
        },
    )
    assert source["data"]["root"] == "data/default"
    assert effective["data"]["root"] == "/datasets/MMAUD/train"
    assert effective["initialization"]["lidar_checkpoint"] == "/models/lidar.pt"
    assert effective["experiment"]["output_dir"] == "/experiments/e5"


def test_unresolved_path_variable_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unresolved environment variable"):
        resolve_project_path("${RDQ_DOES_NOT_EXIST}/data", tmp_path)


def test_literal_dollar_in_filename_is_allowed(tmp_path: Path) -> None:
    assert resolve_project_path("data/price$5.csv", tmp_path) == tmp_path / "data/price$5.csv"


def test_detrex_config_link_is_relative_and_survives_checkout_move(tmp_path: Path) -> None:
    root = tmp_path / "detrex"
    (root / "configs/common").mkdir(parents=True)
    (root / "detectron2/configs/common").mkdir(parents=True)
    destination = ensure_detrex_config_link(root)
    assert destination.is_symlink()
    assert not Path(destination.readlink()).is_absolute()
    detectron_destination = root / "detectron2/detectron2/model_zoo/configs"
    assert detectron_destination.is_symlink()
    assert not Path(detectron_destination.readlink()).is_absolute()
    moved = tmp_path / "moved-detrex"
    root.rename(moved)
    assert (moved / "detrex/config/configs/common").is_dir()
    assert (moved / "detectron2/detectron2/model_zoo/configs/common").is_dir()
