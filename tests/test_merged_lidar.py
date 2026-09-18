from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdq_uav.multimodal.merged_lidar import (  # noqa: E402
    LidarFrameEvent, build_parent, concatenate_frames, isolated_point_keep_mask, load_released_xyz,
    merge_frame_streams, normalize_delta_t, packed_offsets, select_last_history,
    voxelize_level,
)


def event(timestamp: float, sensor_id: int) -> LidarFrameEvent:
    return LidarFrameEvent("seq", timestamp, sensor_id, "Avia" if sensor_id == 0 else "Mid360",
                           Path(f"{timestamp}_{sensor_id}.npy"))


def test_merged_stream_is_deterministic_ordered_and_preserves_sensor_ids():
    avia = [event(3.0, 0), event(1.0, 0), event(2.0, 0)]
    mid = [event(2.0, 1), event(0.0, 1)]
    first = merge_frame_streams((avia, mid))
    second = merge_frame_streams((reversed(mid), reversed(avia)))
    assert first == second
    assert [item.timestamp for item in first] == [0.0, 1.0, 2.0, 2.0, 3.0]
    assert [(item.timestamp, item.sensor_id) for item in first[2:4]] == [(2.0, 0), (2.0, 1)]


def test_history_is_last_20_never_future_and_short_history_is_explicit():
    stream = merge_frame_streams(([event(float(index), index % 2) for index in range(30)],))
    selected = select_last_history(stream, 22.5, 20)
    assert len(selected) == 20 and selected[0].timestamp == 3 and selected[-1].timestamp == 22
    assert all(item.timestamp <= 22.5 for item in selected)
    short = select_last_history(stream, 4.5, 20)
    assert len(short) == 5 and all(item.timestamp <= 4.5 for item in short)


def test_concatenation_conserves_valid_points_and_time_features():
    selected = [event(8.0, 0), event(9.0, 1), event(10.0, 0)]
    loaded = [
        (np.ones((2, 3)), 4, 2), (np.ones((3, 3)) * 2, 3, 0),
        (np.ones((1, 3)) * 3, 1, 0),
    ]
    result = concatenate_frames(selected, 10.0, loaded)
    assert len(result["xyz"]) == 6 == sum(result["per_frame_valid"])
    np.testing.assert_array_equal(result["sensor_id"], [0, 0, 1, 1, 1, 0])
    assert np.all(result["delta_t"] <= 0)
    np.testing.assert_allclose(result["delta_t_norm"], [-1, -1, -.5, -.5, -.5, 0])
    assert result["raw_total"] == 8 and result["invalid_total"] == 2
    assert normalize_delta_t(np.array([0.0]), 0.0)[0] == 0.0


def test_reader_removes_nan_inf_and_confirmed_zero_padding_only():
    raw = np.array([[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0], [0, 0, 0], [-1, 0, 2]], float)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "1.0.npy"
        np.save(path, raw)
        xyz, raw_count, invalid_count = load_released_xyz(path)
    np.testing.assert_array_equal(xyz, [[1, 2, 3], [-1, 0, 2]])
    assert raw_count == 5 and invalid_count == 3


def test_exact_radius_filter_removes_only_zero_neighbor_points():
    points = np.array([[0, 0, 0], [1.99, 0, 0], [10, 0, 0], [12.01, 0, 0], [100, 0, 0]], float)
    keep = isolated_point_keep_mask(points, 2.0)
    np.testing.assert_array_equal(keep, [True, True, False, False, False])
    rng = np.random.default_rng(4)
    random_points = rng.uniform(-8, 8, (100, 3))
    exact = isolated_point_keep_mask(random_points, 2.0)
    brute = np.zeros(100, dtype=bool)
    for index in range(100):
        distances = np.linalg.norm(random_points - random_points[index], axis=1)
        brute[index] = np.any((distances <= 2.0) & (np.arange(100) != index))
    np.testing.assert_array_equal(exact, brute)


def test_voxel_floor_order_invariance_single_and_sensor_composition():
    points = np.array([[-.01, 0, 0], [.01, 0, 0], [.2, .2, .2], [1.1, 0, 0]], float)
    sensors = np.array([0, 1, 0, 1], dtype=np.int8)
    dt = np.array([-1, -.5, -.25, 0.0])
    frames = np.array([0, 1, 2, 3])
    level = voxelize_level(points, sensors, dt, frames, .5)
    assert (-1, 0, 0) in set(map(tuple, level["coords"]))
    assert np.any(level["counts"] == 1)
    assert np.any((level["avia_count"] > 0) & (level["mid360_count"] == 0))
    assert np.any((level["mid360_count"] > 0) & (level["avia_count"] == 0))
    assert np.any((level["avia_count"] > 0) & (level["mid360_count"] > 0))
    permutation = np.array([3, 1, 0, 2])
    shuffled = voxelize_level(points[permutation], sensors[permutation], dt[permutation], frames[permutation], .5)
    np.testing.assert_array_equal(level["coords"], shuffled["coords"])
    np.testing.assert_array_equal(level["counts"], shuffled["counts"])


def test_parent_mapping_slots_and_occupancy_are_exact_for_negative_coords():
    children = np.array([
        [-2, -2, -2], [-1, -1, -1], [0, 0, 0], [0, 0, 1],
        [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ])
    parent = build_parent(children)
    assert parent["error_count"] == 0
    assert np.all(parent["occupancy_mask"].sum(axis=1) == parent["child_count"])
    full = np.flatnonzero(parent["child_count"] == 8)
    assert len(full) == 1 and np.all(parent["occupancy_mask"][full[0]])
    second = build_parent(parent["coords"])
    assert second["error_count"] == 0
    assert np.all(second["occupancy_mask"].sum(axis=1) == second["child_count"])


def test_packed_offsets_are_unpadded_and_correct():
    offsets, batch_index = packed_offsets([3, 0, 2, 4])
    np.testing.assert_array_equal(offsets, [0, 3, 3, 5, 9])
    np.testing.assert_array_equal(batch_index, [0, 0, 0, 2, 2, 3, 3, 3, 3])


def test_tool_filters_only_after_dual_stream_concatenation():
    source = (ROOT / "tools/audit_merged20_sparse_hierarchy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [(getattr(node.func, "id", ""), node.lineno) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    concatenate_line = min(line for name, line in calls if name == "concatenate_frames")
    filter_line = min(line for name, line in calls if name == "isolated_point_keep_mask")
    assert concatenate_line < filter_line
    assert "isolated_point_keep_mask(xyz, 2.0)" in source


def test_no_forbidden_pipeline_or_quadratic_distance_matrix():
    paths = [ROOT / "src/rdq_uav/multimodal/merged_lidar.py",
             ROOT / "tools/audit_merged20_sparse_hierarchy.py"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    assert "import torch" not in text
    assert "import cv2" not in text and "from pil" not in text
    assert "dbscan(" not in text and "transformer(" not in text
    assert "motion_compens" not in text.replace("no compensation", "")
    assert "dynamic_filter" not in text
    assert "points[:, none" not in text and "points[none" not in text
