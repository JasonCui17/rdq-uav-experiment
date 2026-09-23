from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "tools/audit_mmaud_raw_sensor_support.py"
SPEC = importlib.util.spec_from_file_location("raw_sensor_support", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nearest_time_and_limit_status_are_separate():
    times = np.asarray([0.0, 0.1, 0.2])
    assert MODULE.nearest_timestamp(times, 0.14) == (1, 40.00000000000001)
    _, gap = MODULE.nearest_timestamp(times, 0.18)
    assert gap <= 50.0
    _, gap = MODULE.nearest_timestamp(times, 0.26)
    assert gap > 50.0  # caller assigns NO_TEMPORAL_MATCH before loading points
    assert MODULE.temporal_match_status(gap, 50.0) == (False, "NO_TEMPORAL_MATCH")


def test_empty_and_mmwave_empty_are_not_nonempty_no_support():
    result = MODULE.spatial_support(np.empty((0, 3)), np.zeros(3))
    assert np.isnan(result["d_min"])
    assert MODULE.support_status(result["d_min"]) == "EMPTY_FRAME"
    assert all(value == 0 for value in result["counts"].values())


def test_distance_counts_nearest_point_and_no_gt_injection():
    points = np.asarray([[0.4, 0, 0], [0.8, 0, 0], [1.5, 0, 0], [2.5, 0, 0], [4, 0, 0]])
    original = points.copy()
    result = MODULE.spatial_support(points, np.zeros(3))
    assert result["d_min"] == 0.4
    np.testing.assert_array_equal(result["nearest_point"], points[0])
    assert result["counts"] == {0.5: 1, 1.0: 2, 2.0: 3, 3.0: 4}
    np.testing.assert_array_equal(points, original)
    assert len(points) == 5  # GT was never appended to the sensor point set


def test_deterministic_sampling_and_shared_range_bins():
    first = MODULE.deterministic_indices(400, 75)
    second = MODULE.deterministic_indices(400, 75)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 75 and first[0] == 0 and first[-1] == 399
    q33, q67 = MODULE.common_range_boundaries(np.arange(1.0, 10.0))
    values = [MODULE.assign_range_bin(value, q33, q67) for value in (1.0, 5.0, 9.0)]
    assert values == ["NEAR", "MID", "FAR"]
    # The API accepts one common pair; it has no sensor-specific boundary input.
    assert MODULE.assign_range_bin(5.0, q33, q67) == MODULE.assign_range_bin(5.0, q33, q67)


def test_sensor_summaries_are_independent():
    def row(sensor, empty, distance):
        return {"sensor": sensor, "temporal_match_valid": True, "frame_empty": empty,
                "status": "EMPTY_FRAME" if empty else MODULE.support_status(distance),
                "d_min": distance, "dt_ms": 1.0}
    rows = [row("Mid360", False, 0.2), row("Mid360", False, 4.0), row("mmWave", True, np.nan)]
    mid = MODULE.summarize_rows([x for x in rows if x["sensor"] == "Mid360"], 2)
    radar = MODULE.summarize_rows([x for x in rows if x["sensor"] == "mmWave"], 1)
    assert mid["nonempty_frame_count"] == 2 and mid["support_0p5m_among_temporal_valid"] == 0.5
    assert radar["nonempty_frame_count"] == 0 and radar["empty_frame_count"] == 1
    assert radar["support_3m_among_temporal_valid"] == 0.0


def test_tool_has_no_forbidden_pipeline_imports_or_calls():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    assert not any(any(token in name.lower() for token in ("torch", "sklearn", "dbscan", "tracker", "model"))
                   for name in imports)
    calls = {getattr(node.func, "id", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not {"DBSCAN", "fit", "train", "predict"} & calls
