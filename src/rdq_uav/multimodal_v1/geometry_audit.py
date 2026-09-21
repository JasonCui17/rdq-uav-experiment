"""Metric and preregistration contracts for the P4 geometry audit.

This module deliberately does not infer calibration or acceptance thresholds.
It summarizes measurements produced by a calibrated projection pipeline and
only emits PASS/FAIL when a committed threshold contract is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DISTANCE_FIELDS = {
    "nearest_3d_distance_m": "m",
    "image_reprojection_distance_px": "px",
    "nearest_center_distance_px": "px",
    "nearest_bbox_distance_px": "px",
}
PIXEL_RADII = (8, 16, 32, 64)
FEATURE_STRIDES = (4, 8, 16)


def _finite(records: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([record.get(field, np.nan) for record in records], dtype=float)
    return values[np.isfinite(values)]


def summarize_geometry_measurements(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize real and same-sequence shuffled measurement records."""

    summary: dict[str, Any] = {"record_count": len(records), "groups": {}}
    for pairing in ("real", "same_sequence_shuffled"):
        group = [record for record in records if record.get("pairing") == pairing]
        metrics: dict[str, Any] = {"samples": len(group)}
        for field, unit in DISTANCE_FIELDS.items():
            values = _finite(group, field)
            metrics[field] = {
                "unit": unit,
                "count": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "p90": float(np.percentile(values, 90)) if len(values) else None,
                "p95": float(np.percentile(values, 95)) if len(values) else None,
            }
        reprojection = _finite(group, "image_reprojection_distance_px")
        metrics["pixel_coverage"] = {
            f"coverage_at_{radius}px": (
                float(np.mean(reprojection <= radius)) if len(reprojection) else None
            )
            for radius in PIXEL_RADII
        }
        metrics["feature_neighborhood_coverage"] = {}
        for stride in FEATURE_STRIDES:
            field = f"feature_neighborhood_hit_stride_{stride}"
            values = [bool(record[field]) for record in group if field in record]
            metrics["feature_neighborhood_coverage"][f"stride_{stride}_3x3"] = (
                float(np.mean(values)) if values else None
            )
        summary["groups"][pairing] = metrics
    return summary


def flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric_metrics(child, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value):
            result[prefix] = float(value)
    return result


def validate_threshold_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("thresholds_frozen") is not True:
        raise ValueError("formal audit requires thresholds_frozen: true")
    commit = contract.get("threshold_commit")
    if not isinstance(commit, str) or len(commit) < 7:
        raise ValueError("formal audit requires a committed threshold_commit")
    thresholds = contract.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("formal audit requires at least one numeric threshold")
    for metric, rule in thresholds.items():
        if not isinstance(rule, Mapping) or rule.get("op") not in {"le", "ge"}:
            raise ValueError(f"threshold {metric!r} requires op: le|ge")
        value = rule.get("value")
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"threshold {metric!r} requires a finite numeric value")


def evaluate_geometry_gate(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a frozen generic metric contract without inventing thresholds."""

    validate_threshold_contract(contract)
    flattened = flatten_numeric_metrics(summary)
    checks = []
    for metric, rule in contract["thresholds"].items():
        if metric not in flattened:
            raise KeyError(f"threshold metric is absent from report: {metric}")
        actual = flattened[metric]
        expected = float(rule["value"])
        passed = actual <= expected if rule["op"] == "le" else actual >= expected
        checks.append(
            {
                "metric": metric,
                "op": rule["op"],
                "threshold": expected,
                "actual": actual,
                "passed": bool(passed),
            }
        )
    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "threshold_commit": contract["threshold_commit"],
        "checks": checks,
    }
