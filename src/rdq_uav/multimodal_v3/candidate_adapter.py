"""Versioned frozen-cache reader. Ground truth is deliberately excluded."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import Observations


EVIDENCE_FIELDS = ("point_count", "motion_valid", "motion_dynamic",
                   "motion_reliability", "motion_speed_mps")


@dataclass(frozen=True)
class CandidateFrame:
    dense: Observations
    global_pool: Observations
    baseline: Observations
    evidence: dict[int, dict[str, float | bool]]


def from_cache(data, schema_version: int) -> CandidateFrame:
    if schema_version not in (1, 2):
        raise ValueError(f"Unsupported candidate cache schema {schema_version}")
    dense = Observations(data["dense_xyz"], data["dense_score"],
                         data["dense_rank"], data["dense_token_id"])
    global_pool = Observations(data["global_xyz"], data["global_score"],
                               data["global_score"], data["global_token_id"])
    baseline = Observations(data["baseline_xyz"], data["baseline_score"],
                            data["baseline_score"], data["baseline_token_id"])
    if not set(global_pool.token_id).issubset(set(dense.token_id)):
        raise ValueError("Global candidates must be present in dense predictions")
    if not set(baseline.token_id).issubset(set(global_pool.token_id)):
        raise ValueError("Baseline candidate must be present in global pool")
    present = [f"dense_{name}" in data for name in EVIDENCE_FIELDS]
    if schema_version == 1 and any(present):
        raise ValueError("Schema 1 cache cannot contain schema 2 evidence")
    if schema_version == 2 and any(present) and not all(present):
        raise ValueError("Schema 2 evidence fields must be complete")
    evidence = {}
    if all(present):
        arrays = {name: np.asarray(data[f"dense_{name}"]) for name in EVIDENCE_FIELDS}
        if any(array.shape != (len(dense.xyz),) for array in arrays.values()):
            raise ValueError("Dense evidence length differs from candidate count")
        if np.any(arrays["point_count"] < 0):
            raise ValueError("Negative point count")
        for name in ("motion_reliability", "motion_speed_mps"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"Nonfinite {name}")
        if np.any((arrays["motion_reliability"] < 0) |
                  (arrays["motion_reliability"] > 1)) or np.any(arrays["motion_speed_mps"] < 0):
            raise ValueError("Invalid motion evidence")
        for i, token in enumerate(dense.token_id):
            evidence[int(token)] = {
                "point_count": int(arrays["point_count"][i]),
                "motion_valid": bool(arrays["motion_valid"][i]),
                "motion_dynamic": bool(arrays["motion_dynamic"][i]),
                "motion_reliability": float(arrays["motion_reliability"][i]),
                "motion_speed_mps": float(arrays["motion_speed_mps"][i]),
            }
    return CandidateFrame(dense, global_pool, baseline, evidence)
