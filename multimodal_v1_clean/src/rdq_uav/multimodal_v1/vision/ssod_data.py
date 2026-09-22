"""Manifest contracts and deterministic stratified subset selection for vision SSOD."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class VisionLabelRecord:
    sequence_id: str
    query_uid: object
    query_time: float
    image_path: str
    role: str
    gt_xyz_m: tuple[float, float, float]
    range_m: float
    box_xyxy_px: tuple[float, float, float, float] | None
    gt_2d_valid: bool

    @property
    def key(self) -> tuple[str, object]:
        return self.sequence_id, self.query_uid


def _stable_rank(record: Mapping[str, Any], seed: int) -> str:
    payload = f"{seed}|{record['sequence_id']}|{record['query_uid']}|{record['query_time']:.9f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _range_bin(value: float, edges: Sequence[float]) -> int:
    for index, upper in enumerate(edges):
        if value < upper:
            return index
    return len(edges)


def _allocate(total: int, strata_sizes: Mapping[tuple[str, int], int]) -> dict[tuple[str, int], int]:
    if total <= 0 or not strata_sizes:
        return {key: 0 for key in strata_sizes}
    population = sum(strata_sizes.values())
    raw = {key: total * size / population for key, size in strata_sizes.items()}
    base = {key: min(strata_sizes[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = total - sum(base.values())
    order = sorted(strata_sizes, key=lambda key: (-(raw[key] - base[key]), key))
    while remaining > 0:
        progressed = False
        for key in order:
            if base[key] < strata_sizes[key]:
                base[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return base


def assign_stratified_roles(
    records: Sequence[Mapping[str, Any]],
    *,
    labeled_train_fraction: float = 0.04,
    calibration_fraction: float = 0.01,
    range_edges_m: Sequence[float] = (20.0, 40.0, 60.0, 100.0),
    seed: int = 20260922,
    total_limit: int | None = None,
) -> dict[tuple[str, object], str]:
    """Select train/calibration labels by sequence + GT range, without val/test leakage."""
    if labeled_train_fraction < 0 or calibration_fraction < 0 or labeled_train_fraction + calibration_fraction > 1:
        raise ValueError("invalid labeled fractions")
    if not records:
        return {}
    target_total = int(round(len(records) * (labeled_train_fraction + calibration_fraction)))
    if total_limit is not None:
        target_total = min(target_total, int(total_limit))
    target_total = max(1, target_total)
    calibration_total = int(round(target_total * calibration_fraction / max(labeled_train_fraction + calibration_fraction, 1e-12)))
    calibration_total = min(calibration_total, target_total)

    strata: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["sequence_id"]), _range_bin(float(record["range_m"]), range_edges_m))
        strata.setdefault(key, []).append(record)
    for values in strata.values():
        values.sort(key=lambda record: _stable_rank(record, seed))

    selected_alloc = _allocate(target_total, {key: len(values) for key, values in strata.items()})
    selected_by_stratum = {key: values[:selected_alloc[key]] for key, values in strata.items()}
    calibration_alloc = _allocate(calibration_total, {key: len(values) for key, values in selected_by_stratum.items()})

    roles: dict[tuple[str, object], str] = {}
    for key, values in selected_by_stratum.items():
        cal_count = calibration_alloc[key]
        cal_records = sorted(values, key=lambda record: _stable_rank(record, seed + 1))[:cal_count]
        cal_keys = {(str(record["sequence_id"]), record["query_uid"]) for record in cal_records}
        for record in values:
            query_key = (str(record["sequence_id"]), record["query_uid"])
            roles[query_key] = "labeled_calibration" if query_key in cal_keys else "labeled_train"
    return roles


def load_label_manifest(path: str | Path, *, require_boxes: bool = True) -> list[VisionLabelRecord]:
    records: list[VisionLabelRecord] = []
    seen: set[tuple[str, object]] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["sequence_id"]), item["query_uid"])
            if key in seen:
                raise ValueError(f"duplicate manifest key {key} at line {line_number}")
            seen.add(key)
            role = str(item["role"])
            if role not in {"labeled_train", "labeled_calibration"}:
                raise ValueError(f"invalid role {role!r}")
            valid = bool(item.get("gt_2d_valid", False))
            raw_box = item.get("box_xyxy_px")
            box = None if raw_box is None else tuple(float(x) for x in raw_box)
            if box is not None:
                if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"invalid bbox for {key}")
            if require_boxes and (not valid or box is None):
                raise ValueError(f"manifest record {key} has no verified 2D bbox")
            xyz = tuple(float(x) for x in item["gt_xyz_m"])
            if len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ValueError(f"invalid gt_xyz for {key}")
            records.append(VisionLabelRecord(
                sequence_id=key[0], query_uid=key[1], query_time=float(item["query_time"]),
                image_path=str(item["image_path"]), role=role, gt_xyz_m=xyz,
                range_m=float(item["range_m"]), box_xyxy_px=box, gt_2d_valid=valid,
            ))
    return records
