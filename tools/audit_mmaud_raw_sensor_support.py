#!/usr/bin/env python3
"""Read-only G1-4 audit of raw sensor points around MMAUD GT positions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")
SPLIT_PATH = PROJECT_ROOT / "outputs/mmuav_paper_reproduction/splits/splits.json"
DEFAULT_SEQUENCES = (
    "seq0001", "seq0007", "seq0009", "seq0024", "seq0036", "seq0049",
    "seq0054", "seq0068", "seq0075", "seq0089", "seq0098", "seq0102",
)
SENSORS = {
    "Mid360": "lidar_360",
    "Avia": "livox_avia",
    "mmWave": "radar_enhance_pcl",
}
RADII = (0.5, 1.0, 2.0, 3.0)
ASSUMPTION = "EXISTING_MMUAV_COORDINATE_ASSUMPTION"


def deterministic_indices(total: int, requested: int) -> np.ndarray:
    """Unique, evenly spaced indices including both endpoints."""
    if total <= 0 or requested <= 0:
        return np.empty(0, dtype=np.int64)
    if total <= requested:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, requested, dtype=np.int64)


def nearest_timestamp(timestamps: np.ndarray, query: float) -> tuple[int, float]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps):
        raise ValueError("timestamps must be a nonempty 1D array")
    right = int(np.searchsorted(timestamps, query, side="left"))
    candidates = [index for index in (right - 1, right) if 0 <= index < len(timestamps)]
    index = min(candidates, key=lambda item: (abs(float(timestamps[item]) - query), item))
    return index, abs(float(timestamps[index]) - query) * 1000.0


def temporal_match_status(dt_ms: float, limit_ms: float) -> tuple[bool, str]:
    valid = bool(np.isfinite(dt_ms) and dt_ms <= limit_ms)
    return valid, "TEMPORAL_MATCH" if valid else "NO_TEMPORAL_MATCH"


def load_released_xyz(path: Path) -> tuple[np.ndarray, int]:
    """Match reproduction load_xyz semantics; additionally normalize released empty radar."""
    raw = np.asarray(np.load(path, allow_pickle=False))
    raw_rows = int(raw.shape[0]) if raw.ndim >= 1 else 0
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.float64), raw_rows
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"Expected [N,>=3] point cloud at {path}, got {raw.shape}")
    xyz = np.asarray(raw[:, :3], dtype=np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)]
    return xyz, raw_rows


def spatial_support(points: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64).reshape(3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if not len(points):
        return {"d_min": np.nan, "nearest_point": np.full(3, np.nan),
                "counts": {radius: 0 for radius in RADII}}
    distances = np.linalg.norm(points - gt[None, :], axis=1)
    nearest = int(np.argmin(distances))
    return {"d_min": float(distances[nearest]), "nearest_point": points[nearest].copy(),
            "counts": {radius: int(np.count_nonzero(distances <= radius)) for radius in RADII}}


def support_status(d_min: float) -> str:
    if np.isnan(d_min):
        return "EMPTY_FRAME"
    if d_min <= 0.5:
        return "POINT_WITHIN_0P5M"
    if d_min <= 1.0:
        return "POINT_WITHIN_1M"
    if d_min <= 2.0:
        return "POINT_WITHIN_2M"
    if d_min <= 3.0:
        return "POINT_WITHIN_3M"
    return "NONEMPTY_NO_POINT_WITHIN_3M"


def common_range_boundaries(ranges: np.ndarray) -> tuple[float, float]:
    values = np.asarray(ranges, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("ranges must be finite and nonempty")
    q33, q67 = np.quantile(values, [0.33, 0.67])
    return float(q33), float(q67)


def assign_range_bin(value: float, q33: float, q67: float) -> str:
    return "NEAR" if value <= q33 else "MID" if value <= q67 else "FAR"


def finite_stats(values: list[float]) -> dict[str, float | str]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"median": "", "p90": "", "p95": ""}
    return {"median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95))}


def safe_rate(numerator: int, denominator: int) -> float | str:
    return numerator / denominator if denominator else ""


def summarize_rows(rows: list[dict[str, Any]], sampled_gt_count: int) -> dict[str, Any]:
    temporal = [row for row in rows if row["temporal_match_valid"]]
    nonempty = [row for row in temporal if not row["frame_empty"] and row["status"] != "ERROR"]
    empty = [row for row in temporal if row["frame_empty"]]
    d_stats = finite_stats([float(row["d_min"]) for row in nonempty])
    result: dict[str, Any] = {
        "sampled_gt_count": sampled_gt_count,
        "temporal_match_count": len(temporal),
        "temporal_match_rate": safe_rate(len(temporal), sampled_gt_count),
        "nonempty_frame_count": len(nonempty),
        "nonempty_rate_all_sampled_gt": safe_rate(len(nonempty), sampled_gt_count),
        "nonempty_rate_temporal_valid": safe_rate(len(nonempty), len(temporal)),
        "empty_frame_count": len(empty),
        "error_count": sum(row["status"] == "ERROR" for row in rows),
        "median_dt_ms": finite_stats([float(row["dt_ms"]) for row in temporal])["median"],
        "p95_dt_ms": finite_stats([float(row["dt_ms"]) for row in temporal])["p95"],
        "median_d_min_nonempty": d_stats["median"],
        "p90_d_min_nonempty": d_stats["p90"],
        "p95_d_min_nonempty": d_stats["p95"],
    }
    for radius, suffix in zip(RADII, ("0p5m", "1m", "2m", "3m")):
        supported_temporal = sum(
            not row["frame_empty"] and np.isfinite(float(row["d_min"])) and float(row["d_min"]) <= radius
            for row in temporal
        )
        supported_nonempty = sum(float(row["d_min"]) <= radius for row in nonempty)
        result[f"support_{suffix}_among_temporal_valid"] = safe_rate(supported_temporal, len(temporal))
        result[f"support_{suffix}_among_temporal_valid_nonempty"] = safe_rate(supported_nonempty, len(nonempty))
    return result


def verdict(summary: dict[str, Any], near_support: float | str, far_support: float | str) -> str:
    nonempty_count = int(summary["nonempty_frame_count"])
    median = summary["median_d_min_nonempty"]
    nonempty_rate = summary["nonempty_rate_temporal_valid"]
    support2 = summary["support_2m_among_temporal_valid"]
    support2_nonempty = summary["support_2m_among_temporal_valid_nonempty"]
    if nonempty_count >= 20 and median != "" and float(median) > 10.0:
        return "POSSIBLE_COORDINATE_FRAME_PROBLEM"
    if nonempty_rate == "" or float(nonempty_rate) < 0.5 or support2 == "" or float(support2) < 0.25:
        return "SPARSE_OR_EMPTY_SUPPORT"
    if (near_support != "" and far_support != "" and float(near_support) >= 0.25
            and float(near_support) - float(far_support) >= 0.20):
        return "DISTANCE_LIMITED_SUPPORT"
    if float(support2) >= 0.5 and support2_nonempty != "" and float(support2_nonempty) >= 0.65:
        return "STRONG_RAW_SUPPORT"
    return "SPARSE_OR_EMPTY_SUPPORT"


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/own_multimodal_research/g1_raw_sensor_support_audit")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--samples-per-sequence", type=int, default=75)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.data_root.resolve() != DATA_ROOT.resolve():
        raise ValueError("G1-4 uses the fixed official/train data root")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_of = {sequence: split for split in ("train_sub", "validation_sub", "heldout_test_sub")
                for sequence in split_payload[split]}
    sequences = args.sequences[:1] if args.smoke else args.sequences
    requested = 5 if args.smoke else args.samples_per_sequence

    # Timing rule is frozen before any spatial point/GT distance is computed.
    timing_rows, sensor_indexes = [], {}
    limits_ms = {}
    for sensor, directory in SENSORS.items():
        all_periods = []
        for sequence in sequences:
            paths = sorted((args.data_root / sequence / directory).glob("*.npy"), key=lambda p: float(p.stem))
            times = np.asarray([float(path.stem) for path in paths], dtype=np.float64)
            sensor_indexes[(sequence, sensor)] = (times, paths)
            if len(times) > 1:
                all_periods.extend(np.diff(times).tolist())
        period_stats = finite_stats(all_periods)
        median_period_ms = float(period_stats["median"]) * 1000.0
        p95_period_ms = float(period_stats["p95"]) * 1000.0
        limit = max(2.0 * median_period_ms, 50.0)
        limits_ms[sensor] = limit
        timing_rows.append({"sensor": sensor, "directory": directory,
                            "period_interval_count": len(all_periods),
                            "median_period_ms": median_period_ms, "p95_period_ms": p95_period_ms,
                            "temporal_match_limit_ms": limit,
                            "rule": "max(2*global_selected_sequence_median_sensor_period,50ms)",
                            "frozen_reproduction_50ms_reused": False,
                            "note": "Frozen 50ms is trajectory-evaluation tolerance, not raw sensor-to-GT matching."})

    manifest, gt_records = [], []
    for sequence in sequences:
        gt_paths = sorted((args.data_root / sequence / "ground_truth").glob("*.npy"), key=lambda p: float(p.stem))
        for index in deterministic_indices(len(gt_paths), requested):
            path = gt_paths[int(index)]
            gt = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(3)
            record = {"sequence_id": sequence, "gt_timestamp": float(path.stem),
                      "gt_index": int(index), "split_name": split_of[sequence],
                      "selection_rule": f"deterministic_evenly_spaced_{requested}_including_endpoints"}
            manifest.append(record)
            gt_records.append((record, gt))
    q33, q67 = common_range_boundaries(np.asarray([np.linalg.norm(gt) for _, gt in gt_records]))

    detail_rows = []
    for record, gt in gt_records:
        sequence = record["sequence_id"]
        gt_range = float(np.linalg.norm(gt))
        for sensor in SENSORS:
            times, paths = sensor_indexes[(sequence, sensor)]
            base = {"sequence_id": sequence, "split_name": record["split_name"], "sensor": sensor,
                    "sensor_directory": SENSORS[sensor], "gt_timestamp": record["gt_timestamp"],
                    "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2], "gt_range": gt_range,
                    "range_bin": assign_range_bin(gt_range, q33, q67),
                    "temporal_match_limit_ms": limits_ms[sensor],
                    "coordinate_assumption": ASSUMPTION, "error": ""}
            try:
                index, dt_ms = nearest_timestamp(times, float(record["gt_timestamp"]))
                sensor_time, sensor_path = float(times[index]), paths[index]
                temporal_valid, temporal_status = temporal_match_status(dt_ms, limits_ms[sensor])
                base.update({"sensor_timestamp": sensor_time, "sensor_path": str(sensor_path),
                             "dt_ms": dt_ms, "temporal_match_valid": temporal_valid})
                if not temporal_valid:
                    base.update({"raw_num_rows": "", "num_points": "", "frame_empty": "",
                                 "d_min": "", "nearest_point_x": "", "nearest_point_y": "",
                                 "nearest_point_z": "", "n_0p5m": "", "n_1m": "", "n_2m": "",
                                 "n_3m": "", "status": temporal_status})
                else:
                    points, raw_rows = load_released_xyz(sensor_path)
                    support = spatial_support(points, gt)
                    nearest = support["nearest_point"]
                    base.update({"raw_num_rows": raw_rows, "num_points": len(points),
                                 "frame_empty": len(points) == 0, "d_min": support["d_min"],
                                 "nearest_point_x": nearest[0], "nearest_point_y": nearest[1],
                                 "nearest_point_z": nearest[2],
                                 "n_0p5m": support["counts"][0.5], "n_1m": support["counts"][1.0],
                                 "n_2m": support["counts"][2.0], "n_3m": support["counts"][3.0],
                                 "status": support_status(float(support["d_min"]))})
            except Exception as exc:
                base.update({"sensor_timestamp": "", "sensor_path": "", "dt_ms": "",
                             "temporal_match_valid": False, "raw_num_rows": "", "num_points": "",
                             "frame_empty": "", "d_min": "", "nearest_point_x": "",
                             "nearest_point_y": "", "nearest_point_z": "", "n_0p5m": "",
                             "n_1m": "", "n_2m": "", "n_3m": "", "status": "ERROR",
                             "error": f"{type(exc).__name__}: {exc}"})
            detail_rows.append(base)

    per_sensor, distance_rows, sequence_rows = [], [], []
    sensor_summaries = {}
    for sensor in SENSORS:
        rows = [row for row in detail_rows if row["sensor"] == sensor]
        summary = summarize_rows(rows, len(manifest))
        summary = {"sensor": sensor, **summary}
        sensor_summaries[sensor] = summary
        per_sensor.append(summary)
        for range_bin, lower, upper in (("NEAR", 0.0, q33), ("MID", q33, q67), ("FAR", q67, float("inf"))):
            group = [row for row in rows if row["range_bin"] == range_bin]
            sub = summarize_rows(group, len(group))
            distance_rows.append({"sensor": sensor, "range_bin": range_bin,
                                  "range_min": lower, "range_max": upper, "count": len(group),
                                  "temporal_match_rate": sub["temporal_match_rate"],
                                  "nonempty_rate_temporal_valid": sub["nonempty_rate_temporal_valid"],
                                  "median_d_min": sub["median_d_min_nonempty"],
                                  "p90_d_min": sub["p90_d_min_nonempty"],
                                  "support_0p5m_among_temporal_valid": sub["support_0p5m_among_temporal_valid"],
                                  "support_1m_among_temporal_valid": sub["support_1m_among_temporal_valid"],
                                  "support_2m_among_temporal_valid": sub["support_2m_among_temporal_valid"],
                                  "support_3m_among_temporal_valid": sub["support_3m_among_temporal_valid"]})
        for sequence in sequences:
            group = [row for row in rows if row["sequence_id"] == sequence]
            sub = summarize_rows(group, len(group))
            source_paths = sensor_indexes[(sequence, sensor)][1]
            source_parser_nonempty = 0
            for path in source_paths:
                try:
                    points, _ = load_released_xyz(path)
                    source_parser_nonempty += int(len(points) > 0)
                except Exception:
                    pass
            flags = []
            if sub["temporal_match_count"] and sub["nonempty_frame_count"] == 0:
                flags.append("ALL_FRAMES_EMPTY")
            if (sub["nonempty_frame_count"] >= 20 and sub["median_d_min_nonempty"] != ""
                    and float(sub["median_d_min_nonempty"]) > 10.0):
                flags.append("SYSTEMATIC_LARGE_OFFSET")
            sequence_rows.append({"sequence_id": sequence, "split_name": split_of[sequence],
                                  "sensor": sensor, "sample_count": len(group),
                                  "temporal_match_rate": sub["temporal_match_rate"],
                                  "nonempty_rate_all_sampled_gt": sub["nonempty_rate_all_sampled_gt"],
                                  "nonempty_rate_temporal_valid": sub["nonempty_rate_temporal_valid"],
                                  "median_d_min": sub["median_d_min_nonempty"],
                                  "support_1m": sub["support_1m_among_temporal_valid"],
                                  "support_2m": sub["support_2m_among_temporal_valid"],
                                  "support_3m": sub["support_3m_among_temporal_valid"],
                                  "source_frame_count": len(source_paths),
                                  "source_parser_nonempty_frame_count": source_parser_nonempty,
                                  "diagnostic_flag": ";".join(flags) if flags else "NONE"})

    distance_index = {(row["sensor"], row["range_bin"]): row for row in distance_rows}
    verdicts = {}
    for sensor, summary in sensor_summaries.items():
        near = distance_index[(sensor, "NEAR")]["support_2m_among_temporal_valid"]
        far = distance_index[(sensor, "FAR")]["support_2m_among_temporal_valid"]
        verdicts[sensor] = verdict(summary, near, far)
        summary["verdict"] = verdicts[sensor]

    fields = ["sequence_id", "split_name", "sensor", "sensor_directory", "gt_timestamp",
              "sensor_timestamp", "sensor_path", "dt_ms", "temporal_match_limit_ms",
              "temporal_match_valid", "gt_x", "gt_y", "gt_z", "gt_range", "range_bin",
              "raw_num_rows", "num_points", "frame_empty", "d_min", "nearest_point_x",
              "nearest_point_y", "nearest_point_z", "n_0p5m", "n_1m", "n_2m", "n_3m",
              "status", "coordinate_assumption", "error"]
    csv_write(output / "audit_sample_manifest.csv", manifest)
    csv_write(output / "raw_sensor_gt_support.csv", detail_rows, fields)
    csv_write(output / "per_sensor_summary.csv", per_sensor)
    csv_write(output / "distance_stratified_summary.csv", distance_rows)
    csv_write(output / "per_sequence_sensor_summary.csv", sequence_rows)
    csv_write(output / "sensor_timing_summary.csv", timing_rows)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                check=True, text=True, capture_output=True).stdout.strip()
    except Exception:
        commit = "UNKNOWN"
    config = {
        "data_root": str(args.data_root.resolve()), "selected_sequences": sequences,
        "selected_sequence_splits": {sequence: split_of[sequence] for sequence in sequences},
        "sampling_rule": f"deterministic evenly spaced indices including endpoints; {requested} per sequence",
        "sample_count_per_sequence": requested, "total_sampled_gt": len(manifest),
        "temporal_match_rule": "nearest sensor timestamp; valid iff dt <= max(2*global selected-sequence median sensor period,50ms)",
        "temporal_match_limit_by_sensor_ms": limits_ms, "support_radii_m": list(RADII),
        "coordinate_assumption": ASSUMPTION,
        "coordinate_warning": "Engineering assumption from released MMUAV use; not official cross-sensor extrinsic verification.",
        "point_reader": "first XYZ columns, finite rows, remove all-zero padding; matches build_mmuav_cluster_dataset.load_xyz; released empty radar normalized to (0,3)",
        "range_bins": {"shared_across_sensors": True, "q33": q33, "q67": q67,
                       "definition": "NEAR<=Q33; MID<=Q67; FAR>Q67"},
        "diagnostic_rules_fixed_before_interpretation": {
            "SYSTEMATIC_LARGE_OFFSET": "at least 20 nonempty matched frames and median d_min > 10m",
            "verdict_order": "coordinate problem; sparse/empty; distance limited; strong; otherwise sparse",
            "strong": "support_2m temporal>=0.5 and nonempty-conditional>=0.65",
            "distance_limited": "near support_2m>=0.25 and near-minus-far>=0.20",
            "sparse_or_empty": "nonempty temporal rate<0.5 or effective support_2m<0.25",
        },
        "code_commit": commit, "code_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_split_reference": str(SPLIT_PATH.resolve()),
        "frozen_split_assignment_sha256": split_payload["assignment_sha256"],
        "heldout_use": "read-only raw support audit; never fit/train/threshold selection",
        "forbidden_operations": {"model": False, "dbscan": False, "candidate_generation": False,
                                 "tracking": False, "coordinate_fit": False, "data_mutation": False,
                                 "image_or_bbox_read": False},
        "errors": [row for row in detail_rows if row["status"] == "ERROR"],
    }
    (output / "audit_config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def percent(value: Any) -> str:
        return "N/A" if value == "" else f"{100*float(value):.2f}%"
    timing_by = {row["sensor"]: row for row in timing_rows}
    report_table = "\n".join(
        ["| Sensor | Match rate | Median dt ms | P95 dt ms | Limit ms |",
         "| --- | ---: | ---: | ---: | ---: |"] +
        [f"| {sensor} | {percent(sensor_summaries[sensor]['temporal_match_rate'])} | "
         f"{sensor_summaries[sensor]['median_dt_ms']:.3f} | {sensor_summaries[sensor]['p95_dt_ms']:.3f} | "
         f"{timing_by[sensor]['temporal_match_limit_ms']:.3f} |" for sensor in SENSORS]
    )
    def sensor_section(sensor: str) -> str:
        s = sensor_summaries[sensor]
        distance = [distance_index[(sensor, name)] for name in ("NEAR", "MID", "FAR")]
        support = ", ".join(
            f"{suffix}={percent(s[f'support_{suffix}_among_temporal_valid'])} effective / "
            f"{percent(s[f'support_{suffix}_among_temporal_valid_nonempty'])} nonempty-conditional"
            for suffix in ("0p5m", "1m", "2m", "3m")
        )
        trend = "; ".join(
            f"{row['range_bin']}: n={row['count']}, median d_min={row['median_d_min']}, "
            f"support2={percent(row['support_2m_among_temporal_valid'])}" for row in distance
        )
        return (f"Nonempty: {s['nonempty_frame_count']}/{s['temporal_match_count']} temporal-valid "
                f"({percent(s['nonempty_rate_temporal_valid'])}). Nonempty d_min median/P90/P95: "
                f"{s['median_d_min_nonempty']} / {s['p90_d_min_nonempty']} / {s['p95_d_min_nonempty']} m. "
                f"Support: {support}. Distance strata: {trend}. Verdict: **{s['verdict']}**.")

    radar_seq = [row for row in sequence_rows if row["sensor"] == "mmWave"]
    radar_all_empty = sum(row["source_parser_nonempty_frame_count"] == 0 for row in radar_seq)
    radar_some = len(radar_seq) - radar_all_empty
    radar_status_counts = Counter(row["status"] for row in detail_rows if row["sensor"] == "mmWave")
    flags = [row for row in sequence_rows if row["diagnostic_flag"] != "NONE"]
    flag_summary = defaultdict(lambda: defaultdict(list))
    for row in flags:
        for diagnostic in row["diagnostic_flag"].split(";"):
            flag_summary[row["sensor"]][diagnostic].append(row["sequence_id"])
    flag_text = "; ".join(
        f"{sensor} {diagnostic}={','.join(sequence_ids)}"
        for sensor in SENSORS
        for diagnostic, sequence_ids in flag_summary[sensor].items()
    ) or "none"
    dominance = {}
    for sensor in SENSORS:
        supported = Counter(row["sequence_id"] for row in detail_rows
                            if row["sensor"] == sensor and row["temporal_match_valid"]
                            and not row["frame_empty"] and np.isfinite(float(row["d_min"]))
                            and float(row["d_min"]) <= 2.0)
        total = sum(supported.values())
        dominance[sensor] = {"supporting_sequences": len(supported),
                             "largest_sequence_share": max(supported.values()) / total if total else None}
    # Purely descriptive distance change; no verdict selection or radius adjustment.
    declines = {sensor: (float(distance_index[(sensor, "NEAR")]["support_2m_among_temporal_valid"])
                         - float(distance_index[(sensor, "FAR")]["support_2m_among_temporal_valid"]))
                for sensor in SENSORS}
    largest_decline = max(declines, key=declines.get)
    recommendations = {sensor: verdicts[sensor] in {"STRONG_RAW_SUPPORT", "DISTANCE_LIMITED_SUPPORT"}
                       for sensor in SENSORS}
    report = f"""# G1-4 Raw Sensor Support Audit

## 1. Data

Data root: `{args.data_root.resolve()}`. Sampled {len(sequences)} sequences ({', '.join(sequences)})
and {len(manifest)} GT timestamps ({requested} deterministic evenly spaced indices per sequence,
including endpoints). Splits are read from `{SPLIT_PATH.resolve()}`; heldout access is read-only.

Every distance uses **{ASSUMPTION}**. This is the existing released-MMUAV engineering assumption,
not verified official cross-sensor calibration. `gt_range=norm(GT XYZ)` is only a common relative
stratifier and is not claimed to equal a physical sensor range.

## 2. Temporal matching

{report_table}

The limit is fixed independently per sensor as `max(2*median native period, 50 ms)`, computed before
spatial distances. The reproduction's 50 ms value applies to trajectory evaluation and was not
misrepresented as a raw-frame matching rule. `NO_TEMPORAL_MATCH` is excluded from empty/no-support claims.

## 3. Mid360

{sensor_section('Mid360')}

## 4. Avia

{sensor_section('Avia')}

## 5. mmWave

{sensor_section('mmWave')}

Across the {len(radar_seq)} sampled sequences, {radar_all_empty} complete source streams are all-empty
and {radar_some} contain at least one nonempty source frame. Audited matched-frame states:
`{dict(radar_status_counts)}`. Effective support includes empty temporal-valid frames; conditional
support uses only temporal-valid nonempty frames. Therefore `SENSOR_STREAM_EMPTY` is not conflated
with `NONEMPTY_BUT_NO_GT_NEARBY`.

## 6. Distance effect

Shared GT-range boundaries: Q33={q33:.6f}, Q67={q67:.6f}. Effective 2 m near-minus-far changes:
`{json.dumps(declines)}`. The largest observed decline is {largest_decline}. This is descriptive raw
support under the coordinate assumption, not detector recall.

## 7. Sequence consistency

2 m support concentration: `{json.dumps(dominance)}`. Diagnostic sequence flags:
`{flag_text}`. Aggregate support is considered potentially dominated when only
one sequence supports it or the largest supporting sequence contributes more than 50%; see the CSV
for exact per-sequence rates.

## 8. Coordinate sanity

`SYSTEMATIC_LARGE_OFFSET` is emitted only with at least 20 nonempty matched frames and median d_min
above 10 m. Such a flag means **POSSIBLE_COORDINATE_FRAME_MISMATCH**, not sensor failure. Current flags
are listed above; no transform was estimated or applied.

## 9. G1-4 verdict

- Mid360: **{verdicts['Mid360']}**
- Avia: **{verdicts['Avia']}**
- mmWave: **{verdicts['mmWave']}**

These labels describe nearby raw spatial evidence only. They are not detection, candidate, recall,
classification, tracking, or fusion metrics.

## 10. Next-stage recommendation

- Mid360 -> G2 candidate recall: **{'YES' if recommendations['Mid360'] else 'NO'}**
- Avia -> G2 candidate recall: **{'YES' if recommendations['Avia'] else 'NO'}**
- mmWave -> G2 candidate recall: **{'YES' if recommendations['mmWave'] else 'NO'}**

No G2 implementation is included.
"""
    (output / "G1_4_RAW_SENSOR_SUPPORT_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "sequences": len(sequences), "gt_samples": len(manifest),
                      "records": len(detail_rows), "verdicts": verdicts,
                      "recommendations": recommendations, "errors": len(config["errors"])}, indent=2))


if __name__ == "__main__":
    main()
