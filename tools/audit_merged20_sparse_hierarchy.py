#!/usr/bin/env python3
"""G2-0 read-only audit of merged-20 dual-LiDAR sparse preprocessing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdq_uav.multimodal.merged_lidar import (  # noqa: E402
    LidarFrameEvent, build_parent, concatenate_frames, fixed_stat_embedding,
    interpolate_position, isolated_point_keep_mask, load_released_xyz,
    merge_frame_streams, packed_offsets, select_last_history, support_to_gt,
    voxelize_level,
)

DATA_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")
SPLIT_PATH = ROOT / "outputs/mmuav_paper_reproduction/splits/splits.json"
OUTPUT = ROOT / "outputs/own_multimodal_research/g2_merged20_sparse_hierarchy_audit"
SEQUENCES = (
    "seq0001", "seq0007", "seq0009", "seq0024", "seq0036", "seq0049",
    "seq0054", "seq0068", "seq0075", "seq0089", "seq0098", "seq0102",
)
SENSORS = ((0, "Avia", "livox_avia"), (1, "Mid360", "lidar_360"))
ASSUMPTION = "EXISTING_MMUAV_COORDINATE_ASSUMPTION"


def timestamp_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.npy"), key=lambda path: (float(path.stem), str(path)))


def deterministic_indices(total: int, requested: int) -> np.ndarray:
    if total <= requested:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, requested, dtype=np.int64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float], percentiles=(50, 90, 95), include_mean=True) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {}
    result = {"mean": float(np.mean(array))} if include_mean else {}
    result.update({f"p{p}": float(np.percentile(array, p)) for p in percentiles})
    result["max"] = float(np.max(array))
    return result


def histogram_percentiles(histogram: Counter[int], percentiles: tuple[int, ...]) -> dict[str, int]:
    total = sum(histogram.values())
    if not total:
        return {f"p{p}": 0 for p in percentiles} | {"max": 0}
    result = {}
    cumulative = 0
    targets = {p: max(1, int(np.ceil(total * p / 100.0))) for p in percentiles}
    for value in sorted(histogram):
        cumulative += histogram[value]
        for percentile, target in targets.items():
            if f"p{percentile}" not in result and cumulative >= target:
                result[f"p{percentile}"] = int(value)
    result["max"] = int(max(histogram))
    return result


def current_peak_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--samples-per-sequence", type=int, default=30)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.data_root.resolve() != DATA_ROOT.resolve():
        raise ValueError("G2-0 data root is frozen to official/train")
    sequences = args.sequences[:1] if args.smoke else args.sequences
    requested = 2 if args.smoke else args.samples_per_sequence
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_of = {sequence: split for split in ("train_sub", "validation_sub", "heldout_test_sub")
                for sequence in split_payload[split]}

    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    hierarchy_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    token_counts = {"L0": [], "L1": [], "L2": []}
    point_count_hist: Counter[int] = Counter()
    child_hist = {"L1": Counter(), "L2": Counter()}
    errors: list[dict[str, str]] = []
    future_leakage = conservation_errors = hierarchy_errors = occupancy_errors = 0

    for sequence in sequences:
        sequence_dir = args.data_root / sequence
        merge_start = time.perf_counter()
        streams = []
        for sensor_id, sensor_name, directory in SENSORS:
            streams.append([
                LidarFrameEvent(sequence, float(path.stem), sensor_id, sensor_name, path)
                for path in timestamp_paths(sequence_dir / directory)
            ])
        merged = merge_frame_streams(streams)
        merge_ms = (time.perf_counter() - merge_start) * 1000.0
        if any(a.timestamp > b.timestamp for a, b in zip(merged, merged[1:])):
            raise AssertionError(f"Merged stream not ordered: {sequence}")
        gt_paths = timestamp_paths(sequence_dir / "ground_truth")
        gt_times = np.asarray([float(path.stem) for path in gt_paths])
        gt_positions = np.stack([
            np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(3)
            for path in gt_paths
        ])
        chosen = deterministic_indices(len(gt_paths), requested)
        for local_sample_index, gt_index in enumerate(chosen):
            sample_id = f"{sequence}_g{int(gt_index):06d}"
            total_start = time.perf_counter()
            prediction_time = float(gt_times[gt_index])
            gt_xyz = gt_positions[gt_index]
            select_start = time.perf_counter()
            selected = select_last_history(merged, prediction_time, 20)
            select_ms = (time.perf_counter() - select_start) * 1000.0
            if any(event.timestamp > prediction_time for event in selected):
                future_leakage += 1
            avia_frames = sum(event.sensor_id == 0 for event in selected)
            mid_frames = sum(event.sensor_id == 1 for event in selected)
            oldest = selected[0].timestamp if selected else np.nan
            newest = selected[-1].timestamp if selected else np.nan
            duration = prediction_time - oldest if selected else 0.0
            load_start = time.perf_counter()
            loaded = []
            for selected_index, event in enumerate(selected):
                try:
                    item = load_released_xyz(event.file_path)
                    loaded.append(item)
                    frame_rows.append({
                        "sample_id": sample_id, "sequence_id": sequence,
                        "prediction_timestamp": prediction_time, "selected_frame_index": selected_index,
                        "frame_timestamp": event.timestamp, "delta_t": event.timestamp - prediction_time,
                        "sensor_id": event.sensor_id, "sensor_name": event.sensor_name,
                        "file_path": str(event.file_path), "raw_point_count": item[1],
                        "valid_point_count": len(item[0]), "invalid_point_count": item[2],
                    })
                except Exception as exc:
                    errors.append({"sample_id": sample_id, "path": str(event.file_path), "error": repr(exc)})
                    loaded.append((np.empty((0, 3)), 0, 0))
            loading_ms = (time.perf_counter() - load_start) * 1000.0
            concatenate_start = time.perf_counter()
            combined = concatenate_frames(selected, prediction_time, loaded)
            concatenate_ms = (time.perf_counter() - concatenate_start) * 1000.0
            xyz = combined["xyz"]
            sensor_id = combined["sensor_id"]
            delta_t = combined["delta_t"]
            delta_t_norm = combined["delta_t_norm"]
            frame_index = combined["frame_index"]
            if len(xyz) != int(np.sum(combined["per_frame_valid"])):
                conservation_errors += 1
            raw_support = support_to_gt(xyz, gt_xyz)
            filter_start = time.perf_counter()
            keep = isolated_point_keep_mask(xyz, 2.0)
            radius_ms = (time.perf_counter() - filter_start) * 1000.0
            filtered = xyz[keep]
            filtered_sensor = sensor_id[keep]
            filtered_dt = delta_t[keep]
            filtered_frame = frame_index[keep]
            filtered_support = support_to_gt(filtered, gt_xyz)
            level_times = []
            levels = []
            for size in (0.5, 1.0, 2.0):
                start = time.perf_counter()
                level = voxelize_level(
                    filtered, filtered_sensor, filtered_dt, filtered_frame, size, np.zeros(3)
                )
                level_times.append((time.perf_counter() - start) * 1000.0)
                levels.append(level)
                tokens = fixed_stat_embedding(level, 64)
                if tokens.shape != (len(level["coords"]), 64):
                    raise AssertionError("Fixed embedding shape failure")
            l0, l1, l2 = levels
            parent1_start = time.perf_counter()
            parent1 = build_parent(l0["coords"])
            build_l1_ms = (time.perf_counter() - parent1_start) * 1000.0
            parent2_start = time.perf_counter()
            parent2 = build_parent(l1["coords"])
            build_l2_ms = (time.perf_counter() - parent2_start) * 1000.0
            hierarchy_error = int(parent1["error_count"]) + int(parent2["error_count"])
            hierarchy_error += int(set(map(tuple, parent1["coords"])) != set(map(tuple, l1["coords"])))
            hierarchy_error += int(set(map(tuple, parent2["coords"])) != set(map(tuple, l2["coords"])))
            hierarchy_errors += hierarchy_error
            occupancy_error = int(np.count_nonzero(
                parent1["occupancy_mask"].sum(axis=1) != parent1["child_count"]
            )) + int(np.count_nonzero(
                parent2["occupancy_mask"].sum(axis=1) != parent2["child_count"]
            ))
            occupancy_errors += occupancy_error
            point_count_hist.update(map(int, l0["counts"]))
            child_hist["L1"].update(map(int, parent1["child_count"]))
            child_hist["L2"].update(map(int, parent2["child_count"]))
            for name, level in zip(("L0", "L1", "L2"), levels):
                token_counts[name].append(len(level["coords"]))
            old_gt, old_gt_valid = interpolate_position(gt_times, gt_positions, oldest) if selected else (np.full(3, np.nan), False)
            gt_motion = float(np.linalg.norm(gt_xyz - old_gt)) if old_gt_valid else np.nan
            total_ms = (time.perf_counter() - total_start) * 1000.0
            avia_points = int(np.count_nonzero(sensor_id == 0))
            mid_points = int(np.count_nonzero(sensor_id == 1))
            invalid = int(combined["invalid_total"])
            raw_total = int(combined["raw_total"])
            valid_total = len(xyz)
            removed = valid_total - len(filtered)
            window_rows.append({
                "sample_id": sample_id, "sequence_id": sequence, "split_name": split_of[sequence],
                "gt_index": int(gt_index), "prediction_timestamp": prediction_time,
                "selected_frame_count": len(selected), "avia_frame_count": avia_frames,
                "mid360_frame_count": mid_frames, "oldest_frame_timestamp": oldest,
                "newest_frame_timestamp": newest, "window_duration_ms": duration * 1000.0,
                "oldest_delta_t": oldest - prediction_time if selected else "",
                "min_point_delta_t": float(np.min(delta_t)) if len(delta_t) else "",
                "max_point_delta_t": float(np.max(delta_t)) if len(delta_t) else "",
                "min_point_delta_t_norm": float(np.min(delta_t_norm)) if len(delta_t_norm) else "",
                "max_point_delta_t_norm": float(np.max(delta_t_norm)) if len(delta_t_norm) else "",
                "gt_motion_distance": gt_motion, "gt_oldest_interpolation_valid": old_gt_valid,
                "frame_stream_merge_ms_amortized": merge_ms / len(chosen), "select_last20_ms": select_ms,
                "point_loading_ms": loading_ms, "concatenate_ms": concatenate_ms,
                "radius_filter_ms": radius_ms, "voxelize_0p5_ms": level_times[0],
                "voxelize_1m_ms": level_times[1], "voxelize_2m_ms": level_times[2],
                "build_L1_ms": build_l1_ms, "build_L2_ms": build_l2_ms,
                "total_preprocess_ms": total_ms, "peak_cpu_memory_mb": current_peak_mb(),
            })
            noise_rows.append({
                "sample_id": sample_id, "sequence_id": sequence, "raw_total_points": raw_total,
                "valid_total_points": valid_total, "invalid_removed_points": invalid,
                "filtered_total_points": len(filtered), "removed_noise_points": removed,
                "removed_noise_ratio": removed / valid_total if valid_total else 0.0,
                "filter_radius_m": 2.0, "filter_rule": "remove_only_zero_neighbor_points_after_dual_lidar_merge",
                "point_conservation_valid": valid_total == sum(combined["per_frame_valid"]),
            })
            support_rows.append({
                "sample_id": sample_id, "sequence_id": sequence,
                "gt_x": gt_xyz[0], "gt_y": gt_xyz[1], "gt_z": gt_xyz[2],
                "raw_dmin_to_gt": raw_support["d_min"],
                "raw_n_within_0p5m": raw_support["n_0p5m"],
                "raw_n_within_1m": raw_support["n_1m"], "raw_n_within_2m": raw_support["n_2m"],
                "filtered_dmin_to_gt": filtered_support["d_min"],
                "filtered_n_within_0p5m": filtered_support["n_0p5m"],
                "filtered_n_within_1m": filtered_support["n_1m"],
                "filtered_n_within_2m": filtered_support["n_2m"],
            })
            n0, n1, n2 = (len(level["coords"]) for level in levels)
            hierarchy_rows.append({
                "sample_id": sample_id, "sequence_id": sequence,
                "raw_total_points": raw_total, "valid_total_points": valid_total,
                "filtered_total_points": len(filtered), "avia_valid_points": avia_points,
                "mid360_valid_points": mid_points,
                "avia_point_fraction": avia_points / valid_total if valid_total else 0.0,
                "mid360_point_fraction": mid_points / valid_total if valid_total else 0.0,
                "occupied_voxels_0p5": n0, "occupied_voxels_1m": n1,
                "occupied_voxels_2m": n2,
                "reduction_0p5_to_1": 1.0 - n1 / n0 if n0 else 0.0,
                "reduction_1_to_2": 1.0 - n2 / n1 if n1 else 0.0,
                "single_point_voxels_0p5": int(np.count_nonzero(l0["counts"] == 1)),
                "single_point_ratio_0p5": float(np.mean(l0["counts"] == 1)) if n0 else 0.0,
                "avia_only_voxels_0p5": int(np.count_nonzero((l0["avia_count"] > 0) & (l0["mid360_count"] == 0))),
                "mid360_only_voxels_0p5": int(np.count_nonzero((l0["mid360_count"] > 0) & (l0["avia_count"] == 0))),
                "mixed_voxels_0p5": int(np.count_nonzero((l0["avia_count"] > 0) & (l0["mid360_count"] > 0))),
                "child_count_L1_total": int(np.sum(parent1["child_count"])),
                "child_count_L1_mean": float(np.mean(parent1["child_count"])) if len(parent1["child_count"]) else 0.0,
                "child_count_L1_max": int(np.max(parent1["child_count"])) if len(parent1["child_count"]) else 0,
                "child_count_L2_total": int(np.sum(parent2["child_count"])),
                "child_count_L2_mean": float(np.mean(parent2["child_count"])) if len(parent2["child_count"]) else 0.0,
                "child_count_L2_max": int(np.max(parent2["child_count"])) if len(parent2["child_count"]) else 0,
                "parent_mapping_error_count": hierarchy_error,
                "occupancy_mask_error_count": occupancy_error,
                "L0_token_shape": f"{n0}x64", "L1_token_shape": f"{n1}x64", "L2_token_shape": f"{n2}x64",
            })

    # Packed batches use the largest real token-count samples as the memory stress case.
    batch_results = []
    for batch_size in (1, 2, 4):
        try:
            counts = sorted(token_counts["L0"], reverse=True)[:batch_size]
            offsets, batch_index = packed_offsets(counts)
            allocation = np.empty((int(offsets[-1]), 64), dtype=np.float32)
            actual_batch_size = len(counts)
            if len(batch_index) != len(allocation) or len(offsets) != actual_batch_size + 1:
                raise AssertionError("Packed batch structure mismatch")
            batch_results.append({"batch_size": batch_size, "actual_batch_size": actual_batch_size,
                                  "status": "PASS" if actual_batch_size == batch_size else "PASS_AVAILABLE_SAMPLES",
                                  "packed_tokens": len(allocation),
                                  "allocated_mb": (allocation.nbytes + batch_index.nbytes + offsets.nbytes) / 1e6,
                                  "failure_reason": ""})
            del allocation, batch_index
        except MemoryError as exc:
            batch_results.append({"batch_size": batch_size, "actual_batch_size": len(counts),
                                  "status": "OOM", "packed_tokens": "",
                                  "allocated_mb": current_peak_mb(), "failure_reason": repr(exc)})

    for field in ("frame_stream_merge_ms_amortized", "select_last20_ms", "point_loading_ms", "concatenate_ms",
                  "radius_filter_ms", "voxelize_0p5_ms", "voxelize_1m_ms", "voxelize_2m_ms",
                  "build_L1_ms", "build_L2_ms", "total_preprocess_ms", "peak_cpu_memory_mb"):
        summary = stats([float(row[field]) for row in window_rows])
        runtime_rows.append({"metric": field, **summary})
    for row in batch_results:
        runtime_rows.append({"metric": f"packed_batch_{row['batch_size']}", "mean": row["allocated_mb"],
                             "p50": row["packed_tokens"], "p90": "", "p95": "", "max": row["status"]})

    level_summary_rows = []
    for name, size in (("L0", 0.5), ("L1", 1.0), ("L2", 2.0)):
        summary = stats(token_counts[name])
        level_summary_rows.append({"level": name, "voxel_size_m": size, "mean_tokens": summary["mean"],
                                   "median_tokens": summary["p50"], "p90_tokens": summary["p90"],
                                   "p95_tokens": summary["p95"], "max_tokens": summary["max"]})
    point_distribution = histogram_percentiles(point_count_hist, (50, 75, 90, 95, 99))
    child_distribution = {name: histogram_percentiles(hist, (50, 75, 90, 95, 99))
                          for name, hist in child_hist.items()}
    support_summary = {}
    for suffix in ("0p5m", "1m", "2m"):
        before = float(np.mean([int(row[f"raw_n_within_{suffix}"]) > 0 for row in support_rows]))
        after = float(np.mean([int(row[f"filtered_n_within_{suffix}"]) > 0 for row in support_rows]))
        support_summary[suffix] = {"before": before, "after": after, "drop": before - after}
    max_support_drop = max(item["drop"] for item in support_summary.values())
    batch1_ok = next(row["status"] for row in batch_results if row["batch_size"] == 1) == "PASS"
    g3_ready = (future_leakage == 0 and conservation_errors == 0 and hierarchy_errors == 0
                and occupancy_errors == 0 and max_support_drop <= 0.02 and batch1_ok and not errors)

    write_csv(output / "merged_frame_manifest.csv", frame_rows)
    write_csv(output / "per_sample_lidar_window.csv", window_rows)
    write_csv(output / "noise_filter_audit.csv", noise_rows)
    write_csv(output / "gt_support_before_after_filter.csv", support_rows)
    write_csv(output / "sparse_hierarchy_per_sample.csv", hierarchy_rows)
    write_csv(output / "sparse_hierarchy_summary.csv", level_summary_rows)
    write_csv(output / "runtime_memory_summary.csv", runtime_rows)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                check=True, text=True).stdout.strip()
    except Exception:
        commit = "UNKNOWN"
    config = {
        "data_root": str(args.data_root.resolve()), "selected_sequences": sequences,
        "sampling_rule": f"{requested} deterministic evenly spaced GT indices per sequence including endpoints",
        "merged_frame_rule": "merge Avia + Mid360 streams; stable sort by timestamp,sensor_id,path; select last 20 where timestamp<=t0",
        "prediction_time_rule": "t0=GT timestamp", "num_merged_frames": 20,
        "sensor_id_mapping": {"0": "Avia", "1": "Mid360"},
        "delta_t_definition": "frame_timestamp-prediction_timestamp; shared by all points in frame",
        "delta_t_normalization": "delta_t/max(window_duration,float64_eps), clipped to [-1,0]",
        "coordinate_assumption": ASSUMPTION,
        "radius_filter": {"radius_m": 2.0, "rule": "remove only zero-neighbor isolated points after all selected dual-LiDAR points are merged",
                          "implementation": "exact spatial hash candidates plus Euclidean distance; no NxN matrix"},
        "voxel_sizes_m": [0.5, 1.0, 2.0], "origin_rule": "fixed global [0,0,0]; mathematical floor",
        "split_reference": str(SPLIT_PATH.resolve()), "split_assignment_sha256": split_payload["assignment_sha256"],
        "code_commit": commit, "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256((ROOT / "src/rdq_uav/multimodal/merged_lidar.py").read_bytes()).hexdigest(),
        "support_damage_rule": "absolute sample support-rate drop must be <=0.02 at 0.5m,1m,2m",
        "point_cleaning": "existing parser semantics: first XYZ, remove NaN/Inf and all-zero padding only",
        "gt_use": "offline support and motion audit only; never point filtering or input transformation",
        "errors": errors,
    }
    (output / "audit_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    frame_counts = {name: stats([float(row[f"{key}_frame_count"]) for row in window_rows])
                    for name, key in (("Avia", "avia"), ("Mid360", "mid360"))}
    full_windows = [row for row in window_rows if int(row["selected_frame_count"]) == 20]
    full_frame_means = {
        "Avia": float(np.mean([int(row["avia_frame_count"]) for row in full_windows])),
        "Mid360": float(np.mean([int(row["mid360_frame_count"]) for row in full_windows])),
    }
    windows = stats([float(row["window_duration_ms"]) for row in window_rows])
    motions = stats([float(row["gt_motion_distance"]) for row in window_rows])
    raw_points = stats([float(row["raw_total_points"]) for row in noise_rows])
    valid_points = stats([float(row["valid_total_points"]) for row in noise_rows])
    filtered_points = stats([float(row["filtered_total_points"]) for row in noise_rows])
    removed = stats([float(row["removed_noise_ratio"]) for row in noise_rows])
    total_avia = sum(int(row["avia_valid_points"]) for row in hierarchy_rows)
    total_mid = sum(int(row["mid360_valid_points"]) for row in hierarchy_rows)
    l0_total = sum(int(row["occupied_voxels_0p5"]) for row in hierarchy_rows)
    avia_only = sum(int(row["avia_only_voxels_0p5"]) for row in hierarchy_rows)
    mid_only = sum(int(row["mid360_only_voxels_0p5"]) for row in hierarchy_rows)
    mixed = sum(int(row["mixed_voxels_0p5"]) for row in hierarchy_rows)
    single_ratio = sum(int(row["single_point_voxels_0p5"]) for row in hierarchy_rows) / l0_total
    per_sequence_removed = {
        sequence: float(np.mean([float(row["removed_noise_ratio"]) for row in noise_rows if row["sequence_id"] == sequence]))
        for sequence in sequences
    }
    extreme = sorted(noise_rows, key=lambda row: int(row["valid_total_points"]), reverse=True)[:5]
    extreme_summary = [
        {"sample_id": row["sample_id"], "sequence_id": row["sequence_id"],
         "valid_total_points": int(row["valid_total_points"]),
         "filtered_total_points": int(row["filtered_total_points"])}
        for row in extreme
    ]
    runtime_index = {row["metric"]: row for row in runtime_rows}
    report = f"""# G2-0 Merged-20 Sparse Hierarchy Audit

## 1. 统一时间流

All {len(sequences)} sequences produced merged streams. Samples: {len(window_rows)}. Mean frame composition:
Avia={frame_counts['Avia']['mean']:.3f}, Mid360={frame_counts['Mid360']['mean']:.3f}; counts are timestamp-driven,
never forced 10:10. Among {len(full_windows)} full 20-event windows, means are Avia={full_frame_means['Avia']:.3f}
and Mid360={full_frame_means['Mid360']:.3f}. Window duration median/P90/P95/max={windows['p50']:.3f}/{windows['p90']:.3f}/
{windows['p95']:.3f}/{windows['max']:.3f} ms.

## 2. 点数

Per-sample raw rows mean/median/P95/max={raw_points['mean']:.1f}/{raw_points['p50']:.1f}/
{raw_points['p95']:.1f}/{raw_points['max']:.1f}. Valid points={valid_points}. Filtered points={filtered_points}.
Aggregate valid composition: Avia={total_avia/(total_avia+total_mid):.4%}, Mid360={total_mid/(total_avia+total_mid):.4%}.

## 3. Δt

Point delta_t range=[{min(float(row['min_point_delta_t']) for row in window_rows if row['min_point_delta_t']!=''):.6f},
{max(float(row['max_point_delta_t']) for row in window_rows if row['max_point_delta_t']!=''):.6f}] s.
Oldest delta_t median={-windows['p50']/1000:.6f} s; P95 window={windows['p95']:.3f} ms.
Future leakage count={future_leakage}.

## 4. 2m孤立点去噪

Removal ratio mean/median/P90/P95/max={removed['mean']:.6%}/{removed['p50']:.6%}/{removed['p90']:.6%}/
{removed['p95']:.6%}/{removed['max']:.6%}. Per-sequence mean={json.dumps(per_sequence_removed)}.
GT support before/after/drop={json.dumps(support_summary)}. Significant damage threshold is fixed at 2 percentage points;
maximum observed drop={max_support_drop:.6%}.

## 5. 0.5/1/2m Token统计

| Level | Voxel size | Mean | Median | P90 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {row['level']} | {row['voxel_size_m']} | {row['mean_tokens']:.2f} | {row['median_tokens']:.0f} | "
        f"{row['p90_tokens']:.0f} | {row['p95_tokens']:.0f} | {row['max_tokens']:.0f} |"
        for row in level_summary_rows
    ) + f"""

## 6. 每体素点数

L0 points/voxel={json.dumps(point_distribution)}. Single-point voxel ratio={single_ratio:.6%}. No point cap applied.

## 7. 父子结构

L1 child_count={json.dumps(child_distribution['L1'])}; L2 child_count={json.dumps(child_distribution['L2'])}.
Occupancy mask errors={occupancy_errors}; parent mapping errors={hierarchy_errors}.

## 8. 双LiDAR组成

L0 voxel composition: Avia-only={avia_only/l0_total:.6%}, Mid360-only={mid_only/l0_total:.6%},
mixed={mixed/l0_total:.6%}.

## 9. 20帧运动跨度

Window duration ms={json.dumps(windows)}. GT linear-interpolation motion distance m={json.dumps(motions)}.
Motion is audit-only; no compensation was applied.

## 10. Runtime

Amortized stream merge ms={json.dumps(runtime_index['frame_stream_merge_ms_amortized'])};
last-20 selection ms={json.dumps(runtime_index['select_last20_ms'])};
loading ms={json.dumps(runtime_index['point_loading_ms'])}; concatenate ms={json.dumps(runtime_index['concatenate_ms'])};
radius filter ms={json.dumps(runtime_index['radius_filter_ms'])}; L0 voxelization ms={json.dumps(runtime_index['voxelize_0p5_ms'])};
L1/L2 direct voxelization ms={json.dumps([runtime_index['voxelize_1m_ms'], runtime_index['voxelize_2m_ms']])};
L1/L2 parent build ms={json.dumps([runtime_index['build_L1_ms'], runtime_index['build_L2_ms']])};
total ms={json.dumps(runtime_index['total_preprocess_ms'])}. Packed batch tests={json.dumps(batch_results)}.
Peak process CPU memory={max(float(row['peak_cpu_memory_mb']) for row in window_rows):.2f} MiB.

## 11. 异常

Future leakage={future_leakage}; conservation errors={conservation_errors}; hierarchy errors={hierarchy_errors};
occupancy errors={occupancy_errors}; file errors={len(errors)}. Samples with fewer than 20 historical frames=
{sum(int(row['selected_frame_count'])<20 for row in window_rows)}. Largest valid-point samples={json.dumps(extreme_summary)}.
Structural observation under {ASSUMPTION}: valid-point composition is Avia={total_avia/(total_avia+total_mid):.4%}
versus Mid360={total_mid/(total_avia+total_mid):.4%}, and mixed L0 voxels are {mixed/l0_total:.6%}.
This audit records the imbalance but does not estimate or alter any coordinate transform.
No coordinates were fitted, transformed, motion-compensated, or GT-filtered.

## 12. 最终判断

**G3_READY = {'YES' if g3_ready else 'NO'}**
"""
    (output / "G2_0_MERGED20_SPARSE_HIERARCHY_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "sequences": len(sequences), "samples": len(window_rows),
                      "selected_frame_rows": len(frame_rows), "support": support_summary,
                      "future_leakage": future_leakage, "conservation_errors": conservation_errors,
                      "hierarchy_errors": hierarchy_errors, "occupancy_errors": occupancy_errors,
                      "batch": batch_results, "G3_READY": "YES" if g3_ready else "NO"}, indent=2))


if __name__ == "__main__":
    main()
