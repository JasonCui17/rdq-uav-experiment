#!/usr/bin/env python3
"""Non-mutating, single-sequence reproduction of the MMUAV LiDAR tracker.

The preprocessing and StoneSoup tracking equations/parameters are kept equal to
dtc111111/Multi-Modal-UAV at commit
f11b57390effbe9623ee2c7d561afddc8d0cdfa7.  This runner only adapts paths and
adds lossless sidecar logging for multiple tracks at the same timestamp.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SOURCE_COMMIT = "f11b57390effbe9623ee2c7d561afddc8d0cdfa7"
RAW_MODALITIES = ("lidar_360", "livox_avia")


def _npy_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.npy"), key=lambda p: float(p.stem))


def inspect_raw(sequence_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality in RAW_MODALITIES:
        files = _npy_files(sequence_dir / modality)
        if not files:
            raise FileNotFoundError(f"No NPY files in {sequence_dir / modality}")
        sample = np.load(files[0], mmap_mode="r")
        result[modality] = {
            "frames": len(files),
            "first_timestamp": float(files[0].stem),
            "last_timestamp": float(files[-1].stem),
            "sample_shape": list(sample.shape),
            "sample_dtype": str(sample.dtype),
        }
    return result


def source_audit(source_repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source_repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--short")
    algorithm_diff = git(
        "diff",
        "--",
        "point_cloud_processing/tracker/preprocess.py",
        "point_cloud_processing/tracker/extract_feature.py",
        "point_cloud_processing/tracker/lidar_360_detector.py",
        "point_cloud_processing/tracker/fusion_tracking.py",
        "point_cloud_processing/tracker/postprocess.py",
    )
    return {
        "source_repo": str(source_repo),
        "source_commit": commit,
        "source_commit_matches_expected": commit == EXPECTED_SOURCE_COMMIT,
        "source_git_dirty": bool(status),
        "source_git_status": status.splitlines(),
        "source_git_diff_summary": algorithm_diff,
        "engineering_fixes": [
            "Path-only staging keeps generated files outside the official dataset.",
            "tracks_raw.csv records every state from every surviving track, avoiding timestamp filename overwrite.",
            "Legacy timestamp NPY output is flattened to shape (3,) for postprocess compatibility; state values are unchanged.",
        ],
        "algorithm_changes": [],
    }


def import_source_modules(source_repo: Path):
    tracker_dir = source_repo / "point_cloud_processing" / "tracker"
    required = [
        tracker_dir / "preprocess.py",
        tracker_dir / "extract_feature.py",
        tracker_dir / "lidar_360_detector.py",
        tracker_dir / "fusion_tracking.py",
        tracker_dir / "postprocess.py",
        tracker_dir / "lstm_model.pth",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing original MMUAV files: {missing}")
    sys.path.insert(0, str(tracker_dir))
    try:
        preprocess = importlib.import_module("preprocess")
        fusion_tracking = importlib.import_module("fusion_tracking")
        postprocess = importlib.import_module("postprocess")
    finally:
        sys.path.pop(0)
    return preprocess, fusion_tracking, postprocess, tracker_dir / "lstm_model.pth"


def prepare_staging(sequence_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory to preserve prior results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f"{sequence_dir.name}_staging"
    staging.mkdir()
    for modality in RAW_MODALITIES:
        src = (sequence_dir / modality).resolve()
        if not src.is_dir():
            raise FileNotFoundError(src)
        (staging / modality).symlink_to(src, target_is_directory=True)
    gt_src = sequence_dir / "ground_truth"
    if gt_src.is_dir():
        (staging / "ground_truth").symlink_to(gt_src.resolve(), target_is_directory=True)
    for name in ("lidar_360_processed", "livox_avia_processed", "lidar_fusion"):
        (staging / name).mkdir()
    tracking_dir = staging / "tracking_output"
    postprocess_dir = output_dir / "postprocess"
    tracking_dir.mkdir()
    postprocess_dir.mkdir()
    return staging, tracking_dir, postprocess_dir


def existing_staging(sequence_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    staging = output_dir / f"{sequence_dir.name}_staging"
    required = [
        staging / "lidar_360_processed",
        staging / "livox_avia_processed",
        staging / "lidar_fusion",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Cannot resume; missing preprocessed directories: {missing}")
    tracking_dir = staging / "tracking_output"
    postprocess_dir = output_dir / "postprocess"
    tracking_dir.mkdir(exist_ok=True)
    postprocess_dir.mkdir(exist_ok=True)
    if any(tracking_dir.iterdir()):
        raise FileExistsError(
            f"Cannot resume into non-empty tracking directory: {tracking_dir}"
        )
    return staging, tracking_dir, postprocess_dir


def count_outputs(staging: Path) -> dict[str, dict[str, int]]:
    result = {}
    for name in ("lidar_360_processed", "livox_avia_processed", "lidar_fusion"):
        files = _npy_files(staging / name)
        nonempty = sum(np.load(path, mmap_mode="r").size > 0 for path in files)
        result[name] = {"files": len(files), "nonempty_files": nonempty}
    return result


def _state_row(track_id: str, state: Any, age: int, num_updates: int) -> dict[str, Any]:
    vector = np.asarray(state.state_vector, dtype=float).reshape(-1)
    if vector.size < 6:
        raise ValueError(f"Expected 6D StoneSoup state, got {vector.shape}")
    return {
        "timestamp": float(state.timestamp.timestamp()),
        "track_id": track_id,
        "x": vector[0],
        "y": vector[2],
        "z": vector[4],
        "vx": vector[1],
        "vy": vector[3],
        "vz": vector[5],
        "track_age": age,
        "num_updates": num_updates,
    }


def run_source_faithful_tracking(
    staging: Path, tracking_dir: Path, source_tracking: Any
) -> list[dict[str, Any]]:
    """Mirror fusion_tracking.process_sequence and add a sidecar return value."""
    # Imports are from the same StoneSoup API used by the original source.
    from datetime import datetime, timedelta
    from stonesoup.dataassociator.neighbour import NearestNeighbour
    from stonesoup.deleter.error import CovarianceBasedDeleter
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.initiator.simple import MultiMeasurementInitiator
    from stonesoup.measures import Euclidean
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    from stonesoup.predictor.kalman import ExtendedKalmanPredictor
    from stonesoup.types.detection import Detection
    from stonesoup.types.state import GaussianState
    from stonesoup.types.update import GaussianStateUpdate
    from stonesoup.updater.kalman import ExtendedKalmanUpdater

    lidar_data = source_tracking.read_lidar_files(str(staging / "lidar_fusion"))
    if not lidar_data:
        raise RuntimeError("TRACKING failed: lidar_fusion contains no readable frames")

    start_datetime = None
    for timestamp, data in lidar_data.items():
        if data.size != 0:
            timestamp_float = float(timestamp)
            seconds = int(timestamp_float)
            microseconds = int((timestamp_float - seconds) * 1e6)
            start_datetime = datetime.fromtimestamp(seconds) + timedelta(microseconds=microseconds)
            break
    initial_data = next(iter(lidar_data.values()))
    if start_datetime is None or initial_data.size == 0:
        # This intentionally preserves the original early-exit condition.
        raise RuntimeError("TRACKING failed: original tracker requires the first fusion frame to be non-empty")

    prior = GaussianState(
        [[initial_data[0][0]], [0.001], [initial_data[0][1]], [0.001], [initial_data[0][2]], [0.001]],
        np.diag([0.01, 0.1, 0.01, 0.1, 0.01, 0.1]),
        timestamp=start_datetime,
    )
    noise_covar = 0.001
    meas_model = LinearGaussian(
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=np.diag([noise_covar, noise_covar, noise_covar]),
    )
    all_measurements = []
    for timestamp, data in lidar_data.items():
        measurement_set = set()
        if data.size != 0:
            cluster_data = source_tracking.point_cloud_detector(data, eps=1.0, min_samples=1)
            timestamp_float = float(timestamp)
            seconds = int(timestamp_float)
            microseconds = int((timestamp_float - seconds) * 1e6)
            dt = datetime.fromtimestamp(seconds) + timedelta(microseconds=microseconds)
            for detection in cluster_data:
                measurement_set.add(
                    Detection(detection.transpose(), timestamp=dt, measurement_model=meas_model)
                )
            all_measurements.append(measurement_set)

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.15), ConstantVelocity(0.15), ConstantVelocity(0.15)]
    )
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=meas_model)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=30.0)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, Euclidean(), missed_distance=3.0
    )
    data_associator = NearestNeighbour(hypothesiser)
    initiator = MultiMeasurementInitiator(
        prior_state=prior,
        measurement_model=meas_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=1,
    )

    tracks = set()
    for measurements in all_measurements:
        if not measurements:
            continue
        timestamp = next(iter(measurements)).timestamp
        hypotheses = data_associator.associate(tracks, measurements, timestamp)
        associated_measurements = set()
        for track in tracks.copy():
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                track.append(updater.update(hypothesis))
                associated_measurements.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)
        tracks -= deleter.delete_tracks(tracks)
        tracks |= initiator.initiate(measurements - associated_measurements, timestamp)

    rows: list[dict[str, Any]] = []
    for track in tracks:
        track_id = str(track.id)
        update_count = 0
        for age, state in enumerate(track, start=1):
            if isinstance(state, GaussianStateUpdate):
                update_count += 1
            row = _state_row(track_id, state, age, update_count)
            rows.append(row)
            # SOURCE OUTPUT COMPATIBILITY: preserve timestamp filenames and
            # overwrite semantics; flatten only fixes the postprocess shape.
            xyz = np.array([row["x"], row["y"], row["z"]], dtype=float)
            np.save(tracking_dir / f'{row["timestamp"]}.npy', xyz)
    rows.sort(key=lambda r: (r["track_id"], r["timestamp"]))
    return rows


TRACK_FIELDS = [
    "timestamp", "track_id", "x", "y", "z", "vx", "vy", "vz", "track_age", "num_updates"
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_tracks(rows: list[dict[str, Any]], output_dir: Path, sequence_name: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["track_id"]), []).append(row)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for track_id, states in grouped.items():
        states.sort(key=lambda r: float(r["timestamp"]))
        xyz = np.array([[r["x"], r["y"], r["z"]] for r in states])
        label = track_id[:8]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker=".", label=label)
        ax.scatter(*xyz[0], marker="o", s=45)
        ax.scatter(*xyz[-1], marker="X", s=55)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_title(f"MMUAV reproduced 3D LiDAR tracks - {sequence_name}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "track_xyz_3d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for track_id, states in grouped.items():
        states.sort(key=lambda r: float(r["timestamp"]))
        ts = np.array([r["timestamp"] for r in states], dtype=float)
        ts -= ts.min()
        for axis, key in zip(axes, ("x", "y", "z")):
            axis.plot(ts, [r[key] for r in states], marker=".", label=track_id[:8])
            axis.set_ylabel(key)
    axes[-1].set_xlabel("Time since track start (s)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"MMUAV reproduced XYZ vs time - {sequence_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "track_xyz_vs_time.png", dpi=180)
    plt.close(fig)


def gt_diagnostic(rows: list[dict[str, Any]], staging: Path, output_dir: Path) -> bool:
    gt_files = _npy_files(staging / "ground_truth") if (staging / "ground_truth").exists() else []
    if not gt_files:
        return False
    gt_ts = np.array([float(p.stem) for p in gt_files])
    gt_xyz = np.stack([np.asarray(np.load(p), dtype=float).reshape(-1)[:3] for p in gt_files])
    diagnostic = []
    for row in rows:
        idx = int(np.argmin(np.abs(gt_ts - float(row["timestamp"]))))
        item = dict(row)
        item.update(
            gt_timestamp=gt_ts[idx],
            gt_time_gap_ms=abs(gt_ts[idx] - float(row["timestamp"])) * 1000,
            gt_x=gt_xyz[idx, 0], gt_y=gt_xyz[idx, 1], gt_z=gt_xyz[idx, 2],
            raw_coordinate_distance_m=float(
                np.linalg.norm(np.array([row["x"], row["y"], row["z"]]) - gt_xyz[idx])
            ),
        )
        diagnostic.append(item)
    fields = TRACK_FIELDS + [
        "gt_timestamp", "gt_time_gap_ms", "gt_x", "gt_y", "gt_z", "raw_coordinate_distance_m"
    ]
    write_csv(output_dir / "track_vs_gt_diagnostic.csv", diagnostic, fields)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="black", label="raw GT")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["track_id"]), []).append(row)
    for track_id, states in grouped.items():
        xyz = np.array([[r["x"], r["y"], r["z"]] for r in states])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker=".", label=f"track {track_id[:8]}")
    ax.set_title("Raw MMUAV tracks vs raw GT (no coordinate transform)")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "track_vs_gt.png", dpi=180)
    plt.close(fig)
    return True


def run(args: argparse.Namespace) -> None:
    sequence_dir = args.sequence_dir.resolve()
    source_repo = args.source_repo.resolve()
    output_dir = args.output_dir.resolve()
    raw = inspect_raw(sequence_dir)
    audit = source_audit(source_repo)
    preprocess, fusion_tracking, postprocess, checkpoint = import_source_modules(source_repo)

    print("[1] RAW INPUT")
    for name, info in raw.items():
        print(
            f"{name} frames={info['frames']} first={info['first_timestamp']:.6f} "
            f"last={info['last_timestamp']:.6f} shape={tuple(info['sample_shape'])} dtype={info['sample_dtype']}"
        )
    print(f"source_commit={audit['source_commit']} dirty={audit['source_git_dirty']}")
    print(f"checkpoint={checkpoint}")
    if args.dry_run:
        print("dry-run complete; no staging or heavy processing performed")
        return

    if args.resume_preprocessed:
        staging, tracking_dir, postprocess_dir = existing_staging(sequence_dir, output_dir)
    else:
        staging, tracking_dir, postprocess_dir = prepare_staging(sequence_dir, output_dir)
    timings: dict[str, float] = {}
    audit.update(
        input_sequence=str(sequence_dir),
        staging_directory=str(staging),
        raw_input=raw,
        checkpoint_path=str(checkpoint),
    )
    (output_dir / "reproduction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    try:
        if args.resume_preprocessed:
            timings["preprocess_time_sec"] = 0.0
            timings["preprocess_reused"] = True
        else:
            start = time.perf_counter()
            preprocess.process_lidar_livox(str(staging))
            preprocess.process_lidar_360(str(staging), str(checkpoint))
            preprocess.process_fusion(str(staging))
            timings["preprocess_time_sec"] = time.perf_counter() - start
            timings["preprocess_reused"] = False
        processed = count_outputs(staging)
        print("[2] PREPROCESS")
        for name, counts in processed.items():
            print(
                f"{name} files={counts['files']} nonempty_files={counts['nonempty_files']}"
            )
        strict_preprocess_success = all(
            counts["files"] > 0 and counts["nonempty_files"] > 0
            for counts in processed.values()
        )
        if not strict_preprocess_success:
            print(
                "PREPROCESS STRICT CHECK FAILED: at least one branch has no non-empty output; "
                "continuing only because original fusion produced non-empty lidar_fusion."
            )
        if processed["lidar_fusion"]["nonempty_files"] == 0:
            raise RuntimeError(f"PREPROCESS failed: lidar_fusion has no usable points: {processed}")

        start = time.perf_counter()
        rows = run_source_faithful_tracking(staging, tracking_dir, fusion_tracking)
        timings["tracking_time_sec"] = time.perf_counter() - start
        if not rows:
            raise RuntimeError("TRACKING failed: no states from surviving tracks")
        write_csv(output_dir / "tracks_raw.csv", rows, TRACK_FIELDS)
        track_lengths = Counter(str(row["track_id"]) for row in rows)
        print("[3] TRACKING")
        print(f"legacy tracker outputs={len(_npy_files(tracking_dir))}")
        print(f"unique tracks={len(track_lengths)}")
        print(f"unique timestamps={len({row['timestamp'] for row in rows})}")
        print(f"multi-timestamp tracks={sum(length > 1 for length in track_lengths.values())}")

        start = time.perf_counter()
        timestamps, points, new_timestamps = postprocess.load_trajectory_data(
            str(staging), tracking_dir.name
        )
        interval = [0, len(new_timestamps)]
        linear = postprocess.interpolate_trajectory(timestamps, points, new_timestamps, interval)
        spline = postprocess.interpolate_trajectory_spline(
            timestamps, points, new_timestamps, interval, 0.5
        )
        postprocess.plot_trajectories(
            timestamps,
            points,
            new_timestamps,
            spline,
            interval,
            sequence_dir.name,
            str(postprocess_dir / "original_postprocess.png"),
        )
        del linear
        plot_tracks(rows, output_dir, sequence_dir.name)
        has_gt = gt_diagnostic(rows, staging, output_dir)
        timings["postprocess_time_sec"] = time.perf_counter() - start
        print("[4] POSTPROCESS")
        print(f"success=True output={postprocess_dir / 'original_postprocess.png'}")
        print(f"sidecar_3d={output_dir / 'track_xyz_3d.png'}")
        print(f"gt_diagnostic={has_gt}")
        success = {
            "preprocess_outputs": processed,
            "strict_preprocess_success": strict_preprocess_success,
            "tracks_raw_nonempty": bool(rows),
            "unique_tracks": len(track_lengths),
            "unique_timestamps": len({row["timestamp"] for row in rows}),
            "multi_timestamp_tracks": sum(length > 1 for length in track_lengths.values()),
            "track_xyz_3d_exists": (output_dir / "track_xyz_3d.png").is_file(),
        }
        success["strict_reproduction_success"] = bool(
            success["strict_preprocess_success"]
            and success["tracks_raw_nonempty"]
            and success["multi_timestamp_tracks"] > 0
            and success["track_xyz_3d_exists"]
        )
        (output_dir / "reproduction_result.json").write_text(
            json.dumps(success, indent=2), encoding="utf-8"
        )
        print(f"strict_reproduction_success={success['strict_reproduction_success']}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        timings["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output_dir / "runtime_metrics.json").write_text(
            json.dumps(timings, indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-preprocessed",
        action="store_true",
        help="Reuse existing staged preprocess outputs and run tracking/postprocess only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
