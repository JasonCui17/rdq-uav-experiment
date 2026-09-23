#!/usr/bin/env python3
"""Exploratory transfer check for the recovered MMAUD GT -> left-camera mapping.

This tool never fits or adjusts calibration. It reads fixed R/t/time parameters,
interpolates native GT, projects with the existing omni+radtan implementation,
and writes images plus an empty human-review sheet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration.omni import OmniRadtanCamera, transform_points  # noqa: E402
from rdq_uav.calibration.trajectory import PositionTrajectory  # noqa: E402

EXPECTED_OFFSET = -0.12415366829748864
EXPECTED_R = np.asarray([
    [-0.9611438276150616, 0.2760260815671112, -0.0034849579876607343],
    [-0.27603112745837816, -0.960868300366179, 0.023214780323483345],
    [0.003059299188681272, 0.02327469969983467, 0.9997244265508156],
], dtype=np.float64)
EXPECTED_T = np.asarray(
    [0.16965789709994833, 0.01804162364147821, 0.036247986801981685],
    dtype=np.float64,
)
DEFAULT_SEQUENCES = ("seq0001", "seq0024", "seq0049", "seq0069", "seq0089", "seq0102")
CSV_FIELDS = (
    "sample_id", "sequence_id", "image_path", "left_image_path", "image_timestamp",
    "query_gt_timestamp", "gt_x", "gt_y", "gt_z", "range_m", "camera_x", "camera_y",
    "camera_z", "u", "v", "projection_geometry_valid", "in_left_image",
    "projection_valid", "manual_label", "manual_note",
)


def stats_ms(values: np.ndarray) -> tuple[float | str, float | str, float | str]:
    values = np.asarray(values, dtype=np.float64) * 1000.0
    if values.size == 0:
        return "", "", ""
    return float(np.median(values)), float(np.percentile(values, 95)), float(values.max())


def load_fixed_config(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float, OmniRadtanCamera]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("evidence_level") != "REPRODUCIBLE_BUT_NOT_EVALUATION_SAFE":
        raise ValueError("Recovered config must retain its exploratory evidence level")
    left = config["cameras"]["left"]
    rotation = np.asarray(left["rotation_left_camera_from_gt"], dtype=np.float64)
    translation = np.asarray(left["translation_left_camera_from_gt_m"], dtype=np.float64)
    offset = float(config["time"]["time_offset_s"])
    # These exact guards make accidental parameter adjustment a hard failure.
    if not np.array_equal(rotation, EXPECTED_R):
        raise ValueError("Recovered rotation differs from the audited historical rotation")
    if not np.array_equal(translation, EXPECTED_T):
        raise ValueError("Recovered translation differs from the audited historical translation")
    if offset != EXPECTED_OFFSET:
        raise ValueError("Recovered time offset differs from the audited historical offset")
    camera = OmniRadtanCamera.from_config({
        "intrinsics": left["intrinsics"],
        "distortion_coeffs": left["distortion_coeffs"],
        "resolution": left["resolution_wh"],
    })
    return config, rotation, translation, offset, camera


def timed_images(directory: Path) -> list[tuple[float, Path]]:
    rows = []
    for path in directory.glob("*.png"):
        try:
            rows.append((float(path.stem), path))
        except ValueError:
            continue
    return sorted(rows)


def interpolation_widths(query: np.ndarray, gt_times: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = []
    for value in query[valid]:
        right = int(np.searchsorted(gt_times, value, side="left"))
        if right == 0:
            width = gt_times[1] - gt_times[0]
        elif right == len(gt_times):
            width = gt_times[-1] - gt_times[-2]
        elif gt_times[right] == value:
            left_gap = value - gt_times[right - 1]
            right_gap = gt_times[right + 1] - value if right + 1 < len(gt_times) else left_gap
            width = max(left_gap, right_gap)
        else:
            width = gt_times[right] - gt_times[right - 1]
        result.append(width)
    return np.asarray(result, dtype=np.float64)


def audit_sequence(
    root: Path, sequence: str, rotation: np.ndarray, translation: np.ndarray,
    offset: float, camera: OmniRadtanCamera,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequence_dir = root / sequence
    images = timed_images(sequence_dir / "Image")
    if not images:
        raise FileNotFoundError(f"No PNG images in {sequence_dir / 'Image'}")
    trajectory = PositionTrajectory.from_directory(sequence_dir / "ground_truth")
    image_times = np.asarray([item[0] for item in images], dtype=np.float64)
    query_times = image_times + offset
    gt_xyz, query_valid = trajectory.evaluate(query_times)
    camera_xyz = transform_points(gt_xyz, rotation, translation)
    pixels, geometric_valid = camera.project(camera_xyz, require_in_image=False)
    in_image = geometric_valid & camera.in_image(pixels)
    neighbor = np.diff(trajectory.timestamps)
    interp = interpolation_widths(query_times, trajectory.timestamps, query_valid)
    nmed, np95, nmax = stats_ms(neighbor)
    imed, ip95, imax = stats_ms(interp)
    summary = {
        "sequence_id": sequence,
        "num_images": len(images),
        "num_gt_timestamps": len(trajectory.timestamps),
        "num_query_valid": int(query_valid.sum()),
        "valid_ratio": float(query_valid.mean()),
        "num_out_of_range": int((~query_valid).sum()),
        "median_neighbor_gap_ms": nmed,
        "p95_neighbor_gap_ms": np95,
        "max_neighbor_gap_ms": nmax,
        "median_interp_gap_ms": imed,
        "p95_interp_gap_ms": ip95,
        "max_interp_gap_ms": imax,
        "num_geometry_valid": int((query_valid & geometric_valid).sum()),
        "num_in_left_image": int((query_valid & in_image).sum()),
        "in_left_image_ratio": float((query_valid & in_image).mean()),
    }
    candidates = []
    ranges = np.linalg.norm(gt_xyz, axis=1)
    for index, ((image_time, image_path), query_time) in enumerate(zip(images, query_times)):
        candidates.append({
            "index": index, "sequence_id": sequence, "image_timestamp": image_time,
            "query_gt_timestamp": float(query_time), "image_path": image_path,
            "gt_xyz": gt_xyz[index], "range_m": float(ranges[index]),
            "camera_xyz": camera_xyz[index], "pixel": pixels[index],
            "query_valid": bool(query_valid[index]),
            "geometric_valid": bool(geometric_valid[index]), "in_image": bool(in_image[index]),
        })
    return summary, candidates


def select_five(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministic time-stratified selection with range/image-region diversity."""
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates, got {len(candidates)}")
    ranges = np.asarray([row["range_m"] for row in candidates])
    order = np.argsort(ranges, kind="stable")
    range_rank = np.empty(len(candidates), dtype=float)
    range_rank[order] = np.linspace(0.0, 1.0, len(candidates))
    # Every sample comes from a distinct temporal stratum. Targets encourage
    # near-to-far and spatial spread, but never alter calibration parameters.
    uv_targets = ((0.15, 0.20), (0.85, 0.20), (0.50, 0.50), (0.15, 0.80), (0.85, 0.80))
    target_ranges = np.linspace(0.0, 1.0, count)
    edges = np.linspace(0, len(candidates), count + 1, dtype=int)
    selected = []
    for slot in range(count):
        indices = np.arange(edges[slot], edges[slot + 1])
        tu, tv = uv_targets[slot % len(uv_targets)]
        scores = []
        for index in indices:
            row = candidates[int(index)]
            if row["geometric_valid"] and np.isfinite(row["pixel"]).all():
                u, v = row["pixel"]
                spatial = (u / 1280.0 - tu) ** 2 + (v / 960.0 - tv) ** 2
            else:
                spatial = 2.0
            score = abs(range_rank[index] - target_ranges[slot]) + 0.35 * spatial
            scores.append((score, int(index)))
        selected.append(candidates[min(scores)[1]])
    return selected


def render(row: dict[str, Any], destination: Path) -> None:
    with Image.open(row["image_path"]) as packed:
        if packed.size != (2560, 960):
            raise ValueError(f"Expected packed 2560x960 image, got {packed.size}: {row['image_path']}")
        # Historical exact-pixel importer defines left as columns [0:1280].
        left = packed.convert("RGB").crop((0, 0, 1280, 960))
    draw = ImageDraw.Draw(left)
    valid = row["query_valid"] and row["geometric_valid"] and row["in_image"]
    color = (0, 255, 0) if valid else (255, 40, 40)
    if row["geometric_valid"] and np.isfinite(row["pixel"]).all():
        u, v = (float(x) for x in row["pixel"])
        radius = 11
        draw.ellipse((u-radius, v-radius, u+radius, v+radius), outline=color, width=4)
        draw.line((u-18, v, u+18, v), fill=color, width=4)
        draw.line((u, v-18, u, v+18), fill=color, width=4)
    gt = row["gt_xyz"]
    uv = row["pixel"]
    lines = [
        f"{row['sequence_id']}  image={row['image_timestamp']:.6f}",
        f"query_gt={row['query_gt_timestamp']:.6f}",
        f"GT=({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f})  range={row['range_m']:.3f}",
        f"uv=({uv[0]:.2f}, {uv[1]:.2f})" if np.isfinite(uv).all() else "uv=(nan, nan)",
        "VALID_PROJECTION" if valid else "INVALID_PROJECTION",
    ]
    box_height = 18 * len(lines) + 12
    draw.rectangle((6, 6, 720, 6 + box_height), fill=(0, 0, 0), outline=color, width=2)
    for line_index, line in enumerate(lines):
        draw.text((14, 12 + line_index * 18), line, fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    left.save(destination, format="PNG")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = ("sequence_id", "num_images", "num_gt_timestamps", "num_query_valid", "valid_ratio",
              "num_out_of_range", "median_neighbor_gap_ms", "p95_neighbor_gap_ms",
              "median_interp_gap_ms", "p95_interp_gap_ms", "in_left_image_ratio")
    header = "| " + " | ".join(fields) + " |\n| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(
            f"{row[field]:.6f}" if isinstance(row[field], float) else str(row[field]) for field in fields
        ) + " |")
    return header + "\n" + "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/official/train"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "outputs/own_multimodal_research/g0_gt_camera_alignment_audit/recovered_alignment_config.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/own_multimodal_research/g1_gt_camera_transfer_sanity")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--samples-per-sequence", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Use first sequence and exactly two images")
    args = parser.parse_args()
    if args.data_root.resolve() != Path("/home/jasoncui/datasets/MMAUD/official/train"):
        raise ValueError("G1-1 data root is fixed; refusing another root")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config, rotation, translation, offset, camera = load_fixed_config(args.config.resolve())
    sequences = args.sequences[:1] if args.smoke else args.sequences
    count = 2 if args.smoke else args.samples_per_sequence
    summaries, selected = [], []
    for sequence in sequences:
        summary, candidates = audit_sequence(
            args.data_root.resolve(), sequence, rotation, translation, offset, camera
        )
        summaries.append(summary)
        selected.extend(select_five(candidates, count))
    expected = 2 if args.smoke else len(sequences) * args.samples_per_sequence
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected samples, got {len(selected)}")
    projection_dir = output / "projection_samples"
    sample_rows = []
    for ordinal, row in enumerate(selected, start=1):
        sample_id = f"g1_{ordinal:03d}_{row['sequence_id']}_{row['image_timestamp']:.6f}"
        destination = projection_dir / row["sequence_id"] / f"{sample_id}.png"
        render(row, destination)
        gt, pc, uv = row["gt_xyz"], row["camera_xyz"], row["pixel"]
        valid = row["query_valid"] and row["geometric_valid"] and row["in_image"]
        sample_rows.append({
            "sample_id": sample_id, "sequence_id": row["sequence_id"],
            "image_path": str(row["image_path"].resolve()), "left_image_path": str(destination.resolve()),
            "image_timestamp": f"{row['image_timestamp']:.6f}",
            "query_gt_timestamp": f"{row['query_gt_timestamp']:.6f}",
            "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2], "range_m": row["range_m"],
            "camera_x": pc[0], "camera_y": pc[1], "camera_z": pc[2],
            "u": uv[0], "v": uv[1],
            "projection_geometry_valid": row["geometric_valid"], "in_left_image": row["in_image"],
            "projection_valid": valid, "manual_label": "", "manual_note": "",
        })
    pairing_fields = list(summaries[0])
    write_csv(output / "time_pairing_summary.csv", summaries, pairing_fields)
    write_csv(output / "projection_samples.csv", sample_rows, list(CSV_FIELDS))
    table = markdown_table(summaries)
    low_time_risk = sum(row["valid_ratio"] < 0.90 for row in summaries) > len(summaries) / 2
    timing_note = (
        "Most sampled sequences have low valid ratio: fixed-offset transfer has clear timing risk."
        if low_time_risk else
        "Fixed-offset query coverage is high in the selected sequences. This checks temporal range only; it does not validate the offset value."
    )
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    time_report = f"""# Time Pairing Summary

This is a coverage sanity check with the immutable historical offset `{offset:.15f} s`.
`neighbor_gap` summarizes every consecutive native GT timestamp gap. `interp_gap` is the
bracketing GT interval used for each in-range query (for an exact GT timestamp, the larger
adjacent interval). Neither statistic estimates a new offset.

{table}

{timing_note}
"""
    (output / "TIME_PAIRING_SUMMARY.md").write_text(time_report, encoding="utf-8")
    R_text = np.array2string(rotation, precision=15, separator=", ")
    t_text = np.array2string(translation, precision=15, separator=", ")
    report = f"""# GT Camera Transfer Sanity

## 1. Scope

This is an exploratory transfer sanity check of the recovered historical GT-to-left-camera
alignment. It is not formal calibration or evaluation, uses no 2D bbox supervision, and performs
no fitting, parameter search, learning, LiDAR projection, Radar projection, or split modification.

## 2. Immutable parameters and implementation

```text
p_left_camera = R @ p_GT + t
R = {R_text}
t = {t_text}
t_GT = t_image + ({offset:.15f}) s
```

The executable checks exact equality against the audited historical R, t and time offset before
processing. Configuration: `{args.config.resolve()}` (SHA-256 `{config_hash}`). Camera parameters
are the recovered official left-camera Kalibr unified `omni` intrinsics with `radtan` distortion.
Code path: `PositionTrajectory.evaluate` -> `transform_points` -> `OmniRadtanCamera.project`.

The packed image was verified at runtime to be 2560x960. Left is cropped as columns `[0:1280]`,
matching the historical `import_official_2d.py` exact-pixel convention. This ordering is inherited
from that old code/evidence; no current 2D bbox was read.

## 3. Time pairing

{table}

{timing_note}

These ratios only establish that shifted query times lie inside the GT interpolation domain.
They do not prove that `-0.1241536683 s` remains the correct physical clock offset.

## 4. Projection samples

Generated {len(sample_rows)} annotated left-camera PNGs under `{projection_dir.resolve()}` from
{len(sequences)} sequences, {count} per sequence. Each sequence is divided into distinct temporal
strata; within strata, deterministic targets encourage near/mid/far range and image-region spread.
Range is only `norm(GT XYZ)` for sampling and retains the old **CODE_ASSUMED metre** convention.
All candidate image timestamps participate in the automatic projection-validity summary, so the
visual selection cannot hide sequence-level invalid rates.

Open the PNGs and fill `manual_label` in `{(output/'projection_samples.csv').resolve()}` with exactly
one of `GOOD`, `NEAR`, `BAD`, `UNCERTAIN`; use `manual_note` for visible target location, systematic
offset direction, ambiguity, occlusion, or absence. `GOOD/NEAR/BAD` require a visually identifiable
UAV; otherwise use `UNCERTAIN`. These human labels are exploratory and not bbox-based metrics.

## 5. Current automatic status

**READY_FOR_MANUAL_REVIEW**. Automatic checks can confirm interpolation coverage, finite camera
coordinates and whether predicted pixels lie inside the left image. They cannot determine whether
the projected point is on the UAV. No transfer verdict is issued before manual review.

## 6. Final conclusion template

- `TRANSFER_SUPPORTED`: most identifiable targets are GOOD/NEAR without material systematic or sequence-specific displacement.
- `TRANSFER_PARTIALLY_SUPPORTED`: useful alignment exists but fixed or sequence-dependent bias is visible; describe affected sequences and direction.
- `TRANSFER_NOT_SUPPORTED`: projections generally miss identifiable targets or vary incompatibly across sequences.

Required explanation category: `FIXED_SYSTEMATIC_BIAS`, `SEQUENCE_DEPENDENT_BIAS`, or
`VISUALLY_UNDETERMINED`. Updating this section must summarize the completed CSV; it must not refit
R/t or time offset and must not describe the result as formal calibration.

## 7. Known unknowns

- Correct transfer of historical rig geometry to current recordings remains UNKNOWN until review.
- Correct transfer of the historical clock offset remains UNKNOWN; in-range interpolation is not clock validation.
- Current 2D target visibility/location is UNKNOWN without human review.
- This check says nothing about Mid360, Avia, Radar or right-camera extrinsics.
"""
    (output / "GT_CAMERA_TRANSFER_SANITY.md").write_text(report, encoding="utf-8")
    metadata = {
        "mode": "smoke" if args.smoke else "formal_g1_1_output",
        "automatic_status": "READY_FOR_MANUAL_REVIEW",
        "final_transfer_verdict": None,
        "data_root": str(args.data_root.resolve()), "sequences": sequences,
        "sample_count": len(sample_rows), "fixed_parameter_config": str(args.config.resolve()),
        "fixed_parameter_config_sha256": config_hash, "fitting_performed": False,
        "parameter_search_performed": False, "bbox_read": False,
        "point_cloud_read": False, "allowed_manual_labels": ["GOOD", "NEAR", "BAD", "UNCERTAIN"],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(sample_rows),
                      "sequences": sequences, "automatic_status": "READY_FOR_MANUAL_REVIEW"}, indent=2))


if __name__ == "__main__":
    main()
