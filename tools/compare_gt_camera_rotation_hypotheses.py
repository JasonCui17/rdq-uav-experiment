#!/usr/bin/env python3
"""Compare frozen OLD_R and identity on the exact G1-1 samples.

Geometry-only and exploratory: no labels, fitting, optimization, parameter
search, point clouds, or model inference are read or performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.calibration.omni import OmniRadtanCamera, transform_points  # noqa: E402

EXPECTED_R_OLD = np.asarray([
    [-0.9611438276150616, 0.2760260815671112, -0.0034849579876607343],
    [-0.27603112745837816, -0.960868300366179, 0.023214780323483345],
    [0.003059299188681272, 0.02327469969983467, 0.9997244265508156],
], dtype=np.float64)
EXPECTED_T = np.asarray(
    [0.16965789709994833, 0.01804162364147821, 0.036247986801981685],
    dtype=np.float64,
)
EXPECTED_DT = -0.12415366829748864
EXPECTED_SAMPLE_COUNT = 30
MANUAL_FIELDS = (
    "target_visible_manual", "old_label", "identity_label",
    "preferred_rotation", "manual_note",
)
COMPARISON_FIELDS = (
    "sample_id", "sequence_id", "image_path", "visualization_path", "image_timestamp",
    "query_gt_timestamp", "gt_x", "gt_y", "gt_z", "range",
    "old_camera_x", "old_camera_y", "old_camera_z", "old_u", "old_v",
    "old_valid", "old_in_frame", "identity_camera_x", "identity_camera_y",
    "identity_camera_z", "identity_u", "identity_v", "identity_valid",
    "identity_in_frame", "excluded_from_rotation_judgment", "visibility_status",
)
REVIEW_FIELDS = (
    "sample_id", "sequence_id", "image_timestamp", "gt_x", "gt_y", "gt_z", "range",
    "old_u", "old_v", "old_valid", "old_in_frame", "identity_u", "identity_v",
    "identity_valid", "identity_in_frame", "excluded_from_rotation_judgment",
    "visibility_status", *MANUAL_FIELDS,
)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def fixed_parameters(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    left = config["cameras"]["left"]
    old = np.asarray(left["rotation_left_camera_from_gt"], dtype=np.float64)
    translation = np.asarray(left["translation_left_camera_from_gt_m"], dtype=np.float64)
    dt = float(config["time"]["time_offset_s"])
    identity = np.eye(3, dtype=np.float64)
    if not np.array_equal(old, EXPECTED_R_OLD):
        raise ValueError("OLD_R does not exactly equal the audited historical rotation")
    if not np.array_equal(identity, np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        raise AssertionError("Identity hypothesis is not exactly I")
    if not np.array_equal(translation, EXPECTED_T):
        raise ValueError("Translation differs from the frozen G1-1 translation")
    if dt != EXPECTED_DT:
        raise ValueError("Time offset differs from the frozen G1-1 time offset")
    camera = OmniRadtanCamera.from_config({
        "intrinsics": left["intrinsics"],
        "distortion_coeffs": left["distortion_coeffs"],
        "resolution": left["resolution_wh"],
    })
    return config, old, identity, translation, dt, camera


def project_same_gt(gt: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                    camera: OmniRadtanCamera) -> dict:
    camera_xyz = transform_points(gt, rotation, translation)
    uv, valid = camera.project(camera_xyz, require_in_image=False)
    in_frame = bool(valid) and bool(camera.in_image(uv))
    return {"camera_xyz": camera_xyz, "uv": uv, "valid": bool(valid), "in_frame": in_frame}


def draw_cross(draw: ImageDraw.ImageDraw, uv: np.ndarray, color: tuple[int, int, int]) -> None:
    u, v = map(float, uv)
    draw.line((u - 18, v, u + 18, v), fill=color, width=5)
    draw.line((u, v - 18, u, v + 18), fill=color, width=5)
    draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=color, width=3)


def draw_diamond(draw: ImageDraw.ImageDraw, uv: np.ndarray, color: tuple[int, int, int]) -> None:
    u, v = map(float, uv)
    draw.line((u, v - 16, u + 16, v, u, v + 16, u - 16, v, u, v - 16), fill=color, width=5)
    draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=color)


def render(raw_path: Path, destination: Path, row: dict, old: dict, identity: dict) -> None:
    with Image.open(raw_path) as packed:
        if packed.size != (2560, 960):
            raise ValueError(f"Expected 2560x960 packed image, got {packed.size}: {raw_path}")
        image = packed.convert("RGB").crop((0, 0, 1280, 960))
    draw = ImageDraw.Draw(image)
    green, magenta = (20, 255, 20), (255, 30, 230)
    if old["valid"] and old["in_frame"]:
        draw_cross(draw, old["uv"], green)
    if identity["valid"] and identity["in_frame"]:
        draw_diamond(draw, identity["uv"], magenta)
    gt = np.asarray([float(row[f"gt_{axis}"]) for axis in "xyz"])
    lines = [
        f"{row['sequence_id']} image={float(row['image_timestamp']):.6f}",
        f"GT=({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}) range={float(row['range_m']):.3f}",
        f"OLD_R green-cross uv=({old['uv'][0]:.2f},{old['uv'][1]:.2f}) valid={old['valid']} in={old['in_frame']}",
        f"IDENTITY_R magenta-diamond uv=({identity['uv'][0]:.2f},{identity['uv'][1]:.2f}) valid={identity['valid']} in={identity['in_frame']}",
    ]
    if row["sequence_id"] == "seq0001":
        lines.append("TARGET_NOT_RELIABLY_VISIBLE_BY_MANUAL_REVIEW; exclude rotation judgment")
    height = len(lines) * 18 + 12
    draw.rectangle((6, 6, 925, 6 + height), fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    for index, line in enumerate(lines):
        color = green if index == 2 else magenta if index == 3 else (255, 255, 255)
        draw.text((14, 12 + 18 * index), line, fill=color)
    # A color-independent marker legend is included in the text and marker shape.
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


def distribution(values: list[float]) -> dict[str, float | None]:
    array = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not len(array):
        return {"min": None, "median": None, "p95": None, "max": None}
    return {"min": float(array.min()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=PROJECT_ROOT / "outputs/own_multimodal_research/g1_gt_camera_transfer_sanity/projection_samples.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "outputs/own_multimodal_research/g0_gt_camera_alignment_audit/recovered_alignment_config.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/own_multimodal_research/g1_rotation_hypothesis_comparison")
    parser.add_argument("--smoke", action="store_true", help="Process the first two source rows")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config, old_r, identity_r, translation, dt, camera = fixed_parameters(args.config.resolve())
    with args.samples.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"G1-1 must contain exactly {EXPECTED_SAMPLE_COUNT} samples, got {len(source_rows)}")
    rows_to_process = source_rows[:2] if args.smoke else source_rows
    comparison, review = [], []
    for source in rows_to_process:
        gt = np.asarray([float(source[f"gt_{axis}"]) for axis in "xyz"], dtype=np.float64)
        # One shared GT object, translation, dt, and camera are used by both calls.
        old = project_same_gt(gt, old_r, translation, camera)
        identity = project_same_gt(gt, identity_r, translation, camera)
        expected_query = float(source["image_timestamp"]) + dt
        if abs(expected_query - float(source["query_gt_timestamp"])) > 5e-7:
            raise ValueError(f"G1-1 query timestamp mismatch for {source['sample_id']}")
        destination = output / "visualizations" / source["sequence_id"] / f"{source['sample_id']}_old_vs_identity.png"
        render(Path(source["image_path"]), destination, source, old, identity)
        uncertain = source["sequence_id"] == "seq0001"
        item = {
            "sample_id": source["sample_id"], "sequence_id": source["sequence_id"],
            "image_path": source["image_path"], "visualization_path": str(destination),
            "image_timestamp": source["image_timestamp"], "query_gt_timestamp": source["query_gt_timestamp"],
            "gt_x": source["gt_x"], "gt_y": source["gt_y"], "gt_z": source["gt_z"],
            "range": source["range_m"],
            "old_camera_x": old["camera_xyz"][0], "old_camera_y": old["camera_xyz"][1],
            "old_camera_z": old["camera_xyz"][2], "old_u": old["uv"][0], "old_v": old["uv"][1],
            "old_valid": old["valid"], "old_in_frame": old["in_frame"],
            "identity_camera_x": identity["camera_xyz"][0], "identity_camera_y": identity["camera_xyz"][1],
            "identity_camera_z": identity["camera_xyz"][2], "identity_u": identity["uv"][0],
            "identity_v": identity["uv"][1], "identity_valid": identity["valid"],
            "identity_in_frame": identity["in_frame"],
            "excluded_from_rotation_judgment": uncertain,
            "visibility_status": "TARGET_NOT_RELIABLY_VISIBLE_BY_MANUAL_REVIEW" if uncertain else "PENDING_MANUAL_REVIEW",
        }
        comparison.append(item)
        review.append({**item, **{field: "" for field in MANUAL_FIELDS}})
    write_csv(output / "rotation_hypothesis_comparison.csv", comparison, COMPARISON_FIELDS)
    write_csv(output / "rotation_hypothesis_review.csv", review, REVIEW_FIELDS)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in comparison:
        grouped[row["sequence_id"]].append(row)
    per_sequence = []
    for sequence, rows in grouped.items():
        record = {"sequence_id": sequence, "sample_count": len(rows),
                  "excluded_from_rotation_judgment": sequence == "seq0001"}
        for prefix in ("old", "identity"):
            record[f"{prefix}_valid_count"] = sum(bool(row[f"{prefix}_valid"]) for row in rows)
            record[f"{prefix}_in_frame_count"] = sum(bool(row[f"{prefix}_in_frame"]) for row in rows)
            record[f"{prefix}_invalid_count"] = sum(not bool(row[f"{prefix}_valid"]) for row in rows)
            record[f"{prefix}_valid_ratio"] = record[f"{prefix}_valid_count"] / len(rows)
            record[f"{prefix}_in_frame_ratio"] = record[f"{prefix}_in_frame_count"] / len(rows)
            for axis in ("u", "v"):
                record.update({f"{prefix}_{axis}_{key}": value for key, value in distribution(
                    [float(row[f"{prefix}_{axis}"]) for row in rows if bool(row[f"{prefix}_valid"])]
                ).items()})
        per_sequence.append(record)
    sequence_fields = tuple(per_sequence[0])
    write_csv(output / "per_sequence_geometry_summary.csv", per_sequence, sequence_fields)
    overall = {}
    for prefix in ("old", "identity"):
        overall[prefix] = {
            "valid_count": sum(bool(row[f"{prefix}_valid"]) for row in comparison),
            "in_frame_count": sum(bool(row[f"{prefix}_in_frame"]) for row in comparison),
            "invalid_count": sum(not bool(row[f"{prefix}_valid"]) for row in comparison),
            "u": distribution([float(row[f"{prefix}_u"]) for row in comparison if bool(row[f"{prefix}_valid"])]),
            "v": distribution([float(row[f"{prefix}_v"]) for row in comparison if bool(row[f"{prefix}_valid"])]),
        }
    table_header = "| sequence | n | OLD valid/in-frame/invalid | Identity valid/in-frame/invalid | seq0001 excluded |"
    table_rule = "| --- | ---: | --- | --- | --- |"
    table_rows = [
        f"| {row['sequence_id']} | {row['sample_count']} | {row['old_valid_count']}/{row['old_in_frame_count']}/{row['old_invalid_count']} | "
        f"{row['identity_valid_count']}/{row['identity_in_frame_count']}/{row['identity_invalid_count']} | {row['excluded_from_rotation_judgment']} |"
        for row in per_sequence
    ]
    table = "\n".join([table_header, table_rule, *table_rows])
    report = f"""# G1-2 Rotation Hypothesis Comparison

## 1. Goal

Compare the frozen historical `OLD_R` against exactly `R=I` on the same 30 G1-1 samples.
This is an exploratory geometry and manual-review aid, not calibration or formal evaluation.

## 2. Fixed variables

- Source sample IDs, raw images, timestamps, query GT timestamps and GT XYZ: unchanged from `{args.samples.resolve()}`.
- Translation: `{translation.tolist()}` for both branches.
- Time offset: `{dt:.17g} s` for both branches; G1-1 query timestamps are checked against `image_timestamp + dt`.
- Camera: the same single `OmniRadtanCamera` object using recovered Kalibr unified omni+radtan parameters.
- Crop: the same raw packed image columns `[0:1280]`, yielding 1280x960 left images.
- No bbox, YOLO output, point cloud, model output, fitting, optimization, or rotation search is read or performed.

## 3. Rotation hypotheses

```text
OLD_R =
{np.array2string(old_r, precision=15)}

IDENTITY_R =
{np.array2string(identity_r, precision=1)}
```

Only `R` changes between branches. `OLD_R` and `IDENTITY_R` are checked by exact array equality.

## 4. Automatic geometry summary

`valid` means the omni projection is numerically/geometrically defined. `in-frame` additionally
requires `0 <= u < 1280` and `0 <= v < 960`. No distance to an image target is computed.

- OLD_R: valid `{overall['old']['valid_count']}`, in-frame `{overall['old']['in_frame_count']}`, invalid `{overall['old']['invalid_count']}`.
- IDENTITY_R: valid `{overall['identity']['valid_count']}`, in-frame `{overall['identity']['in_frame_count']}`, invalid `{overall['identity']['invalid_count']}`.
- OLD_R u distribution: `{json.dumps(overall['old']['u'])}`; v: `{json.dumps(overall['old']['v'])}`.
- IDENTITY_R u distribution: `{json.dumps(overall['identity']['u'])}`; v: `{json.dumps(overall['identity']['v'])}`.

{table}

These counts cannot select a rotation hypothesis. In-frame frequency and proximity to image center
are not treated as accuracy.

## 5. Manual review instructions

Review the 30 PNGs in `{(output/'visualizations').resolve()}`. OLD_R is a green cross/circle;
IDENTITY_R is a magenta diamond. Text labels identify both markers without relying on color.
Fill `{(output/'rotation_hypothesis_review.csv').resolve()}`:

- `target_visible_manual`: `YES`, `NO`, `UNCERTAIN`.
- `old_label` / `identity_label`: `GOOD`, `NEAR`, `BAD`, `UNCERTAIN`.
- `preferred_rotation`: `OLD_R`, `IDENTITY_R`, `BOTH_BAD`, `BOTH_SIMILAR`, `UNCERTAIN`.

All human-entry fields are initially empty. Every seq0001 row is pre-marked with
`TARGET_NOT_RELIABLY_VISIBLE_BY_MANUAL_REVIEW` and `excluded_from_rotation_judgment=True`; it must
not contribute to hypothesis preference. This does not assign BAD to either projection.

After review, compare labels only where `target_visible_manual == YES`. Report GOOD/NEAR/BAD totals
for each hypothesis and preference counts for OLD_R, IDENTITY_R and BOTH_BAD.

## 6. Current status

**READY_FOR_MANUAL_REVIEW**. No automatic OLD_R-versus-Identity conclusion is made.

## 7. Final verdict placeholder

Pending completed human review. Select exactly one:

- `IDENTITY_STRONGLY_SUPPORTED`
- `IDENTITY_PARTIALLY_SUPPORTED`
- `SEQUENCE_DEPENDENT`
- `INCONCLUSIVE`
"""
    (output / "ROTATION_HYPOTHESIS_COMPARISON.md").write_text(report, encoding="utf-8")
    metadata = {
        "status": "READY_FOR_MANUAL_REVIEW", "final_verdict": None,
        "source_samples": str(args.samples.resolve()),
        "source_samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "source_row_count": len(source_rows), "processed_row_count": len(comparison),
        "same_gt_for_both": True, "same_translation_for_both": True,
        "same_dt_for_both": True, "same_camera_for_both": True,
        "same_crop_for_both": True, "fit_performed": False,
        "rotation_search_performed": False, "bbox_or_yolo_read": False,
        "point_cloud_read": False, "seq0001_auto_bad": False,
        "automatic_geometry": overall,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "processed": len(comparison),
                      "status": metadata["status"], "geometry": overall}, indent=2))


if __name__ == "__main__":
    main()
