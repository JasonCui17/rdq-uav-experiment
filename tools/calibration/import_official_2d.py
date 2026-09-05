#!/usr/bin/env python3
"""Map MMAUD's official 2D labels back to timestamp-named V1 images.

The released detector images are exact pixel crops of the left 1280x960 half
of the timestamp-named 2560x960 stereo images.  Filename indices (b1_..., ...)
are *not* timestamps, so this tool deliberately matches decoded pixel hashes
instead of assuming an index/time formula.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent
PREFIX_TO_CLASS = {
    "b1": "Mavic2",
    "b2": "Mavic3",
    "b3": "Pham4",
    "b4": "Avata",
    "b5": "M300",
}


def pixel_hash(path_and_half: tuple[str, str]) -> tuple[str, str, str]:
    """Return path, decoded-pixel hash and error text (empty on success)."""
    path_text, half = path_and_half
    image = cv2.imread(path_text, cv2.IMREAD_COLOR)
    if image is None:
        return path_text, "", "cv2.imread returned None"
    if half == "left":
        if image.shape[:2] != (960, 2560):
            return path_text, "", f"expected 2560x960 stereo image, got {image.shape}"
        image = image[:, :1280]
    elif half == "full":
        if image.shape[:2] != (960, 1280):
            return path_text, "", f"expected 1280x960 official image, got {image.shape}"
    else:
        return path_text, "", f"unsupported half mode: {half}"
    digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
    return path_text, digest, ""


def parallel_hash(paths: list[Path], half: str, workers: int) -> list[tuple[str, str, str]]:
    jobs = [(str(path), half) for path in paths]
    if workers <= 1:
        return [pixel_hash(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(pixel_hash, jobs, chunksize=8))


def load_manifest_index(manifest_paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in manifest_paths:
        if not path.exists():
            continue
        split = path.stem
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["_experiment_split"] = split
                by_class[row["class_name"]].append(row)
    for rows in by_class.values():
        rows.sort(key=lambda row: float(row["image_time"]))
    return by_class


def nearest_manifest_row(
    manifest_index: dict[str, list[dict[str, str]]], class_name: str, image_time: float
) -> tuple[dict[str, str] | None, float]:
    rows = manifest_index.get(class_name, [])
    if not rows:
        return None, float("inf")
    times = [float(row["image_time"]) for row in rows]
    insertion = bisect.bisect_left(times, image_time)
    candidate_indices = [index for index in (insertion - 1, insertion) if 0 <= index < len(rows)]
    best_index = min(candidate_indices, key=lambda index: abs(times[index] - image_time))
    return rows[best_index], image_time - times[best_index]


def official_split(path: Path) -> str:
    parts = path.parts
    if "train2017" in parts:
        return "official_train"
    if "val2017" in parts:
        return "official_val"
    if "test" in parts:
        return "official_test"
    return "official_unknown"


def load_or_build_raw_cache(
    v1_root: Path,
    cache_path: Path,
    workers: int,
    rebuild: bool,
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    if cache_path.exists() and not rebuild:
        with cache_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        source_paths: list[Path] = []
        for class_name in PREFIX_TO_CLASS.values():
            source_paths.extend(sorted((v1_root / class_name / "image").glob("*.png")))
        print(f"hashing {len(source_paths)} timestamp stereo images with {workers} workers")
        results = parallel_hash(source_paths, "left", workers)
        rows = []
        for path_text, digest, error in results:
            path = Path(path_text)
            rows.append(
                {
                    "class_name": path.parent.parent.name,
                    "image_path": str(path.resolve()),
                    "image_time": path.stem,
                    "pixel_hash": digest,
                    "error": error,
                }
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"raw hash cache: {cache_path.resolve()}")

    index: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["pixel_hash"]:
            index[f'{row["class_name"]}:{row["pixel_hash"]}'].append(row["image_path"])
    return index, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map official MMAUD_2D boxes to timestamp-named V1 stereo images"
    )
    parser.add_argument(
        "--official-root", type=Path, default=DEFAULT_DATA_ROOT / "official_2d_detection"
    )
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_DATA_ROOT / "v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_2d_timestamp_mapping.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "calibration/cache/raw_left_pixel_hashes.csv",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force intentionally")
    if not args.official_root.is_dir():
        raise FileNotFoundError(args.official_root)
    if not args.v1_root.is_dir():
        raise FileNotFoundError(args.v1_root)

    raw_index, raw_rows = load_or_build_raw_cache(
        args.v1_root, args.cache, args.workers, args.rebuild_cache
    )
    decode_errors = [row for row in raw_rows if row["error"]]
    print(f"raw decode errors: {len(decode_errors)}")

    official_paths = sorted(args.official_root.rglob("b*.png"))
    print(f"hashing {len(official_paths)} official 2D images")
    official_results = parallel_hash(official_paths, "full", args.workers)
    manifest_index = load_manifest_index(
        [PROJECT_ROOT / f"manifests/{split}.csv" for split in ("train", "val", "test")]
    )

    output_rows: list[dict[str, str | int | float]] = []
    error_count = 0
    unmatched_count = 0
    ambiguous_count = 0
    for path_text, digest, error in official_results:
        image_path = Path(path_text)
        prefix, index_text = image_path.stem.split("_", maxsplit=1)
        class_name = PREFIX_TO_CLASS[prefix]
        candidates = raw_index.get(f"{class_name}:{digest}", []) if digest else []
        if error:
            status = "official_decode_error"
            error_count += 1
        elif not candidates:
            status = "unmatched"
            unmatched_count += 1
        elif len(candidates) > 1:
            status = "ambiguous_duplicate_pixels"
            ambiguous_count += 1
        else:
            status = "exact"

        matched_path = Path(candidates[0]) if len(candidates) == 1 else None
        nearest_row, dt_to_manifest = (
            nearest_manifest_row(manifest_index, class_name, float(matched_path.stem))
            if matched_path
            else (None, float("inf"))
        )
        label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if "/test/images/" in str(image_path):
            label_path = Path(str(image_path).replace("/test/images/", "/test/labels/")).with_suffix(
                ".txt"
            )
        label = label_path.read_text(encoding="utf-8").strip().split()
        if len(label) != 5:
            raise ValueError(f"Expected one YOLO box in {label_path}, got {label}")
        _, x, y, width, height = label
        x, y, width, height = map(float, (x, y, width, height))
        output_rows.append(
            {
                "class_name": class_name,
                "official_prefix": prefix,
                "official_index": int(index_text),
                "official_split": official_split(image_path),
                "official_image_path": str(image_path.resolve()),
                "official_label_path": str(label_path.resolve()),
                "camera": "left",
                "bbox_x_center_norm": x,
                "bbox_y_center_norm": y,
                "bbox_width_norm": width,
                "bbox_height_norm": height,
                "u": x * 1280.0,
                "v": y * 960.0,
                "bbox_width_px": width * 1280.0,
                "bbox_height_px": height * 960.0,
                "image_path": str(matched_path.resolve()) if matched_path else "",
                "image_time": matched_path.stem if matched_path else "",
                "nearest_sample_id": nearest_row["sample_id"] if nearest_row else "",
                "nearest_manifest_image_time": nearest_row["image_time"] if nearest_row else "",
                "dt_to_manifest_s": dt_to_manifest if nearest_row else "",
                "experiment_split": nearest_row["_experiment_split"] if nearest_row else "",
                "match_status": status,
                "match_candidates": len(candidates),
                "decode_error": error,
            }
        )

    output_rows.sort(key=lambda row: (str(row["class_name"]), int(row["official_index"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    exact = sum(row["match_status"] == "exact" for row in output_rows)
    print(f"mapping: {args.output.resolve()}")
    print(
        f"official={len(output_rows)} exact={exact} unmatched={unmatched_count} "
        f"ambiguous={ambiguous_count} official_decode_errors={error_count}"
    )
    if decode_errors:
        print("raw images that could not be decoded:")
        for row in decode_errors[:20]:
            print(f'  {row["image_path"]}: {row["error"]}')
    if exact != len(output_rows):
        raise SystemExit("Mapping is incomplete; inspect non-exact rows before calibration")


if __name__ == "__main__":
    main()
