#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully decode all images used by manifests")
    parser.add_argument(
        "--root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/v1")
    )
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "manifests")
    args = parser.parse_args()

    relative_paths = set()
    for split in ("train", "val", "test"):
        manifest = args.manifest_dir / f"{split}.csv"
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            relative_paths.update(row["image_path"] for row in csv.DictReader(handle))
    paths = sorted(relative_paths)
    failures = []
    for index, relative_path in enumerate(paths, start=1):
        path = args.root / relative_path
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (2560, 960):
                    raise ValueError(f"unexpected size {image.size}")
                if image.mode != "RGB":
                    raise ValueError(f"unexpected mode {image.mode}")
        except Exception as exc:
            failures.append((relative_path, type(exc).__name__, str(exc)))
            print("BAD", relative_path, type(exc).__name__, exc, flush=True)
        if index % 500 == 0 or index == len(paths):
            print(f"checked={index}/{len(paths)} bad={len(failures)}", flush=True)
    if failures:
        raise SystemExit(f"IMAGE_VERIFY_FAILED: {len(failures)} image(s)")
    print("IMAGE_VERIFY_OK")


if __name__ == "__main__":
    main()
