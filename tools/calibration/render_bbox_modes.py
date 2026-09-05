#!/usr/bin/env python3
"""Render paired Full/Erase/Foreground-only/Crop samples for visual audit."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.data.transforms import DualFisheyeTransform  # noqa: E402


MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor * STD + MEAN).clamp(0, 1)
    array = tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests_oracle_left_fixed256_bbox/val.csv",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/v1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "calibration/official_bbox_modes_montage.jpg",
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--context-scale", type=float, default=2.0)
    args = parser.parse_args()

    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        name = row["class_name"]
        if counts[name] < args.samples_per_class:
            selected.append(row)
            counts[name] += 1
    modes = ("full", "erase", "foreground_only", "crop")
    transforms = {
        mode: DualFisheyeTransform(
            [256, 256],
            False,
            image_mode="oracle_left",
            bbox_mode=mode,
            bbox_context_scale=args.context_scale,
        )
        for mode in modes
    }
    header_height = 24
    canvas = Image.new(
        "RGB", (len(modes) * 256, len(selected) * (256 + header_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(selected):
        with Image.open(args.root / row["image_path"]) as source:
            source.load()
            for column, mode in enumerate(modes):
                rendered = to_image(transforms[mode](source, row)[0])
                x = column * 256
                y = row_index * (256 + header_height)
                canvas.paste(rendered, (x, y + header_height))
                draw.text((x + 4, y + 4), f"{row['class_name']} | {mode}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
