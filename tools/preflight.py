#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check environment and experiment inputs")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    failures = []
    print("python", sys.version.replace("\n", " "))
    for name in ("numpy", "PIL", "yaml", "torch", "torchvision", "timm", "cv2"):
        try:
            module = importlib.import_module(name)
            print(name, "OK", getattr(module, "__version__", ""))
        except Exception as exc:
            failures.append(f"dependency {name}: {exc}")
            print(name, "FAIL", exc)

    try:
        import torch

        print("cuda_available", torch.cuda.is_available())
        print("cuda_device_count", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            print("gpu", index, torch.cuda.get_device_name(index))
    except ImportError:
        pass

    root = Path(config["data"]["root"])
    for class_name in config["data"]["classes"]:
        for modality in ("ground_truth", "image", "radar_enhance_pcl"):
            path = root / class_name / modality
            if not path.is_dir():
                failures.append(f"missing directory: {path}")

    manifest_dir = Path(config["data"]["manifest_dir"])
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.csv"
        if not path.is_file():
            failures.append(f"missing manifest: {path}")
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        print(split, "samples", len(rows), "classes", dict(Counter(row["class_name"] for row in rows)))
    stats_path = manifest_dir / "radar_stats.json"
    if stats_path.is_file():
        print("radar_stats", json.loads(stats_path.read_text(encoding="utf-8")))
    else:
        failures.append(f"missing radar stats: {stats_path}")

    checkpoint_cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    torch_files = list(checkpoint_cache.glob("*resnet18*")) if checkpoint_cache.is_dir() else []
    # timm 通常通过 huggingface_hub 下载该模型，因此同时检查两种缓存布局。
    huggingface_cache = Path.home() / ".cache" / "huggingface" / "hub"
    huggingface_files = (
        list(huggingface_cache.glob("models--timm--resnet18*"))
        if huggingface_cache.is_dir()
        else []
    )
    resnet18_files = torch_files + huggingface_files
    print("cached_resnet18_weights", [str(path) for path in resnet18_files])
    if bool(config["model"]["backbone"]["pretrained"]) and not resnet18_files:
        print("WARNING pretrained ResNet18 is not cached; first run may require network access")

    if failures:
        print("\nPREFLIGHT_FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print("\nPREFLIGHT_OK")


if __name__ == "__main__":
    main()
