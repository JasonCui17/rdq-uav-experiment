#!/usr/bin/env python3
"""Audit current B4 and prepare a fresh run with its trained MTCE radar baseline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_architecture import (
    MOTION_KEYS, validate_b4_radar_config, audit_b4_radar_detector,
)


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def radar_state(payload):
    prefix = "network.lidar.detector."
    state = payload.get("state_dict", payload)
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError("Checkpoint has no network.lidar.detector weights")
    return selected


def checkpoint_summary(payload):
    state = radar_state(payload)
    point_mlp = state.get("voxel_embed.point_mlp.0.weight")
    projection = state.get("voxel_embed.proj.weight")
    learned = point_mlp is not None and projection is not None and projection.shape[1] in (288, 544)
    norms = {key: float(state[key].float().norm()) if key in state else None for key in MOTION_KEYS}
    return {"epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
            "learned_sbe_weights": learned, "mtce_projection_norms": norms}


def audit_current(config, root):
    init = config["initialization"]
    path = resolve(root, init["lidar_config"])
    voxel = yaml.safe_load(path.read_text())["model"]["voxel"]
    report = {"lidar_config": str(path), "configured_embedding": voxel.get("embedding"),
              "configured_mtce": voxel.get("motion_centroid", {}).get("enabled", False),
              "b0_checkpoint": init.get("b0_checkpoint")}
    if init.get("b0_checkpoint"):
        checkpoint = resolve(root, init["b0_checkpoint"])
        if checkpoint.is_file():
            report["checkpoint"] = checkpoint_summary(torch.load(checkpoint, map_location="cpu", weights_only=False))
    return report


def prepare(config, checkpoint, root, *, radar_config=None, run_output="outputs/b4_learnsbe_mtce_seed42"):
    if config["experiment"]["stage"] != "B4":
        raise ValueError("Source configuration must be B4")
    checkpoint = resolve(root, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    baseline = payload.get("hyper_parameters", {}).get("config", {})
    if baseline.get("experiment", {}).get("stage") != "B0":
        raise ValueError("Choose a trained B0 MTCE checkpoint with architecture metadata")
    reference = radar_config or baseline.get("initialization", {}).get("lidar_config")
    if reference is None:
        raise ValueError("B0 checkpoint has no lidar_config; supply --radar-config explicitly")
    path = resolve(root, reference)
    radar_cfg = yaml.safe_load(path.read_text())
    structure = validate_b4_radar_config(radar_cfg, config["data"]["radar_history_s"])
    detector = LiDARUAVDetector(radar_cfg).eval()
    selected = radar_state(payload)
    bad = [name for name, tensor in selected.items() if not torch.isfinite(tensor).all()]
    if bad:
        raise FloatingPointError("Non-finite radar checkpoint weights: " + ", ".join(bad[:8]))
    detector.load_state_dict(selected, strict=True)
    report = {**structure, **audit_b4_radar_detector(detector), **checkpoint_summary(payload)}
    result = deepcopy(config)
    result["initialization"]["lidar_config"] = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    result["initialization"]["b0_checkpoint"] = str(checkpoint.relative_to(root)) if checkpoint.is_relative_to(root) else str(checkpoint)
    b1 = resolve(root, result["initialization"]["b1_checkpoint"])
    if not b1.is_file():
        raise FileNotFoundError(f"Existing B1 checkpoint does not exist: {b1}")
    result["experiment"]["name"] = "b4_learnsbe_mtce_seed42"
    result["experiment"]["output_dir"] = run_output
    if resolve(root, run_output).exists():
        raise FileExistsError("Use a fresh output directory for the changed radar baseline: " + run_output)
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/multimodal_v2/configs/b4_aligned_pairs.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Trained B0 LearnSBE + MTCE checkpoint; omit to audit only")
    parser.add_argument("--radar-config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/multimodal_v2/configs/b4_learnsbe_mtce_server.yaml"))
    parser.add_argument("--run-output", default="outputs/b4_learnsbe_mtce_seed42")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(ROOT, args.config).read_text())
    print("Current B4:", json.dumps(audit_current(config, ROOT), indent=2, ensure_ascii=False))
    if args.checkpoint is None:
        return
    output = resolve(ROOT, args.output)
    if output.exists():
        raise FileExistsError(f"New configuration already exists: {output}; choose another --output")
    result, report = prepare(config, args.checkpoint, ROOT, radar_config=args.radar_config, run_output=args.run_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    print("Verified trained radar:", json.dumps(report, indent=2, ensure_ascii=False))
    print("Prepared:", output)
    print("LearnSBE and MTCE are active but frozen in B4. Start a fresh run; do not resume the old B4.")


if __name__ == "__main__":
    main()
