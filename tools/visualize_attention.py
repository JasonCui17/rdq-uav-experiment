#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.experiment import make_dataset  # noqa: E402
from rdq_uav.models import build_model  # noqa: E402


def to_rgb_image(normalized: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (normalized.cpu() * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-camera cross-attention overlays")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("attention_overlay.png"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if config["model"]["variant"] not in {"learned_query", "rdq"}:
        raise ValueError("Attention visualization requires learned_query or rdq")
    dataset = make_dataset(config, args.split)
    sample = dataset[args.index]
    model = build_model(config["model"], load_backbone_pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        output = model(
            sample["image"].unsqueeze(0),
            sample["radar"].unsqueeze(0),
            sample["radar_mask"].unsqueeze(0),
            return_attention=True,
        )
    attention = output["attention"]
    if attention is None or output["visual_grid"] is None:
        raise RuntimeError("Model did not return cross-attention")
    height, width = output["visual_grid"]
    weights = attention[0, :, 0].mean(dim=0).cpu().numpy()
    views = int(sample["image"].shape[0])
    if weights.size != views * height * width:
        raise RuntimeError("Attention token count does not match the camera grid")
    weight_min, weight_max = float(weights.min()), float(weights.max())
    weights = (weights - weight_min) / max(weight_max - weight_min, 1e-12)

    overlays = []
    for camera_index in range(views):
        image_rgb = to_rgb_image(sample["image"][camera_index])
        camera_weights = weights[
            camera_index * height * width : (camera_index + 1) * height * width
        ].reshape(height, width)
        heat = cv2.resize(
            camera_weights, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlays.append(cv2.addWeighted(image_bgr, 0.55, heat_bgr, 0.45, 0))
    canvas = np.concatenate(overlays, axis=1)
    prediction = int(output["logits"].argmax(dim=1))
    text = f"id={sample['sample_id']} target={int(sample['label'])} pred={prediction}"
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Failed to write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
