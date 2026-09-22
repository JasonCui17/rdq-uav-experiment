#!/usr/bin/env python3
"""Real-data smoke test for P5 Geometry Bi-HCI.

Runs one real MMAUD multimodal query through:
  LiDAR V2 -> HCI0/1/2 -> shared DINO-Swin-T -> DINO head

This is a correctness smoke test, not a training/evaluation benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETREX = ROOT / "third_party/detrex"
DEFAULT_DINO_CONFIG = (
    DEFAULT_DETREX
    / "projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py"
)
DEFAULT_DINO_CHECKPOINT = (
    ROOT
    / "checkpoints/dino_swin_t/"
    "dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"
)
DEFAULT_LIDAR_CONFIG = ROOT / "configs/lidar_uav_v2.yaml"
DEFAULT_CAMERA_CONFIG = ROOT / "configs/calibration/mmaud_v1_omni.yaml"
DEFAULT_GEOMETRY = ROOT / "calibration/official_left_p4_current_geometry.json"
DEFAULT_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")
DEFAULT_SPLIT_FILE = ROOT / "outputs/mmuav_paper_reproduction/splits/splits.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split", default="validation_sub")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--max-image-gap-s", type=float, default=0.04)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--detrex", type=Path, default=DEFAULT_DETREX)
    parser.add_argument("--dino-config", type=Path, default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--lidar-config", type=Path, default=DEFAULT_LIDAR_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--geometry-calibration", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--dino-short-edge", type=int, default=800)
    parser.add_argument("--dino-max-size", type=int, default=1333)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def dino_eval_resize(
    image: Image.Image,
    *,
    short_edge: int,
    max_size: int,
) -> tuple[Image.Image, tuple[float, float]]:
    width, height = image.size
    scale = float(short_edge) / float(min(width, height))
    if float(max(width, height)) * scale > float(max_size):
        scale = float(max_size) / float(max(width, height))
    new_height = int(float(height) * scale + 0.5)
    new_width = int(float(width) * scale + 0.5)
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return resized, (new_width / width, new_height / height)


def load_left_rgb(path: str | Path, camera_width: int, camera_height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if height < camera_height:
        raise ValueError(
            f"image height {height} is smaller than calibrated height {camera_height}"
        )
    if width < camera_width:
        raise ValueError(
            f"image width {width} is smaller than calibrated width {camera_width}"
        )
    # MMAUD official UVC frames may contain left/right 1280x960 images
    # concatenated horizontally. The calibrated left camera is the left crop.
    return image.crop((0, 0, camera_width, camera_height))


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device)


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def choose_sample(dataset, image_index, explicit_index: int | None) -> tuple[int, object]:
    """Choose a genuinely bimodal query: valid RGB match and non-empty LiDAR."""

    def inspect(index: int) -> tuple[object, int]:
        record = dataset.records[index]
        match = image_index.match(record["sequence_id"], record["query_time"])
        if not match.valid:
            return match, 0
        query = dataset[index]
        return match, int(len(query["points"]))

    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= len(dataset):
            raise IndexError(f"sample-index {explicit_index} outside [0,{len(dataset)})")
        match, point_count = inspect(explicit_index)
        if not match.valid:
            raise RuntimeError("requested sample has no valid left-image match")
        if point_count == 0:
            raise RuntimeError(
                "requested sample has zero causal LiDAR points; choose another sample"
            )
        return explicit_index, match

    empty_lidar_matches = 0
    for index in range(len(dataset)):
        match, point_count = inspect(index)
        if not match.valid:
            continue
        if point_count == 0:
            empty_lidar_matches += 1
            continue
        if empty_lidar_matches:
            print(
                f"[sample-select] skipped {empty_lidar_matches} image-matched "
                "queries with zero causal LiDAR points"
            )
        return index, match

    raise RuntimeError(
        "no genuinely bimodal query found in split: "
        "need both a valid left image and non-empty causal LiDAR"
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    sys.path.insert(0, str(args.detrex))
    sys.path.insert(0, str(args.detrex / "detectron2"))
    sys.path.insert(0, str(ROOT / "src"))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from rdq_uav.lidar_v2 import LiDARUAVDetector
    from rdq_uav.lidar_v2.data import LiDARUAVDataset
    from rdq_uav.multimodal_v1 import (
        DINOAdapter,
        GeometryBiHCIStack,
        LiDARV2PyramidAdapter,
        P5MultimodalBackbone,
        collate_multimodal_queries,
        load_left_projection_context,
        make_interaction_context,
    )
    from rdq_uav.multimodal_v1.data import LeftImageIndex, MultimodalQueryDataset

    required = (
        args.dino_config,
        args.dino_checkpoint,
        args.lidar_config,
        args.camera_config,
        args.geometry_calibration,
        args.split_file,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    camera_cfg = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    left_cfg = camera_cfg["cameras"]["left"]
    camera_width, camera_height = map(int, left_cfg["resolution"])

    geometry_cfg = json.loads(
        args.geometry_calibration.read_text(encoding="utf-8")
    )
    time_offset_s = float(geometry_cfg["time_offset_s"])

    lidar_dataset = LiDARUAVDataset(
        args.root,
        args.split_file,
        args.split,
        max_events=args.max_events,
    )
    image_index = LeftImageIndex(
        args.root,
        time_offset_s=time_offset_s,
        max_abs_gap_s=args.max_image_gap_s,
    )
    sample_index, match = choose_sample(
        lidar_dataset, image_index, args.sample_index
    )
    multimodal_dataset = MultimodalQueryDataset(
        lidar_dataset,
        image_index,
        calibration_handle=args.geometry_calibration,
    )
    sample = multimodal_dataset[sample_index]
    batch = collate_multimodal_queries([sample])

    if sample["left_image_path"] is None:
        raise RuntimeError("selected query unexpectedly has no image")
    left_image = load_left_rgb(
        sample["left_image_path"], camera_width, camera_height
    )
    resized_image, scale_xy = dino_eval_resize(
        left_image,
        short_edge=args.dino_short_edge,
        max_size=args.dino_max_size,
    )
    resized_tensor = image_to_tensor(resized_image, device)
    dino_inputs = [
        {
            "image": resized_tensor,
            "height": resized_image.height,
            "width": resized_image.width,
        }
    ]

    dino_cfg = LazyConfig.load(str(args.dino_config))
    dino_cfg.model.device = str(device)
    detector = instantiate(dino_cfg.model).to(device).eval()
    DetectionCheckpointer(detector).load(str(args.dino_checkpoint))
    dino = DINOAdapter(detector).eval()

    lidar_cfg = yaml.safe_load(args.lidar_config.read_text(encoding="utf-8"))
    lidar_detector = LiDARUAVDetector(lidar_cfg).to(device).eval()
    radar = LiDARV2PyramidAdapter(detector=lidar_detector).eval()

    hci = GeometryBiHCIStack(
        vision_dims=(96, 192, 384),
        feature_strides=(4, 8, 16),
        radar_dim=128,
        interaction_dim=128,
        num_heads=4,
        gate_bias_init=-4.6,
    ).to(device).eval()

    p5 = P5MultimodalBackbone(
        radar=radar,
        vision=dino.swin,
        interaction=hci,
    ).to(device).eval()

    # Move only tensors needed by LiDAR/HCI to the selected device; retain
    # metadata lists/strings on CPU.
    lidar_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    with torch.inference_mode():
        preprocessed = detector.preprocess_image(dino_inputs)
        projection = load_left_projection_context(
            args.camera_config,
            args.geometry_calibration,
            image_scale_xy=torch.tensor([scale_xy], dtype=torch.float32),
            device=device,
        )
        interaction_context = make_interaction_context(
            batch["calibration_handle"],
            lidar_batch["m_R"],
            lidar_batch["m_V"],
            projection,
        )

        p5_output = p5(
            lidar_batch,
            preprocessed.tensor,
            interaction_context,
            return_aux=True,
        )

        image_masks = preprocessed.tensor.new_zeros(
            preprocessed.tensor.shape[0],
            preprocessed.tensor.shape[-2],
            preprocessed.tensor.shape[-1],
        )
        dino_output = dino.forward_from_pyramid(
            p5_output.vision,
            image_masks,
        )

    hci_report = []
    for index, aux in enumerate(p5_output.hci_aux):
        if aux is None:
            raise RuntimeError("return_aux=True but an HCI stage returned no aux")
        valid = aux["valid_projection_mask"]
        active = aux["active_radar_mask"]
        radar_neighbors = aux["radar_neighbor_count"]
        vision_support = aux["vision_support_count"]
        gate_r = aux["gate_R"]
        gate_v = aux["gate_V"]
        hci_report.append(
            {
                "stage": index,
                "edge_count": int(aux["edge_count"]),
                "radar_tokens": int(valid.numel()),
                "valid_projected_tokens": int(valid.sum().item()),
                "active_radar_tokens": int(active.sum().item()),
                "projection_valid_rate": (
                    float(valid.float().mean().item()) if valid.numel() else 0.0
                ),
                "radar_neighbor_count_max": (
                    int(radar_neighbors.max().item())
                    if radar_neighbors.numel()
                    else 0
                ),
                "vision_support_count_max": (
                    int(vision_support.max().item())
                    if vision_support.numel()
                    else 0
                ),
                "gate_R_mean": (
                    float(gate_r[active].mean().item())
                    if bool(active.any())
                    else 0.0
                ),
                "gate_R_max": (
                    float(gate_r.max().item()) if gate_r.numel() else 0.0
                ),
                "gate_V_mean_supported": (
                    float(gate_v[vision_support > 0].mean().item())
                    if bool((vision_support > 0).any())
                    else 0.0
                ),
                "gate_V_max": (
                    float(gate_v.max().item()) if gate_v.numel() else 0.0
                ),
            }
        )

    finite_checks = {
        "radar_logits": finite(p5_output.radar["logits"]),
        "radar_pred_xyz": finite(p5_output.radar["pred_xyz"]),
        "radar_fine_features": finite(p5_output.radar["fine_features"]),
        "vision_v0": finite(p5_output.vision.features[0]),
        "vision_v1": finite(p5_output.vision.features[1]),
        "vision_v2": finite(p5_output.vision.features[2]),
        "vision_v3": finite(p5_output.vision.features[3]),
        "dino_pred_logits": finite(dino_output["pred_logits"]),
        "dino_pred_boxes": finite(dino_output["pred_boxes"]),
        "dino_decoder_query_features": finite(
            dino_output["decoder_query_features"]
        ),
    }
    all_finite = all(finite_checks.values())
    all_stages_engaged = all(item["edge_count"] > 0 for item in hci_report)

    report = {
        "status": "PASS" if all_finite and all_stages_engaged else "FAIL",
        "purpose": "P5 real-data forward smoke test; not an accuracy benchmark",
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        ),
        "sample": {
            "dataset_index": sample_index,
            "sequence_id": sample["sequence_id"],
            "query_uid": sample["query_uid"],
            "query_time": float(sample["query_time"]),
            "image_time": float(sample["image_time"]),
            "image_query_gap_s": float(sample["image_query_gap_s"]),
            "left_image_path": str(sample["left_image_path"]),
            "point_count": int(len(sample["points"])),
        },
        "image": {
            "camera_source_wh": [camera_width, camera_height],
            "resized_wh": [resized_image.width, resized_image.height],
            "scale_xy": [float(scale_xy[0]), float(scale_xy[1])],
            "preprocessed_tensor_hw": [
                int(preprocessed.tensor.shape[-2]),
                int(preprocessed.tensor.shape[-1]),
            ],
        },
        "radar_token_counts": p5_output.radar["aux_stats"]["token_counts"],
        "hci": hci_report,
        "shapes": {
            "radar_fine_features": list(
                p5_output.radar["fine_features"].shape
            ),
            "vision": [
                list(value.shape) for value in p5_output.vision.features
            ],
            "dino_pred_logits": list(dino_output["pred_logits"].shape),
            "dino_pred_boxes": list(dino_output["pred_boxes"].shape),
            "dino_decoder_query_features": list(
                dino_output["decoder_query_features"].shape
            ),
        },
        "finite": finite_checks,
        "all_three_hci_stages_engaged": all_stages_engaged,
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
