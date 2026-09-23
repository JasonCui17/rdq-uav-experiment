#!/usr/bin/env python3
"""Run the official detrex reference and the P2 adapter on one shared model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETREX = ROOT / "third_party/detrex"
DEFAULT_CONFIG = DEFAULT_DETREX / "projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detrex", type=Path, default=DEFAULT_DETREX)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max().item())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.detrex))
    sys.path.insert(0, str(args.detrex / "detectron2"))
    sys.path.insert(0, str(ROOT / "src"))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from rdq_uav.multimodal_v1.vision import DINOAdapter

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    config = LazyConfig.load(str(args.config))
    config.model.device = args.device
    detector = instantiate(config.model).to(args.device).eval()
    DetectionCheckpointer(detector).load(str(args.checkpoint))

    captured: dict[str, object] = {}
    handles = []
    for index, layer in enumerate(detector.backbone.layers):
        def capture_stage(module, inputs, output, index=index):
            captured[f"stage{index}"] = tuple(
                item.detach().clone() if torch.is_tensor(item) else item for item in output
            )
        handles.append(layer.register_forward_hook(capture_stage))
    handles.append(
        detector.backbone.register_forward_hook(
            lambda module, inputs, output: captured.update(
                backbone={key: value.detach().clone() for key, value in output.items()}
            )
        )
    )
    handles.append(
        detector.transformer.register_forward_hook(
            lambda module, inputs, output: captured.update(
                transformer=tuple(value.detach().clone() for value in output)
            )
        )
    )

    torch.manual_seed(42)
    image = torch.rand(3, 224, 224, device=args.device) * 255
    batch = [{"image": image, "height": 224, "width": 224}]
    with torch.inference_mode():
        detections = detector(batch)
    for handle in handles:
        handle.remove()

    transformer = captured["transformer"]
    decoder_states, initial_reference, intermediate_references, _, _ = transformer
    final_level = int(decoder_states.shape[0]) - 1
    with torch.inference_mode():
        reference_logits = detector.class_embed[final_level](decoder_states[-1])
        reference_delta = detector.bbox_embed[final_level](decoder_states[-1])
        reference_point = intermediate_references[final_level - 1]
        point = reference_point.clamp(0, 1)
        inverse = torch.log(point.clamp(min=1e-3) / (1 - point).clamp(min=1e-3))
        reference_boxes = (reference_delta + inverse).sigmoid()

        reference_stages = []
        for index in range(4):
            raw, height, width, *_ = captured[f"stage{index}"]
            if index == 0:
                exposed = raw
            else:
                exposed = getattr(detector.backbone, f"norm{index}")(raw)
            reference_stages.append(
                exposed.view(-1, height, width, detector.backbone.num_features[index])
                .permute(0, 3, 1, 2)
                .contiguous()
            )

    adapter = DINOAdapter(detector).eval()
    with torch.inference_mode():
        adapted = adapter(batch)
        writeback = adapter(
            batch, pre_stage_transform=lambda stage: stage.tokens
        )

    diffs = {
        **{
            f"V{index}": max_diff(reference_stages[index], adapted["pyramid"].features[index])
            for index in range(4)
        },
        "pred_logits": max_diff(reference_logits, adapted["pred_logits"]),
        "pred_boxes": max_diff(reference_boxes, adapted["pred_boxes"]),
        "decoder_query_features": max_diff(
            decoder_states[-1], adapted["decoder_query_features"]
        ),
        "identity_writeback_pred_logits": max_diff(
            adapted["pred_logits"], writeback["pred_logits"]
        ),
        "identity_writeback_pred_boxes": max_diff(
            adapted["pred_boxes"], writeback["pred_boxes"]
        ),
    }
    shapes = {
        f"V{index}": list(adapted["pyramid"].features[index].shape)
        for index in range(4)
    }
    report = {
        "status": "PASS" if max(diffs.values()) <= 1e-6 else "FAIL",
        "reference": "IDEA-Research/detrex DINO Swin-Tiny 4-scale",
        "detrex_commit": "e244e6c3da3e84566728c52c21fb061d23ce0e2f",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "reference_detections": len(detections[0]["instances"]),
        "stage_shapes": shapes,
        "stage_strides": [4, 8, 16, 32],
        "dino_neck_shapes": [list(value.shape) for value in adapted["multi_level_features"]],
        "object_query_source": "final DINO decoder hidden state (inter_states[-1])",
        "object_query_shape": list(adapted["decoder_query_features"].shape),
        "max_abs_diff": diffs,
        "single_shared_backbone": adapter.detector.backbone is adapter.swin.backbone,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
