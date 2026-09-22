#!/usr/bin/env python3
"""Real MMAUD forward smoke for the complete frozen P5 -> P6 -> P7 chain.

This intentionally uses untrained multimodal heads and a freshly adapted
single-class UAV DINO classification head. It validates tensor/geometry/model
integration only; predicted boxes/XYZ are not an accuracy result.
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
DEFAULT_DINO_CONFIG = DEFAULT_DETREX / "projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py"
DEFAULT_DINO_CHECKPOINT = ROOT / "checkpoints/dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"
DEFAULT_LIDAR_CONFIG = ROOT / "configs/lidar_uav_v2.yaml"
DEFAULT_P6_CONFIG = ROOT / "configs/multimodal_v1/p6_candidates.yaml"
DEFAULT_CAMERA_CONFIG = ROOT / "configs/calibration/mmaud_v1_omni.yaml"
DEFAULT_GEOMETRY = ROOT / "calibration/official_left_p4_current_geometry.json"
DEFAULT_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")
DEFAULT_SPLIT_FILE = ROOT / "outputs/mmuav_paper_reproduction/splits/splits.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    p.add_argument("--split", default="validation_sub")
    p.add_argument("--stats-split", default="train_sub")
    p.add_argument("--max-events", type=int, default=20)
    p.add_argument("--max-image-gap-s", type=float, default=0.04)
    p.add_argument("--sample-index", type=int)
    p.add_argument("--detrex", type=Path, default=DEFAULT_DETREX)
    p.add_argument("--dino-config", type=Path, default=DEFAULT_DINO_CONFIG)
    p.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    p.add_argument("--lidar-config", type=Path, default=DEFAULT_LIDAR_CONFIG)
    p.add_argument("--p6-config", type=Path, default=DEFAULT_P6_CONFIG)
    p.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    p.add_argument("--geometry-calibration", type=Path, default=DEFAULT_GEOMETRY)
    p.add_argument("--dino-short-edge", type=int, default=800)
    p.add_argument("--dino-max-size", type=int, default=1333)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-report", type=int, default=5)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def dino_eval_resize(image: Image.Image, *, short_edge: int, max_size: int):
    width, height = image.size
    scale = float(short_edge) / float(min(width, height))
    if float(max(width, height)) * scale > float(max_size):
        scale = float(max_size) / float(max(width, height))
    nh = int(float(height) * scale + 0.5)
    nw = int(float(width) * scale + 0.5)
    return image.resize((nw, nh), Image.Resampling.BILINEAR), (nw / width, nh / height)


def load_left_rgb(path: str | Path, camera_width: int, camera_height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.width < camera_width or image.height < camera_height:
        raise ValueError(f"image {image.size} smaller than calibrated {(camera_width,camera_height)}")
    return image.crop((0, 0, camera_width, camera_height))


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def choose_sample(dataset, image_index, explicit_index: int | None):
    def inspect(index: int):
        record = dataset.records[index]
        match = image_index.match(record["sequence_id"], record["query_time"])
        if not match.valid:
            return match, 0
        return match, int(len(dataset[index]["points"]))

    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= len(dataset):
            raise IndexError("sample-index outside dataset")
        match, point_count = inspect(explicit_index)
        if not match.valid or point_count == 0:
            raise RuntimeError("requested sample must have valid RGB and non-empty causal LiDAR")
        return explicit_index, match

    skipped = 0
    for index in range(len(dataset)):
        match, point_count = inspect(index)
        if not match.valid:
            continue
        if point_count == 0:
            skipped += 1
            continue
        if skipped:
            print(f"[sample-select] skipped {skipped} image-matched queries with zero causal LiDAR points")
        return index, match
    raise RuntimeError("no genuinely bimodal query found")


def train_xyz_stats(dataset) -> tuple[np.ndarray, np.ndarray, int]:
    values = []
    for record in dataset.records:
        path = record.get("target_path")
        if path is None:
            continue
        xyz = np.load(path, allow_pickle=False).reshape(3).astype(np.float64)
        if np.isfinite(xyz).all():
            values.append(xyz)
    if not values:
        raise RuntimeError("stats split contains no valid 3D GT")
    arr = np.stack(values)
    mean = arr.mean(0)
    std = arr.std(0)
    if np.any(std <= 1e-8):
        raise RuntimeError(f"degenerate train XYZ std: {std.tolist()}")
    return mean, std, len(arr)


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.float()).all().item())


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    sys.path.insert(0, str(args.detrex))
    sys.path.insert(0, str(args.detrex / "detectron2"))
    sys.path.insert(0, str(ROOT / "src"))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from rdq_uav.lidar_v2 import LiDARUAVDetector
    from rdq_uav.lidar_v2.data import LiDARUAVDataset
    from rdq_uav.lidar_v2.selector import CandidateSelector
    from rdq_uav.multimodal_v1 import (
        DINOAdapter, GeometryBiHCIStack, LiDARV2PyramidAdapter,
        P5MultimodalBackbone, collate_multimodal_queries,
        load_left_projection_context, make_interaction_context,
    )
    from rdq_uav.multimodal_v1.candidate import HYP_RV, HYP_R, HYP_V
    from rdq_uav.multimodal_v1.data import LeftImageIndex, MultimodalQueryDataset
    from rdq_uav.multimodal_v1.model import FullMultimodalV1
    from rdq_uav.multimodal_v1.vision.uav_dino import adapt_dino_class_head_to_single_uav

    required = (args.dino_config,args.dino_checkpoint,args.lidar_config,args.p6_config,
                args.camera_config,args.geometry_calibration,args.split_file)
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    camera_cfg = yaml.safe_load(args.camera_config.read_text())
    camera_width, camera_height = map(int, camera_cfg["cameras"]["left"]["resolution"])
    geometry_cfg = json.loads(args.geometry_calibration.read_text())
    time_offset_s = float(geometry_cfg["time_offset_s"])

    dataset = LiDARUAVDataset(args.root,args.split_file,args.split,max_events=args.max_events)
    stats_dataset = LiDARUAVDataset(args.root,args.split_file,args.stats_split,max_events=args.max_events)
    xyz_mean, xyz_std, xyz_stats_count = train_xyz_stats(stats_dataset)
    image_index = LeftImageIndex(args.root,time_offset_s=time_offset_s,max_abs_gap_s=args.max_image_gap_s)
    sample_index, _ = choose_sample(dataset,image_index,args.sample_index)
    multimodal_dataset = MultimodalQueryDataset(dataset,image_index,calibration_handle=args.geometry_calibration)
    sample = multimodal_dataset[sample_index]
    batch = collate_multimodal_queries([sample])

    left = load_left_rgb(sample["left_image_path"],camera_width,camera_height)
    resized, scale_xy = dino_eval_resize(left,short_edge=args.dino_short_edge,max_size=args.dino_max_size)
    dino_inputs=[{"image":image_to_tensor(resized,device),"height":resized.height,"width":resized.width}]

    dino_cfg=LazyConfig.load(str(args.dino_config)); dino_cfg.model.device=str(device)
    detector=instantiate(dino_cfg.model).to(device).eval()
    DetectionCheckpointer(detector).load(str(args.dino_checkpoint))
    adapt_dino_class_head_to_single_uav(detector)
    detector.eval()
    dino=DINOAdapter(detector).eval()

    lidar_cfg=yaml.safe_load(args.lidar_config.read_text())
    lidar_detector=LiDARUAVDetector(lidar_cfg).to(device).eval()
    radar=LiDARV2PyramidAdapter(detector=lidar_detector).eval()
    hci=GeometryBiHCIStack(vision_dims=(96,192,384),feature_strides=(4,8,16),radar_dim=128,
                           interaction_dim=128,num_heads=4,gate_bias_init=-4.6).to(device).eval()
    backbone=P5MultimodalBackbone(radar=radar,vision=dino.swin,interaction=hci).to(device).eval()

    p6_cfg=yaml.safe_load(args.p6_config.read_text())
    selector=CandidateSelector(p6_cfg["candidate"]["radar"])
    model=FullMultimodalV1(
        backbone=backbone,dino=dino,radar_selector=selector,
        radar_topk=int(p6_cfg["candidate"]["radar"]["final_topk"]),
        vision_pre_topk=int(p6_cfg["candidate"]["rgb"]["pre_topk"]),
        vision_topk=int(p6_cfg["candidate"]["rgb"]["final_topk"]),
        vision_nms_iou=float(p6_cfg["candidate"]["rgb"]["nms_iou"]),
        geometry_gate_px=float(p6_cfg["association"]["geometry_gate_px"]),
        xyz_mean=tuple(float(x) for x in xyz_mean),
        xyz_std=tuple(float(x) for x in xyz_std),
    ).to(device).eval()

    lidar_batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    with torch.inference_mode():
        preprocessed=detector.preprocess_image(dino_inputs)
        projection=load_left_projection_context(args.camera_config,args.geometry_calibration,
            image_scale_xy=torch.tensor([scale_xy],dtype=torch.float32),device=device)
        context=make_interaction_context(batch["calibration_handle"],lidar_batch["m_R"],lidar_batch["m_V"],projection)
        output=model(lidar_batch,preprocessed.tensor,context,return_aux=True)

    if output.aux is None:
        raise RuntimeError("return_aux=True produced no aux")
    radar_candidates=output.aux["radar_candidates"]
    rgb_candidates=output.aux["rgb_candidates"]
    pre_h=output.aux["hypotheses_pre_postprocess"]
    type_counts={
        "H_RV": int((pre_h.hypothesis_type==HYP_RV).sum().item()),
        "H_R": int((pre_h.hypothesis_type==HYP_R).sum().item()),
        "H_V": int((pre_h.hypothesis_type==HYP_V).sum().item()),
    }
    conservation = pre_h.n == radar_candidates.n + rgb_candidates.n - type_counts["H_RV"]

    finite_checks={
        "fused_score": finite(output.fused_score),
        "box_xyxy_px": finite(output.box_xyxy_px),
        "xyz": finite(output.xyz),
        "c_2d": finite(output.c_2d),
        "c_3d": finite(output.c_3d),
        "gate_weights": finite(output.gate_weights),
    }
    gate_sum_ok = bool(torch.allclose(output.gate_weights.sum(1),torch.ones_like(output.fused_score),atol=1e-5,rtol=1e-5)) if output.fused_score.numel() else True

    order=torch.argsort(output.fused_score,descending=True)[:max(0,args.top_report)]
    gt_xyz=torch.as_tensor(sample["target_xyz"],dtype=torch.float32,device=device) if sample.get("target_valid",False) else None
    top=[]
    for idx in order.tolist():
        item={
            "rank":len(top)+1,
            "fused_score":float(output.fused_score[idx].item()),
            "hypothesis_type":int(output.hypothesis_type[idx].item()),
            "box_xyxy_px":[float(x) for x in output.box_xyxy_px[idx].tolist()],
            "xyz_m":[float(x) for x in output.xyz[idx].tolist()],
            "c_2d":float(output.c_2d[idx].item()),
            "c_3d":float(output.c_3d[idx].item()),
            "gate_weights":[float(x) for x in output.gate_weights[idx].tolist()],
        }
        if gt_xyz is not None:
            item["xyz_error_to_gt_m"] = float(torch.linalg.vector_norm(output.xyz[idx]-gt_xyz).item())
        top.append(item)

    status = all(finite_checks.values()) and gate_sum_ok and conservation and pre_h.n>0 and output.fused_score.numel()>0
    report={
        "status":"PASS" if status else "FAIL",
        "purpose":"Complete P5->P6->P7 real-data structural forward smoke; NOT an accuracy benchmark",
        "accuracy_valid":False,
        "untrained_components":["single-class DINO classification head","P6 RGB feature projection","Reliability Gate","Typed Shared Query","P7 Transformer decoder","fusion prediction heads"],
        "device":torch.cuda.get_device_name(device) if device.type=="cuda" else str(device),
        "sample":{
            "dataset_index":sample_index,"sequence_id":sample["sequence_id"],"query_uid":sample["query_uid"],
            "query_time":float(sample["query_time"]),"image_time":float(sample["image_time"]),
            "image_query_gap_s":float(sample["image_query_gap_s"]),"point_count":int(len(sample["points"])),
            "gt_xyz_m":[float(x) for x in torch.as_tensor(sample["target_xyz"]).tolist()] if sample.get("target_valid",False) else None,
        },
        "train_only_xyz_normalization":{"split":args.stats_split,"count":xyz_stats_count,"mean":xyz_mean.tolist(),"std":xyz_std.tolist()},
        "p6":{
            "radar_candidates":radar_candidates.n,"rgb_candidates":rgb_candidates.n,
            "hypotheses_pre_postprocess":pre_h.n,"hypothesis_type_counts":type_counts,
            "matched_pairs":type_counts["H_RV"],"conservation_pass":conservation,
            "geometry_gate_px":float(p6_cfg["association"]["geometry_gate_px"]),
        },
        "p7":{
            "predictions_after_postprocess":int(output.fused_score.numel()),
            "gate_weights_sum_to_one":gate_sum_ok,
            "finite":finite_checks,
            "top_predictions":top,
        },
    }
    text=json.dumps(report,indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n")
    if report["status"]!="PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
