#!/usr/bin/env python3
"""Train the frozen Multimodal V1.1 E5 architecture on real MMAUD queries.

This entry point composes the already-audited P1--P7 modules.  It does not
retrain E0 or redefine any model/loss mathematics.  Run ``--max-updates 2``
as the real-data correctness gate before starting the complete schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/multimodal_v1/e5_full_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resize_wh(source_wh: tuple[int, int], short_edge: int, max_size: int) -> tuple[int, int]:
    width, height = source_wh
    scale = float(short_edge) / float(min(width, height))
    if max(width, height) * scale > max_size:
        scale = float(max_size) / float(max(width, height))
    return int(width * scale + 0.5), int(height * scale + 0.5)


def load_left_rgb(path: str | Path, source_wh: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = source_wh
    if image.width < width or image.height < height:
        raise ValueError(f"image {image.size} smaller than calibrated {source_wh}: {path}")
    return image.crop((0, 0, width, height))


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device)


class E5QueryDataset(Dataset):
    """One causal LiDAR query + nearest RGB image + optional verified 2D GT."""

    def __init__(self, lidar_dataset, image_index, manifest, sequence_ids, calibration_handle):
        self.lidar_dataset = lidar_dataset
        self.image_index = image_index
        self.calibration_handle = str(calibration_handle)
        allowed = set(sequence_ids)
        self.box_by_image: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for record in manifest:
            if record.sequence_id not in allowed or not record.gt_2d_valid or record.box_xyxy_px is None:
                continue
            key = (record.sequence_id, Path(record.image_path).name)
            box = tuple(record.box_xyxy_px)
            if key in self.box_by_image and self.box_by_image[key] != box:
                raise ValueError(f"conflicting verified boxes for {key}")
            self.box_by_image[key] = box
        self.indices = []
        self.matches = []
        for index, record in enumerate(lidar_dataset.records):
            match = image_index.match(record["sequence_id"], record["query_time"])
            self.indices.append(index)
            self.matches.append(match)
        if not self.indices:
            raise RuntimeError("split has no LiDAR query with a valid RGB binding")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = dict(self.lidar_dataset[self.indices[index]])
        match = self.matches[index]
        image_key = None if match.path is None else (query["sequence_id"], match.path.name)
        box = None if image_key is None else self.box_by_image.get(image_key)
        query.update(
            left_image_path=None if match.path is None else str(match.path),
            image_time=match.image_time,
            image_query_gap_s=match.gap_s,
            calibration_handle=self.calibration_handle,
            m_R=bool(len(query["points"]) > 0),
            m_V=bool(match.valid and match.path is not None),
            gt_box_xyxy_px=torch.zeros(4) if box is None else torch.tensor(box, dtype=torch.float32),
            gt_2d_valid=box is not None,
        )
        return query

    @property
    def labeled_indices(self) -> list[int]:
        result = []
        for index, match in enumerate(self.matches):
            sequence = self.lidar_dataset.records[self.indices[index]]["sequence_id"]
            if match.path is not None and (sequence, match.path.name) in self.box_by_image:
                result.append(index)
        return result


def collate_e5(samples):
    from rdq_uav.multimodal_v1.data import collate_multimodal_queries

    batch = collate_multimodal_queries(samples)
    batch["gt_box_xyxy_px"] = torch.stack([item["gt_box_xyxy_px"] for item in samples])
    batch["gt_2d_valid"] = torch.tensor([item["gt_2d_valid"] for item in samples], dtype=torch.bool)
    return batch


class GateSampler(Sampler[int]):
    """Put verified-box samples first for the two-update correctness gate."""

    def __init__(self, dataset: E5QueryDataset, seed: int):
        labeled = dataset.labeled_indices
        remaining = [index for index in range(len(dataset)) if index not in set(labeled)]
        generator = random.Random(seed)
        generator.shuffle(labeled)
        generator.shuffle(remaining)
        self.indices = labeled + remaining

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def xyz_stats(dataset) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for record in dataset.records:
        path = record.get("target_path")
        if path is not None:
            xyz = np.load(path, allow_pickle=False).reshape(3).astype(np.float64)
            if np.isfinite(xyz).all():
                values.append(xyz)
    if not values:
        raise RuntimeError("training split has no finite 3D targets")
    array = np.stack(values)
    std = array.std(0)
    if np.any(std <= 1e-8):
        raise RuntimeError(f"degenerate training XYZ standard deviation: {std.tolist()}")
    return array.mean(0), std


@dataclass
class Runtime:
    model: torch.nn.Module
    lidar_detector: torch.nn.Module
    dino_detector: torch.nn.Module
    radar_criterion: torch.nn.Module
    new_modules: tuple[torch.nn.Module, ...]
    camera_wh: tuple[int, int]
    camera_config: Path
    geometry_calibration: Path
    short_edge: int
    max_size: int
    initialization: dict[str, Any]


def build_runtime(cfg: dict[str, Any], train_lidar, device: torch.device) -> Runtime:
    init = cfg["initialization"]
    detrex = resolve(init["dino_root"])
    for path in (ROOT / "src", detrex / "detectron2", detrex):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from rdq_uav.lidar_v2 import LiDARUAVDetector
    from rdq_uav.lidar_v2.loss import CandidateLoss
    from rdq_uav.lidar_v2.selector import CandidateSelector
    from rdq_uav.multimodal_v1 import DINOAdapter, GeometryBiHCIStack, LiDARV2PyramidAdapter, P5MultimodalBackbone
    from rdq_uav.multimodal_v1.loss import FusionLoss
    from rdq_uav.multimodal_v1.model import FullMultimodalV1
    from rdq_uav.multimodal_v1.training import load_lidar_checkpoint_strict
    from rdq_uav.multimodal_v1.vision.uav_dino import adapt_dino_class_head_to_single_uav

    required = [resolve(value) for key, value in init.items() if key.endswith(("config", "checkpoint")) or key == "p6_config"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing initialization files: {missing}")

    dino_cfg = LazyConfig.load(str(resolve(init["dino_config"])))
    dino_cfg.model.device = str(device)
    dino_detector = instantiate(dino_cfg.model).to(device)
    DetectionCheckpointer(dino_detector).load(str(resolve(init["dino_checkpoint"])))
    adapt_dino_class_head_to_single_uav(dino_detector)
    dino = DINOAdapter(dino_detector)

    lidar_cfg = yaml.safe_load(resolve(init["lidar_config"]).read_text())
    lidar_detector = LiDARUAVDetector(lidar_cfg).to(device)
    lidar_payload = load_lidar_checkpoint_strict(lidar_detector, str(resolve(init["lidar_checkpoint"])))
    radar = LiDARV2PyramidAdapter(detector=lidar_detector)
    hci = GeometryBiHCIStack(
        vision_dims=(96, 192, 384), feature_strides=(4, 8, 16), radar_dim=128,
        interaction_dim=128, num_heads=4, gate_bias_init=-4.6,
    ).to(device)
    backbone = P5MultimodalBackbone(radar=radar, vision=dino.swin, interaction=hci)
    p6_cfg = yaml.safe_load(resolve(init["p6_config"]).read_text())
    selector = CandidateSelector(p6_cfg["candidate"]["radar"])
    mean, std = xyz_stats(train_lidar)
    loss_cfg = cfg["loss"]
    fusion_loss = FusionLoss(
        lambda_cls=loss_cfg["lambda_cls"], lambda_2d=loss_cfg["lambda_2d"],
        lambda_3d=loss_cfg["lambda_3d"], lambda_valid=loss_cfg["lambda_valid"],
        focal_alpha=loss_cfg["focal_alpha"], focal_gamma=loss_cfg["focal_gamma"],
    )
    model = FullMultimodalV1(
        backbone=backbone, dino=dino, radar_selector=selector,
        radar_topk=int(p6_cfg["candidate"]["radar"]["final_topk"]),
        vision_pre_topk=int(p6_cfg["candidate"]["rgb"]["pre_topk"]),
        vision_topk=int(p6_cfg["candidate"]["rgb"]["final_topk"]),
        vision_nms_iou=float(p6_cfg["candidate"]["rgb"]["nms_iou"]),
        geometry_gate_px=float(p6_cfg["association"]["geometry_gate_px"]),
        xyz_mean=tuple(map(float, mean)), xyz_std=tuple(map(float, std)), fusion_loss=fusion_loss,
    ).to(device)
    data_cfg = cfg["data"]
    camera_path = resolve(data_cfg["camera_config"])
    camera_cfg = yaml.safe_load(camera_path.read_text())
    camera_wh = tuple(map(int, camera_cfg["cameras"]["left"]["resolution"]))
    return Runtime(
        model, lidar_detector, dino_detector, CandidateLoss(lidar_cfg),
        (hci, model.rgb_candidates, model.reliability_gate, model.shared_query, model.decoder, model.heads),
        camera_wh, camera_path, resolve(data_cfg["geometry_calibration"]),
        int(data_cfg["dino_short_edge"]), int(data_cfg["dino_max_size"]),
        {
            "lidar_checkpoint": str(resolve(init["lidar_checkpoint"])),
            "lidar_epoch": lidar_payload.get("epoch"),
            "dino_checkpoint": str(resolve(init["dino_checkpoint"])),
        },
    )


def prepare_batch(batch: dict[str, Any], runtime: Runtime, device: torch.device):
    from rdq_uav.multimodal_v1 import load_left_projection_context, make_interaction_context
    from rdq_uav.multimodal_v1.loss import FusionTargets
    from rdq_uav.multimodal_v1.vision.ssod import ViewTransform

    tensors = {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}
    images, transforms, scales = [], [], []
    for path in batch["left_image_path"]:
        source = Image.new("RGB", runtime.camera_wh) if path is None else load_left_rgb(path, runtime.camera_wh)
        view_wh = resize_wh(source.size, runtime.short_edge, runtime.max_size)
        view = source.resize(view_wh, Image.Resampling.BILINEAR)
        transform = ViewTransform(source.size, view_wh, False)
        images.append({"image": image_tensor(view, device), "height": view.height, "width": view.width})
        transforms.append(transform)
        scales.append(transform.scale_xy)
    preprocessed = runtime.dino_detector.preprocess_image(images)
    projection = load_left_projection_context(
        runtime.camera_config, runtime.geometry_calibration,
        image_scale_xy=torch.tensor(scales, dtype=torch.float32), device=device,
    )
    context = make_interaction_context(
        batch["calibration_handle"], tensors["m_R"], tensors["m_V"], projection,
    )
    targets = FusionTargets(
        tensors["gt_box_xyxy_px"], tensors["gt_2d_valid"],
        tensors["target_xyz"], tensors["target_valid"],
    )
    return tensors, preprocessed.tensor, context, targets, transforms


def optimizer_groups(runtime: Runtime, cfg: dict[str, Any]):
    rates = cfg["training"]["learning_rates"]
    radar_ids = {id(p) for p in runtime.lidar_detector.parameters()}
    swin_ids = {id(p) for p in runtime.dino_detector.backbone.parameters()}
    dino_ids = {id(p) for p in runtime.dino_detector.parameters()} - swin_ids
    new_ids = {id(p) for module in runtime.new_modules for p in module.parameters()}
    definitions = (
        ("new_modules", new_ids, float(rates["new_modules"])),
        ("radar", radar_ids, float(rates["radar"])),
        ("dino_head", dino_ids, float(rates["dino_head"])),
        ("swin_last_two", swin_ids, float(rates["swin_last_two"])),
    )
    by_id = {id(parameter): parameter for parameter in runtime.model.parameters()}
    groups, assigned = [], set()
    for name, identifiers, lr in definitions:
        parameters = [by_id[item] for item in identifiers if item in by_id and item not in assigned]
        assigned.update(id(parameter) for parameter in parameters)
        if parameters:
            groups.append({"params": parameters, "lr": lr, "base_lr": lr, "name": name})
    missing = set(by_id) - assigned
    if missing:
        names = [name for name, parameter in runtime.model.named_parameters() if id(parameter) in missing]
        raise RuntimeError(f"optimizer grouping missed parameters: {names[:20]}")
    return groups


def activate_stage(runtime: Runtime, optimizer, cfg: dict[str, Any], epoch: int) -> str:
    from rdq_uav.multimodal_v1.training import set_e5_trainable, stage_for_epoch

    stage = stage_for_epoch(cfg["training"]["stages"], epoch)
    set_e5_trainable(
        stage=stage, lidar_detector=runtime.lidar_detector,
        dino_detector=runtime.dino_detector, new_modules=runtime.new_modules,
    )
    active = {
        "T1": {"new_modules"},
        "T2": {"new_modules", "radar", "dino_head"},
        "T3": {"new_modules", "radar", "dino_head", "swin_last_two"},
    }[stage]
    for group in optimizer.param_groups:
        group["stage_active"] = group["name"] in active
    return stage


def schedule_lr(optimizer, step: int, total: int, cfg: dict[str, Any]) -> float:
    training = cfg["training"]
    warmup = max(1, round(total * float(training["warmup_fraction"])))
    if step <= warmup:
        factor = step / warmup
    else:
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        ratio = float(training["final_lr_ratio"])
        factor = ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor if group.get("stage_active", False) else 0.0
    return factor


def forward_losses(runtime: Runtime, batch, device, cfg, *, return_aux: bool = True):
    from rdq_uav.multimodal_v1.training import combine_e5_losses, supervised_dino_loss

    lidar, images, context, targets, transforms = prepare_batch(batch, runtime, device)
    output = runtime.model(lidar, images, context, targets=targets, return_aux=return_aux)
    if output.aux is None or output.losses is None:
        raise RuntimeError("training forward requires fusion losses and auxiliary backbone outputs")
    radar = runtime.radar_criterion(output.aux["p5"].radar, lidar)
    vision, vision_parts, vision_supervised = supervised_dino_loss(
        runtime.dino_detector, output.aux["dino"],
        gt_box_xyxy_source=targets.gt_box_xyxy_px,
        gt_2d_valid=targets.gt_2d_valid, transforms=transforms,
    )
    fusion = output.losses["loss"]
    loss_cfg = cfg["loss"]
    total = combine_e5_losses(
        radar["loss"], vision, fusion,
        lambda_r=float(loss_cfg["lambda_R"]), lambda_v=float(loss_cfg["lambda_V"]),
        lambda_f=float(loss_cfg["lambda_F"]),
    )
    return total, radar, vision, vision_parts, vision_supervised, output, targets


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


@torch.no_grad()
def validate(runtime: Runtime, loader, device, cfg, max_samples: int | None = None) -> dict[str, Any]:
    runtime.model.eval()
    sums = {"loss": 0.0, "loss_R": 0.0, "loss_V": 0.0, "loss_F": 0.0}
    xyz_errors, box_ious = [], []
    batches = samples = vision_labels = 0
    for batch in loader:
        if max_samples is not None and samples >= max_samples:
            break
        total, radar, vision, _, labeled, output, targets = forward_losses(runtime, batch, device, cfg)
        sums["loss"] += float(total.float())
        sums["loss_R"] += float(radar["loss"].float())
        sums["loss_V"] += float(vision.float())
        sums["loss_F"] += float(output.losses["loss"].float())
        batches += 1
        samples += len(batch["sample_id"])
        vision_labels += labeled
        for index in range(len(batch["sample_id"])):
            mask = output.batch_index == index
            if not bool(mask.any()):
                continue
            local = torch.nonzero(mask, as_tuple=False).flatten()
            top = local[torch.argmax(output.fused_score[local].float())]
            if bool(targets.gt_3d_valid[index]):
                xyz_errors.append(float(torch.linalg.vector_norm(output.xyz[top].float() - targets.gt_xyz[index].float())))
            if bool(targets.gt_2d_valid[index]):
                a, b = output.box_xyxy_px[top].float(), targets.gt_box_xyxy_px[index].float()
                lt, rb = torch.maximum(a[:2], b[:2]), torch.minimum(a[2:], b[2:])
                inter = torch.prod((rb - lt).clamp_min(0))
                union = torch.prod((a[2:] - a[:2]).clamp_min(0)) + torch.prod((b[2:] - b[:2]).clamp_min(0)) - inter
                box_ious.append(float(inter / union) if float(union) > 0 else 0.0)
    denominator = max(1, batches)
    result = {key: value / denominator for key, value in sums.items()}
    result.update(
        samples=samples, batches=batches, vision_supervised=vision_labels,
        final_3d_count=len(xyz_errors),
        final_3d_success_05m=float(np.mean(np.asarray(xyz_errors) <= 0.5)) if xyz_errors else float("nan"),
        final_3d_success_1m=float(np.mean(np.asarray(xyz_errors) <= 1.0)) if xyz_errors else float("nan"),
        final_3d_success_2m=float(np.mean(np.asarray(xyz_errors) <= 2.0)) if xyz_errors else float("nan"),
        final_3d_mean_error=float(np.mean(xyz_errors)) if xyz_errors else float("nan"),
        final_3d_median_error=percentile(xyz_errors, 50),
        final_3d_p90_error=percentile(xyz_errors, 90),
        vision_top1_iou_mean=float(np.mean(box_ious)) if box_ious else float("nan"),
    )
    return result


def better(metrics: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current_key = (
        float(metrics["final_3d_success_1m"]),
        -float(metrics["final_3d_median_error"]),
    )
    best_key = (float(best["final_3d_success_1m"]), -float(best["final_3d_median_error"]))
    return current_key > best_key


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(runtime, optimizer, cfg, epoch, step, best_metrics):
    return {
        "model_state": runtime.model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "global_optimizer_step": step, "best_metrics": best_metrics,
        "effective_config": cfg, "initialization": runtime.initialization, "rng_state": rng_state(),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    output = resolve(args.output or cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    resume_path = None
    if args.resume:
        resume_path = output / cfg["checkpoint"]["last"] if args.resume == "auto" else resolve(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    elif (output / cfg["checkpoint"]["last"]).exists():
        raise FileExistsError(f"{output} already has last.pt; pass --resume auto instead of overwriting it")

    raw_device = str(args.device)
    if raw_device.isdigit():
        raw_device = f"cuda:{raw_device}"
    if raw_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(raw_device)
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sys.path.insert(0, str(ROOT / "src"))
    from rdq_uav.lidar_v2.data import LiDARUAVDataset
    from rdq_uav.multimodal_v1.data import LeftImageIndex
    from rdq_uav.multimodal_v1.vision.ssod_data import load_label_manifest

    data_cfg = cfg["data"]
    root, split_file = resolve(data_cfg["root"]), resolve(data_cfg["split_file"])
    train_lidar = LiDARUAVDataset(root, split_file, data_cfg["train_split"], max_events=int(data_cfg["max_events"]))
    val_lidar = LiDARUAVDataset(root, split_file, data_cfg["val_split"], max_events=int(data_cfg["max_events"]))
    geometry = json.loads(resolve(data_cfg["geometry_calibration"]).read_text())
    image_index = LeftImageIndex(root, time_offset_s=float(geometry["time_offset_s"]), max_abs_gap_s=float(data_cfg["max_image_gap_s"]))
    manifest = load_label_manifest(resolve(data_cfg["annotation_manifest"]), require_boxes=False)
    train_sequences = {str(record["sequence_id"]) for record in train_lidar.records}
    val_sequences = {str(record["sequence_id"]) for record in val_lidar.records}
    train_dataset = E5QueryDataset(train_lidar, image_index, manifest, train_sequences, data_cfg["geometry_calibration"])
    val_dataset = E5QueryDataset(val_lidar, image_index, manifest, val_sequences, data_cfg["geometry_calibration"])
    if args.train_limit is not None:
        train_dataset.indices = train_dataset.indices[:args.train_limit]; train_dataset.matches = train_dataset.matches[:args.train_limit]
    if args.val_limit is not None:
        val_dataset.indices = val_dataset.indices[:args.val_limit]; val_dataset.matches = val_dataset.matches[:args.val_limit]
    training_cfg = cfg["training"]
    sampler = GateSampler(train_dataset, seed) if args.max_updates is not None else None
    train_loader = DataLoader(
        train_dataset, batch_size=int(training_cfg["batch_size"]), sampler=sampler,
        shuffle=sampler is None, num_workers=int(data_cfg["num_workers"]), collate_fn=collate_e5,
        pin_memory=device.type == "cuda", persistent_workers=int(data_cfg["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(training_cfg["batch_size"]), shuffle=False,
        num_workers=int(data_cfg["num_workers"]), collate_fn=collate_e5,
        pin_memory=device.type == "cuda", persistent_workers=int(data_cfg["num_workers"]) > 0,
    )
    runtime = build_runtime(cfg, train_lidar, device)
    optimizer = torch.optim.AdamW(optimizer_groups(runtime, cfg), weight_decay=float(training_cfg["weight_decay"]))
    accumulate = int(training_cfg["accumulate"])
    epochs = int(training_cfg["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulate)
    total_updates = epochs * updates_per_epoch
    start_epoch, global_step, best_metrics = 1, 0, None
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu")
        runtime.model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_optimizer_step"])
        best_metrics = payload.get("best_metrics")
        if payload.get("rng_state") is not None:
            restore_rng(payload["rng_state"])

    cfg["runtime"] = {
        "device": str(device), "train_queries": len(train_dataset), "val_queries": len(val_dataset),
        "updates_per_epoch": updates_per_epoch, "total_planned_updates": total_updates,
        "max_updates_gate": args.max_updates,
    }
    (output / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "effective_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"E5 train={len(train_dataset)} val={len(val_dataset)} epochs={epochs} updates={total_updates}")
    print(f"initialization={runtime.initialization}")

    csv_path = output / "training_log.csv"
    csv_fields = ["epoch", "stage", "optimizer_step", "loss", "loss_R", "loss_V", "loss_F", "loss_cls_R", "loss_reg_R", "vision_supervised", "lr_factor", "seconds"]
    if not csv_path.exists():
        with csv_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writeheader()
    log_every = int(cfg["logging"]["log_every_updates"])
    amp_enabled = bool(training_cfg["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if training_cfg["amp_dtype"] == "bfloat16" else torch.float16
    stop = False
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        runtime.model.train()
        stage = activate_stage(runtime, optimizer, cfg, epoch)
        optimizer.zero_grad(set_to_none=True)
        sums = {key: 0.0 for key in ("loss", "loss_R", "loss_V", "loss_F", "loss_cls_R", "loss_reg_R")}
        batches = vision_count = pending = 0
        for batch_index, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                total, radar, vision, _, labeled, model_output, _ = forward_losses(runtime, batch, device, cfg)
                scaled = total / accumulate
            if not bool(torch.isfinite(total.float())):
                raise FloatingPointError(f"non-finite E5 loss at epoch={epoch} batch={batch_index}")
            scaled.backward()
            pending += 1; batches += 1; vision_count += labeled
            values = {
                "loss": total, "loss_R": radar["loss"], "loss_V": vision,
                "loss_F": model_output.losses["loss"], "loss_cls_R": radar["loss_cls"], "loss_reg_R": radar["loss_reg"],
            }
            for key, value in values.items():
                sums[key] += float(value.detach().float())
            boundary = pending == accumulate or batch_index == len(train_loader)
            if boundary:
                if pending != accumulate:
                    correction = accumulate / pending
                    for parameter in runtime.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in runtime.model.parameters() if parameter.requires_grad],
                    float(training_cfg["grad_clip_norm"]),
                )
                if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                    raise FloatingPointError(f"non-finite gradient at epoch={epoch} batch={batch_index}")
                global_step += 1
                factor = schedule_lr(optimizer, global_step, total_updates, cfg)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0
                if global_step % log_every == 0 or args.max_updates is not None:
                    print(f"epoch={epoch} stage={stage} update={global_step} loss={float(total):.5f} R={float(radar['loss']):.5f} V={float(vision):.5f} F={float(model_output.losses['loss']):.5f}")
                if args.max_updates is not None and global_step >= args.max_updates:
                    stop = True
                    break
        row = {
            "epoch": epoch, "stage": stage, "optimizer_step": global_step,
            **{key: value / max(1, batches) for key, value in sums.items()},
            "vision_supervised": vision_count, "lr_factor": factor if global_step else 0.0,
            "seconds": time.time() - epoch_start,
        }
        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writerow(row)
        atomic_torch_save(checkpoint(runtime, optimizer, cfg, epoch, global_step, best_metrics), output / cfg["checkpoint"]["last"])
        if stop:
            report = {"status": "PASS", "mode": "max_updates_gate", "optimizer_updates": global_step, "epoch_partial": epoch, "losses": row}
            (output / "max_updates_report.json").write_text(json.dumps(report, indent=2, allow_nan=True))
            print(json.dumps(report, indent=2, allow_nan=True))
            return
        if epoch % int(cfg["validation"]["every_epochs"]) == 0:
            metrics = validate(runtime, val_loader, device, cfg, cfg["validation"].get("max_samples"))
            metrics.update(epoch=epoch, global_optimizer_step=global_step)
            with (output / "validation_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=True) + "\n")
            if better(metrics, best_metrics):
                best_metrics = metrics
                atomic_torch_save(checkpoint(runtime, optimizer, cfg, epoch, global_step, best_metrics), output / cfg["checkpoint"]["best"])
            atomic_torch_save(checkpoint(runtime, optimizer, cfg, epoch, global_step, best_metrics), output / cfg["checkpoint"]["last"])
            print(f"validation epoch={epoch} success@1m={metrics['final_3d_success_1m']:.4f} median={metrics['final_3d_median_error']:.4f}")
    summary = {"status": "COMPLETE", "epochs": epochs, "optimizer_updates": global_step, "best": best_metrics}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
