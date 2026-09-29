"""MMAUD single-query data and preprocessing for Multimodal V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml

from rdq_uav.lidar_v2.data import LiDARUAVDataset
from rdq_uav.multimodal_v1 import LeftImageIndex, make_interaction_context
from rdq_uav.multimodal_v1.contracts import ProjectionContext
from rdq_uav.multimodal_v1.data import collate_multimodal_queries
from rdq_uav.multimodal_v1.vision.ssod import ViewTransform
from rdq_uav.multimodal_v1.vision.ssod_data import load_label_manifest

from .loss import MultimodalTargets


def resize_wh(source_wh: tuple[int, int], short_edge: int, max_size: int) -> tuple[int, int]:
    width, height = source_wh
    scale = float(short_edge) / min(width, height)
    if max(width, height) * scale > max_size:
        scale = float(max_size) / max(width, height)
    return int(width * scale + 0.5), int(height * scale + 0.5)


def prepare_image(path: str | Path | None, source_wh: tuple[int, int],
                  short_edge: int, max_size: int) -> dict[str, torch.Tensor]:
    if path is None:
        source = Image.new("RGB", source_wh)
    else:
        with Image.open(path) as handle:
            source = handle.convert("RGB")
        if source.width < source_wh[0] or source.height < source_wh[1]:
            raise ValueError(f"image {source.size} smaller than calibration {source_wh}: {path}")
        source = source.crop((0, 0, source_wh[0], source_wh[1]))
    view_wh = resize_wh(source_wh, short_edge, max_size)
    array = np.asarray(source.resize(view_wh, Image.Resampling.BILINEAR), dtype=np.uint8).copy()
    return {
        "image_uint8": torch.from_numpy(array).permute(2, 0, 1).contiguous(),
        "image_source_wh": torch.tensor(source_wh, dtype=torch.long),
        "image_view_wh": torch.tensor(view_wh, dtype=torch.long),
        "image_scale_xy": torch.tensor((view_wh[0] / source_wh[0], view_wh[1] / source_wh[1])),
    }


class MultimodalV2Dataset(Dataset):
    """One causal latest-20 LiDAR query plus nearest left image and masked GT."""

    def __init__(self, lidar_dataset: LiDARUAVDataset, image_index: LeftImageIndex,
                 manifest: Sequence[Any], sequence_ids: Sequence[str],
                 calibration_handle: str | Path, *, camera_wh: tuple[int, int],
                 short_edge: int, max_size: int) -> None:
        self.lidar_dataset = lidar_dataset
        self.image_index = image_index
        self.calibration_handle = str(calibration_handle)
        self.camera_wh = tuple(map(int, camera_wh))
        self.short_edge, self.max_size = int(short_edge), int(max_size)
        allowed = set(sequence_ids)
        self.box_by_image: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for record in manifest:
            if record.sequence_id in allowed and record.gt_2d_valid and record.box_xyxy_px is not None:
                key = (record.sequence_id, Path(record.image_path).name)
                box = tuple(map(float, record.box_xyxy_px))
                if key in self.box_by_image and self.box_by_image[key] != box:
                    raise ValueError(f"conflicting boxes for {key}")
                self.box_by_image[key] = box
        self.matches = [
            image_index.match(record["sequence_id"], record["query_time"])
            for record in lidar_dataset.records
        ]

    def __len__(self) -> int:
        return len(self.lidar_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = dict(self.lidar_dataset[index]); match = self.matches[index]
        key = None if match.path is None else (query["sequence_id"], match.path.name)
        box = None if key is None else self.box_by_image.get(key)
        query.update(
            left_image_path=None if match.path is None else str(match.path),
            image_time=match.image_time, image_query_gap_s=match.gap_s,
            calibration_handle=self.calibration_handle,
            m_R=bool(len(query["points"]) > 0), m_V=bool(match.valid and match.path is not None),
            gt_box_xyxy_px=torch.zeros(4) if box is None else torch.tensor(box),
            gt_2d_valid=box is not None,
        )
        query.update(prepare_image(query["left_image_path"], self.camera_wh, self.short_edge, self.max_size))
        return query


def collate_multimodal_v2(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    batch = collate_multimodal_queries(samples)
    batch["gt_box_xyxy_px"] = torch.stack([item["gt_box_xyxy_px"] for item in samples]).float()
    batch["gt_2d_valid"] = torch.tensor([bool(item["gt_2d_valid"]) for item in samples])
    shapes = {tuple(item["image_uint8"].shape) for item in samples}
    if len(shapes) != 1:
        raise ValueError(f"image views in a batch must share shape, got {sorted(shapes)}")
    for key in ("image_uint8", "image_source_wh", "image_view_wh", "image_scale_xy"):
        batch[key] = torch.stack([item[key] for item in samples])
    return batch


def expand_projection(base: ProjectionContext, scale_xy: torch.Tensor) -> ProjectionContext:
    count = len(scale_xy)
    return ProjectionContext(
        base.rotation_camera_from_radar.expand(count, -1, -1),
        base.translation_camera_from_radar_m.expand(count, -1),
        base.intrinsics.expand(count, -1), base.distortion.expand(count, -1),
        base.image_size_wh.expand(count, -1), scale_xy,
    )


def prepare_model_batch(batch: Mapping[str, Any], dino_detector: Any,
                        projection_base: ProjectionContext, device: torch.device):
    cpu_only = {"image_uint8", "image_view_wh"}
    moved = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) and key not in cpu_only else value
        for key, value in batch.items()
    }
    image_batch = batch["image_uint8"].to(device=device, dtype=torch.float32, non_blocking=True)
    inputs, transforms = [], []
    for index in range(len(image_batch)):
        source_wh = tuple(map(int, batch["image_source_wh"][index].tolist()))
        view_wh = tuple(map(int, batch["image_view_wh"][index].tolist()))
        inputs.append({"image": image_batch[index], "height": view_wh[1], "width": view_wh[0]})
        transforms.append(ViewTransform(source_wh, view_wh, False))
    images = dino_detector.preprocess_image(inputs)
    image_mask = torch.ones(
        (len(inputs), images.tensor.shape[-2], images.tensor.shape[-1]),
        dtype=torch.bool, device=device,
    )
    for index, (height, width) in enumerate(images.image_sizes):
        image_mask[index, :height, :width] = False
    projection = expand_projection(projection_base, moved["image_scale_xy"])
    context = make_interaction_context(
        batch["calibration_handle"], moved["m_R"], moved["m_V"], projection,
    )
    targets = MultimodalTargets(
        moved["target_xyz"], moved["target_valid"],
        moved["gt_box_xyxy_px"], moved["gt_2d_valid"],
    )
    return moved, images.tensor, image_mask, context, targets, transforms


def build_datasets(config: Mapping[str, Any], root: Path):
    data = config["data"]
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    dataset_root, split_path = resolve(data["root"]), resolve(data["split_file"])
    train_lidar = LiDARUAVDataset(dataset_root, split_path, data["train_split"], max_events=int(data["max_events"]))
    val_lidar = LiDARUAVDataset(dataset_root, split_path, data["val_split"], max_events=int(data["max_events"]))
    train_sequences = {record["sequence_id"] for record in train_lidar.records}
    val_sequences = {record["sequence_id"] for record in val_lidar.records}
    if train_sequences & val_sequences:
        raise RuntimeError("sequence-level train/validation leakage")
    geometry = json.loads(resolve(data["geometry_calibration"]).read_text())
    image_index = LeftImageIndex(dataset_root, time_offset_s=float(geometry["time_offset_s"]),
                                 max_abs_gap_s=float(data["max_image_gap_s"]))
    manifest = load_label_manifest(resolve(data["annotation_manifest"]), require_boxes=False)
    camera = yaml.safe_load(resolve(data["camera_config"]).read_text())
    args = dict(camera_wh=tuple(camera["cameras"]["left"]["resolution"]),
                short_edge=int(data["dino_short_edge"]), max_size=int(data["dino_max_size"]))
    train = MultimodalV2Dataset(train_lidar, image_index, manifest, train_sequences,
                                data["geometry_calibration"], **args)
    val = MultimodalV2Dataset(val_lidar, image_index, manifest, val_sequences,
                              data["geometry_calibration"], **args)
    return train_lidar, train, val
