"""V2 data boundary: query -> Sample -> packed Batch -> model inputs.

Flat fields keep the existing detector interfaces. Radar: points/delta_t/
sensor_id/m_R. Vision: image_uint8/vision_delta_t/m_V and resize dimensions.
Target: target_xyz/target_valid/gt_box_xyxy_px/gt_2d_valid. All remaining
identity, absolute time and file fields are audit metadata, not model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml

from rdq_uav.multimodal.merged_lidar import LidarFrameEvent, load_released_xyz, merge_frame_streams
from .geometry import ProjectionContext
from .loss import MultimodalTargets

@dataclass(frozen=True)
class ViewTransform:
    """Invertible resize + optional horizontal flip from source camera pixels."""

    source_wh: tuple[int, int]
    view_wh: tuple[int, int]
    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        sw, sh = self.source_wh
        vw, vh = self.view_wh
        if min(sw, sh, vw, vh) <= 0:
            raise ValueError("source/view dimensions must be positive")

    @property
    def scale_xy(self) -> tuple[float, float]:
        sw, sh = self.source_wh
        vw, vh = self.view_wh
        return vw / sw, vh / sh

    def source_points_to_view(self, points_xy: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points_xy)
        if points.shape[-1] != 2:
            raise ValueError("points must end in XY")
        sx, sy = self.scale_xy
        out = points * points.new_tensor((sx, sy))
        if self.horizontal_flip:
            out = torch.stack((out.new_tensor(float(self.view_wh[0])) - out[..., 0], out[..., 1]), dim=-1)
        return out

    def source_boxes_to_view(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        boxes = torch.as_tensor(boxes_xyxy)
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must end in XYXY")
        sx, sy = self.scale_xy
        x1, y1, x2, y2 = (boxes * boxes.new_tensor((sx, sy, sx, sy))).unbind(-1)
        if self.horizontal_flip:
            width = boxes.new_tensor(float(self.view_wh[0]))
            x1, x2 = width - x2, width - x1
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def view_boxes_to_source(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        boxes = torch.as_tensor(boxes_xyxy)
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must end in XYXY")
        x1, y1, x2, y2 = boxes.unbind(-1)
        if self.horizontal_flip:
            width = boxes.new_tensor(float(self.view_wh[0]))
            x1, x2 = width - x2, width - x1
        sx, sy = self.scale_xy
        return torch.stack((x1 / sx, y1 / sy, x2 / sx, y2 / sy), dim=-1)


@dataclass(frozen=True)
class ImageLabelRecord:
    sequence_id: str
    image_path: str
    box_xyxy_px: tuple[float, ...] | None
    gt_2d_valid: bool


def load_label_manifest(path: str | Path, *, require_boxes: bool = True) -> list[ImageLabelRecord]:
    """Read image supervision only; query UID/3D GT do not define 2D identity."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            valid = bool(item.get("gt_2d_valid", False))
            raw = item.get("box_xyxy_px")
            box = None if raw is None else tuple(map(float, raw))
            if box is not None and (len(box) != 4 or not np.isfinite(box).all()
                                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError(f"invalid box at manifest line {line_number}")
            if valid and box is None:
                raise ValueError(f"valid label has no box at manifest line {line_number}")
            if require_boxes and not valid:
                raise ValueError(f"missing verified box at manifest line {line_number}")
            seq, image_path = str(item["sequence_id"]), str(item["image_path"])
            if not seq or not image_path:
                raise ValueError(f"empty image label identity at line {line_number}")
            records.append(ImageLabelRecord(seq, image_path, box, valid))
    return records


SENSORS = ((0, "Avia", "livox_avia"), (1, "Mid360", "lidar_360"))


def _timestamp_paths(directory: Path, suffix: str) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.glob(f"*.{suffix}"), key=lambda p: (float(p.stem), str(p))))
    if any(not np.isfinite(float(p.stem)) for p in paths):
        raise ValueError(f"non-finite filename timestamp in {directory}")
    return paths


@dataclass(frozen=True)
class LeftImageMatch:
    sequence_id: str
    path: Path | None
    image_time: float | None
    delta_t: float | None
    valid: bool


class LeftImageIndex:
    """Sequence-local latest historical frame within [query_time-gap, query_time]."""
    def __init__(self, root: str | Path, sequence_ids: Sequence[str], max_image_gap_s: float = 1.0):
        self.max_image_gap_s = float(max_image_gap_s)
        if not np.isfinite(self.max_image_gap_s) or self.max_image_gap_s < 0:
            raise ValueError("max_image_gap_s must be finite and non-negative")
        self.image_index_by_sequence = {}
        for seq in sequence_ids:
            paths = _timestamp_paths(Path(root)/seq/"Image", "png")
            self.image_index_by_sequence[seq] = (np.asarray([float(p.stem) for p in paths]), paths)

    def match(self, sequence_id: str, query_time: float) -> LeftImageMatch:
        if not np.isfinite(query_time):
            raise ValueError("query_time must be finite")
        times, paths = self.image_index_by_sequence[sequence_id]
        if not len(times):
            return LeftImageMatch(sequence_id, None, None, None, False)
        # right includes a frame exactly at query_time; future frames are never candidates.
        index = int(np.searchsorted(times, query_time, side="right"))-1
        if index < 0:
            return LeftImageMatch(sequence_id, None, None, None, False)
        image_time = float(times[index])
        delta_t = image_time-query_time
        valid = query_time-self.max_image_gap_s <= image_time <= query_time
        # Out-of-range historical frame is deliberately not an input or GT key.
        return LeftImageMatch(sequence_id, paths[index] if valid else None,
                              image_time if valid else None, delta_t if valid else None, valid)


def resize_wh(source_wh: tuple[int, int], short_edge: int, max_size: int) -> tuple[int, int]:
    width, height = source_wh
    if min(width, height, short_edge, max_size) <= 0:
        raise ValueError("image dimensions and resize limits must be positive")
    scale = min(float(short_edge)/min(width, height), float(max_size)/max(width, height))
    return int(width*scale+0.5), int(height*scale+0.5)


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
        "image_scale_xy": torch.tensor((view_wh[0]/source_wh[0], view_wh[1]/source_wh[1])),
    }


def validate_sample(sample: Mapping[str, Any]) -> None:
    """Check identities, inclusive time window, shapes and single-source validity."""
    seq, tq = sample["sequence_id"], float(sample["query_time"])
    history, gap = float(sample["radar_history_s"]), float(sample["max_image_gap_s"])
    if not seq or not np.isfinite(tq):
        raise ValueError("invalid query identity")
    times, sequences = sample["event_timestamps"], sample["event_sequence_ids"]
    if len(times) != len(sequences) or len(times) != sample["event_count"]:
        raise ValueError("event metadata length mismatch")
    if any(s != seq for s in sequences):
        raise ValueError("radar event sequence mismatch")
    if any(not np.isfinite(t) or not tq-history <= t <= tq for t in times):
        raise ValueError("radar event outside causal history window")
    points = sample["points"]
    n = len(points)
    if points.shape != (n, 3) or not torch.isfinite(points).all():
        raise ValueError("points must be finite [N,3]")
    for key in ("delta_t", "sensor_id", "supervision_recent_mask"):
        if sample[key].shape != (n,):
            raise ValueError(f"{key} must be [N]")
    dt = sample["delta_t"]
    # Float32 conversion only: do not expand the event selection window.
    tolerance = np.finfo(np.float32).eps*max(1.0, history)
    if not torch.isfinite(dt).all() or bool(((dt > 0) | (dt < -history-tolerance)).any()):
        raise ValueError("point relative time outside causal history window")
    if sample["sensor_id"].dtype != torch.long or bool(((sample["sensor_id"] < 0) | (sample["sensor_id"] > 1)).any()):
        raise ValueError("sensor_id must be long with 0=Avia, 1=Mid360")
    if bool(sample["m_R"]) != (n > 0):
        raise ValueError("m_R must equal nonempty valid points")
    if sample["m_V"]:
        it = sample["image_time"]
        if sample["image_sequence_id"] != seq or sample["left_image_path"] is None:
            raise ValueError("image sequence/path mismatch")
        if it is None or not np.isfinite(it) or not tq-gap <= it <= tq:
            raise ValueError("valid image outside causal history window")
        if sample["vision_delta_t"] != it-tq:
            raise ValueError("vision relative time mismatch")
    elif sample["left_image_path"] is not None or sample["image_time"] is not None or sample["gt_2d_valid"]:
        raise ValueError("invalid image must not carry an input path/time or valid box")
    if sample["target_xyz"].shape != (3,) or sample["gt_box_xyxy_px"].shape != (4,):
        raise ValueError("target shapes must be [3]/[4]")
    if sample["target_valid"] and (not torch.isfinite(sample["target_xyz"]).all() or sample["target_timestamp"] != tq):
        raise ValueError("valid XYZ target must be finite and refer to query_time")
    if sample["gt_2d_valid"]:
        box = sample["gt_box_xyxy_px"]
        if not torch.isfinite(box).all() or bool((box[2:] <= box[:2]).any()):
            raise ValueError("valid box must be finite, non-degenerate source-pixel xyxy")


class MultimodalV2Dataset(Dataset):
    """Construct a complete Sample directly from each unique query_time.

    Query records currently originate from sequence ground_truth filenames.
    Train-only empty-input filtering happens here at initialization, including
    events whose released XYZ rows are all invalid. Cached validity is one bool
    per event; complete clouds are not retained or capped.
    """
    def __init__(self, root: str | Path, sequence_ids: Sequence[str], manifest: Sequence[Any], *,
                 camera_wh: tuple[int, int], short_edge: int, max_size: int,
                 radar_history_s: float = 1.0, max_image_gap_s: float = 1.0,
                 filter_empty: bool = False) -> None:
        self.root = Path(root)
        self.camera_wh = tuple(map(int, camera_wh))
        self.short_edge, self.max_size = int(short_edge), int(max_size)
        self.radar_history_s = float(radar_history_s)
        self.max_image_gap_s = float(max_image_gap_s)
        if not np.isfinite(self.radar_history_s) or self.radar_history_s < 0:
            raise ValueError("radar_history_s must be finite and non-negative")
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("duplicate sequence in split")
        self.image_index = LeftImageIndex(self.root, sequence_ids, max_image_gap_s)
        self.image_index_by_sequence = self.image_index.image_index_by_sequence
        self.radar_events_by_sequence = {}
        self.radar_times_by_sequence = {}
        self.box_by_image = {}
        for record in manifest:
            if record.sequence_id not in sequence_ids or not record.gt_2d_valid or record.box_xyxy_px is None:
                continue
            key = (record.sequence_id, Path(record.image_path).name)
            box = tuple(map(float, record.box_xyxy_px))
            if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"invalid box for {key}")
            if key in self.box_by_image and self.box_by_image[key] != box:
                raise ValueError(f"conflicting boxes for {key}")
            self.box_by_image[key] = box
        records = []
        seen_times = set()
        for seq in sequence_ids:
            events = merge_frame_streams([
                [LidarFrameEvent(seq, float(p.stem), sid, name, p)
                 for p in _timestamp_paths(self.root/seq/directory, "npy")]
                for sid, name, directory in SENSORS])
            self.radar_events_by_sequence[seq] = events
            self.radar_times_by_sequence[seq] = np.asarray([e.timestamp for e in events], dtype=np.float64)
            for path in _timestamp_paths(self.root/seq/"ground_truth", "npy"):
                tq = float(path.stem)
                if tq in seen_times:
                    raise ValueError(f"duplicate query_time {tq}; query_time must uniquely identify a Sample")
                seen_times.add(tq)
                records.append(dict(sequence_id=seq, query_time=tq, target_path=path,
                                    sample_id=f"{seq}_query_{path.stem}"))
        self.query_records = []
        self.filtered_empty_queries = 0
        self._event_validity = {}
        for record in records:
            seq, tq = record["sequence_id"], record["query_time"]
            match = self.image_index.match(seq, tq)
            if filter_empty and not match.valid and not self._has_valid_radar(seq, tq):
                self.filtered_empty_queries += 1
                continue
            self.query_records.append(record)
        # records remains an audit alias used by existing tooling, no wrapper Dataset.
        self.records = self.query_records

    def select_radar_events(self, sequence_id: str, query_time: float) -> list[LidarFrameEvent]:
        if not np.isfinite(query_time):
            raise ValueError("query_time must be finite")
        times = self.radar_times_by_sequence[sequence_id]
        start = int(np.searchsorted(times, query_time-self.radar_history_s, side="left"))
        stop = int(np.searchsorted(times, query_time, side="right"))
        events = self.radar_events_by_sequence[sequence_id][start:stop]
        if any(e.sequence_id != sequence_id or not query_time-self.radar_history_s <= e.timestamp <= query_time for e in events):
            raise ValueError("radar index sequence/window mismatch")
        return events

    def _has_valid_radar(self, sequence_id: str, query_time: float) -> bool:
        for event in self.select_radar_events(sequence_id, query_time):
            if event.file_path not in self._event_validity:
                self._event_validity[event.file_path] = bool(len(load_released_xyz(event.file_path)[0]))
            if self._event_validity[event.file_path]:
                return True
        return False

    def __len__(self) -> int:
        return len(self.query_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.query_records[index]
        seq, tq = record["sequence_id"], record["query_time"]
        events = self.select_radar_events(seq, tq)
        parts, times, sensors, recent = [], [], [], []
        for i, event in enumerate(events):
            points = load_released_xyz(event.file_path)[0].astype(np.float32)
            parts.append(points)
            times.append(np.full(len(points), event.timestamp-tq, dtype=np.float32))
            sensors.append(np.full(len(points), event.sensor_id, dtype=np.int64))
            # Preserve the frozen LiDAR diagnostic loss's latest-four semantics.
            # This mask never limits observation selection or enters the encoder.
            recent.append(np.full(len(points), i >= max(0, len(events)-4), dtype=bool))
        def packed(values, shape, dtype):
            return torch.from_numpy(np.concatenate(values) if values else np.empty(shape, dtype=dtype))
        points = packed(parts, (0, 3), np.float32)
        match = self.image_index.match(seq, tq)
        box = None if match.path is None else self.box_by_image.get((seq, match.path.name))
        xyz = torch.as_tensor(np.load(record["target_path"], allow_pickle=False).reshape(3), dtype=torch.float32)
        sample = {
            # Radar
            "points": points, "delta_t": packed(times, (0,), np.float32),
            "sensor_id": packed(sensors, (0,), np.int64), "m_R": bool(len(points)),
            # Vision; missing relative time is zero storage masked by m_V.
            "m_V": match.valid, "vision_delta_t": 0.0 if match.delta_t is None else match.delta_t,
            # Target
            "target_xyz": xyz, "target_valid": bool(torch.isfinite(xyz).all()),
            "target_timestamp": tq, "gt_box_xyxy_px": torch.zeros(4) if box is None else torch.tensor(box),
            "gt_2d_valid": box is not None,
            # Diagnostic supervision only, not an observation feature.
            "supervision_recent_mask": packed(recent, (0,), bool),
            # Meta
            "sequence_id": seq, "query_time": tq, "sample_id": record["sample_id"],
            "num_samples": 1, "event_count": len(events),
            "event_timestamps": [e.timestamp for e in events],
            "event_sequence_ids": [e.sequence_id for e in events],
            "image_time": match.image_time, "left_image_path": None if match.path is None else str(match.path),
            "image_sequence_id": seq if match.valid else None,
            "radar_history_s": self.radar_history_s, "max_image_gap_s": self.max_image_gap_s,
        }
        sample.update(prepare_image(match.path, self.camera_wh, self.short_edge, self.max_size))
        validate_sample(sample)
        return sample


# Batch construction: no identity deduplication or fixed-point padding.
def collate_multimodal_v2(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    for sample in samples:
        validate_sample(sample)
    n = len(samples)
    counts = torch.tensor([len(s["points"]) for s in samples], dtype=torch.long)
    batch = {k: torch.cat([s[k] for s in samples]) for k in
             ("points", "delta_t", "sensor_id", "supervision_recent_mask")}
    batch.update(num_samples=n, point_counts=counts,
                 point_batch_index=torch.repeat_interleave(torch.arange(n), counts))
    for key in ("m_R", "m_V", "target_valid", "gt_2d_valid"):
        batch[key] = torch.tensor([bool(s[key]) for s in samples], dtype=torch.bool)
    shapes = {tuple(s["image_uint8"].shape) for s in samples}
    if len(shapes) != 1:
        raise ValueError(f"image views in a batch must share shape, got {sorted(shapes)}")
    for key in ("target_xyz", "gt_box_xyxy_px", "image_uint8", "image_source_wh", "image_view_wh", "image_scale_xy"):
        batch[key] = torch.stack([s[key] for s in samples])
    batch["vision_delta_t"] = torch.tensor([s["vision_delta_t"] for s in samples], dtype=torch.float32)
    for key in ("query_time", "target_timestamp", "image_time"):
        batch[key] = torch.tensor([float('nan') if s[key] is None else s[key] for s in samples], dtype=torch.float64)
    batch["event_count"] = torch.tensor([s["event_count"] for s in samples], dtype=torch.long)
    for key in ("sequence_id", "sample_id", "left_image_path", "event_timestamps", "event_sequence_ids", "image_sequence_id"):
        batch[key] = [s[key] for s in samples]
    return batch


def expand_projection(base: ProjectionContext, scale_xy: torch.Tensor) -> ProjectionContext:
    count = len(scale_xy)
    result = ProjectionContext(
        base.rotation_camera_from_radar.expand(count, -1, -1),
        base.translation_camera_from_radar_m.expand(count, -1),
        base.intrinsics.expand(count, -1), base.distortion.expand(count, -1),
        base.image_size_wh.expand(count, -1), scale_xy)
    result.validate(count)
    return result


# Model preparation: absolute timestamps, identities and paths never cross this boundary.
def prepare_model_batch(batch: Mapping[str, Any], dino_detector: Any,
                        projection_base: ProjectionContext, device: torch.device):
    model_keys = ("points", "delta_t", "sensor_id", "supervision_recent_mask", "point_counts",
                  "point_batch_index", "m_R", "m_V", "vision_delta_t", "target_xyz", "target_valid",
                  "gt_box_xyxy_px", "gt_2d_valid", "image_source_wh", "image_scale_xy")
    moved = {key: batch[key].to(device, non_blocking=True) for key in model_keys}
    moved["num_samples"] = int(batch["num_samples"])
    # validate_batch currently checks the image field, but detectors use preprocessed images below.
    moved["image_uint8"] = batch["image_uint8"]
    image_batch = batch["image_uint8"].to(device=device, dtype=torch.float32, non_blocking=True)
    inputs, transforms = [], []
    for index in range(len(image_batch)):
        source_wh = tuple(map(int, batch["image_source_wh"][index].tolist()))
        view_wh = tuple(map(int, batch["image_view_wh"][index].tolist()))
        inputs.append({"image": image_batch[index], "height": view_wh[1], "width": view_wh[0]})
        transforms.append(ViewTransform(source_wh, view_wh, False))
    images = dino_detector.preprocess_image(inputs)
    image_mask = torch.ones((len(inputs), images.tensor.shape[-2], images.tensor.shape[-1]),
                            dtype=torch.bool, device=device)
    for index, (height, width) in enumerate(images.image_sizes):
        # Padding mask describes layout only. Keep finite DINO execution for
        # placeholders; m_V gates candidates/evidence in the model.
        image_mask[index, :height, :width] = False
    projection = expand_projection(projection_base, moved["image_scale_xy"])
    targets = MultimodalTargets(moved["target_xyz"], moved["target_valid"],
                                moved["gt_box_xyxy_px"], moved["gt_2d_valid"])
    return moved, images.tensor, image_mask, projection, targets, transforms


def build_datasets(config: Mapping[str, Any], root: Path):
    data = config["data"]
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root/path
    dataset_root = resolve(data["root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MMAUD root missing: {dataset_root}")
    splits = json.loads(resolve(data["split_file"]).read_text())
    train_sequences, val_sequences = splits[data["train_split"]], splits[data["val_split"]]
    if set(train_sequences) & set(val_sequences):
        raise RuntimeError("sequence-level train/validation leakage")
    geometry = json.loads(resolve(data["geometry_calibration"]).read_text())
    # This Sample contract uses raw image_time-query_time, not an implicit clock correction.
    # Current audited calibration is zero. A nonzero offset requires an explicit new contract.
    if float(geometry.get("time_offset_s", 0.0)) != 0.0:
        raise ValueError("raw historical-image contract requires time_offset_s=0; nonzero offset needs explicit review")
    manifest = load_label_manifest(resolve(data["annotation_manifest"]), require_boxes=False)
    camera = yaml.safe_load(resolve(data["camera_config"]).read_text())
    args = dict(camera_wh=tuple(camera["cameras"]["left"]["resolution"]),
                short_edge=int(data["dino_short_edge"]), max_size=int(data["dino_max_size"]),
                radar_history_s=float(data.get("radar_history_s", 1.0)),
                max_image_gap_s=float(data.get("max_image_gap_s", 1.0)))
    train = MultimodalV2Dataset(dataset_root, train_sequences, manifest, filter_empty=True, **args)
    val = MultimodalV2Dataset(dataset_root, val_sequences, manifest, filter_empty=False, **args)
    train_times = {r["query_time"] for r in train.query_records}
    if any(r["query_time"] in train_times for r in val.query_records):
        raise ValueError("duplicate query_time across train/validation")
    return train, val
