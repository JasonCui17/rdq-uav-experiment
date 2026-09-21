"""Multimodal query binding with explicit time and modality contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rdq_uav.lidar_v2.data import assert_query_integrity, collate_lidar_samples


QueryKey = tuple[str, object]


@dataclass(frozen=True)
class LeftImageMatch:
    sequence_id: str
    path: Path | None
    image_time: float | None
    query_time: float
    gap_s: float | None
    valid: bool


class LeftImageIndex:
    """Sequence-local nearest left-image matching with calibrated clock offset.

    The project convention is ``query_time = image_time + time_offset_s``;
    consequently matching minimizes
    ``abs(query_time - (image_time + time_offset_s))``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        time_offset_s: float,
        image_directory: str = "Image",
        max_abs_gap_s: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.time_offset_s = float(time_offset_s)
        self.image_directory = image_directory
        self.max_abs_gap_s = (
            None if max_abs_gap_s is None else float(max_abs_gap_s)
        )
        if not np.isfinite(self.time_offset_s):
            raise ValueError("time_offset_s must be finite")
        if self.max_abs_gap_s is not None and self.max_abs_gap_s < 0:
            raise ValueError("max_abs_gap_s must be non-negative")
        self._cache: dict[str, tuple[np.ndarray, tuple[Path, ...]]] = {}

    def _sequence(self, sequence_id: str) -> tuple[np.ndarray, tuple[Path, ...]]:
        if not sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if sequence_id not in self._cache:
            paths = tuple(
                sorted(
                    (self.root / sequence_id / self.image_directory).glob("*.png"),
                    key=lambda path: (float(path.stem), str(path)),
                )
            )
            times = np.asarray([float(path.stem) for path in paths], dtype=np.float64)
            self._cache[sequence_id] = (times, paths)
        return self._cache[sequence_id]

    def match(self, sequence_id: str, query_time: float) -> LeftImageMatch:
        query_time = float(query_time)
        if not np.isfinite(query_time):
            raise ValueError("query_time must be finite")
        times, paths = self._sequence(sequence_id)
        if len(times) == 0:
            return LeftImageMatch(sequence_id, None, None, query_time, None, False)
        corrected_times = times + self.time_offset_s
        insertion = int(np.searchsorted(corrected_times, query_time))
        candidates = {max(0, insertion - 1), min(len(times) - 1, insertion)}
        index = min(
            candidates,
            key=lambda item: (abs(query_time - corrected_times[item]), item),
        )
        gap = query_time - float(corrected_times[index])
        valid = self.max_abs_gap_s is None or abs(gap) <= self.max_abs_gap_s
        return LeftImageMatch(
            sequence_id=sequence_id,
            path=paths[index] if valid else None,
            image_time=float(times[index]) if valid else None,
            query_time=query_time,
            gap_s=float(gap) if valid else None,
            valid=valid,
        )


def query_key(record: Mapping[str, Any]) -> QueryKey:
    sequence_id = record.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("query record requires a non-empty sequence_id")
    if "query_uid" not in record:
        raise KeyError("query record requires stable query_uid")
    return sequence_id, record["query_uid"]


def deduplicate_multimodal_queries(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], torch.Tensor]:
    """Stable within-batch UQP mapping keyed by ``(sequence_id, query_uid)``.

    This only constructs identities.  It does not cache features across a
    forward or optimizer update.
    """

    unique: list[Mapping[str, Any]] = []
    key_to_index: dict[QueryKey, int] = {}
    inverse: list[int] = []
    signatures: dict[QueryKey, tuple[float, str | None]] = {}
    for record in records:
        key = query_key(record)
        signature = (
            float(record["query_time"]),
            None
            if record.get("left_image_path") is None
            else str(record["left_image_path"]),
        )
        if key not in key_to_index:
            key_to_index[key] = len(unique)
            signatures[key] = signature
            unique.append(record)
        elif signatures[key] != signature:
            raise ValueError(
                f"conflicting records for stable multimodal query key {key}: "
                f"{signatures[key]} != {signature}"
            )
        inverse.append(key_to_index[key])
    return unique, torch.tensor(inverse, dtype=torch.long)


class MultimodalQueryDataset(Dataset):
    """Bind each causal LiDAR query to its nearest left RGB reference."""

    def __init__(
        self,
        lidar_dataset: Dataset,
        image_index: LeftImageIndex,
        *,
        calibration_handle: str | Path,
    ) -> None:
        self.lidar_dataset = lidar_dataset
        self.image_index = image_index
        self.calibration_handle = str(calibration_handle)

    def __len__(self) -> int:
        return len(self.lidar_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = dict(self.lidar_dataset[index])
        assert_query_integrity(query, require_events=True)
        match = self.image_index.match(query["sequence_id"], query["query_time"])
        query.update(
            left_image_path=None if match.path is None else str(match.path),
            image_time=match.image_time,
            image_query_gap_s=match.gap_s,
            calibration_handle=self.calibration_handle,
            m_R=bool(len(query["points"]) > 0),
            m_V=match.valid,
        )
        return query


def collate_multimodal_queries(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pack unique multimodal queries while retaining image references/masks."""

    batch = collate_lidar_samples(samples)
    batch.update(
        left_image_path=[sample.get("left_image_path") for sample in samples],
        image_time=torch.tensor(
            [
                float("nan")
                if sample.get("image_time") is None
                else float(sample["image_time"])
                for sample in samples
            ],
            dtype=torch.float64,
        ),
        image_query_gap_s=torch.tensor(
            [
                float("nan")
                if sample.get("image_query_gap_s") is None
                else float(sample["image_query_gap_s"])
                for sample in samples
            ],
            dtype=torch.float64,
        ),
        calibration_handle=[sample["calibration_handle"] for sample in samples],
        m_R=torch.tensor([bool(sample["m_R"]) for sample in samples]),
        m_V=torch.tensor([bool(sample["m_V"]) for sample in samples]),
    )
    return batch
