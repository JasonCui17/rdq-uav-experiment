"""Candidate-generation subset of the Multi-Modal-UAV LiDAR pipeline.

The tracking/Kalman/postprocess stages are intentionally omitted. The processing
constants and ordering below follow dtc111111/Multi-Modal-UAV at commit
f11b57390effbe9623ee2c7d561afddc8d0cdfa7.

The upstream MIT notice is retained in ``MULTI_MODAL_UAV_LICENSE.txt``.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SOURCE_REPO = "dtc111111/Multi-Modal-UAV"
SOURCE_COMMIT = "f11b57390effbe9623ee2c7d561afddc8d0cdfa7"


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """Original NumPy FPS, including its random first point."""
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def _dbscan_labels(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Import sklearn only when the non-dry processing path is actually used."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover - environment-dependent guard
        raise RuntimeError(
            "scikit-learn is required for the MMUAV processing path. "
            "Install the repository requirements before a full run."
        ) from exc
    return DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_


def _summary(values: list[int] | list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.quantile(array, 0.90)),
    }


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
def process_lidar_livox(
    livox_avia_data: Mapping[str, np.ndarray], output_folder: str | Path, max_pts: int = 100
) -> dict[str, Any]:
    """Process each Avia frame independently using original zero-filter + FPS."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    raw_counts: list[int] = []
    filtered_counts: list[int] = []
    processed_counts: list[int] = []
    fps_frames = 0
    for timestamp, data in livox_avia_data.items():
        raw_counts.append(int(data.shape[0]))
        mask = np.any(data != 0, axis=1)
        filtered_data = data[mask]
        filtered_counts.append(int(filtered_data.shape[0]))
        if filtered_data.shape[0] > max_pts:
            filtered_data = farthest_point_sample(filtered_data, max_pts)
            fps_frames += 1
        processed_counts.append(int(filtered_data.shape[0]))
        np.save(output_folder / f"{timestamp}.npy", filtered_data)
    return {
        "input_frames": len(livox_avia_data), "max_pts": max_pts,
        "raw_points_per_frame": _summary(raw_counts),
        "nonzero_points_per_frame": _summary(filtered_counts),
        "processed_points_per_frame": _summary(processed_counts),
        "fps_triggered_frames": fps_frames,
    }


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
def extract_feature_set_predict(
    data: np.ndarray,
    labels: np.ndarray,
    time_ind: np.ndarray,
    frame_indices: tuple[int, ...] = tuple(range(1, 21)),
) -> tuple[np.ndarray, np.ndarray]:
    """Build one [20,9] mean/std/range sequence per non-noise cluster.

    The upstream implementation iterates ``set(time_ind)``. Here the explicit 1..20
    list preserves its intended 20-frame input and its zero fill for frames in which
    a cluster has no point. This is required by the published 20-frame pipeline and
    does not add a new feature or decision rule.
    """
    unique_labels = set(labels)
    feature_set_list = []
    cluster_label_list = []
    for k in unique_labels:
        if k == -1:
            continue
        feature_set = np.array([])
        class_member_mask = labels == k
        for time in frame_indices:
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                xyz_std = np.std(masked_data, axis=0)
                xyz_range = np.max(masked_data, axis=0) - np.min(masked_data, axis=0)
            else:
                xyz_mean = np.zeros(3)
                xyz_std = np.zeros(3)
                xyz_range = np.zeros(3)
            feature = np.concatenate((xyz_mean, xyz_std, xyz_range), axis=0).reshape(1, -1)
            if np.size(feature_set) == 0:
                feature_set = feature
            else:
                feature_set = np.vstack([feature_set, feature])
        feature_set_list.append(feature_set)
        cluster_label_list.append(k)
    if not feature_set_list:
        return np.empty((0, len(frame_indices), 9)), np.empty((0, 1), dtype=np.int64)
    feature_set_all = np.stack(feature_set_list, axis=0)
    cluster_label_set_all = np.array(cluster_label_list).reshape(-1, 1)
    return feature_set_all, cluster_label_set_all


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
class MyLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

    def detect(self, feature_set: np.ndarray, model_path: str | Path) -> torch.Tensor:
        inputs = torch.tensor(feature_set, dtype=torch.float32)
        self.load_state_dict(torch.load(model_path))
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted


def load_original_checkpoint(model_path: str | Path) -> MyLSTMClassifier:
    """Validate the original 9→64→2 checkpoint without running inference."""
    model = MyLSTMClassifier(input_size=9, hidden_size=64, num_layers=1, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _accumulate_lidar_360_blocks(
    lidar_360_data: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Reproduce non-overlapping 20-frame blocks plus upstream final-20 block."""
    total_frame = len(lidar_360_data)
    accumulated_lidar_360_data: dict[str, np.ndarray] = {}
    accumulated_frame_timestamps: list[str] = []
    accumulated_all_timestamps: dict[str, list[str]] = {}
    accumulated_data = np.array([])

    for timestamp, data in lidar_360_data.items():
        accumulated_frame_timestamps.append(timestamp)
        mask = np.any(data != 0, axis=1)
        filtered_data = data[mask]
        frame_ind = len(accumulated_frame_timestamps) * np.ones([filtered_data.shape[0], 1])
        data_with_ind = np.concatenate((frame_ind, filtered_data), axis=1)
        if np.size(accumulated_data) == 0:
            accumulated_data = data_with_ind
        else:
            accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
        if len(accumulated_frame_timestamps) == 20:
            accumulated_lidar_360_data[accumulated_frame_timestamps[-1]] = np.array(accumulated_data)
            accumulated_all_timestamps[accumulated_frame_timestamps[-1]] = accumulated_frame_timestamps
            accumulated_frame_timestamps = []
            accumulated_data = np.array([])

    # SOURCE-FAITHFUL: the last 20 frames are processed as an extra block, not a
    # sliding window. If its key duplicates a prior block, dict assignment matches
    # the upstream overwrite behavior.
    accumulated_frame_timestamps = []
    accumulated_data = np.array([])
    for idx, (timestamp, data) in enumerate(lidar_360_data.items()):
        if idx >= total_frame - 20:
            accumulated_frame_timestamps.append(timestamp)
            mask = np.any(data != 0, axis=1)
            filtered_data = data[mask]
            frame_ind = len(accumulated_frame_timestamps) * np.ones([filtered_data.shape[0], 1])
            data_with_ind = np.concatenate((frame_ind, filtered_data), axis=1)
            if np.size(accumulated_data) == 0:
                accumulated_data = data_with_ind
            else:
                accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
            if len(accumulated_frame_timestamps) == 20:
                accumulated_lidar_360_data[accumulated_frame_timestamps[-1]] = np.array(accumulated_data)
                accumulated_all_timestamps[accumulated_frame_timestamps[-1]] = accumulated_frame_timestamps
    return accumulated_lidar_360_data, accumulated_all_timestamps


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
def process_lidar_360(
    lidar_360_data: Mapping[str, np.ndarray], output_folder: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    blocks, block_timestamps = _accumulate_lidar_360_blocks(lidar_360_data)
    model = MyLSTMClassifier(9, 64, 1, 2)
    raw_counts = [int(data.shape[0]) for data in lidar_360_data.values()]
    dbscan_clusters = 0
    positive_clusters = 0
    processed_nonempty = 0
    skipped_empty_blocks = 0

    for timestamp, data_with_ind in blocks.items():
        frame_time = block_timestamps[timestamp]
        point_cloud_data = {key: [] for key in frame_time}
        if data_with_ind.size == 0:
            skipped_empty_blocks += 1
        else:
            time_ind = data_with_ind[:, 0]
            data = data_with_ind[:, 1:]
            labels = _dbscan_labels(data, eps=2, min_samples=10)
            feature_set, cluster_labels = extract_feature_set_predict(data, labels, time_ind)
            dbscan_clusters += int(feature_set.shape[0])
            if feature_set.shape[0]:
                if feature_set.shape[1:] != (20, 9):
                    raise RuntimeError(f"Expected LSTM features [K,20,9], got {feature_set.shape}")
                predicted = model.detect(feature_set, model_path).cpu().numpy()
                predicted_labels = cluster_labels.reshape(-1)[predicted == 1]
                positive_clusters += int(len(predicted_labels))
                for predicted_label in predicted_labels:
                    det_data = data[labels == predicted_label]
                    det_time_ind = time_ind[labels == predicted_label].astype(int) - 1
                    for ind, frame_index in enumerate(det_time_ind):
                        point_cloud_data[frame_time[frame_index]].append(det_data[ind].tolist())
        for frame_name in frame_time:
            saved_pts = np.array(point_cloud_data[frame_name])
            np.save(output_folder / f"{frame_name}.npy", saved_pts)
            processed_nonempty += int(saved_pts.size != 0)
    return {
        "input_frames": len(lidar_360_data),
        "raw_points_per_frame": _summary(raw_counts),
        "accumulated_20_frame_blocks": len(blocks),
        "dbscan_eps": 2, "dbscan_min_samples": 10,
        "dbscan_clusters": dbscan_clusters,
        "lstm_positive_clusters": positive_clusters,
        "processed_output_frames": len(list(output_folder.glob("*.npy"))),
        "processed_nonempty_frames": processed_nonempty,
        "empty_blocks_skipped_safely": skipped_empty_blocks,
    }


def read_lidar_files(directory: str | Path) -> dict[str, np.ndarray]:
    """Equivalent of upstream read_lidar_files(), with numeric timestamp order."""
    paths = sorted(Path(directory).glob("*.npy"), key=lambda path: float(path.stem))
    return {path.stem: np.load(path, allow_pickle=False) for path in paths}


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
def process_fusion(
    livox_avia_data: Mapping[str, np.ndarray], lidar_360_data: Mapping[str, np.ndarray],
    output_folder: str | Path,
) -> dict[str, Any]:
    merged_dict = {**livox_avia_data, **lidar_360_data}
    lidar_data = {key: merged_dict[key] for key in sorted(merged_dict, key=lambda x: float(x))}
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    accumulated_timestamps = []
    accumulated_data = np.array([])
    for timestamp, data in lidar_data.items():
        if data.size != 0:
            accumulated_timestamps.append(timestamp)
            mask = np.any(data != 0, axis=1)
            filtered_data = data[mask]
            frame_ind = len(accumulated_timestamps) * np.ones([filtered_data.shape[0], 1])
            data_with_ind = np.concatenate((frame_ind, filtered_data), axis=1)
            if np.size(accumulated_data) == 0:
                accumulated_data = data_with_ind
            else:
                accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
    accumulated_data = np.array(accumulated_data)
    audit = {
        "input_livox_avia_frames": len(livox_avia_data),
        "input_lidar_360_frames": len(lidar_360_data),
        "timestamp_collisions_lidar360_overwrites_avia": len(set(livox_avia_data) & set(lidar_360_data)),
        "fusion_timestamps": len(lidar_data), "nonempty_input_timestamps": len(accumulated_timestamps),
        "accumulated_points": int(accumulated_data.shape[0]),
        "max_process_points_exclusive": 50000,
        "dbscan_eps": 1, "dbscan_min_samples": 10,
        "cannot_process": bool(accumulated_data.shape[0] >= 5e4),
        "warning": None, "nonempty_output_frames": 0,
    }
    if accumulated_data.shape[0] >= 5e4:
        audit["warning"] = "Cannot process: upstream accumulated_data.shape[0] >= 5e4 behavior retained"
        return audit
    if accumulated_data.size == 0:
        audit["warning"] = "No non-empty processed LiDAR input"
        return audit

    time_ind = accumulated_data[:, 0]
    data = accumulated_data[:, 1:]
    labels = _dbscan_labels(data, eps=1, min_samples=10)
    detections = accumulated_data[labels != -1]
    point_cloud_data = {key: [] for key in accumulated_timestamps}
    det_data = detections[:, 1:]
    det_time_ind = detections[:, 0].astype(int) - 1
    for ind, frame_index in enumerate(det_time_ind):
        point_cloud_data[accumulated_timestamps[frame_index]].append(det_data[ind].tolist())
    for frame_name in accumulated_timestamps:
        saved_pts = np.array(point_cloud_data[frame_name])
        np.save(output_folder / f"{frame_name}.npy", saved_pts)
        audit["nonempty_output_frames"] += int(saved_pts.size != 0)
    audit["dbscan_clusters"] = int(len(set(labels)) - (1 if -1 in labels else 0))
    audit["dbscan_noise_points"] = int(np.count_nonzero(labels == -1))
    audit["output_frames"] = len(accumulated_timestamps)
    return audit


# SOURCE-FAITHFUL:
# copied/adapted from Multi-Modal-UAV fusion_tracking.py
# commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7
def point_cloud_detector(filtered_data: np.ndarray, eps: float = 1, min_samples: int = 1) -> np.ndarray:
    labels = _dbscan_labels(filtered_data, eps=eps, min_samples=min_samples)
    unique_labels = set(labels)
    cluster_centers = []
    for k in unique_labels:
        if k == -1:
            continue
        class_mask = labels == k
        cluster_centers.append(np.mean(filtered_data[class_mask], axis=0))
    return np.array(cluster_centers)


def write_candidate_sidecar(
    fusion_data: Mapping[str, np.ndarray], output_folder: str | Path,
) -> dict[str, Any]:
    """Save original detector clusters plus point labels for later recall evaluation."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    cluster_counts: list[int] = []
    cluster_sizes_all: list[int] = []
    candidate_frames = 0
    for timestamp, points in fusion_data.items():
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float64)
            labels = np.empty((0,), dtype=np.int64)
            centers = np.empty((0, 3), dtype=np.float64)
            cluster_sizes = np.empty((0,), dtype=np.int64)
        else:
            points = np.asarray(points)
            labels = _dbscan_labels(points, eps=1, min_samples=1)
            unique_labels = [label for label in set(labels) if label != -1]
            centers = np.asarray([np.mean(points[labels == label], axis=0) for label in unique_labels])
            cluster_sizes = np.asarray([np.count_nonzero(labels == label) for label in unique_labels])
            candidate_frames += 1
        cluster_counts.append(int(len(centers)))
        cluster_sizes_all.extend(cluster_sizes.tolist())
        np.savez_compressed(
            output_folder / f"{timestamp}.npz", points=points, labels=labels,
            centers=centers, cluster_sizes=cluster_sizes,
        )
    return {
        "candidate_frames": candidate_frames,
        "output_frames": len(fusion_data),
        "detector_eps": 1, "detector_min_samples": 1,
        "clusters_per_frame": _summary(cluster_counts),
        "points_per_cluster": _summary(cluster_sizes_all),
    }
