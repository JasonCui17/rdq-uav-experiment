"""Source-faithful external baselines used by RDQ-UAV experiments."""

from .mmuav_preprocess import (
    MyLSTMClassifier,
    extract_feature_set_predict,
    farthest_point_sample,
    point_cloud_detector,
    process_fusion,
    process_lidar_360,
    process_lidar_livox,
    write_candidate_sidecar,
)

__all__ = [
    "MyLSTMClassifier",
    "extract_feature_set_predict",
    "farthest_point_sample",
    "point_cloud_detector",
    "process_fusion",
    "process_lidar_360",
    "process_lidar_livox",
    "write_candidate_sidecar",
]
