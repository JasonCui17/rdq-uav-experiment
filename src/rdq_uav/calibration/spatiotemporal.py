from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .omni import OmniRadtanCamera, transform_points
from .trajectory import PositionTrajectory


@dataclass(frozen=True)
class CenterObservation:
    sequence_id: str
    camera: str
    image_time: float
    u: float
    v: float
    weight: float = 1.0


@dataclass
class SpatiotemporalSolution:
    rotation_camera_from_gt: dict[str, np.ndarray]
    translation_camera_from_gt: dict[str, np.ndarray]
    time_offset_s: float
    residuals_px: np.ndarray
    success: bool
    message: str

    def summary(self) -> dict:
        residual_norm = np.linalg.norm(self.residuals_px.reshape(-1, 2), axis=1)
        return {
            "success": self.success,
            "message": self.message,
            "time_convention": "gt_query_time = image_time + time_offset_s",
            "time_offset_s": float(self.time_offset_s),
            "mean_reprojection_error_px": float(residual_norm.mean()),
            "median_reprojection_error_px": float(np.median(residual_norm)),
            "p95_reprojection_error_px": float(np.percentile(residual_norm, 95)),
            "max_reprojection_error_px": float(residual_norm.max()),
            "observation_count": int(len(residual_norm)),
            "cameras": {
                name: {
                    "rotation_camera_from_gt": rotation.tolist(),
                    "translation_camera_from_gt_m": self.translation_camera_from_gt[name].tolist(),
                    "camera_center_in_gt_m": (
                        -rotation.T @ self.translation_camera_from_gt[name]
                    ).tolist(),
                }
                for name, rotation in self.rotation_camera_from_gt.items()
            },
        }


def reprojection_errors(
    observations: Sequence[CenterObservation],
    trajectories: Mapping[str, PositionTrajectory],
    cameras: Mapping[str, OmniRadtanCamera],
    rotations: Mapping[str, np.ndarray],
    translations: Mapping[str, np.ndarray],
    time_offset_s: float,
) -> np.ndarray:
    errors = np.full((len(observations), 2), np.nan, dtype=np.float64)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, observation in enumerate(observations):
        groups.setdefault((observation.sequence_id, observation.camera), []).append(index)
    for (sequence_id, camera_name), indices_list in groups.items():
        indices = np.asarray(indices_list, dtype=int)
        query_times = np.asarray(
            [observations[index].image_time for index in indices], dtype=np.float64
        ) + time_offset_s
        points_gt, valid_time = trajectories[sequence_id].evaluate(query_times)
        points_camera = transform_points(
            points_gt, rotations[camera_name], translations[camera_name]
        )
        pixels, valid_projection = cameras[camera_name].project(points_camera)
        observed = np.asarray(
            [[observations[index].u, observations[index].v] for index in indices],
            dtype=np.float64,
        )
        valid = valid_time & valid_projection
        errors[indices[valid]] = pixels[valid] - observed[valid]
    return errors


def fit_spatiotemporal_calibration(
    observations: Sequence[CenterObservation],
    trajectories: Mapping[str, PositionTrajectory],
    cameras: Mapping[str, OmniRadtanCamera],
    initial_rotation: Mapping[str, np.ndarray],
    initial_translation: Mapping[str, np.ndarray],
    initial_time_offset_s: float = 0.0,
    max_time_offset_s: float = 0.25,
    max_translation_m: float = 2.0,
    robust_scale_px: float = 5.0,
) -> SpatiotemporalSolution:
    """Jointly refine per-camera SE(3) extrinsics and one clock offset.

    The clock convention is ``p_gt(image_time + offset)``. At least six
    spatially diverse observations per camera are required; dozens spanning
    changes in direction and speed are strongly preferred.
    """
    camera_names = tuple(cameras)
    if set(camera_names) != set(initial_rotation) or set(camera_names) != set(initial_translation):
        raise ValueError("Initial transforms must be supplied for every camera")
    counts = {name: sum(obs.camera == name for obs in observations) for name in camera_names}
    if any(count < 6 for count in counts.values()):
        raise ValueError(f"Need at least six observations per camera, got {counts}")

    initial = []
    for name in camera_names:
        initial.extend(Rotation.from_matrix(np.asarray(initial_rotation[name])).as_rotvec())
        initial.extend(np.asarray(initial_translation[name], dtype=np.float64).reshape(3))
    # Do not optimize the offset directly around zero. SciPy's relative finite
    # difference step would then be smaller than the ULP of ~1.7e9 Unix epoch
    # timestamps, producing an all-zero time Jacobian. Keep the parameter near
    # one and map one parameter unit to 100 seconds; its numerical step is then
    # resolvable while the explicit bounds still enforce max_time_offset_s.
    time_parameter_scale_s = 100.0
    initial.append(1.0 + float(initial_time_offset_s) / time_parameter_scale_s)
    initial = np.asarray(initial, dtype=np.float64)

    # Pre-group observations so every optimizer evaluation performs only one
    # vectorized interpolation/projection call per sequence-camera pair.
    grouped: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    group_indices: dict[tuple[str, str], list[int]] = {}
    for index, observation in enumerate(observations):
        group_indices.setdefault((observation.sequence_id, observation.camera), []).append(index)
    for key, indices_list in group_indices.items():
        indices = np.asarray(indices_list, dtype=int)
        grouped[key] = {
            "indices": indices,
            "times": np.asarray(
                [observations[index].image_time for index in indices], dtype=np.float64
            ),
            "pixels": np.asarray(
                [[observations[index].u, observations[index].v] for index in indices],
                dtype=np.float64,
            ),
            "scales": np.sqrt(
                np.maximum(
                    np.asarray(
                        [observations[index].weight for index in indices], dtype=np.float64
                    ),
                    1e-6,
                )
            ),
        }

    def unpack(parameters: np.ndarray):
        rotations: dict[str, np.ndarray] = {}
        translations: dict[str, np.ndarray] = {}
        for index, name in enumerate(camera_names):
            start = index * 6
            rotations[name] = Rotation.from_rotvec(parameters[start : start + 3]).as_matrix()
            translations[name] = parameters[start + 3 : start + 6]
        offset = (float(parameters[-1]) - 1.0) * time_parameter_scale_s
        return rotations, translations, offset

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotations, translations, offset = unpack(parameters)
        result = np.full((len(observations), 2), 500.0, dtype=np.float64)
        for (sequence_id, camera_name), group in grouped.items():
            points_gt, valid_time = trajectories[sequence_id].evaluate(group["times"] + offset)
            points_camera = transform_points(
                points_gt, rotations[camera_name], translations[camera_name]
            )
            pixels, valid_projection = cameras[camera_name].project(points_camera)
            valid = valid_time & valid_projection
            indices = group["indices"][valid]
            result[indices] = (
                group["scales"][valid, None] * (pixels[valid] - group["pixels"][valid])
            )
        return result.reshape(-1)

    lower = np.full_like(initial, -np.inf)
    upper = np.full_like(initial, np.inf)
    for index in range(len(camera_names)):
        start = index * 6
        lower[start + 3 : start + 6] = -max_translation_m
        upper[start + 3 : start + 6] = max_translation_m
    lower[-1] = 1.0 - max_time_offset_s / time_parameter_scale_s
    upper[-1] = 1.0 + max_time_offset_s / time_parameter_scale_s
    optimization = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=robust_scale_px,
        x_scale="jac",
        max_nfev=5000,
    )
    rotations, translations, offset = unpack(optimization.x)
    return SpatiotemporalSolution(
        rotation_camera_from_gt=rotations,
        translation_camera_from_gt=translations,
        time_offset_s=offset,
        residuals_px=residual(optimization.x),
        success=bool(optimization.success),
        message=str(optimization.message),
    )
