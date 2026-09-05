from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OmniRadtanCamera:
    """Kalibr unified omnidirectional camera with radtan distortion.

    Intrinsic order follows Kalibr: ``[xi, fu, fv, pu, pv]``.
    Distortion order is ``[k1, k2, p1, p2]``.
    """

    xi: float
    fu: float
    fv: float
    pu: float
    pv: float
    k1: float
    k2: float
    p1: float
    p2: float
    width: int
    height: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OmniRadtanCamera":
        intrinsic = [float(value) for value in config["intrinsics"]]
        distortion = [float(value) for value in config["distortion_coeffs"]]
        resolution = [int(value) for value in config["resolution"]]
        if len(intrinsic) != 5 or len(distortion) != 4 or len(resolution) != 2:
            raise ValueError("Expected 5 intrinsics, 4 radtan coefficients and 2D resolution")
        return cls(*intrinsic, *distortion, *resolution)

    def project(
        self, points_camera: np.ndarray, *, require_in_image: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-frame 3D points to pixels.

        Returns ``(pixels, valid)``. The projection uses the exact model named
        by the official Kalibr files; it is not OpenCV's equidistant fisheye
        model and not a pinhole approximation.
        """
        points = np.asarray(points_camera, dtype=np.float64)
        original_shape = points.shape
        if original_shape[-1:] != (3,):
            raise ValueError(f"Expected (..., 3) points, got {original_shape}")
        flat = points.reshape(-1, 3)
        distance = np.linalg.norm(flat, axis=1)
        denominator = flat[:, 2] + self.xi * distance
        valid = np.isfinite(flat).all(axis=1) & (distance > 1e-12) & (denominator > 1e-12)

        normalized = np.full((len(flat), 2), np.nan, dtype=np.float64)
        normalized[valid] = flat[valid, :2] / denominator[valid, None]
        x = normalized[:, 0]
        y = normalized[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        x_distorted = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        y_distorted = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        pixels = np.stack(
            (self.fu * x_distorted + self.pu, self.fv * y_distorted + self.pv), axis=1
        )
        if require_in_image:
            valid &= (
                (pixels[:, 0] >= 0.0)
                & (pixels[:, 0] < self.width)
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 1] < self.height)
            )
        return pixels.reshape(original_shape[:-1] + (2,)), valid.reshape(original_shape[:-1])

    def in_image(self, pixels: np.ndarray, margin: float = 0.0) -> np.ndarray:
        pixels = np.asarray(pixels)
        return (
            np.isfinite(pixels).all(axis=-1)
            & (pixels[..., 0] >= margin)
            & (pixels[..., 0] < self.width - margin)
            & (pixels[..., 1] >= margin)
            & (pixels[..., 1] < self.height - margin)
        )


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply ``p_camera = R_camera_from_reference p_reference + t``."""
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return points @ rotation.T + translation
