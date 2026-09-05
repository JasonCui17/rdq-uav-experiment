from __future__ import annotations

import numpy as np

from .omni import OmniRadtanCamera, transform_points


def sphere_projection_extent(
    center_reference: np.ndarray,
    radius_m: float,
    camera: OmniRadtanCamera,
    rotation_camera_from_reference: np.ndarray,
    translation_camera_from_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected center and width/height of a sampled 3D sphere."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    directions = np.asarray(
        [direction for direction in np.ndindex(3, 3, 3) if direction != (1, 1, 1)],
        dtype=np.float64,
    ) - 1.0
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0)
    points = np.concatenate(
        (center_reference[None], center_reference[None] + radius_m * directions), axis=0
    )
    pixels, valid = camera.project(
        transform_points(
            points, rotation_camera_from_reference, translation_camera_from_reference
        )
    )
    if not bool(valid[0]) or int(valid.sum()) < 4:
        raise ValueError("Sphere center or too many boundary points have invalid projections")
    boundary = pixels[1:][valid[1:]]
    return pixels[0], boundary.max(axis=0) - boundary.min(axis=0)


def clamped_square_roi(
    center_uv: np.ndarray,
    extent_wh: np.ndarray,
    image_width: int,
    image_height: int,
    context_scale: float,
    min_side_px: int,
    max_side_px: int,
) -> tuple[int, int, int, int]:
    """Build an in-image square ROI without padding-induced border shortcuts."""
    if context_scale <= 0:
        raise ValueError("context_scale must be positive")
    side = int(
        np.ceil(
            np.clip(
                float(np.max(extent_wh)) * context_scale,
                min_side_px,
                min(max_side_px, image_width, image_height),
            )
        )
    )
    u, v = map(float, center_uv)
    x1 = int(round(u - side / 2))
    y1 = int(round(v - side / 2))
    x1 = min(max(x1, 0), image_width - side)
    y1 = min(max(y1, 0), image_height - side)
    return x1, y1, x1 + side, y1 + side


def centered_square_roi(
    center_uv: np.ndarray,
    side_px: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Return a strictly centered in-image crop, or None instead of shifting."""
    side = int(side_px)
    if side <= 0 or side > min(image_width, image_height):
        raise ValueError("side_px must fit inside the image")
    u, v = map(float, center_uv)
    x1 = int(round(u - side / 2))
    y1 = int(round(v - side / 2))
    x2, y2 = x1 + side, y1 + side
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return None
    return x1, y1, x2, y2
