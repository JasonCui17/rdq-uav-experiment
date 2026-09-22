"""Load MMAUD left-camera calibration into the tensor-only P5 contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import yaml

from .contracts import InteractionContext, ProjectionContext


def load_left_projection_context(
    camera_config: str | Path,
    geometry_calibration: str | Path,
    *,
    image_scale_xy: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ProjectionContext:
    """Parse calibration once, outside model forward, then replicate over batch."""

    scale = torch.as_tensor(image_scale_xy, dtype=dtype, device=device)
    if scale.ndim == 1:
        if scale.shape != (2,):
            raise ValueError("1D image_scale_xy must have shape [2]")
        scale = scale.unsqueeze(0)
    if scale.ndim != 2 or scale.shape[1] != 2:
        raise ValueError("image_scale_xy must have shape [B,2]")
    batch_size = int(scale.shape[0])

    config = yaml.safe_load(Path(camera_config).read_text(encoding="utf-8"))
    fitted = json.loads(Path(geometry_calibration).read_text(encoding="utf-8"))
    left_config = config["cameras"]["left"]
    if left_config.get("model") != "omni":
        raise ValueError("P5 currently requires the calibrated omni camera model")
    if left_config.get("distortion_model") != "radtan":
        raise ValueError("P5 currently requires radtan distortion")
    if fitted.get("time_convention") != "gt_query_time = image_time + time_offset_s":
        raise ValueError("unexpected camera calibration time convention")

    left_geometry = fitted["cameras"]["left"]
    rotation = torch.tensor(
        left_geometry["rotation_camera_from_gt"], dtype=dtype, device=device
    )
    translation = torch.tensor(
        left_geometry["translation_camera_from_gt_m"], dtype=dtype, device=device
    )
    intrinsics = torch.tensor(
        left_config["intrinsics"], dtype=dtype, device=device
    )
    distortion = torch.tensor(
        left_config["distortion_coeffs"], dtype=dtype, device=device
    )
    image_size = torch.tensor(
        left_config["resolution"], dtype=dtype, device=device
    )

    projection = ProjectionContext(
        rotation_camera_from_radar=rotation.unsqueeze(0).repeat(batch_size, 1, 1),
        translation_camera_from_radar_m=translation.unsqueeze(0).repeat(batch_size, 1),
        intrinsics=intrinsics.unsqueeze(0).repeat(batch_size, 1),
        distortion=distortion.unsqueeze(0).repeat(batch_size, 1),
        image_size_wh=image_size.unsqueeze(0).repeat(batch_size, 1),
        image_scale_xy=scale,
    )
    projection.validate(batch_size)
    return projection


def make_interaction_context(
    calibration_handles: Sequence[str],
    m_R: torch.Tensor,
    m_V: torch.Tensor,
    projection: ProjectionContext,
) -> InteractionContext:
    """Bind parsed projection tensors to one multimodal batch."""

    context = InteractionContext(
        calibration_handle=tuple(str(value) for value in calibration_handles),
        m_R=m_R.to(device=projection.rotation_camera_from_radar.device, dtype=torch.bool),
        m_V=m_V.to(device=projection.rotation_camera_from_radar.device, dtype=torch.bool),
        projection=projection,
    )
    context.validate(projection.batch_size)
    return context
