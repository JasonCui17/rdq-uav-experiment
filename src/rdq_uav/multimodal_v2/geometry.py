"""V2 camera projection; calibration mathematics preserved from the audited implementation.

Fixed calibration lives here, never in a Sample. The existing LiDAR pyramid
layout remains in its backbone adapter and is unchanged by this data refactor.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import torch
import yaml

@dataclass(frozen=True)
class ProjectionContext:
    """Tensor-only left-camera geometry consumed by P5.

    The current MMAUD coordinate audit establishes that released LiDAR XYZ and
    the GT reference frame are the same reference frame. Therefore the fitted
    camera-from-GT transform is packed here as camera-from-radar without any
    extra frame conversion.

    Intrinsic order: [xi, fu, fv, pu, pv].
    Distortion order: [k1, k2, p1, p2].
    image_size_wh is the calibrated source image size [width, height].
    image_scale_xy maps calibrated source pixels to the resized DINO tensor.
    """

    rotation_camera_from_radar: torch.Tensor
    translation_camera_from_radar_m: torch.Tensor
    intrinsics: torch.Tensor
    distortion: torch.Tensor
    image_size_wh: torch.Tensor
    image_scale_xy: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.rotation_camera_from_radar.shape[0])

    def validate(self, batch_size: int) -> None:
        expected = {
            "rotation_camera_from_radar": (batch_size, 3, 3),
            "translation_camera_from_radar_m": (batch_size, 3),
            "intrinsics": (batch_size, 5),
            "distortion": (batch_size, 4),
            "image_size_wh": (batch_size, 2),
            "image_scale_xy": (batch_size, 2),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if not bool(torch.isfinite(value.float()).all()):
                raise ValueError(f"{name} must be finite")
        if not bool((self.image_size_wh > 0).all()):
            raise ValueError("image_size_wh must be positive")
        if not bool((self.image_scale_xy > 0).all()):
            raise ValueError("image_scale_xy must be positive")


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


def project_omni_radtan(
    points_radar: torch.Tensor,
    batch_index: torch.Tensor,
    projection: ProjectionContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project radar-frame XYZ with Kalibr omni + radtan in pure torch."""

    if points_radar.ndim != 2 or points_radar.shape[1] != 3:
        raise ValueError(f"points_radar must be [N,3], got {tuple(points_radar.shape)}")
    if not points_radar.dtype.is_floating_point:
        raise TypeError("points_radar must use a floating-point dtype")
    if batch_index.shape != (len(points_radar),):
        raise ValueError("batch_index must have shape [N]")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must be torch.long")
    if len(points_radar) == 0:
        return points_radar.new_empty((0, 2)), torch.zeros(
            0, dtype=torch.bool, device=points_radar.device
        )
    if int(batch_index.min()) < 0 or int(batch_index.max()) >= projection.batch_size:
        raise IndexError("batch_index references a sample outside ProjectionContext")

    device = points_radar.device
    projection_tensors = (
        projection.rotation_camera_from_radar,
        projection.translation_camera_from_radar_m,
        projection.intrinsics,
        projection.distortion,
        projection.image_size_wh,
        projection.image_scale_xy,
    )
    if any(value.device != device for value in projection_tensors):
        raise ValueError("ProjectionContext tensors must be on the same device as radar points")

    dtype = points_radar.dtype
    rotation = projection.rotation_camera_from_radar[batch_index].to(dtype=dtype)
    translation = projection.translation_camera_from_radar_m[batch_index].to(dtype=dtype)
    intrinsics = projection.intrinsics[batch_index].to(dtype=dtype)
    distortion = projection.distortion[batch_index].to(dtype=dtype)
    image_size = projection.image_size_wh[batch_index].to(dtype=dtype)

    points_camera = torch.bmm(rotation, points_radar.unsqueeze(-1)).squeeze(-1)
    points_camera = points_camera + translation

    distance = torch.linalg.vector_norm(points_camera, dim=1)
    xi, fu, fv, pu, pv = intrinsics.unbind(dim=1)
    k1, k2, p1, p2 = distortion.unbind(dim=1)
    denominator = points_camera[:, 2] + xi * distance

    eps = torch.finfo(dtype).eps
    valid = (
        torch.isfinite(points_camera).all(dim=1)
        & torch.isfinite(denominator)
        & (distance > eps)
        & (denominator > eps)
    )
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    x = points_camera[:, 0] / safe_denominator
    y = points_camera[:, 1] / safe_denominator
    r2 = x.square() + y.square()
    radial = 1.0 + k1 * r2 + k2 * r2.square()
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x.square())
    y_distorted = y * radial + p1 * (r2 + 2.0 * y.square()) + 2.0 * p2 * x * y
    u = fu * x_distorted + pu
    v = fv * y_distorted + pv
    pixels = torch.stack((u, v), dim=1)

    width = image_size[:, 0]
    height = image_size[:, 1]
    valid = (
        valid
        & torch.isfinite(pixels).all(dim=1)
        & (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
    )
    return pixels, valid




DEFAULT_GEOMETRY_GATE = dict(mode="inverse_range", min_px=8.0, max_px=48.0,
                             reference_range_m=20.0, reference_margin_px=16.0)


def validate_geometry_gate(config=None) -> dict:
    """One public configuration for association and V<-R admission."""
    import math
    cfg = dict(DEFAULT_GEOMETRY_GATE)
    if config is not None:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown geometry gate keys: {unknown}")
        cfg.update(config)
    if cfg["mode"] != "inverse_range":
        raise ValueError("geometry gate mode must be inverse_range")
    for key in set(cfg)-{"mode"}:
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]):
            raise ValueError("geometry gate values must be finite")
    if not (0 <= cfg["min_px"] <= cfg["reference_margin_px"] <= cfg["max_px"]
            and cfg["reference_range_m"] > 0):
        raise ValueError("invalid geometry gate limits/reference")
    return cfg


def adaptive_box_margin_px(xyz_m: torch.Tensor, *, min_px: float, max_px: float,
                           reference_range_m: float, reference_margin_px: float) -> torch.Tensor:
    """Radar-origin Euclidean range -> source-camera pixel tolerance [N].

    Empirical inverse-range heuristic, not a calibrated reprojection covariance.
    Does not change the projection or attention distance bias.
    """
    cfg = validate_geometry_gate(dict(min_px=min_px, max_px=max_px,
                                      reference_range_m=reference_range_m,
                                      reference_margin_px=reference_margin_px))
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3 or not torch.isfinite(xyz_m).all():
        raise ValueError("geometry gate requires finite XYZ [N,3]")
    with torch.autocast(device_type=xyz_m.device.type, enabled=False):
        range_m = torch.linalg.vector_norm(xyz_m.float(), dim=1).clamp_min(1e-3)
        k = (cfg["reference_margin_px"]-cfg["min_px"])*cfg["reference_range_m"]
        return (cfg["min_px"]+k/range_m).clamp(cfg["min_px"], cfg["max_px"])


def geometry_gate_margins(xyz_m: torch.Tensor, config=None) -> torch.Tensor:
    cfg = validate_geometry_gate(config)
    return adaptive_box_margin_px(xyz_m, **{k:v for k,v in cfg.items() if k != "mode"})
