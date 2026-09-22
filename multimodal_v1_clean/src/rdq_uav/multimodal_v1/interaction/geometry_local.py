"""Torch-only projection and sparse 3x3 geometry edges for P5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..contracts import InteractionContext, ProjectionContext


@dataclass(frozen=True)
class GeometryEdges:
    """Sparse bipartite edges between radar tokens and visual feature cells."""

    radar_index: torch.Tensor
    vision_index: torch.Tensor
    relative_index_v_to_r: torch.Tensor
    relative_index_r_to_v: torch.Tensor
    valid_projection_mask: torch.Tensor
    active_radar_mask: torch.Tensor
    projected_pixels: torch.Tensor
    anchor_xy: torch.Tensor

    @property
    def edge_count(self) -> int:
        return int(self.radar_index.numel())


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


class GeometryLocal(nn.Module):
    """Projection -> discrete 3x3 local cell incidence for one HCI level."""

    def __init__(self, feature_stride: int) -> None:
        super().__init__()
        if feature_stride <= 0:
            raise ValueError("feature_stride must be positive")
        self.feature_stride = int(feature_stride)
        offsets = torch.tensor(
            [
                [-1, -1], [0, -1], [1, -1],
                [-1, 0], [0, 0], [1, 0],
                [-1, 1], [0, 1], [1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("offsets_xy", offsets, persistent=False)

    def forward(
        self,
        radar_centers: torch.Tensor,
        radar_batch_index: torch.Tensor,
        *,
        batch_size: int,
        feature_height: int,
        feature_width: int,
        context: InteractionContext,
    ) -> GeometryEdges:
        if feature_height <= 0 or feature_width <= 0:
            raise ValueError("feature_height and feature_width must be positive")
        context.validate(batch_size)
        if context.projection.batch_size != batch_size:
            raise ValueError("ProjectionContext batch size does not match vision batch")
        if radar_batch_index.device != radar_centers.device:
            raise ValueError("radar_batch_index and radar_centers must share a device")
        if context.m_R.device != radar_centers.device or context.m_V.device != radar_centers.device:
            raise ValueError("modality masks must be on the same device as radar features")

        pixels, source_valid = project_omni_radtan(
            radar_centers, radar_batch_index, context.projection
        )
        if len(radar_centers) == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=radar_centers.device)
            empty_bool = torch.empty(0, dtype=torch.bool, device=radar_centers.device)
            return GeometryEdges(
                radar_index=empty_long,
                vision_index=empty_long,
                relative_index_v_to_r=empty_long,
                relative_index_r_to_v=empty_long,
                valid_projection_mask=empty_bool,
                active_radar_mask=empty_bool,
                projected_pixels=pixels,
                anchor_xy=torch.empty((0, 2), dtype=torch.long, device=radar_centers.device),
            )

        scale = context.projection.image_scale_xy[radar_batch_index].to(
            dtype=radar_centers.dtype
        )
        scaled_pixels = pixels * scale
        anchor_xy = torch.floor(scaled_pixels / float(self.feature_stride)).to(torch.long)
        # DINO may pad a batch to a larger common tensor. Do not let the
        # 3x3 neighborhood read padded visual cells outside each sample's
        # resized, non-padded image extent.
        resized_wh = (
            context.projection.image_size_wh[radar_batch_index].to(
                dtype=radar_centers.dtype
            )
            * scale
        )
        valid_feature_wh = torch.ceil(
            resized_wh / float(self.feature_stride)
        ).to(torch.long)
        valid_feature_wh[:, 0].clamp_(max=feature_width)
        valid_feature_wh[:, 1].clamp_(max=feature_height)

        anchor_in_feature = (
            source_valid
            & (anchor_xy[:, 0] >= 0)
            & (anchor_xy[:, 0] < valid_feature_wh[:, 0])
            & (anchor_xy[:, 1] >= 0)
            & (anchor_xy[:, 1] < valid_feature_wh[:, 1])
        )
        geometry_valid = anchor_in_feature
        cross_modal = context.m_R[radar_batch_index] & context.m_V[radar_batch_index]
        active_radar = geometry_valid & cross_modal

        offsets = self.offsets_xy.to(device=radar_centers.device)
        neighbors = anchor_xy[:, None, :] + offsets[None, :, :]
        neighbor_valid = (
            active_radar[:, None]
            & (neighbors[..., 0] >= 0)
            & (neighbors[..., 0] < valid_feature_wh[:, None, 0])
            & (neighbors[..., 1] >= 0)
            & (neighbors[..., 1] < valid_feature_wh[:, None, 1])
        )

        radar_grid = torch.arange(
            len(radar_centers), device=radar_centers.device, dtype=torch.long
        )[:, None].expand(-1, 9)
        relative_grid = torch.arange(
            9, device=radar_centers.device, dtype=torch.long
        )[None, :].expand(len(radar_centers), -1)
        batch_grid = radar_batch_index[:, None].expand(-1, 9)

        radar_index = radar_grid[neighbor_valid]
        relative_v_to_r = relative_grid[neighbor_valid]
        relative_r_to_v = 8 - relative_v_to_r
        selected_neighbors = neighbors[neighbor_valid]
        selected_batch = batch_grid[neighbor_valid]
        local_vision = (
            selected_neighbors[:, 1] * feature_width + selected_neighbors[:, 0]
        )
        vision_index = (
            selected_batch * (feature_height * feature_width) + local_vision
        ).to(torch.long)

        return GeometryEdges(
            radar_index=radar_index,
            vision_index=vision_index,
            relative_index_v_to_r=relative_v_to_r,
            relative_index_r_to_v=relative_r_to_v,
            valid_projection_mask=geometry_valid,
            active_radar_mask=active_radar,
            projected_pixels=pixels,
            anchor_xy=anchor_xy,
        )
