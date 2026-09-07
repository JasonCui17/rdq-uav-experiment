from __future__ import annotations

import math

import torch


def stitched_token_centers(
    feature_height: int,
    feature_width: int,
    views: int,
    image_height: int,
    view_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return token centers in stitched-canvas pixels in tokenizer order."""
    if min(feature_height, feature_width, views, image_height, view_width) <= 0:
        raise ValueError("Feature and image geometry must be positive")
    y = (torch.arange(feature_height, device=device, dtype=dtype) + 0.5) * (
        image_height / feature_height
    )
    local_x = (torch.arange(feature_width, device=device, dtype=dtype) + 0.5) * (
        view_width / feature_width
    )
    per_view = []
    for view in range(views):
        yy, xx = torch.meshgrid(y, local_x + view * view_width, indexing="ij")
        per_view.append(torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1))
    return torch.cat(per_view, dim=0)


def attention_center_diagnostics(
    attention: torch.Tensor,
    gt_center_normalized: torch.Tensor,
    token_centers_px: torch.Tensor,
    *,
    stitched_width: int,
    image_height: int,
    cell_width: float,
    cell_height: float,
) -> dict[str, torch.Tensor]:
    """Measure where genuine per-head MHA weights place their spatial mass.

    ``attention`` must be ``[B, heads, 1, tokens]``.  ``best_head_error_px``
    is an oracle over heads per sample and is intentionally diagnostic only.
    """
    if attention.ndim != 4 or attention.shape[2] != 1:
        raise ValueError(f"Expected attention [B,H,1,N], got {tuple(attention.shape)}")
    if attention.shape[-1] != token_centers_px.shape[0]:
        raise ValueError(
            f"Attention tokens {attention.shape[-1]} != coordinate tokens "
            f"{token_centers_px.shape[0]}"
        )
    if gt_center_normalized.ndim != 2 or gt_center_normalized.shape[-1] != 2:
        raise ValueError("GT centers must be [B,2] normalized CXCY")

    eps = torch.finfo(attention.dtype).eps
    per_head = attention[:, :, 0, :].clamp_min(0)
    per_head = per_head / per_head.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean_attention = per_head.mean(dim=1)
    mean_attention = mean_attention / mean_attention.sum(dim=-1, keepdim=True).clamp_min(eps)

    gt_scale = gt_center_normalized.new_tensor([stitched_width, image_height])
    gt_px = gt_center_normalized * gt_scale
    mean_center = torch.einsum("bn,nc->bc", mean_attention, token_centers_px)
    head_centers = torch.einsum("bhn,nc->bhc", per_head, token_centers_px)
    peak_center = token_centers_px[mean_attention.argmax(dim=-1)]

    mean_error = torch.linalg.vector_norm(mean_center - gt_px, dim=-1)
    peak_error = torch.linalg.vector_norm(peak_center - gt_px, dim=-1)
    head_errors = torch.linalg.vector_norm(head_centers - gt_px[:, None, :], dim=-1)
    best_head_error, best_head_index = head_errors.min(dim=1)

    token_errors = torch.linalg.vector_norm(
        token_centers_px[None, :, :] - gt_px[:, None, :], dim=-1
    )
    grid_oracle_error, grid_oracle_index = token_errors.min(dim=1)

    log_token_count = math.log(attention.shape[-1])
    entropy_per_head = -(per_head * per_head.clamp_min(eps).log()).sum(dim=-1)
    normalized_entropy = entropy_per_head / log_token_count

    delta = (token_centers_px[None, :, :] - gt_px[:, None, :]).abs()
    near_1 = (delta[..., 0] <= cell_width) & (delta[..., 1] <= cell_height)
    near_2 = (delta[..., 0] <= 2 * cell_width) & (delta[..., 1] <= 2 * cell_height)

    return {
        "mean_attention_center_px": mean_center,
        "mean_attention_error_px": mean_error,
        "peak_attention_center_px": peak_center,
        "peak_attention_error_px": peak_error,
        "per_head_expected_center_px": head_centers,
        "per_head_error_px": head_errors,
        "best_head_error_px": best_head_error,
        "best_head_index": best_head_index,
        "grid_oracle_error_px": grid_oracle_error,
        "grid_oracle_index": grid_oracle_index,
        "attention_entropy": entropy_per_head.mean(dim=1),
        "attention_entropy_normalized": normalized_entropy.mean(dim=1),
        "gt_mass_1cell": (mean_attention * near_1).sum(dim=-1),
        "gt_mass_2cell": (mean_attention * near_2).sum(dim=-1),
    }
