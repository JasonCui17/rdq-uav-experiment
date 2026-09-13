"""M2 PAPER-INSPIRED RECONSTRUCTION; no GT is supplied to forward."""
import numpy as np
import torch
from torch import nn


class CenterRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128),
                                       nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(259, 128), nn.ReLU(), nn.Linear(128, 64),
                                  nn.ReLU(), nn.Linear(64, 3))

    def forward(self, local_xyz, geometric_center):
        features = self.point_mlp(local_xyz).amax(dim=1)
        return self.head(torch.cat([features, geometric_center], dim=-1))


def sample_local(points, center, n, seed):
    """The supplied full-cluster center is never recomputed after sampling."""
    if not len(points):
        raise ValueError("Empty accepted cluster")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), n, replace=len(points) < n)
    return (points[indices] - center).astype(np.float32)


def regression_metrics(pred, gt):
    if len(pred) == 0:
        return {"samples": 0}
    if np.shape(pred) != np.shape(gt) or not np.isfinite(pred).all():
        raise ValueError("Invalid/missing predictions; evaluation samples cannot be dropped")
    residual = np.asarray(pred) - np.asarray(gt)
    squared = residual ** 2
    error = np.linalg.norm(residual, axis=1)
    return {"samples": len(pred), "MSE_coord": float(squared.mean()),
            "MSE_3D": float(squared.sum(1).mean()),
            "RMSE_coord": float(np.sqrt(squared.mean())),
            "RMSE_3D": float(np.sqrt(squared.sum(1).mean())),
            "mean_3d_error": float(error.mean()), "median_3d_error": float(np.median(error)),
            **{f"MSE_{axis}": float(squared[:, i].mean()) for i, axis in enumerate("xyz")},
            **{f"MAE_{axis}": float(np.abs(residual[:, i]).mean()) for i, axis in enumerate("xyz")}}


def verify_evaluation_ids(actual, frozen):
    if actual != frozen or len(set(actual)) != len(actual):
        raise ValueError("Frozen validation sample IDs changed or duplicated")
