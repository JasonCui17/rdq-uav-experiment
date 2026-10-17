"""A1: exact archived whole-voxel point MLP embedding, owned by V2."""
import torch
from torch import nn


def segment_sum(x, index, n):
    return x.new_zeros((n,) + x.shape[1:]).index_add_(0, index, x)


def segment_max(x, index, n):
    out = x.new_full((n,) + x.shape[1:], -torch.inf)
    expanded = index.view(-1, *([1] * (x.ndim - 1))).expand_as(x)
    return out.scatter_reduce_(0, expanded, x, reduce="amax", include_self=True)


class LegacyVoxelEmbed(nn.Module):
    """[P,5] -> MLP 5/32/64 -> voxel max+mean+log count -> [V0,128]."""
    def __init__(self, dim=128, point_hidden=32, point_out=64, voxel_size=.5):
        super().__init__()
        self.voxel_size = voxel_size
        self.sensor_embedding = nn.Embedding(2, 1)
        self.point_mlp = nn.Sequential(nn.Linear(5, point_hidden), nn.GELU(),
                                       nn.Linear(point_hidden, point_out), nn.GELU())
        self.proj = nn.Linear(2 * point_out + 1, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, batch, hierarchy):
        level = hierarchy.levels[0]
        inv = hierarchy.point_to_l0
        local = (batch["points"] - level.centers[inv]) / self.voxel_size
        inputs = torch.cat((local, self.sensor_embedding(batch["sensor_id"]),
                            batch["delta_t"][:, None] / 1.0), 1)
        features = self.point_mlp(inputs)
        n = len(level.coords)
        maximum = segment_max(features, inv, n)
        mean = segment_sum(features, inv, n) / level.point_count[:, None]
        density = torch.log1p(level.point_count.float())[:, None]
        return self.norm(self.proj(torch.cat((maximum, mean, density), 1)))
