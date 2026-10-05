"""A2: point MLP -> eight ordered slots -> VQSA -> voxel token."""
import torch
from torch import nn
from .lidar_legacy import segment_sum, segment_max
from .radar_sbe import subvoxel_coordinates, VoxelQuerySlotAggregation


class LearnedSBEVoxelEmbed(nn.Module):
    """One 5->C point layer; per-slot mean/max/log count/occupancy.

    Geometry is voxel-local, as in Legacy. Slot position is separately fed
    into VQSA. Relative time is an input feature; no exponential pooling or
    11-dimensional physical descriptor is used in this variant.
    """
    def __init__(self, dim=128, voxel_size=.5, point_dim=32):
        super().__init__()
        if point_dim not in (16, 32):
            raise ValueError("learned_sbe.point_dim must be 16 or 32")
        self.voxel_size, self.point_dim = voxel_size, point_dim
        self.sensor_embedding = nn.Embedding(2, 1)
        self.point_mlp = nn.Sequential(nn.Linear(5, point_dim), nn.GELU())
        self.slot_dim = 2 * point_dim + 2
        self.vqsa = VoxelQuerySlotAggregation(self.slot_dim, 16, 2, 0.)
        self.proj = nn.Linear(8 * self.slot_dim + 16, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, batch, hierarchy, return_debug=False):
        level, inv = hierarchy.levels[0], hierarchy.point_to_l0
        # Keep physical coordinate and count construction FP32 under AMP.
        with torch.autocast(device_type=batch["points"].device.type, enabled=False):
            local = (batch["points"].float() - level.centers[inv].float()) / self.voxel_size
            slot, _ = subvoxel_coordinates(local)
            index = inv * 8 + slot
            count = torch.bincount(index, minlength=len(level.coords) * 8)
            dt = batch["delta_t"].float()
            if not torch.isfinite(dt).all() or (dt > 0).any():
                raise ValueError("Learned-SBE requires finite causal delta_t <= 0")
        inputs = torch.cat((local, self.sensor_embedding(batch["sensor_id"]), dt[:, None]), 1)
        features = self.point_mlp(inputs)
        n = len(level.coords) * 8
        # Accumulate learned feature reductions in FP32 for stable AMP means.
        with torch.autocast(device_type=features.device.type, enabled=False):
            mean = segment_sum(features.float(), index, n) / count.clamp_min(1)[:, None]
            maximum = segment_max(features.float(), index, n)
            maximum = maximum.masked_fill((count == 0)[:, None], 0.)
            extras = torch.stack((count.float().log1p(), (count > 0).float()), 1)
            slots = torch.cat((mean, maximum, extras), 1).reshape(-1, 8, self.slot_dim)
            counts = count.reshape(-1, 8)
        dynamic = self.vqsa(slots, counts)
        projection_input = torch.cat((slots.flatten(1), dynamic), 1)
        tokens = self.norm(self.proj(projection_input))
        if return_debug:
            return tokens, {"slot_features": slots, "slot_count": counts,
                            "dynamic_summary": dynamic, "projection_input": projection_input}
        return tokens
