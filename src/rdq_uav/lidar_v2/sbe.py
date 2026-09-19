"""SBE-Lite v1: ordered physical statistics before voxel-level expansion.

Uses existing point_to_l0; no point-level learned feature or sensor embedding.
Geometry/time statistics are FP32, even under autocast. Coordinates are meters,
point delta_t is seconds relative to query time, and packed query identity is
already encoded in hierarchy.point_to_l0.
"""
from __future__ import annotations
import torch
from torch import nn

SLOT_DESCRIPTOR = (
    'mean_residual_x', 'mean_residual_y', 'mean_residual_z',
    'max_abs_residual_x', 'max_abs_residual_y', 'max_abs_residual_z',
    'log_point_count', 'occupancy', 'avia_ratio', 'latest_age', 'temporal_std',
)


def subvoxel_coordinates(q):
    """Normalized L0-local [P,3] -> slots [P], residuals [P,3].

    Slot = 4*bx+2*by+bz; bits clamp floor(2*(q+.5)) to {0,1}.
    Residuals are relative to slot centers +/-0.25, not L0 centers.
    """
    bits=torch.floor(2*(q+.5)).clamp(0,1).long()
    slot=4*bits[:,0]+2*bits[:,1]+bits[:,2]
    center=bits.to(q.dtype)*.5-.25
    return slot,q-center


class SBELiteVoxelEmbed(nn.Module):
    """Packed raw points -> [V0,8,11] statistics -> [V0,128] tokens.

    Slots are always 0..7 and feature ordering is SLOT_DESCRIPTOR. Empty slots
    are exactly all-zero. Only proj/norm are learned; neither pooling nor point
    processing expands to a learned point feature. No GT is accessed.
    """
    def __init__(self,dim=128,voxel_size=.5,config=None):
        super().__init__()
        expected=dict(subdivisions=2,slots=8,slot_dim=11,flattened_dim=88,output_dim=dim)
        if config is not None and any(config.get(k)!=v for k,v in expected.items()):
            raise ValueError(f'SBE-Lite v1 requires {expected}')
        self.voxel_size=voxel_size;self.slots=8;self.slot_dim=len(SLOT_DESCRIPTOR)
        self.proj=nn.Linear(self.slots*self.slot_dim,dim)
        self.norm=nn.LayerNorm(dim,eps=1e-5)

    def slot_statistics(self,batch,hierarchy):
        """Return FP32 [V0,8,11]; reductions vectorized over sub_id=v*8+slot."""
        with torch.autocast(device_type=batch['points'].device.type,enabled=False):
            points=batch['points'].float();dt=batch['delta_t'].float()
            level=hierarchy.levels[0];inv=hierarchy.point_to_l0
            q=(points-level.centers[inv].float())/self.voxel_size
            slot,residual=subvoxel_coordinates(q)
            index=inv*self.slots+slot;n=len(level.coords)*self.slots
            count=torch.bincount(index,minlength=n).float();denom=count.clamp_min(1)
            occupied=count>0
            sum_r=points.new_zeros((n,3)).index_add_(0,index,residual)
            max_r=points.new_zeros((n,3))
            max_r.scatter_reduce_(0,index[:,None].expand(-1,3),residual.abs(),reduce='amax',include_self=True)
            avia=points.new_zeros(n).index_add_(0,index,(batch['sensor_id']==0).float())
            sum_dt=points.new_zeros(n).index_add_(0,index,dt)
            sum_dt2=points.new_zeros(n).index_add_(0,index,dt.square())
            max_dt=points.new_full((n,),-torch.inf)
            max_dt.scatter_reduce_(0,index,dt,reduce='amax',include_self=True)
            max_dt=torch.where(occupied,max_dt,torch.zeros_like(max_dt))
            variance=(sum_dt2/denom-(sum_dt/denom).square()).clamp_min(0)
            # Exact singleton std, including roundoff-sensitive values.
            variance=torch.where(count>1,variance,torch.zeros_like(variance))
            extras=torch.stack((count.log1p(),occupied.float(),avia/denom,
                                (-max_dt).clamp_min(0),variance.sqrt()),dim=1)
            stats=torch.cat((sum_r/denom[:,None],max_r,extras),dim=1)
            stats=stats.masked_fill(~occupied[:,None],0.)
            return stats.reshape(len(level.coords),self.slots,self.slot_dim)

    def forward(self,batch,hierarchy):
        stats=self.slot_statistics(batch,hierarchy)
        return self.norm(self.proj(stats.flatten(1)))
