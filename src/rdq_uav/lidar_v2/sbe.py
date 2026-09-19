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


def ordered_slot_centers(dtype=torch.float32):
    """Fixed normalized centers [8,3] in slot=4*x+2*y+z order."""
    bits=torch.tensor([[(slot>>2)&1,(slot>>1)&1,slot&1] for slot in range(8)],dtype=dtype)
    return bits*.5-.25


def subvoxel_coordinates(q):
    """Normalized L0-local [P,3] -> slots [P], residuals [P,3].

    Slot = 4*bx+2*by+bz; bits clamp floor(2*(q+.5)) to {0,1}.
    Residuals are relative to slot centers +/-0.25, not L0 centers.
    """
    bits=torch.floor(2*(q+.5)).clamp(0,1).long()
    slot=4*bits[:,0]+2*bits[:,1]+bits[:,2]
    center=bits.to(q.dtype)*.5-.25
    return slot,q-center


class VoxelQuerySlotAggregation(nn.Module):
    """One learned voxel query cross-attends to eight masked SBE slots."""
    def __init__(self,slot_dim=11,embed_dim=16,heads=2,dropout=0.):
        super().__init__()
        self.slot_dim=slot_dim;self.embed_dim=embed_dim;self.heads=heads
        self.register_buffer('slot_centers',ordered_slot_centers(),persistent=True)
        self.slot_embed=nn.Sequential(nn.Linear(slot_dim+3,embed_dim),nn.GELU())
        self.voxel_query=nn.Parameter(torch.empty(1,1,embed_dim))
        self.attention=nn.MultiheadAttention(embed_dim,heads,dropout=dropout,batch_first=True)
        nn.init.trunc_normal_(self.voxel_query,std=.02,a=-.04,b=.04)
        nn.init.trunc_normal_(self.attention.in_proj_weight,std=.02,a=-.04,b=.04)
        nn.init.zeros_(self.attention.in_proj_bias)

    def forward(self,slot_stats,slot_count,return_attention=False):
        """[V,8,11]+integer [V,8] -> dynamic [V,16]."""
        if slot_stats.ndim!=3 or slot_stats.shape[1:]!=(8,self.slot_dim):
            raise ValueError(f'Expected slot_stats [V,8,{self.slot_dim}], got {tuple(slot_stats.shape)}')
        if slot_count.shape!=slot_stats.shape[:2]:raise ValueError('slot_count shape mismatch')
        occupied=slot_count>0
        if bool((occupied.sum(1)<1).any()):raise AssertionError('VQSA received an all-masked real voxel')
        centers=self.slot_centers.to(device=slot_stats.device,dtype=slot_stats.dtype).expand(len(slot_stats),-1,-1)
        slot_input=torch.cat((slot_stats,centers),-1)
        slot_tokens=self.slot_embed(slot_input)
        query=self.voxel_query.to(slot_tokens.dtype).expand(len(slot_stats),-1,-1)
        if not len(slot_stats):
            dynamic_sequence=slot_tokens.new_empty((0,1,self.embed_dim));weights=slot_tokens.new_empty((0,self.heads,1,8))
        else:
            dynamic_sequence,weights=self.attention(query,slot_tokens,slot_tokens,key_padding_mask=~occupied,
                need_weights=return_attention,average_attn_weights=False)
        dynamic=dynamic_sequence.squeeze(1)
        if return_attention:return dynamic,dict(slot_input=slot_input,slot_tokens=slot_tokens,voxel_query=query,
            dynamic_sequence=dynamic_sequence,key_padding_mask=~occupied,attention=weights)
        return dynamic


class SBELiteVoxelEmbed(nn.Module):
    """Packed points -> ordered physical slots + VQSA -> [V0,128].

    Slots are always 0..7 and feature ordering is SLOT_DESCRIPTOR. Empty slots
    are exactly all-zero. Learned operations occur only after physical slot
    statistics exist; no point processing expands to a learned feature. No GT
    is accessed.
    """
    def __init__(self,dim=128,voxel_size=.5,config=None):
        super().__init__()
        expected=dict(subdivisions=2,slots=8,slot_dim=11,flattened_dim=88,output_dim=dim)
        if config is not None and any(config.get(k)!=v for k,v in expected.items()):
            raise ValueError(f'SBE-Lite v1 requires {expected}')
        self.voxel_size=voxel_size;self.slots=8;self.slot_dim=len(SLOT_DESCRIPTOR)
        vqsa={} if config is None else config.get('vqsa',{})
        if not bool(vqsa.get('enabled',True)):raise ValueError('VQSA-v1 is required by the final SBE embedding')
        embed_dim=int(vqsa.get('embed_dim',16));heads=int(vqsa.get('heads',2));dropout=float(vqsa.get('dropout',0.))
        if embed_dim!=16 or heads!=2 or dropout!=0.:raise ValueError('VQSA-v1 requires embed_dim=16, heads=2, dropout=0')
        self.vqsa=VoxelQuerySlotAggregation(self.slot_dim,embed_dim,heads,dropout)
        self.proj=nn.Linear(self.slots*self.slot_dim+embed_dim,dim)
        self.norm=nn.LayerNorm(dim,eps=1e-5)

    def slot_statistics(self,batch,hierarchy,return_counts=False):
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
            stats=stats.reshape(len(level.coords),self.slots,self.slot_dim)
            counts=count.reshape(len(level.coords),self.slots).long()
            return (stats,counts) if return_counts else stats

    def forward(self,batch,hierarchy,return_debug=False):
        stats,counts=self.slot_statistics(batch,hierarchy,return_counts=True)
        if return_debug:
            dynamic,debug=self.vqsa(stats,counts,return_attention=True)
        else:dynamic=self.vqsa(stats,counts)
        projection_input=torch.cat((stats.flatten(1),dynamic),1)
        token=self.norm(self.proj(projection_input))
        if return_debug:
            debug.update(slot_stats=stats,slot_count=counts,dynamic_summary=dynamic,
                projection_input=projection_input,token=token)
            return token,debug
        return token
