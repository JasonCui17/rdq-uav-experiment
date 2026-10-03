"""SBE-Lite time-weighted v2: ordered statistics before voxel-level expansion.

Uses existing point_to_l0; no point-level learned feature or sensor embedding.
Geometry/time statistics are FP32, even under autocast. Spatial means/std
use exponential time weights; density, sensor ratio and temporal std remain
unweighted. The eight slots retain their fixed ordering and cross-attention. Coordinates are meters,
point delta_t is seconds relative to query time, and packed query identity is
already encoded in hierarchy.point_to_l0.
"""
from __future__ import annotations
import math
import torch
from torch import nn

STATISTICS_VERSION = "time_weighted_v2"

SLOT_DESCRIPTOR = (
    'weighted_mean_residual_x', 'weighted_mean_residual_y', 'weighted_mean_residual_z',
    'weighted_std_residual_x', 'weighted_std_residual_y', 'weighted_std_residual_z',
    'log_point_count', 'occupancy', 'avia_ratio', 'mean_time_weight', 'temporal_std',
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
    # PyTorch 2.1 CUDA SDPA uses a grid dimension for the packed voxel batch
    # and fails when it exceeds 65,535. Splitting that independent dimension
    # is mathematically exact: attention never mixes different voxels.
    CUDA_ATTENTION_BATCH_LIMIT=65535

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

    def _attend(self,query,slot_tokens,occupied,return_attention,limit):
        dynamic_parts=[];weight_parts=[]
        for start in range(0,len(slot_tokens),limit):
            stop=min(start+limit,len(slot_tokens))
            dynamic_part,weight_part=self.attention(query[start:stop],slot_tokens[start:stop],slot_tokens[start:stop],
                key_padding_mask=~occupied[start:stop],need_weights=return_attention,average_attn_weights=False)
            dynamic_parts.append(dynamic_part)
            if return_attention:weight_parts.append(weight_part)
        return torch.cat(dynamic_parts,0),torch.cat(weight_parts,0) if return_attention else None

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
            limit=self.CUDA_ATTENTION_BATCH_LIMIT if slot_tokens.device.type=='cuda' else len(slot_tokens)
            dynamic_sequence,weights=self._attend(query,slot_tokens,occupied,return_attention,limit)
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
            raise ValueError(f'SBE-Lite slot layout requires {expected}')
        statistics = {} if config is None else config
        self.statistics_version = statistics.get('statistics_version', STATISTICS_VERSION)
        if self.statistics_version != STATISTICS_VERSION:
            raise ValueError(f'SBE statistics_version must be {STATISTICS_VERSION}')
        self.time_half_life_s = float(statistics.get('time_half_life_s', 0.2))
        if (not math.isfinite(self.time_half_life_s)
                or not torch.finfo(torch.float32).tiny <= self.time_half_life_s <= torch.finfo(torch.float32).max):
            raise ValueError('time_half_life_s must be positive and representable in FP32')
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
            if not bool(torch.isfinite(dt).all()) or bool((dt > 0).any()):
                raise ValueError('SBE requires finite causal point delta_t <= 0')
            # w_i = 2 ** (delta_t_i / half_life). Normalize geometry
            # per slot using relative exponents to prevent all-weight underflow
            # for old observations / short half lives. Each occupied slot has
            # at least one relative weight of exactly 1.
            max_dt=points.new_full((n,),-torch.inf)
            max_dt.scatter_reduce_(0,index,dt,reduce='amax',include_self=True)
            relative_weight=torch.exp2((dt-max_dt[index])/self.time_half_life_s)
            weight_sum=points.new_zeros(n).index_add_(0,index,relative_weight)
            weight_denom=weight_sum.clamp_min(1)
            weighted_sum=points.new_zeros((n,3)).index_add_(0,index,residual*relative_weight[:,None])
            mean=weighted_sum/weight_denom[:,None]
            centered=residual-mean[index]
            weighted_square=points.new_zeros((n,3)).index_add_(0,index,centered.square()*relative_weight[:,None])
            spatial_variance=(weighted_square/weight_denom[:,None]).clamp_min(0)
            spatial_variance=torch.where(count[:,None]>1,spatial_variance,torch.zeros_like(spatial_variance))
            spatial_std=spatial_variance.sqrt()
            # Absolute (not per-slot-normalized) freshness survives when every
            # point in a voxel is old. Very old weights may legitimately be 0.
            time_weight=torch.exp2(dt/self.time_half_life_s)
            mean_time_weight=points.new_zeros(n).index_add_(0,index,time_weight)/denom
            avia=points.new_zeros(n).index_add_(0,index,(batch['sensor_id']==0).float())
            sum_dt=points.new_zeros(n).index_add_(0,index,dt)
            sum_dt2=points.new_zeros(n).index_add_(0,index,dt.square())
            variance=(sum_dt2/denom-(sum_dt/denom).square()).clamp_min(0)
            # Exact singleton std, including roundoff-sensitive values.
            variance=torch.where(count>1,variance,torch.zeros_like(variance))
            extras=torch.stack((count.log1p(),occupied.float(),avia/denom,
                                mean_time_weight,variance.sqrt()),dim=1)
            stats=torch.cat((mean,spatial_std,extras),dim=1)
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
