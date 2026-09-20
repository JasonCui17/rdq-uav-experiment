"""LiDAR UAV V2 Query-Causal with selectable SBE-Lite / legacy voxel embedding."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
from .sbe import SBELiteVoxelEmbed
from .geometry import HierarchyBuilder, SparseHierarchy, SparseLevel, decode_residual

def segment_sum(x,index,n):
    out=x.new_zeros((n,)+x.shape[1:]); return out.index_add_(0,index,x)
def segment_max(x,index,n):
    out=x.new_full((n,)+x.shape[1:],-torch.inf); out.scatter_reduce_(0,index.view(-1,*([1]*(x.ndim-1))).expand_as(x),x,reduce="amax",include_self=True); return out

class LegacyVoxelEmbed(nn.Module):
    """Packed points [P,3]+sensor+seconds -> L0 tokens [N0,D]."""
    def __init__(self,dim=128,point_hidden=32,point_out=64,voxel_size=.5):
        super().__init__(); self.voxel_size=voxel_size
        self.sensor_embedding=nn.Embedding(2,1); self.point_mlp=nn.Sequential(nn.Linear(5,point_hidden),nn.GELU(),nn.Linear(point_hidden,point_out),nn.GELU())
        self.proj=nn.Linear(2*point_out+1,dim); self.norm=nn.LayerNorm(dim,eps=1e-5)
    def forward(self,batch,hierarchy):
        level=hierarchy.levels[0]; inv=hierarchy.point_to_l0
        local=(batch["points"]-level.centers[inv])/self.voxel_size
        pf=torch.cat((local,self.sensor_embedding(batch["sensor_id"]),batch["delta_t"][:,None]/1.0),1)
        h=self.point_mlp(pf); n=len(level.coords); maximum=segment_max(h,inv,n); mean=segment_sum(h,inv,n)/level.point_count[:,None]
        return self.norm(self.proj(torch.cat((maximum,mean,torch.log1p(level.point_count.float())[:,None]),1)))

class SparseMerge(nn.Module):
    """Geometry-preserving child-to-parent sparse aggregation.

    Existing children retain their signed octant position in ``self.child``.
    The parent projection additionally receives the complete ordered eight-slot
    occupancy topology and log raw-point density.  Slot order is shared with
    SBE-Lite: ``4*x + 2*y + z``.
    """
    def __init__(self,dim=128):
        super().__init__(); self.child_norm=nn.LayerNorm(dim,eps=1e-5); self.child=nn.Sequential(nn.Linear(dim+3,dim),nn.GELU())
        self.occupancy_slots=8; self.parent=nn.Linear(2*dim+1+self.occupancy_slots,dim); self.out_norm=nn.LayerNorm(dim,eps=1e-5)

    @staticmethod
    def octant_occupancy(child_level,parent_level,parent_map):
        """Return ``local [Nc,3]``, ``slot [Nc]``, and binary ``[Np,8]``."""
        local=child_level.coords-2*parent_level.coords[parent_map]
        if bool(((local<0)|(local>1)).any()):
            bad=local[((local<0)|(local>1)).any(1)][:8].tolist()
            raise AssertionError(f'Invalid child octant coordinates: {bad}')
        slot=4*local[:,0]+2*local[:,1]+local[:,2]
        n=len(parent_level.coords); occupancy=parent_level.centers.new_zeros((n,8))
        flat=parent_map*8+slot
        occupancy.view(-1).index_fill_(0,flat,1.)
        child_count=torch.bincount(parent_map,minlength=n)
        occupied_count=occupancy.sum(1).to(child_count.dtype)
        if not torch.equal(occupied_count,child_count):
            raise AssertionError('Octant occupancy count does not match unique child voxel count')
        return local,slot,occupancy

    def parent_input(self,x,child_level,parent_level,parent_map):
        """Build ordered ``[Np, 2*D+1+8]`` EOOE projection input."""
        local,slot,occupancy=self.octant_occupancy(child_level,parent_level,parent_map)
        rel=local.to(x.dtype)-.5
        u=self.child(torch.cat((self.child_norm(x),rel),1)); n=len(parent_level.coords)
        child_count=torch.bincount(parent_map,minlength=n)
        mx=segment_max(u,parent_map,n); mean=segment_sum(u,parent_map,n)/child_count[:,None]
        density=torch.log1p(parent_level.point_count.to(x.dtype))[:,None]
        return torch.cat((mx,mean,density,occupancy.to(x.dtype)),1),occupancy,slot

    def forward(self,x,child_level,parent_level,parent_map):
        parent_input,_,_=self.parent_input(x,child_level,parent_level,parent_map)
        return self.out_norm(self.parent(parent_input))

class PositionEncoding(nn.Module):
    def __init__(self,dim=128): super().__init__(); self.net=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,dim))
    def forward(self,centers): return self.net(centers/100.)

def morton_order(coords:torch.Tensor)->torch.Tensor:
    if not len(coords): return torch.empty(0,dtype=torch.long,device=coords.device)
    q=(coords-coords.amin(0)).long(); code=torch.zeros(len(q),dtype=torch.long,device=q.device)
    for bit in range(20):
        code|=((q[:,0]>>bit)&1)<<(3*bit); code|=((q[:,1]>>bit)&1)<<(3*bit+1); code|=((q[:,2]>>bit)&1)<<(3*bit+2)
    return torch.argsort(code,stable=True)

class RelativeBias(nn.Module):
    def __init__(self,heads=4,relative_range=32):
        super().__init__(); self.range=relative_range; self.tables=nn.Parameter(torch.zeros(3,heads,2*relative_range+1))
    def forward(self,coords):
        if coords.ndim==2:
            offsets=(coords[None,:,:]-coords[:,None,:]).clamp(-self.range,self.range)+self.range
            return sum(self.tables[axis,:,offsets[:,:,axis]] for axis in range(3))
        if coords.ndim!=3:raise ValueError(f"coords must be [N,3] or [G,N,3], got {tuple(coords.shape)}")
        offsets=(coords[:,None,:,:]-coords[:,:,None,:]).clamp(-self.range,self.range)+self.range
        # Table indexing yields [heads,groups,N,N]; restore [groups,heads,N,N].
        return sum(self.tables[axis,:,offsets[:,:,:,axis]].permute(1,0,2,3) for axis in range(3))

class SpatialBlock(nn.Module):
    """Pre-LN self-attention on caller-provided within-sample token groups."""
    def __init__(self,dim=128,heads=4,ffn_dim=256):
        super().__init__(); self.heads=heads; self.head_dim=dim//heads
        self.norm1=nn.LayerNorm(dim,eps=1e-5); self.qkv=nn.Linear(dim,3*dim); self.proj=nn.Linear(dim,dim)
        self.norm2=nn.LayerNorm(dim,eps=1e-5); self.ffn=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(),nn.Linear(ffn_dim,dim))
    def forward_group(self,x,coords,bias):
        n=len(x); qkv=self.qkv(self.norm1(x)).reshape(n,3,self.heads,self.head_dim).permute(1,2,0,3); q,k,v=qkv
        logits=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)+bias(coords)
        y=(torch.softmax(logits.float(),-1).to(v.dtype)@v).transpose(0,1).reshape(n,-1)
        y=x+self.proj(y); return y+self.ffn(self.norm2(y))
    def forward_local_groups(self,x,coords,groups,bias):
        """Evaluate independent local windows in one padded batched kernel."""
        if not groups:return torch.empty_like(x)
        lengths=torch.tensor([len(idx) for idx in groups],device=x.device,dtype=torch.long)
        width=int(lengths.max());flat_ids=torch.cat(groups);group_id=torch.repeat_interleave(torch.arange(len(groups),device=x.device),lengths)
        starts=torch.repeat_interleave(torch.cumsum(lengths,0)-lengths,lengths);position=torch.arange(len(flat_ids),device=x.device)-starts
        padded=x.new_zeros((len(groups),width,x.shape[1]));padded_coords=coords.new_zeros((len(groups),width,3));valid=torch.zeros((len(groups),width),device=x.device,dtype=torch.bool)
        padded[group_id,position]=x[flat_ids];padded_coords[group_id,position]=coords[flat_ids];valid[group_id,position]=True
        qkv=self.qkv(self.norm1(padded)).reshape(len(groups),width,3,self.heads,self.head_dim).permute(2,0,3,1,4);q,k,v=qkv
        logits=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)+bias(padded_coords)
        logits=logits.masked_fill(~valid[:,None,None,:],-torch.inf)
        attended=(torch.softmax(logits.float(),-1).to(v.dtype)@v).transpose(1,2).reshape(len(groups),width,-1)
        y=padded+self.proj(attended);z=y+self.ffn(self.norm2(y));out=torch.empty_like(x);out[flat_ids]=z[group_id,position]
        return out
    def forward(self,x,coords,groups,bias,batched_local=False):
        if batched_local:return self.forward_local_groups(x,coords,groups,bias)
        out=torch.empty_like(x)
        for idx in groups: out[idx]=self.forward_group(x[idx],coords[idx],bias)
        return out

def local_groups(level:SparseLevel,window:int,shift:int)->list[torch.Tensor]:
    groups=[]
    for batch in torch.unique(level.batch_index,sorted=True):
        ids=torch.nonzero(level.batch_index==batch).flatten(); ordered=ids[morton_order(level.coords[ids])]; n=len(ordered)
        starts=[0] if shift==0 else [0,shift]
        if shift==0: starts=list(range(0,n,window))
        else: starts=[0]+list(range(shift,n,window))
        for j,start in enumerate(starts):
            end=min(n, shift if j==0 and shift else start+window)
            if end>start: groups.append(ordered[start:end])
    return groups

def global_groups(level): return [torch.nonzero(level.batch_index==b).flatten() for b in torch.unique(level.batch_index,sorted=True)]

class SpatialTransformer(nn.Module):
    def __init__(self,dim,heads,ffn_dim,blocks,window_size,window_shift,relative_range,global_attention=False):
        super().__init__(); self.position=PositionEncoding(dim); self.bias=RelativeBias(heads,relative_range)
        self.blocks=nn.ModuleList([SpatialBlock(dim,heads,ffn_dim) for _ in range(blocks)])
        self.window_size=window_size; self.window_shift=window_shift; self.global_attention=global_attention
    def forward(self,x,level):
        x=x+self.position(level.centers)
        for i,block in enumerate(self.blocks):
            groups=global_groups(level) if self.global_attention else local_groups(level,self.window_size,0 if i%2==0 else self.window_shift)
            x=block(x,level.coords,groups,self.bias,batched_local=not self.global_attention)
        return x

class SparseUp(nn.Module):
    def __init__(self,dim=128):
        super().__init__(); self.parent_norm=nn.LayerNorm(dim,eps=1e-5); self.parent=nn.Linear(dim,dim)
        self.norm=nn.LayerNorm(dim,eps=1e-5); self.ffn=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
    def forward(self,fine,coarse,parent_map):
        y=fine+self.parent(self.parent_norm(coarse))[parent_map]; return y+self.ffn(self.norm(y))

class CandidateHead(nn.Module):
    def __init__(self,dim=128,hidden=64,residual_scale_m=1.):
        super().__init__()
        if float(residual_scale_m)!=1.:raise ValueError('residual_scale_m must equal 1.0 for current LiDAR V2 coordinate contract')
        self.residual_scale_m=float(residual_scale_m)
        self.cls=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,1)); self.reg=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,3))
    def forward(self,q,centers):
        logits=self.cls(q).squeeze(-1); residual=self.reg(q); return logits,residual,decode_residual(residual,centers,self.residual_scale_m)

class LiDARUAVDetector(nn.Module):
    """Public V2-base interface: packed point batch -> fine-token candidate fields."""
    def __init__(self,cfg:dict[str,Any]):
        super().__init__(); m=cfg["model"] if "model" in cfg else cfg; d=m["dim"]; tr=m["transformer"]
        self.version=m["name"]; self.hierarchy=HierarchyBuilder(tuple(m["voxel"]["scales"]))
        embedding=m['voxel'].get('embedding','legacy')
        if embedding=='sbe_lite':
            self.voxel_embed=SBELiteVoxelEmbed(d,m['voxel']['scales'][0],m['voxel'].get('sbe'))
        elif embedding=='legacy':
            # Archived compatibility path only; the frozen V2 contract rejects it.
            ph=m['voxel'].get('point_dims',(5,32,64))
            self.voxel_embed=LegacyVoxelEmbed(d,ph[1],ph[2],m['voxel']['scales'][0])
        else:raise ValueError(f'Unknown voxel embedding: {embedding}')
        self.merge01=SparseMerge(d); self.merge12=SparseMerge(d)
        common=(d,tr["heads"],tr["ffn_dim"],tr["blocks_per_level"][0],tr["window_size"],tr["window_shift"],tr["relative_range"])
        self.encoder0=SpatialTransformer(*common,global_attention=False)
        self.encoder1=SpatialTransformer(d,tr["heads"],tr["ffn_dim"],tr["blocks_per_level"][1],tr["window_size"],tr["window_shift"],tr["relative_range"],False)
        self.encoder2=SpatialTransformer(d,tr["heads"],tr["ffn_dim"],tr["blocks_per_level"][2],tr["window_size"],tr["window_shift"],tr["relative_range"],True)
        self.up21=SparseUp(d); self.up10=SparseUp(d); self.final_norm=nn.LayerNorm(d,eps=1e-5); self.head=CandidateHead(d,m["head"]["hidden_dim"],m["head"]["residual_scale_m"])
        self.apply(self._init)
        if isinstance(self.voxel_embed,LegacyVoxelEmbed):
            nn.init.constant_(self.voxel_embed.sensor_embedding.weight[0],0)
            nn.init.constant_(self.voxel_embed.sensor_embedding.weight[1],1)
        for module in self.modules():
            if isinstance(module,RelativeBias): nn.init.zeros_(module.tables)
        nn.init.constant_(self.head.cls[-1].bias,math.log(.01/.99))
    @staticmethod
    def _init(module):
        if isinstance(module,nn.Linear): nn.init.trunc_normal_(module.weight,std=.02,a=-.04,b=.04); nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
    def spatial_forward(self,batch):
        h=self.hierarchy(batch["points"],batch["point_batch_index"]); l0,l1,l2=h.levels
        f0=self.encoder0(self.voxel_embed(batch,h),l0); f1=self.encoder1(self.merge01(f0,l0,l1,h.parent_l0_to_l1),l1)
        f2=self.encoder2(self.merge12(f1,l1,l2,h.parent_l1_to_l2),l2)
        d1=self.up21(f1,f2,h.parent_l1_to_l2); q=self.final_norm(self.up10(f0,d1,h.parent_l0_to_l1)); logits,residual,pred=self.head(q,l0.centers)
        return {"logits":logits,"residual_xyz":residual,"pred_xyz":pred,"fine_features":q,"voxel_centers":l0.centers,
                "source_token_id":torch.arange(len(q),device=q.device),"batch_index":l0.batch_index,"layouts":h,
                "aux_stats":{"token_counts":[len(x.coords) for x in h.levels],"num_samples":int(batch["spatial_num_samples"]),"attention_backend":"explicit_pytorch_scaled_dot_product_with_additive_axis_bias"}}

    def forward(self,batch):
        """Packed query points -> dense spatial candidate fields."""
        return self.spatial_forward(batch)

    def load_pre_sbe_weights(self,state_dict):
        """Load all downstream spatial weights; skip only legacy voxel_embed.*.

        Checks full key coverage and shapes before mutating parameters. Does not
        accept an incomplete spatial-only checkpoint or unexplained new keys.
        Returns the explicit skipped/missing key report; SBE keeps initialization.
        """
        if not isinstance(self.voxel_embed,SBELiteVoxelEmbed):
            raise ValueError('load_pre_sbe_weights requires SBE-Lite model')
        own=self.state_dict();prefix='voxel_embed.'
        downstream={k for k in own if not k.startswith(prefix)}
        provided={k for k in state_dict if not k.startswith(prefix)}
        missing=sorted(downstream-provided);unexpected=sorted(provided-downstream)
        mismatched=sorted(k for k in downstream&provided if own[k].shape!=state_dict[k].shape)
        if missing or unexpected or mismatched:
            raise ValueError(f'Downstream mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatched}')
        selected={k:state_dict[k] for k in downstream}
        result=self.load_state_dict(selected,strict=False)
        expected_missing=sorted(k for k in own if k.startswith(prefix))
        if sorted(result.missing_keys)!=expected_missing or result.unexpected_keys:
            raise RuntimeError(f'Unexpected load result: {result}')
        return dict(skipped_source_keys=sorted(k for k in state_dict if k.startswith(prefix)),
                    expected_missing_keys=expected_missing,unexpected_missing_keys=[],unexpected_keys=[])
