"""LiDAR UAV V2 spatial candidate model with SBE-Lite voxel embedding."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
from .radar_sbe import SBELiteVoxelEmbed
from .lidar_legacy import LegacyVoxelEmbed
from .lidar_learned_sbe import LearnedSBEVoxelEmbed
from .radar_temporal_centroid import TemporalCentroidFeatures


def build_voxel_embedding(model_cfg):
    voxel = model_cfg['voxel']
    name = voxel.get('embedding', 'sbe_lite')
    dim, size = model_cfg['dim'], voxel['scales'][0]
    if name == 'sbe_lite':
        return SBELiteVoxelEmbed(dim, size, voxel.get('sbe'))
    if name == 'legacy':
        return LegacyVoxelEmbed(dim, 32, 64, size)
    if name == 'learned_sbe':
        return LearnedSBEVoxelEmbed(dim, size, int(voxel.get('learned_sbe', {}).get('point_dim', 32)),
                                   motion_enabled=bool(voxel.get("motion_centroid", {}).get("enabled", False)))
    raise ValueError(f'Unknown voxel embedding: {name}')
from .radar_geometry import HierarchyBuilder, SparseHierarchy, SparseLevel, decode_residual

def segment_sum(x,index,n):
    out=x.new_zeros((n,)+x.shape[1:]); return out.index_add_(0,index,x)
def segment_max(x,index,n):
    out=x.new_full((n,)+x.shape[1:],-torch.inf); out.scatter_reduce_(0,index.view(-1,*([1]*(x.ndim-1))).expand_as(x),x,reduce="amax",include_self=True); return out

class SparseMerge(nn.Module):
    """Geometry-preserving child-to-parent sparse aggregation.

    Existing children retain their signed octant position in ``self.child``.
    The parent projection additionally receives the complete ordered eight-slot
    occupancy topology and log raw-point density.  Slot order is shared with
    SBE-Lite: ``4*x + 2*y + z``.
    """
    def __init__(self,dim=128,motion_enabled=False):
        super().__init__(); self.child_norm=nn.LayerNorm(dim,eps=1e-5); self.child=nn.Sequential(nn.Linear(dim+3,dim),nn.GELU())
        self.occupancy_slots=8; self.parent=nn.Linear(2*dim+1+self.occupancy_slots,dim); self.out_norm=nn.LayerNorm(dim,eps=1e-5)
        self.trace_observer=None
        self.motion_proj=nn.Linear(3,dim,bias=False) if motion_enabled else None

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
        parent_input=torch.cat((mx,mean,density,occupancy.to(x.dtype)),1)
        if self.trace_observer is not None:
            self.trace_observer(dict(child_to_parent=parent_map,child_octant=slot,
                child_feature=u,child_count=child_count,mean=mean,maximum=mx,
                occupancy=occupancy,density=density,parent_input=parent_input))
        return parent_input,occupancy,slot

    def forward(self,x,child_level,parent_level,parent_map,motion_features=None):
        parent_input,_,_=self.parent_input(x,child_level,parent_level,parent_map)
        base=self.parent(parent_input)
        if self.motion_proj is not None:
            if motion_features is None or motion_features.shape != (len(base),3):
                raise ValueError("Parent MTCE features must have shape [Np,3]")
            base=base+self.motion_proj(motion_features.to(base.dtype))
        return self.out_norm(base)

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
        self.trace_observer=None; self.trace_focus_token=0
    def forward_group(self,x,coords,bias,group_ids=None):
        n=len(x); normalized=self.norm1(x); qkv=self.qkv(normalized).reshape(n,3,self.heads,self.head_dim).permute(1,2,0,3); q,k,v=qkv
        position_bias=bias(coords)
        logits=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)+position_bias
        weights=torch.softmax(logits.float(),-1).to(v.dtype)
        attended=(weights@v).transpose(0,1).reshape(n,-1)
        y=x+self.proj(attended); output=y+self.ffn(self.norm2(y))
        if self.trace_observer is not None and group_ids is not None:
            self.trace_observer(dict(token_ids=group_ids,coords=coords,input=x,norm1=normalized,
                q=q,k=k,v=v,position_bias=position_bias,logits=logits,weights=weights,attended=attended,
                attention_residual=y,output=output,mask=torch.zeros(n,dtype=torch.bool,device=x.device)))
        return output
    def forward_local_groups(self,x,coords,groups,bias):
        """Evaluate independent local windows in one padded batched kernel."""
        if not groups:return torch.empty_like(x)
        lengths=torch.tensor([len(idx) for idx in groups],device=x.device,dtype=torch.long)
        width=int(lengths.max());flat_ids=torch.cat(groups);group_id=torch.repeat_interleave(torch.arange(len(groups),device=x.device),lengths)
        starts=torch.repeat_interleave(torch.cumsum(lengths,0)-lengths,lengths);position=torch.arange(len(flat_ids),device=x.device)-starts
        padded=x.new_zeros((len(groups),width,x.shape[1]));padded_coords=coords.new_zeros((len(groups),width,3));valid=torch.zeros((len(groups),width),device=x.device,dtype=torch.bool)
        padded[group_id,position]=x[flat_ids];padded_coords[group_id,position]=coords[flat_ids];valid[group_id,position]=True
        normalized=self.norm1(padded)
        qkv=self.qkv(normalized).reshape(len(groups),width,3,self.heads,self.head_dim).permute(2,0,3,1,4);q,k,v=qkv
        position_bias=bias(padded_coords)
        logits=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)+position_bias
        logits=logits.masked_fill(~valid[:,None,None,:],-torch.inf)
        weights=torch.softmax(logits.float(),-1).to(v.dtype)
        attended=(weights@v).transpose(1,2).reshape(len(groups),width,-1)
        y=padded+self.proj(attended);z=y+self.ffn(self.norm2(y));out=torch.empty_like(x);out[flat_ids]=z[group_id,position]
        if self.trace_observer is not None:
            target=next((i for i,ids in enumerate(groups) if bool((ids==self.trace_focus_token).any())),0)
            count=len(groups[target]);sl=slice(0,count)
            self.trace_observer(dict(token_ids=groups[target],coords=padded_coords[target,sl],
                input=padded[target,sl],norm1=normalized[target,sl],q=q[target,:,sl],
                k=k[target,:,sl],v=v[target,:,sl],position_bias=position_bias[target,:,sl,sl],logits=logits[target,:,sl,sl],
                weights=weights[target,:,sl,sl],attended=attended[target,sl],
                attention_residual=y[target,sl],output=z[target,sl],mask=~valid[target,sl]))
        return out
    def forward(self,x,coords,groups,bias,batched_local=False):
        if batched_local:return self.forward_local_groups(x,coords,groups,bias)
        out=torch.empty_like(x)
        for idx in groups:
            trace_ids=idx if self.trace_observer is not None and bool((idx==self.trace_focus_token).any()) else None
            out[idx]=self.forward_group(x[idx],coords[idx],bias,trace_ids)
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
        self.trace_observer=None
    def forward(self,x,level):
        x=x+self.position(level.centers)
        for i,block in enumerate(self.blocks):
            groups=global_groups(level) if self.global_attention else local_groups(level,self.window_size,0 if i%2==0 else self.window_shift)
            if self.trace_observer is not None:
                self.trace_observer(i, groups, 0 if self.global_attention or i%2==0 else self.window_shift)
            x=block(x,level.coords,groups,self.bias,batched_local=not self.global_attention)
        return x

class SparseUp(nn.Module):
    def __init__(self,dim=128):
        super().__init__(); self.parent_norm=nn.LayerNorm(dim,eps=1e-5); self.parent=nn.Linear(dim,dim)
        self.norm=nn.LayerNorm(dim,eps=1e-5); self.ffn=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.trace_observer=None
    def forward(self,fine,coarse,parent_map):
        parent_feature=self.parent(self.parent_norm(coarse))[parent_map]
        y=fine+parent_feature
        output=y+self.ffn(self.norm(y))
        if self.trace_observer is not None:
            self.trace_observer(dict(skip=fine,coarse=coarse,parent_map=parent_map,
                                     parent_feature=parent_feature,residual=y,output=output))
        return output

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
        # Opt-in A/B initialization isolates frontend RNG consumption so that
        # all downstream initial weights match for an identical experiment seed.
        matched_init = bool(m['voxel'].get('matched_frontend_init', False))
        frontend_seed = int(m['voxel'].get('frontend_seed', 42))
        if matched_init:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(frontend_seed)
                self.voxel_embed = build_voxel_embedding(m)
        else:
            self.voxel_embed = build_voxel_embedding(m)
        motion_cfg=m["voxel"].get("motion_centroid",{})
        self.motion_enabled=bool(motion_cfg.get("enabled",False))
        if self.motion_enabled and (m["voxel"].get("embedding") != "learned_sbe" or d != 128):
            raise ValueError("MTCE-3 requires Learned-SBE with dim=128")
        if self.motion_enabled and (motion_cfg.get("injection","pre_norm_residual") != "pre_norm_residual" or not motion_cfg.get("zero_init",True)):
            raise ValueError("MTCE-3 requires pre_norm_residual and zero_init=true")
        self.temporal_centroid=TemporalCentroidFeatures(motion_cfg) if self.motion_enabled else None
        self.merge01=SparseMerge(d,self.motion_enabled); self.merge12=SparseMerge(d,self.motion_enabled)
        common=(d,tr["heads"],tr["ffn_dim"],tr["blocks_per_level"][0],tr["window_size"],tr["window_shift"],tr["relative_range"])
        self.encoder0=SpatialTransformer(*common,global_attention=False)
        self.encoder1=SpatialTransformer(d,tr["heads"],tr["ffn_dim"],tr["blocks_per_level"][1],tr["window_size"],tr["window_shift"],tr["relative_range"],False)
        self.encoder2=SpatialTransformer(d,tr["heads"],tr["ffn_dim"],tr["blocks_per_level"][2],tr["window_size"],tr["window_shift"],tr["relative_range"],True)
        self.up21=SparseUp(d); self.up10=SparseUp(d); self.final_norm=nn.LayerNorm(d,eps=1e-5); self.head=CandidateHead(d,m["head"]["hidden_dim"],m["head"]["residual_scale_m"])
        self.trace_observer=None
        if matched_init:
            for child in self.children():
                if child is self.voxel_embed:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(frontend_seed + 1)
                        child.apply(self._init)
                else:
                    child.apply(self._init)
        else:
            self.apply(self._init)
        if isinstance(self.voxel_embed, (LegacyVoxelEmbed, LearnedSBEVoxelEmbed)):
            nn.init.constant_(self.voxel_embed.sensor_embedding.weight[0], 0.)
            nn.init.constant_(self.voxel_embed.sensor_embedding.weight[1], 1.)
        for module in self.modules():
            if isinstance(module,RelativeBias): nn.init.zeros_(module.tables)
        nn.init.constant_(self.head.cls[-1].bias,math.log(.01/.99))
        if self.motion_enabled:
            for module in (self.voxel_embed,self.merge01,self.merge12):
                nn.init.zeros_(module.motion_proj.weight)
        # Opt-in heads are constructed AFTER baseline initialization so adding
        # them does not perturb the original backbone's RNG sequence.
        quality_cfg = m.get("quality_head", {})
        self.quality_head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
                             if quality_cfg.get("enabled", False) else None)
        self.metadata_proj = nn.Linear(2, d, bias=False) if m.get("head_metadata", False) else None
        if self.metadata_proj is not None:
            nn.init.zeros_(self.metadata_proj.weight)
        query_cfg = m.get("query_decoder", {})
        self.query_decoder = None
        if query_cfg.get("enabled", False):
            from .radar_query_decoder import RadarQueryDecoder
            self.query_decoder = RadarQueryDecoder(d, query_cfg)
            if self.quality_head is not None:
                raise ValueError("query decoder and dense quality head are separate experiments")
    @staticmethod
    def _init(module):
        if isinstance(module,nn.Linear):
            nn.init.trunc_normal_(module.weight,std=.02,a=-.04,b=.04)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
    def spatial_forward(self,batch):
        h=self.hierarchy(batch["points"],batch["point_batch_index"]); l0,l1,l2=h.levels
        if self.trace_observer is not None: self.trace_observer(h)
        motion,temporal_stats=self.temporal_centroid(batch,h) if self.motion_enabled else ([None]*3,[])
        embedding=self.voxel_embed(batch,h,motion_features=motion[0]) if self.motion_enabled else self.voxel_embed(batch,h)
        f0=self.encoder0(embedding,l0)
        f1=self.encoder1(self.merge01(f0,l0,l1,h.parent_l0_to_l1,motion[1]),l1)
        f2=self.encoder2(self.merge12(f1,l1,l2,h.parent_l1_to_l2,motion[2]),l2)
        d1=self.up21(f1,f2,h.parent_l1_to_l2); q=self.final_norm(self.up10(f0,d1,h.parent_l0_to_l1))
        if self.metadata_proj is not None:
            metadata=torch.stack((torch.log1p(torch.linalg.vector_norm(l0.centers.float(),dim=1)),
                                  torch.log1p(l0.point_count.float())),1)
            q=q+self.metadata_proj(metadata.to(q.dtype))
        logits,residual,pred=self.head(q,l0.centers)
        output = {"logits":logits,"residual_xyz":residual,"pred_xyz":pred,"fine_features":q,"voxel_centers":l0.centers,
                "source_token_id":torch.arange(len(q),device=q.device),"batch_index":l0.batch_index,"layouts":h,"motion_features":motion if self.motion_enabled else None,"temporal_stats":temporal_stats,
                "aux_stats":{"token_counts":[len(x.coords) for x in h.levels],"num_samples":int(batch["num_samples"]),"attention_backend":"explicit_pytorch_scaled_dot_product_with_additive_axis_bias"}}

        if self.quality_head is not None:
            output["quality_logits"] = self.quality_head(q).squeeze(-1)
        return output

    def forward(self,batch):
        """Packed query points -> dense candidates or opt-in set predictions."""
        dense = self.spatial_forward(batch)
        return self.query_decoder(dense) if self.query_decoder is not None else dense

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
