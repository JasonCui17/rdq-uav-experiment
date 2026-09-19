"""Query-level temporal modules. All times are seconds, XYZ is reference meters."""
import math
import torch
from torch import nn

class CandidateAwareQueryPool(nn.Module):
    """Per-query softmax(logits) over ALL fine tokens -> [Nq,D], no hard selection."""
    def __init__(self,dim=128):
        super().__init__();self.missing_lidar_token=nn.Parameter(torch.zeros(dim))
        self.xyz_embed=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,dim))
        self.norm=nn.LayerNorm(dim,eps=1e-5)
    def forward(self,outputs,num_samples):
        tokens=[];observed=[]
        for i in range(num_samples):
            mask=outputs['batch_index']==i;has=bool(mask.any());observed.append(has)
            if has:
                a=torch.softmax(outputs['logits'][mask].float(),dim=0)
                f=(a[:,None]*outputs['fine_features'][mask].float()).sum(0)
                p=(a[:,None]*outputs['pred_xyz'][mask].float()).sum(0)
                tokens.append(self.norm(f+self.xyz_embed(p/100.)))
            else:tokens.append(self.missing_lidar_token)
        return torch.stack(tokens),torch.tensor(observed,device=outputs['logits'].device,dtype=torch.bool)

    @torch.no_grad()
    def diagnostics(self,outputs,num_samples,gt_xyz=None,radius_m=1.):
        """Small per-query readout diagnostics; never changes forward or loss."""
        rows=[]
        for spatial_query_idx in range(num_samples):
            mask=outputs['batch_index']==spatial_query_idx;count=int(mask.sum())
            if not count:
                rows.append(dict(candidate_count=0,pool_entropy=0.,max_objectness_probability=0.,
                    reference_xyz=None,pooled_xyz=None,gt_near_pool_weight=None));continue
            logits=outputs['logits'][mask].float();weights=torch.softmax(logits,0);xyz=outputs['pred_xyz'][mask].float()
            reference=xyz[torch.argmax(logits)];pooled=(weights[:,None]*xyz).sum(0)
            near=None
            if gt_xyz is not None:
                near=float(weights[torch.linalg.vector_norm(xyz-gt_xyz[spatial_query_idx].float(),dim=1)<=radius_m].sum())
            rows.append(dict(candidate_count=count,pool_entropy=float(-(weights*weights.clamp_min(1e-12).log()).sum()),
                max_objectness_probability=float(torch.sigmoid(logits.max())),reference_xyz=reference.cpu().tolist(),
                pooled_xyz=pooled.cpu().tolist(),gt_near_pool_weight=near))
        return rows

class TimeEncoding(nn.Module):
    """Float64 timestamp differences first, then FP32 [tau,dt_prev] -> [B,T,D]."""
    def __init__(self,dim=128,hidden_dim=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,dim))
    def forward(self,times,valid):
        tau=times-times[:,:1];dt=torch.cat((torch.zeros_like(times[:,:1]),times[:,1:]-times[:,:-1]),1)
        features=torch.stack((tau,dt),-1).float().masked_fill(~valid[:,:,None],0.)
        return self.net(features)

def causal_mask(length,device=None):
    return torch.full((length,length),float('-inf'),device=device).triu(1)

class TemporalBlock(nn.Module):
    """Pre-LN per-clip causal MHSA; absent observation != invalid query."""
    def __init__(self,dim,heads,ffn_dim,dropout):
        super().__init__();self.heads=heads;self.head_dim=dim//heads
        self.norm1=nn.LayerNorm(dim,eps=1e-5);self.qkv=nn.Linear(dim,3*dim);self.proj=nn.Linear(dim,dim)
        self.norm2=nn.LayerNorm(dim,eps=1e-5);self.ffn=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn_dim,dim))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,valid):
        B,T,D=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(B,T,3,self.heads,self.head_dim).permute(2,0,3,1,4)
        logits=(q@k.transpose(-2,-1)).float()/math.sqrt(self.head_dim)+causal_mask(T,x.device)
        allowed=valid[:,None,:] & torch.ones((T,T),device=x.device,dtype=torch.bool).tril()[None,:,:]
        # For padded queries with no keys, temporarily permit their own diagonal;
        # their outputs are zeroed and never usable as keys by valid queries.
        allowed=allowed | ((~valid)[:,:,None]&torch.eye(T,device=x.device,dtype=torch.bool)[None,:,:])
        logits=logits.masked_fill(~allowed[:,None,:,:],-torch.inf)
        weights=self.dropout(torch.softmax(logits,-1)).to(v.dtype)
        attended=(weights@v).transpose(1,2).reshape(B,T,D)
        y=x+self.proj(attended);z=y+self.ffn(self.norm2(y))
        return z.masked_fill(~valid[:,:,None],0.)

class CausalTemporalTransformer(nn.Module):
    def __init__(self,dim=128,heads=4,ffn_dim=256,blocks=2,dropout=0.):
        super().__init__();self.blocks=nn.ModuleList([TemporalBlock(dim,heads,ffn_dim,dropout) for _ in range(blocks)])
    def forward(self,x,valid):
        for block in self.blocks:x=block(x,valid)
        return x

class TemporalXYZHead(nn.Module):
    """[B,T,128] -> absolute reference-frame XYZ [B,T,3], meters."""
    def __init__(self,dim=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,3))
    def forward(self,x):return self.net(x)
