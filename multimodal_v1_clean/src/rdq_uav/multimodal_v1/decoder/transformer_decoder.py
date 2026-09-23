from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DecoderAux:
    active_batch_index: torch.Tensor
    memory_key_padding_mask: torch.Tensor
    target_key_padding_mask: torch.Tensor


class FusionTransformerDecoder(nn.Module):
    def __init__(self, dim: int=128, heads: int=4, ffn_dim: int=256, layers: int=2, dropout: float=.1, vision_dim: int=384) -> None:
        super().__init__()
        if dim!=128 or layers!=2:
            raise ValueError('Frozen V1 decoder requires dim=128 and layers=2')
        self.radar_proj=nn.Identity()
        self.vision_proj=nn.Linear(vision_dim,dim)
        layer=nn.TransformerDecoderLayer(d_model=dim,nhead=heads,dim_feedforward=ffn_dim,dropout=dropout,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,num_layers=layers,norm=nn.LayerNorm(dim))

    def forward(self, query: torch.Tensor, query_batch_index: torch.Tensor,
                radar_r2: torch.Tensor, radar_batch_index: torch.Tensor,
                vision_v2: torch.Tensor, *, sample_m_R: torch.Tensor|None=None,
                sample_m_V: torch.Tensor|None=None, vision_padding_mask: torch.Tensor|None=None,
                return_aux: bool=False):
        if query.ndim!=2 or query.shape[1]!=128: raise ValueError('query must be [N,128]')
        if query_batch_index.shape!=(len(query),): raise ValueError('query_batch_index must be [N]')
        if radar_r2.ndim!=2 or radar_r2.shape[1]!=128: raise ValueError('radar_r2 must be [Nr,128]')
        if radar_batch_index.shape!=(len(radar_r2),): raise ValueError('radar_batch_index must be [Nr]')
        if vision_v2.ndim!=4 or vision_v2.shape[1]!=self.vision_proj.in_features: raise ValueError('vision_v2 must be [B,C,H,W]')
        if len(query)==0:
            aux=DecoderAux(query_batch_index.new_empty(0),torch.empty((0,0),device=query.device,dtype=torch.bool),torch.empty((0,0),device=query.device,dtype=torch.bool)) if return_aux else None
            return query,aux
        B=vision_v2.shape[0]
        if sample_m_R is None: sample_m_R=torch.ones(B,device=query.device,dtype=torch.bool)
        if sample_m_V is None: sample_m_V=torch.ones(B,device=query.device,dtype=torch.bool)
        if sample_m_R.shape!=(B,) or sample_m_V.shape!=(B,): raise ValueError('sample modality masks must be [B]')
        if vision_padding_mask is None:
            vision_mask_flat=torch.zeros((B,vision_v2.shape[-2]*vision_v2.shape[-1]),device=query.device,dtype=torch.bool)
        else:
            if vision_padding_mask.dtype!=torch.bool or vision_padding_mask.ndim!=3 or vision_padding_mask.shape[0]!=B:
                raise ValueError('vision_padding_mask must be bool [B,H,W]')
            vision_mask_flat=F.interpolate(
                vision_padding_mask[:,None].float(),size=vision_v2.shape[-2:],mode='nearest'
            )[:,0].to(torch.bool).flatten(1)
        active=torch.unique(query_batch_index,sorted=True)
        qgroups=[torch.nonzero(query_batch_index==b).flatten() for b in active]
        qmax=max(len(x) for x in qgroups)
        tgt=query.new_zeros((len(active),qmax,128)); tmask=torch.ones((len(active),qmax),device=query.device,dtype=torch.bool)
        for i,ids in enumerate(qgroups): tgt[i,:len(ids)]=query[ids]; tmask[i,:len(ids)]=False
        vf=self.vision_proj(vision_v2.permute(0,2,3,1).reshape(B,-1,vision_v2.shape[1]))
        rg=self.radar_proj(radar_r2)
        memories=[]; memory_masks=[]
        for b in active.tolist():
            parts=[]; masks=[]
            if bool(sample_m_R[b]):
                rid=torch.nonzero(radar_batch_index==b).flatten()
                if len(rid):
                    parts.append(rg[rid]); masks.append(torch.zeros(len(rid),device=query.device,dtype=torch.bool))
            if bool(sample_m_V[b]):
                parts.append(vf[b]); masks.append(vision_mask_flat[b])
            if not parts: raise RuntimeError('active hypothesis sample has no decoder memory')
            memories.append(torch.cat(parts,0)); memory_masks.append(torch.cat(masks,0))
        mmax=max(len(x) for x in memories)
        memory=query.new_zeros((len(active),mmax,128)); mmask=torch.ones((len(active),mmax),device=query.device,dtype=torch.bool)
        for i,m in enumerate(memories): memory[i,:len(m)]=m; mmask[i,:len(m)]=memory_masks[i]
        decoded=self.decoder(tgt,memory,tgt_key_padding_mask=tmask,memory_key_padding_mask=mmask)
        out=torch.empty_like(query)
        for i,ids in enumerate(qgroups): out[ids]=decoded[i,:len(ids)]
        aux=DecoderAux(active,mmask,tmask) if return_aux else None
        return out,aux
