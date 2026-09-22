from __future__ import annotations

import torch
from torch import nn
from .hypothesis import HypothesisSet


class TypedSharedQuery(nn.Module):
    def __init__(self, feature_dim: int=128, association_dim: int=4, query_dim: int=128) -> None:
        super().__init__()
        self.missing_r=nn.Parameter(torch.zeros(feature_dim))
        self.missing_v=nn.Parameter(torch.zeros(feature_dim))
        self.type_input=nn.Embedding(3,16)
        self.type_output=nn.Embedding(3,query_dim)
        in_dim=2*feature_dim+association_dim+2+16
        self.net=nn.Sequential(nn.Linear(in_dim,256),nn.GELU(),nn.Linear(256,query_dim),nn.LayerNorm(query_dim))

    def forward(self,h: HypothesisSet) -> torch.Tensor:
        if h.n==0: return h.radar_feature.new_empty((0,128))
        rf=torch.where(h.m_R[:,None],h.radar_feature,self.missing_r[None].expand(h.n,-1))
        vf=torch.where(h.m_V[:,None],h.vision_feature,self.missing_v[None].expand(h.n,-1))
        presence=torch.stack((h.m_R,h.m_V),1).to(rf.dtype)
        x=torch.cat((rf,vf,h.association_info,presence,self.type_input(h.hypothesis_type)),1)
        return self.net(x)+self.type_output(h.hypothesis_type)
