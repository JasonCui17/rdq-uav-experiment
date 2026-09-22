from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from .hypothesis import HypothesisSet


@dataclass(frozen=True)
class ReliabilityGateOutput:
    weights: torch.Tensor
    joint_logit: torch.Tensor
    joint_score: torch.Tensor
    fused_score: torch.Tensor


class ReliabilityGate(nn.Module):
    def __init__(self, feature_dim: int=128, association_dim: int=4, type_dim: int=16, hidden_dim: int=128) -> None:
        super().__init__()
        self.missing_r=nn.Parameter(torch.zeros(feature_dim))
        self.missing_v=nn.Parameter(torch.zeros(feature_dim))
        self.type_embedding=nn.Embedding(3,type_dim)
        in_dim=2*feature_dim+association_dim+2+type_dim
        self.gate=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,3))
        self.joint=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))

    def _input(self,h: HypothesisSet) -> torch.Tensor:
        rf=torch.where(h.m_R[:,None],h.radar_feature,self.missing_r[None].expand(h.n,-1))
        vf=torch.where(h.m_V[:,None],h.vision_feature,self.missing_v[None].expand(h.n,-1))
        presence=torch.stack((h.m_R,h.m_V),1).to(rf.dtype)
        typ=self.type_embedding(h.hypothesis_type)
        return torch.cat((rf,vf,h.association_info,presence,typ),1)

    def forward(self,h: HypothesisSet) -> ReliabilityGateOutput:
        if h.n==0:
            z=h.radar_feature.new_empty((0,))
            return ReliabilityGateOutput(h.radar_feature.new_empty((0,3)),z,z,z)
        x=self._input(h)
        logits=self.gate(x)
        neg=torch.finfo(logits.dtype).min
        logits=logits.clone()
        logits[:,0]=torch.where(h.m_R,logits[:,0],torch.full_like(logits[:,0],neg))
        logits[:,1]=torch.where(h.m_V,logits[:,1],torch.full_like(logits[:,1],neg))
        weights=torch.softmax(logits.float(),dim=1).to(logits.dtype)
        joint_logit=self.joint(x).squeeze(1)
        joint_score=torch.sigmoid(joint_logit)
        fused=weights[:,0]*h.radar_score+weights[:,1]*h.vision_score+weights[:,2]*joint_score
        return ReliabilityGateOutput(weights,joint_logit,joint_score,fused)
