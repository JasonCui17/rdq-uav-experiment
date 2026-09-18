"""Non-learned packed sparse hierarchy with fixed global origin."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class SparseLevel:
    coords: torch.Tensor       # [N,3] integer voxel coordinate
    batch_index: torch.Tensor  # [N]
    centers: torch.Tensor      # [N,3] meters
    point_count: torch.Tensor  # [N]

@dataclass
class SparseHierarchy:
    levels: list[SparseLevel]
    point_to_l0: torch.Tensor
    parent_l0_to_l1: torch.Tensor
    parent_l1_to_l2: torch.Tensor

def _unique(keys: torch.Tensor):
    unique,inverse,counts=torch.unique(keys,dim=0,sorted=True,return_inverse=True,return_counts=True)
    return unique,inverse,counts

class HierarchyBuilder:
    def __init__(self, scales=(.5,1.,2.)):
        if tuple(scales)!=(.5,1.,2.): raise ValueError("V1 requires scales 0.5,1,2")
        self.scales=tuple(scales)
    def __call__(self,points:torch.Tensor,point_batch:torch.Tensor)->SparseHierarchy:
        g0=torch.floor(points/self.scales[0]).to(torch.long)
        keys0=torch.cat((point_batch[:,None].long(),g0),1); u0,p2v,c0=_unique(keys0)
        l0=SparseLevel(u0[:,1:],u0[:,0],(u0[:,1:].float()+.5)*self.scales[0],c0)
        g1=torch.div(l0.coords,2,rounding_mode="floor"); u1,v01,k1=_unique(torch.cat((l0.batch_index[:,None],g1),1))
        c1=torch.zeros(len(u1),device=points.device,dtype=torch.long).index_add_(0,v01,l0.point_count)
        l1=SparseLevel(u1[:,1:],u1[:,0],(u1[:,1:].float()+.5)*self.scales[1],c1)
        g2=torch.div(l1.coords,2,rounding_mode="floor"); u2,v12,k2=_unique(torch.cat((l1.batch_index[:,None],g2),1))
        c2=torch.zeros(len(u2),device=points.device,dtype=torch.long).index_add_(0,v12,l1.point_count)
        l2=SparseLevel(u2[:,1:],u2[:,0],(u2[:,1:].float()+.5)*self.scales[2],c2)
        return SparseHierarchy([l0,l1,l2],p2v,v01,v12)
