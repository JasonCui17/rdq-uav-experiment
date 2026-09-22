from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from ..candidate.hypothesis import HypothesisSet,HYP_RV,HYP_R,HYP_V


@dataclass(frozen=True)
class FusionHeadOutput:
    box_xyxy_px: torch.Tensor
    xyz: torch.Tensor
    c2d_logit: torch.Tensor
    c3d_logit: torch.Tensor


class FusionPredictionHeads(nn.Module):
    def __init__(self, dim: int=128, xyz_mean=(0.,0.,0.), xyz_std=(1.,1.,1.)) -> None:
        super().__init__()
        self.box_head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,4))
        self.xyz_head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,3))
        self.c2d_head=nn.Linear(dim,1); self.c3d_head=nn.Linear(dim,1)
        self.register_buffer('xyz_mean',torch.tensor(xyz_mean,dtype=torch.float32))
        self.register_buffer('xyz_std',torch.tensor(xyz_std,dtype=torch.float32).clamp_min(1e-6))

    @staticmethod
    def _box_prior_delta(prior: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x1,y1,x2,y2=prior.unbind(1)
        w=(x2-x1).clamp_min(1.); h=(y2-y1).clamp_min(1.)
        cx=.5*(x1+x2); cy=.5*(y1+y2)
        dx,dy,dw,dh=delta.unbind(1)
        ncx=cx+w*dx; ncy=cy+h*dy
        nw=w*torch.exp(dw.clamp(-4.,4.)); nh=h*torch.exp(dh.clamp(-4.,4.))
        return torch.stack((ncx-.5*nw,ncy-.5*nh,ncx+.5*nw,ncy+.5*nh),1)

    def forward(self, decoded: torch.Tensor, h: HypothesisSet, projected_radar_xy: torch.Tensor,
                source_image_size_wh: torch.Tensor) -> FusionHeadOutput:
        if decoded.shape!=(h.n,128): raise ValueError('decoded must match hypotheses [N,128]')
        if projected_radar_xy.shape!=(h.n,2): raise ValueError('projected_radar_xy must be [N,2] aligned to hypotheses')
        delta_box=self.box_head(decoded)
        raw_xyz=self.xyz_head(decoded)
        box=torch.zeros((h.n,4),device=decoded.device,dtype=decoded.dtype)
        xyz=torch.zeros((h.n,3),device=decoded.device,dtype=decoded.dtype)
        rv_or_v=(h.hypothesis_type==HYP_RV)|(h.hypothesis_type==HYP_V)
        if bool(rv_or_v.any()): box[rv_or_v]=self._box_prior_delta(h.vision_box_xyxy_px[rv_or_v],delta_box[rv_or_v])
        ronly=h.hypothesis_type==HYP_R
        if bool(ronly.any()):
            idx=torch.nonzero(ronly).flatten(); wh=source_image_size_wh[h.batch_index[idx]].to(decoded.dtype)
            center=projected_radar_xy[idx] + torch.tanh(delta_box[idx,:2])*(.25*wh)
            size=torch.exp(delta_box[idx,2:].clamp(-8.,0.))*wh
            box[idx]=torch.cat((center-.5*size,center+.5*size),1)
        has_r=(h.hypothesis_type==HYP_RV)|(h.hypothesis_type==HYP_R)
        if bool(has_r.any()): xyz[has_r]=h.radar_xyz[has_r]+raw_xyz[has_r]
        vonly=h.hypothesis_type==HYP_V
        if bool(vonly.any()): xyz[vonly]=raw_xyz[vonly]*self.xyz_std.to(decoded.dtype)+self.xyz_mean.to(decoded.dtype)
        # Clip boxes to calibrated source frame, preserving differentiability almost everywhere.
        wh=source_image_size_wh[h.batch_index].to(decoded.dtype)
        x1=box[:,0].clamp_min(0); y1=box[:,1].clamp_min(0)
        x2=box[:,2].clamp_min(0); y2=box[:,3].clamp_min(0)
        x1=torch.minimum(x1,wh[:,0]); x2=torch.minimum(x2,wh[:,0])
        y1=torch.minimum(y1,wh[:,1]); y2=torch.minimum(y2,wh[:,1])
        box=torch.stack((x1,y1,x2,y2),1)
        return FusionHeadOutput(box,xyz,self.c2d_head(decoded).squeeze(1),self.c3d_head(decoded).squeeze(1))
