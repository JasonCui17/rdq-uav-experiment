from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .candidate.hypothesis import HypothesisSet,HYP_RV,HYP_R,HYP_V


@dataclass(frozen=True)
class FusionTargets:
    gt_box_xyxy_px: torch.Tensor
    gt_2d_valid: torch.Tensor
    gt_xyz: torch.Tensor
    gt_3d_valid: torch.Tensor

    def validate(self,batch_size:int)->None:
        if self.gt_box_xyxy_px.shape!=(batch_size,4): raise ValueError('gt_box_xyxy_px must be [B,4]')
        if self.gt_2d_valid.shape!=(batch_size,) or self.gt_2d_valid.dtype!=torch.bool: raise ValueError('gt_2d_valid must be bool [B]')
        if self.gt_xyz.shape!=(batch_size,3): raise ValueError('gt_xyz must be [B,3]')
        if self.gt_3d_valid.shape!=(batch_size,) or self.gt_3d_valid.dtype!=torch.bool: raise ValueError('gt_3d_valid must be bool [B]')


def _box_iou_diag(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    x1=torch.maximum(a[:,0],b[:,0]); y1=torch.maximum(a[:,1],b[:,1]); x2=torch.minimum(a[:,2],b[:,2]); y2=torch.minimum(a[:,3],b[:,3])
    inter=(x2-x1).clamp_min(0)*(y2-y1).clamp_min(0)
    aa=(a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0); bb=(b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0)
    union=aa+bb-inter
    return torch.where(union>0,inter/union,torch.zeros_like(union))


def _giou_loss(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    iou=_box_iou_diag(a,b)
    cx1=torch.minimum(a[:,0],b[:,0]); cy1=torch.minimum(a[:,1],b[:,1]); cx2=torch.maximum(a[:,2],b[:,2]); cy2=torch.maximum(a[:,3],b[:,3])
    ca=(cx2-cx1).clamp_min(0)*(cy2-cy1).clamp_min(0)
    x1=torch.maximum(a[:,0],b[:,0]); y1=torch.maximum(a[:,1],b[:,1]); x2=torch.minimum(a[:,2],b[:,2]); y2=torch.minimum(a[:,3],b[:,3])
    inter=(x2-x1).clamp_min(0)*(y2-y1).clamp_min(0)
    aa=(a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0); bb=(b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0)
    union=aa+bb-inter
    giou=iou-torch.where(ca>0,(ca-union)/ca,torch.zeros_like(ca))
    return 1.-giou


def _xyxy_to_norm_cxcywh(box:torch.Tensor,wh:torch.Tensor)->torch.Tensor:
    cx=.5*(box[:,0]+box[:,2])/wh[:,0]; cy=.5*(box[:,1]+box[:,3])/wh[:,1]
    w=(box[:,2]-box[:,0])/wh[:,0]; h=(box[:,3]-box[:,1])/wh[:,1]
    return torch.stack((cx,cy,w,h),1)


class FusionLoss:
    def __init__(self, *, lambda_cls=1.,lambda_2d=2.,lambda_3d=2.,lambda_valid=.25,focal_alpha=.25,focal_gamma=2.) -> None:
        self.lambda_cls=float(lambda_cls); self.lambda_2d=float(lambda_2d); self.lambda_3d=float(lambda_3d); self.lambda_valid=float(lambda_valid)
        self.alpha=float(focal_alpha); self.gamma=float(focal_gamma)

    def __call__(self, *, fused_score:torch.Tensor, pred_box:torch.Tensor,pred_xyz:torch.Tensor,c2d_logit:torch.Tensor,c3d_logit:torch.Tensor,
                 hypotheses:HypothesisSet,targets:FusionTargets,source_image_size_wh:torch.Tensor)->dict[str,torch.Tensor]:
        n=hypotheses.n; zero=fused_score.sum()*0
        B=int(source_image_size_wh.shape[0]); targets.validate(B)
        if n==0:
            return {k:zero for k in ('loss','loss_cls','loss_2d','loss_3d','loss_valid','loss_c2d','loss_c3d')}
        b=hypotheses.batch_index; gt2=targets.gt_box_xyxy_px[b]; gt3=targets.gt_xyz[b]; v2=targets.gt_2d_valid[b]; v3=targets.gt_3d_valid[b]
        evidence_iou=torch.zeros(n,device=fused_score.device,dtype=fused_score.dtype)
        if bool(hypotheses.vision_box_valid.any()):
            m=hypotheses.vision_box_valid & v2; evidence_iou[m]=_box_iou_diag(hypotheses.vision_box_xyxy_px[m],gt2[m])
        evidence_err=torch.full((n,),torch.inf,device=fused_score.device,dtype=fused_score.dtype)
        if bool(hypotheses.radar_xyz_valid.any()):
            m=hypotheses.radar_xyz_valid & v3; evidence_err[m]=torch.linalg.vector_norm(hypotheses.radar_xyz[m]-gt3[m],dim=1)
        rv=hypotheses.hypothesis_type==HYP_RV; rr=hypotheses.hypothesis_type==HYP_R; vv=hypotheses.hypothesis_type==HYP_V
        positive=(rv & v2 & v3 & (evidence_iou>=.5)&(evidence_err<=1.)) | (rr & v3 & (evidence_err<=1.)) | (vv & v2 & (evidence_iou>=.5))
        ignore=(rv & (((v3)&(evidence_err>1.)&(evidence_err<=2.)) | ((v2)&(evidence_iou>=.3)&(evidence_iou<.5)))) | (rr&v3&(evidence_err>1.)&(evidence_err<=2.)) | (vv&v2&(evidence_iou>=.3)&(evidence_iou<.5))
        supervisable=(rv & v2 & v3) | (rr & v3) | (vv & v2)
        cls_mask=supervisable & ~ignore
        if bool(cls_mask.any()):
            p=fused_score[cls_mask].clamp(1e-6,1-1e-6); y=positive[cls_mask].to(p.dtype)
            ce=-(y*torch.log(p)+(1-y)*torch.log1p(-p)); pt=y*p+(1-y)*(1-p); alpha=y*self.alpha+(1-y)*(1-self.alpha)
            lcls=(alpha*(1-pt).pow(self.gamma)*ce).mean()
        else: lcls=zero
        reg2=positive & v2; reg3=positive & v3
        if bool(reg2.any()):
            wh=source_image_size_wh[b[reg2]].to(pred_box.dtype)
            l1=F.l1_loss(_xyxy_to_norm_cxcywh(pred_box[reg2],wh),_xyxy_to_norm_cxcywh(gt2[reg2],wh))
            giou=_giou_loss(pred_box[reg2],gt2[reg2]).mean(); l2=l1+2.*giou
        else: l2=zero
        l3=F.smooth_l1_loss(pred_xyz[reg3],gt3[reg3]) if bool(reg3.any()) else zero
        # Reliability target uses prediction quality, detached, and is supervised whenever GT exists regardless of modality provenance.
        if bool(v2.any()):
            t2=_box_iou_diag(pred_box[v2],gt2[v2]).detach().clamp(0,1); lc2=F.binary_cross_entropy_with_logits(c2d_logit[v2],t2)
        else: lc2=zero
        if bool(v3.any()):
            err=torch.linalg.vector_norm(pred_xyz[v3]-gt3[v3],dim=1); t3=torch.exp(-err/1.).detach(); lc3=F.binary_cross_entropy_with_logits(c3d_logit[v3],t3)
        else: lc3=zero
        lvalid=lc2+lc3
        total=self.lambda_cls*lcls+self.lambda_2d*l2+self.lambda_3d*l3+self.lambda_valid*lvalid
        return {'loss':total,'loss_cls':lcls,'loss_2d':l2,'loss_3d':l3,'loss_valid':lvalid,'loss_c2d':lc2,'loss_c3d':lc3,
                'num_positive':positive.sum().to(torch.long),'num_ignore':ignore.sum().to(torch.long)}
