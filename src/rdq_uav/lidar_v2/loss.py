"""GT label construction and sample-normalized V2-base candidate losses."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class CandidateLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__(); c=cfg["loss"] if "loss" in cfg else cfg
        self.alpha=float(c["focal_alpha"]); self.gamma=float(c["focal_gamma"]); self.beta=float(c["smooth_l1_beta"]); self.reg_weight=float(c["reg_weight"])
    def labels(self,outputs,batch):
        h=outputs["layouts"]; inv=h.point_to_l0; n=len(outputs["logits"]); gt=batch["gt_xyz"][outputs["batch_index"]]
        point_gt=batch["gt_xyz"][batch["point_batch_index"]]; dist=torch.linalg.vector_norm(batch["points"]-point_gt,dim=1)
        all_min=dist.new_full((n,),torch.inf); all_min.scatter_reduce_(0,inv,dist,reduce="amin",include_self=True)
        recent_min=dist.new_full((n,),torch.inf); ridx=inv[batch["recent_mask"]]; rdist=dist[batch["recent_mask"]]
        if len(ridx): recent_min.scatter_reduce_(0,ridx,rdist,reduce="amin",include_self=True)
        positive=recent_min<=1.; ignore=(~positive)&(all_min<=2.); negative=all_min>2.
        return positive,ignore,negative,(gt-outputs["voxel_centers"])/1.0
    def forward(self,outputs,batch):
        pos,ignore,neg,target=self.labels(outputs,batch); losses=[]; cls_items=[]; reg_items=[]; supervised=0; no_support=0
        num_pos=num_neg=num_ignore=0
        for b in range(len(batch["gt_xyz"])):
            token=outputs["batch_index"]==b; p=pos&token; i=ignore&token; n=neg&token; np_=int(p.sum()); num_pos+=np_; num_neg+=int(n.sum()); num_ignore+=int(i.sum())
            if np_==0: no_support+=1; continue
            supervised+=1; valid=p|n; logits=outputs["logits"][valid]; targets=p[valid].float()
            ce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none"); prob=torch.sigmoid(logits); pt=targets*prob+(1-targets)*(1-prob)
            alpha=targets*self.alpha+(1-targets)*(1-self.alpha); cls=(alpha*(1-pt).pow(self.gamma)*ce).sum()/max(1,np_)
            reg=F.smooth_l1_loss(outputs["residual_xyz"][p],target[p],beta=self.beta,reduction="none").sum()/max(1,np_)
            cls_items.append(cls); reg_items.append(reg); losses.append(cls+self.reg_weight*reg)
        zero=outputs["logits"].sum()*0
        mean=lambda xs: torch.stack(xs).mean() if xs else zero
        return {"loss":mean(losses),"loss_cls":mean(cls_items),"loss_reg":mean(reg_items),"num_pos":num_pos,"num_neg":num_neg,
                "num_ignore":num_ignore,"num_supervised_samples":supervised,"num_no_current_support":no_support,
                "positive_mask":pos,"ignore_mask":ignore,"negative_mask":neg}
