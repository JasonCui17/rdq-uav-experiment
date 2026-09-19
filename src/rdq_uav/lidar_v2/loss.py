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
        if 'spatial_target_xyz' not in batch and 'target_xyz' not in batch:
            n=len(outputs['logits']);mask=torch.zeros(n,dtype=torch.bool,device=outputs['logits'].device)
            return mask,mask.clone(),mask.clone(),outputs['pred_xyz'].new_zeros((n,3))
        spatial_gt=batch.get('spatial_target_xyz',batch['target_xyz']);spatial_valid=batch.get('spatial_target_valid',batch['target_valid'])
        h=outputs["layouts"]; inv=h.point_to_l0; n=len(outputs["logits"]); gt=spatial_gt[outputs["batch_index"]]
        point_gt=spatial_gt[batch["point_batch_index"]]; dist=torch.linalg.vector_norm(batch["points"]-point_gt,dim=1)
        all_min=dist.new_full((n,),torch.inf); all_min.scatter_reduce_(0,inv,dist,reduce="amin",include_self=True)
        recent_min=dist.new_full((n,),torch.inf); ridx=inv[batch["recent_mask"]]; rdist=dist[batch["recent_mask"]]
        if len(ridx): recent_min.scatter_reduce_(0,ridx,rdist,reduce="amin",include_self=True)
        positive=recent_min<=1.; ignore=(~positive)&(all_min<=2.); negative=all_min>2.
        valid=spatial_valid[outputs['batch_index']]
        return positive&valid,ignore&valid,negative&valid,(gt-outputs["voxel_centers"])/1.0
    def forward(self,outputs,batch):
        pos,ignore,neg,target=self.labels(outputs,batch);spatial_n=int(batch.get('spatial_num_samples',batch['num_samples']))
        spatial_valid=batch.get('spatial_target_valid',batch['target_valid']);zero=outputs["logits"].sum()*0
        sample_total=outputs['logits'].new_zeros(spatial_n,dtype=torch.float32)
        sample_cls=sample_total.clone();sample_reg=sample_total.clone();sample_supervised=torch.zeros(spatial_n,dtype=torch.bool,device=sample_total.device)
        sample_no_support=torch.zeros_like(sample_supervised);sample_pos=torch.zeros(spatial_n,dtype=torch.long,device=sample_total.device)
        sample_neg=sample_pos.clone();sample_ignore=sample_pos.clone()
        for b in range(spatial_n):
            if not bool(spatial_valid[b]):continue
            token=outputs["batch_index"]==b; p=pos&token; i=ignore&token; n=neg&token; np_=int(p.sum())
            sample_pos[b]=np_;sample_neg[b]=n.sum();sample_ignore[b]=i.sum()
            if np_==0:sample_no_support[b]=True;continue
            sample_supervised[b]=True;valid=p|n;logits=outputs["logits"][valid].float();targets=p[valid].float()
            ce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none"); prob=torch.sigmoid(logits); pt=targets*prob+(1-targets)*(1-prob)
            alpha=targets*self.alpha+(1-targets)*(1-self.alpha); cls=(alpha*(1-pt).pow(self.gamma)*ce).sum()/max(1,np_)
            reg=F.smooth_l1_loss(outputs["residual_xyz"][p].float(),target[p].float(),beta=self.beta,reduction="none").sum()/max(1,np_)
            sample_cls[b]=cls;sample_reg[b]=reg;sample_total[b]=cls+self.reg_weight*reg
        occurrence=batch.get('occurrence_to_unique',torch.arange(spatial_n,device=sample_total.device))
        occurrence_valid=batch['query_valid_mask'].flatten();mapped=occurrence[occurrence_valid]
        supervised_occurrence=sample_supervised[mapped];supervised_ids=mapped[supervised_occurrence]
        mean=lambda x:x[supervised_ids].mean() if len(supervised_ids) else zero
        num_pos=int(sample_pos[mapped].sum());num_neg=int(sample_neg[mapped].sum());num_ignore=int(sample_ignore[mapped].sum())
        return {"loss":mean(sample_total),"loss_cls":mean(sample_cls),"loss_reg":mean(sample_reg),"num_pos":num_pos,"num_neg":num_neg,
                "num_ignore":num_ignore,"num_supervised_samples":int(supervised_occurrence.sum()),"num_no_current_support":int(sample_no_support[mapped].sum()),
                "positive_mask":pos,"ignore_mask":ignore,"negative_mask":neg}

class TemporalPositionLoss(nn.Module):
    """Mean over XYZ axes and valid query targets, including no-current-support."""
    def __init__(self,cfg):
        super().__init__();self.beta=float(cfg['loss']['temporal_smooth_l1_beta'])
    def forward(self,outputs,batch):
        mask=batch['target_valid_clip']&batch['query_valid_mask']
        if not bool(mask.any()):return outputs['temporal_pred_xyz'].sum()*0
        return F.smooth_l1_loss(outputs['temporal_pred_xyz'][mask].float(),batch['target_xyz_clip'][mask].float(),beta=self.beta,reduction='mean')

class QueryCausalLoss(CandidateLoss):
    """Spatial V1 loss plus query-wise temporal position supervision."""
    def __init__(self,cfg):
        super().__init__(cfg);self.temporal=TemporalPositionLoss(cfg);self.weight=float(cfg['loss']['temporal_weight'])
    def forward(self,outputs,batch):
        result=super().forward(outputs,batch);spatial=result['loss'];temporal=self.temporal(outputs,batch)
        result.update(spatial_loss=spatial,temporal_loss=temporal,loss=spatial+self.weight*temporal,
                      num_temporal_supervised=int((batch['target_valid_clip']&batch['query_valid_mask']).sum()))
        return result
