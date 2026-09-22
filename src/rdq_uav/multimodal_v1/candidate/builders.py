from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .candidate_set import CandidateSet


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    x1,y1,x2,y2 = boxes.unbind(1)
    areas = (x2-x1).clamp_min(0) * (y2-y1).clamp_min(0)
    order = torch.argsort(scores, descending=True, stable=True)
    keep=[]
    while len(order):
        i=order[0]; keep.append(i)
        if len(order)==1: break
        rest=order[1:]
        xx1=torch.maximum(x1[i],x1[rest]); yy1=torch.maximum(y1[i],y1[rest])
        xx2=torch.minimum(x2[i],x2[rest]); yy2=torch.minimum(y2[i],y2[rest])
        inter=(xx2-xx1).clamp_min(0)*(yy2-yy1).clamp_min(0)
        union=areas[i]+areas[rest]-inter
        iou=torch.where(union>0,inter/union,torch.zeros_like(union))
        order=rest[iou<=iou_threshold]
    return torch.stack(keep)


def _cxcywh_norm_to_source_xyxy(
    boxes: torch.Tensor,
    source_wh: torch.Tensor,
) -> torch.Tensor:
    # DINO boxes are normalized to the per-sample resized image; normalized
    # geometry is invariant to the resize, so direct scaling by calibrated
    # source W/H recovers the canonical source-pixel frame.
    cx,cy,w,h=boxes.unbind(-1)
    x1=(cx-.5*w)*source_wh[...,0,None]
    y1=(cy-.5*h)*source_wh[...,1,None]
    x2=(cx+.5*w)*source_wh[...,0,None]
    y2=(cy+.5*h)*source_wh[...,1,None]
    return torch.stack((x1,y1,x2,y2),-1)


class RadarCandidateBuilder:
    def __init__(self, selector: Any) -> None:
        self.selector=selector

    def __call__(self, radar_output: dict[str, Any]) -> CandidateSet:
        per_batch=self.selector(radar_output)
        chunks=[]
        for b,result in enumerate(per_batch):
            cand=result['nms']
            n=len(cand['score'])
            if not n: continue
            chunks.append((b,cand))
        device=radar_output['logits'].device
        dtype=radar_output['fine_features'].dtype
        if not chunks:
            return CandidateSet.empty(source='radar',device=device,dtype=dtype)
        score=torch.cat([c['score'].to(dtype=dtype) for _,c in chunks])
        feature=torch.cat([c['feature'] for _,c in chunks])
        xyz=torch.cat([c['xyz'] for _,c in chunks])
        batch_index=torch.cat([torch.full((len(c['score']),),b,device=device,dtype=torch.long) for b,c in chunks])
        source_index=torch.cat([c['source_token_id'].long() for _,c in chunks])
        n=len(score)
        return CandidateSet(score,feature,xyz,torch.ones(n,device=device,dtype=torch.bool),
                            torch.zeros((n,4),device=device,dtype=dtype),torch.zeros(n,device=device,dtype=torch.bool),
                            batch_index,'radar',source_index)


class RGBCandidateBuilder(nn.Module):
    def __init__(self, query_dim: int=256, feature_dim: int=128, pre_topk: int=100, final_topk: int=50, nms_iou: float=.7) -> None:
        super().__init__()
        self.feature_proj=nn.Linear(query_dim,feature_dim)
        self.pre_topk=int(pre_topk); self.final_topk=int(final_topk); self.nms_iou=float(nms_iou)

    def forward(self, dino_output: dict[str,torch.Tensor], source_image_size_wh: torch.Tensor) -> CandidateSet:
        logits=dino_output['pred_logits']; boxes=dino_output['pred_boxes']; query=dino_output['decoder_query_features']
        if logits.ndim!=3 or boxes.shape[:2]!=logits.shape[:2] or query.shape[:2]!=logits.shape[:2]:
            raise ValueError('DINO logits/boxes/query features must agree on [B,Q]')
        if source_image_size_wh.shape!=(logits.shape[0],2):
            raise ValueError('source_image_size_wh must be [B,2]')
        # Final model is one-class UAV. Until that head is adapted, C>1 is
        # intentionally rejected instead of silently treating COCO classes as UAV.
        if logits.shape[-1] != 1:
            raise ValueError(f'RGB CandidateSet requires 1-class UAV logits, got C={logits.shape[-1]}')
        score=torch.sigmoid(logits[...,0].float()).to(query.dtype)
        source_boxes=_cxcywh_norm_to_source_xyxy(boxes,source_image_size_wh.to(boxes.dtype))
        projected=self.feature_proj(query)
        out=[]
        for b in range(logits.shape[0]):
            qn=score.shape[1]
            ids=torch.arange(qn,device=score.device,dtype=torch.long)
            order=torch.argsort(score[b],descending=True,stable=True)[:self.pre_topk]
            kept_local=_nms_xyxy(source_boxes[b,order],score[b,order],self.nms_iou)[:self.final_topk]
            keep=order[kept_local]
            if len(keep): out.append((b,keep))
        if not out:
            return CandidateSet.empty(source='rgb',device=query.device,dtype=query.dtype)
        idx=torch.cat([keep for _,keep in out])
        batch=torch.cat([torch.full((len(keep),),b,device=query.device,dtype=torch.long) for b,keep in out])
        scores=torch.cat([score[b,keep] for b,keep in out])
        feats=torch.cat([projected[b,keep] for b,keep in out])
        bxs=torch.cat([source_boxes[b,keep] for b,keep in out])
        n=len(idx)
        return CandidateSet(scores,feats,torch.zeros((n,3),device=query.device,dtype=query.dtype),
                            torch.zeros(n,device=query.device,dtype=torch.bool),bxs,
                            torch.ones(n,device=query.device,dtype=torch.bool),batch,'rgb',idx)
