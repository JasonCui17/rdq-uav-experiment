from __future__ import annotations
import torch


def _iou_one(box:torch.Tensor, boxes:torch.Tensor)->torch.Tensor:
    x1=torch.maximum(box[0],boxes[:,0]); y1=torch.maximum(box[1],boxes[:,1]); x2=torch.minimum(box[2],boxes[:,2]); y2=torch.minimum(box[3],boxes[:,3])
    inter=(x2-x1).clamp_min(0)*(y2-y1).clamp_min(0)
    a=(box[2]-box[0]).clamp_min(0)*(box[3]-box[1]).clamp_min(0)
    b=(boxes[:,2]-boxes[:,0]).clamp_min(0)*(boxes[:,3]-boxes[:,1]).clamp_min(0)
    union=a+b-inter
    return torch.where(union>0,inter/union,torch.zeros_like(union))


def joint_suppression_indices(score:torch.Tensor,boxes:torch.Tensor,xyz:torch.Tensor,batch_index:torch.Tensor,*,iou_threshold:float=.7,radius_m:float=1.,final_topk:int=50)->torch.Tensor:
    kept=[]
    for b in torch.unique(batch_index,sorted=True).tolist():
        ids=torch.nonzero(batch_index==b).flatten(); order=ids[torch.argsort(score[ids],descending=True,stable=True)]
        local=[]
        for idx in order.tolist():
            if local:
                prev=torch.tensor(local,device=score.device,dtype=torch.long)
                duplicate=(_iou_one(boxes[idx],boxes[prev])>iou_threshold) & (torch.linalg.vector_norm(xyz[prev]-xyz[idx],dim=1)<radius_m)
                if bool(duplicate.any()): continue
            local.append(idx)
            if len(local)>=final_topk: break
        kept.extend(local)
    return torch.tensor(kept,device=score.device,dtype=torch.long)
