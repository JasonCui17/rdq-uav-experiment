"""Training/evaluation utilities depending only on detector public outputs."""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any
import numpy as np
import torch
from .contracts import require_occurrence_aligned_evaluation

def move_batch(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def optimizer_groups(model,weight_decay):
    decay=[]; no_decay=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim==1 or name.endswith("bias") or "sensor_embedding" in name or ".bias.tables" in name: no_decay.append(p)
        else: decay.append(p)
    return [{"params":decay,"weight_decay":weight_decay},{"params":no_decay,"weight_decay":0.}]

class UpdateScheduler:
    def __init__(self,optimizer,total_updates,warmup_fraction,base_lr,final_lr):
        self.optimizer=optimizer; self.total=max(1,total_updates); self.warmup=max(1,round(self.total*warmup_fraction)); self.base=base_lr; self.final=final_lr; self.updates=0; self._set(0.)
    def _set(self,fraction):
        lr=self.base*fraction
        for group in self.optimizer.param_groups: group["lr"]=lr
    def _apply_update(self):
        if self.updates<=self.warmup: factor=self.updates/self.warmup
        else:
            progress=(self.updates-self.warmup)/max(1,self.total-self.warmup)
            factor=(self.final/self.base)+(1-self.final/self.base)*.5*(1+math.cos(math.pi*min(1.,progress)))
        self._set(factor)
    def prepare_first_update(self):
        """Set LR for update one while keeping ``updates`` as completed updates."""
        if self.updates==0:
            self.updates=1; self._apply_update(); self.updates=0
    def step(self):
        self.updates+=1
        next_update=min(self.total,self.updates+1); completed=self.updates
        self.updates=next_update; self._apply_update(); self.updates=completed
    def state_dict(self): return {"updates":self.updates,"total":self.total,"warmup":self.warmup}
    def load_state_dict(self,state):
        self.total=int(state["total"]); self.warmup=int(state["warmup"])
        self.updates=int(state["updates"])
        next_update=min(self.total,self.updates+1); completed=self.updates
        self.updates=next_update; self._apply_update(); self.updates=completed

@torch.no_grad()
def evaluate_batch(outputs,batch,selector,criterion):
    require_occurrence_aligned_evaluation(batch)
    selected=selector(outputs); pos,_,_,_=criterion.labels(outputs,batch); rows=[]
    score=batch.get('score_mask')
    score=torch.ones(len(selected),dtype=torch.bool,device=batch['target_valid'].device) if score is None else score.flatten()
    for b,item in enumerate(selected):
        if not bool(score[b]) or not bool(batch['target_valid'][b]):continue
        gt=batch["target_xyz"][b]; current=bool(torch.any(pos & (outputs["batch_index"]==b)))
        recent_points=batch["supervision_recent_mask"]&(batch["point_batch_index"]==b)
        recent_neighbors=int(torch.count_nonzero(torch.linalg.vector_norm(batch["points"][recent_points]-gt,dim=1)<=1.))
        row={"sample_id":batch["sample_id"][b],"sequence_id":batch["sequence_id"][b],"t0":float(batch["query_time"][b]),"support_group":"CURRENT_SUPPORT" if current else "NO_CURRENT_SUPPORT","recent_neighbor_group":"1" if recent_neighbors==1 else "2-3" if recent_neighbors in (2,3) else "4+" if recent_neighbors>=4 else "0"}
        for kind in ("raw","nms"):
            dist=torch.linalg.vector_norm(item[kind]["xyz"]-gt,dim=1)
            row[f"{kind}_distances"]=dist.cpu().tolist(); row[f"{kind}_count"]=len(dist)
        row['query_time']=float(batch['query_time'][b])
        row['target_timestamp']=float(batch['target_timestamp'][b])
        rows.append(row)
    return rows

def summarize_metrics(rows):
    def one(group):
        out={"samples":len(group)}
        for kind in ("raw","nms"):
            for k in (1,5,10,20):
                for radius in (.5,1.,2.): out[f"{kind}_recall_at_{k}_{radius:g}m"]=float(np.mean([bool(r[f"{kind}_distances"][:k]) and min(r[f"{kind}_distances"][:k])<=radius for r in group])) if group else 0.
            top=[r[f"{kind}_distances"][0] if r[f"{kind}_distances"] else np.inf for r in group]; finite=np.asarray([x for x in top if np.isfinite(x)])
            for radius in (.5,1.,2.): out[f"{kind}_top1_success_{radius:g}m"]=float(np.mean(np.asarray(top)<=radius)) if group else 0.
            out[f"{kind}_coverage"]=float(np.mean([r[f"{kind}_count"]>0 for r in group])) if group else 0.
            for label,fn in (("mean",np.mean),("median",np.median),("p90",lambda x:np.percentile(x,90)),("p95",lambda x:np.percentile(x,95))): out[f"{kind}_top1_error_{label}"]=float(fn(finite)) if len(finite) else float("inf")
            oracle=[min(r[f"{kind}_distances"][:10]) if r[f"{kind}_distances"][:10] else np.inf for r in group]; out[f"{kind}_oracle_top10_error"]=float(np.mean(oracle)) if oracle else float("inf")
        return out
    result={"all":one(rows)}
    for key in ("CURRENT_SUPPORT","NO_CURRENT_SUPPORT"): result[key.lower()]=one([r for r in rows if r["support_group"]==key])
    result["per_sequence"]={seq:one([r for r in rows if r["sequence_id"]==seq]) for seq in sorted({r["sequence_id"] for r in rows})}
    result["per_recent_neighbor_group"]={key:one([r for r in rows if r["recent_neighbor_group"]==key]) for key in ("0","1","2-3","4+")}
    return result
