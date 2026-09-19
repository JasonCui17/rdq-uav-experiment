"""Stable raw Top-K and Euclidean radius-suppressed candidate selection."""
from __future__ import annotations
import torch

class CandidateSelector:
    VERSION="stable_topk100_radius1m_v1"
    def __init__(self,cfg):
        c=cfg["selector"] if "selector" in cfg else cfg; self.raw_topk=int(c["raw_topk"]); self.pre=int(c["pre_nms_topk"]); self.radius=float(c["nms_radius_m"]); self.final=int(c["final_topk"])
    @staticmethod
    def _order(scores,ids):
        # stable preserves ascending source id as tie break after its pre-sort.
        base=torch.argsort(ids,stable=True); return base[torch.argsort(scores[base],descending=True,stable=True)]
    def _gather(self,o,idx):
        return {"xyz":o["pred_xyz"][idx],"score":torch.sigmoid(o["logits"][idx].float()),"feature":o["fine_features"][idx],"source_token_id":o["source_token_id"][idx]}
    def __call__(self,o):
        results=[]
        for b in range(int(o.get("aux_stats",{}).get("num_samples",int(o["batch_index"].max())+1 if len(o["batch_index"]) else 1))):
            ids=torch.nonzero(o["batch_index"]==b).flatten()
            rank_logits=o["logits"][ids].float()
            order=ids[self._order(rank_logits,o["source_token_id"][ids])]
            raw=order[:self.raw_topk]; pool=order[:self.pre]; kept=[]
            for idx in pool:
                if not kept or torch.all(torch.linalg.vector_norm(o["pred_xyz"][torch.stack(kept)]-o["pred_xyz"][idx],dim=1)>self.radius): kept.append(idx)
                if len(kept)>=self.final: break
            final=torch.stack(kept) if kept else pool[:0]; results.append({"raw":self._gather(o,raw),"nms":self._gather(o,final)})
        return results
