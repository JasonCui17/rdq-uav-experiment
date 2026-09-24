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
            raw=order[:self.raw_topk]; pool=order[:self.pre]
            # Compute the exact legacy radius predicate in one device region,
            # then transfer only the <=100 x <=100 boolean matrix once. The
            # greedy stable-order loop remains identical, without one host
            # synchronization and kernel sequence per candidate.
            if len(pool):
                xyz=o["pred_xyz"][pool]
                suppressed=(torch.linalg.vector_norm(xyz[:,None]-xyz[None,:],dim=2)<=self.radius).cpu()
                kept_local=[]
                for local_idx in range(len(pool)):
                    if not kept_local or not bool(suppressed[local_idx,kept_local].any()):
                        kept_local.append(local_idx)
                    if len(kept_local)>=self.final: break
                final=pool[torch.tensor(kept_local,device=pool.device,dtype=torch.long)]
            else: final=pool
            results.append({"raw":self._gather(o,raw),"nms":self._gather(o,final)})
        return results
