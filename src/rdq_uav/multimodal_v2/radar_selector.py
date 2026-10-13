"""Stable raw Top-K and Euclidean radius-suppressed candidate selection."""
from __future__ import annotations
import torch

class CandidateSelector:
    VERSION="stable_topk100_radius1m_v1"
    def __init__(self,cfg, *, raw_only=False):
        self.raw_only=bool(raw_only)
        c=cfg["selector"] if "selector" in cfg else cfg; self.raw_topk=int(c["raw_topk"]); self.pre=int(c["pre_nms_topk"]); self.radius=float(c["nms_radius_m"]); self.final=int(c["final_topk"]); self.trace_observer=None
    @staticmethod
    def _order(scores,ids):
        # stable preserves ascending source id as tie break after its pre-sort.
        base=torch.argsort(ids,stable=True); return base[torch.argsort(scores[base],descending=True,stable=True)]
    @staticmethod
    def ranking_scores(o):
        score = torch.sigmoid(o["logits"].float())
        if "quality_logits" in o:
            score = score * torch.sigmoid(o["quality_logits"].float())
        return score

    def _gather(self,o,idx):
        return {"xyz":o["pred_xyz"][idx],"score":self.ranking_scores(o)[idx],"feature":o["fine_features"][idx],"source_token_id":o["source_token_id"][idx]}
    def __call__(self,o):
        results=[]
        for b in range(int(o.get("aux_stats",{}).get("num_samples",int(o["batch_index"].max())+1 if len(o["batch_index"]) else 1))):
            ids=torch.nonzero(o["batch_index"]==b).flatten()
            # Baseline keeps logit order exactly (no sigmoid saturation ties).
            rank_logits=(self.ranking_scores(o)[ids] if "quality_logits" in o else o["logits"][ids].float())
            order=ids[self._order(rank_logits,o["source_token_id"][ids])]
            raw=order[:self.raw_topk]; pool=order[:self.pre]
            if self.raw_only:
                results.append({"raw":self._gather(o,raw), "nms":self._gather(o,raw)})
                continue
            if o.get("prediction_kind") == "query":
                # One-to-one set predictions use no radius NMS. All K queries
                # participate before the final reporting cap.
                results.append({"raw":self._gather(o,raw), "nms":self._gather(o,order[:self.final])})
                continue
            # Compute the exact legacy radius predicate in one device region,
            # then transfer only the <=100 x <=100 boolean matrix once. The
            # greedy stable-order loop remains identical, without one host
            # synchronization and kernel sequence per candidate.
            suppressed=None; kept_local=[]; evaluated_count=0
            if len(pool):
                xyz=o["pred_xyz"][pool]
                suppressed=(torch.linalg.vector_norm(xyz[:,None]-xyz[None,:],dim=2)<=self.radius).cpu()
                for local_idx in range(len(pool)):
                    evaluated_count=local_idx+1
                    if not kept_local or not bool(suppressed[local_idx,kept_local].any()):
                        kept_local.append(local_idx)
                    if len(kept_local)>=self.final: break
                final=pool[torch.tensor(kept_local,device=pool.device,dtype=torch.long)]
            else: final=pool
            if self.trace_observer is not None:
                self.trace_observer(dict(batch_index=b,order=order,raw=raw,pool=pool,
                    final=final,suppressed=suppressed,kept_local=kept_local,
                    radius_m=self.radius,final_topk=self.final,evaluated_count=evaluated_count))
            results.append({"raw":self._gather(o,raw),"nms":self._gather(o,final)})
        return results
