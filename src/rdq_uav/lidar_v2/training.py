"""Formal-training support for the independent V2-base model graph."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from rdq_uav.multimodal.merged_lidar import load_released_xyz

from .runtime import evaluate_batch, move_batch, summarize_metrics


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        self.total += float(value) * float(weight)
        self.weight += float(weight)

    @property
    def value(self) -> float:
        return self.total / self.weight if self.weight else 0.0


class EpochTracker:
    """Weighted epoch means and token/sample counters for progress and CSV logs."""
    def __init__(self) -> None:
        self.loss = RunningMean(); self.cls = RunningMean(); self.reg = RunningMean()
        self.pos = RunningMean(); self.neg = RunningMean(); self.l0 = RunningMean()
        self.supervised = 0; self.no_support = 0; self.batches = 0

    def update(self, losses: dict[str, Any], l0_tokens: int) -> None:
        weight = int(losses["num_supervised_samples"])
        if weight:
            self.loss.update(float(losses["loss"].detach()), weight)
            self.cls.update(float(losses["loss_cls"].detach()), weight)
            self.reg.update(float(losses["loss_reg"].detach()), weight)
        self.pos.update(int(losses["num_pos"])); self.neg.update(int(losses["num_neg"])); self.l0.update(l0_tokens)
        self.supervised += int(losses["num_supervised_samples"])
        self.no_support += int(losses["num_no_current_support"]); self.batches += 1


def finite_or_raise(name: str, tensor: torch.Tensor, context: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"[ERROR] Non-finite {name} at {context}")


def inspect_dataset_timing(dataset: Any, indices: Iterable[int]) -> list[dict[str, Any]]:
    """Load deterministic samples and assert t0/GT/history causality."""
    rows = []
    for index in indices:
        sample = dataset[int(index)]
        events = list(sample["event_timestamps"])
        t0 = float(sample["t0"]); gt_timestamp = float(sample["gt_timestamp"])
        if t0 != gt_timestamp:
            raise AssertionError(f"GT timestamp mismatch for {sample['sample_id']}: {t0} != {gt_timestamp}")
        if events and max(events) > t0:
            raise AssertionError(f"Future event violation for {sample['sample_id']}: {max(events)} > {t0}")
        delta = np.asarray(events, dtype=np.float64) - t0 if events else np.empty(0)
        if len(delta) and float(delta.max()) > 0:
            raise AssertionError(f"Positive delta_t for {sample['sample_id']}")
        if not torch.isfinite(sample["gt_xyz"]).all():
            raise AssertionError(f"Non-finite GT for {sample['sample_id']}")
        rows.append({
            "loader_source": sample["gt_source"], "sequence_id": sample["sequence_id"],
            "sample_id": sample["sample_id"], "t0": t0, "gt_timestamp": gt_timestamp,
            "gt_x": float(sample["gt_xyz"][0]), "gt_y": float(sample["gt_xyz"][1]),
            "gt_z": float(sample["gt_xyz"][2]),
            "latest_event_timestamp": max(events) if events else None,
            "oldest_selected_event_timestamp": min(events) if events else None,
            "num_selected_events": len(events),
            "delta_t_min": float(delta.min()) if len(delta) else None,
            "delta_t_max": float(delta.max()) if len(delta) else None,
            "recent_event_count": min(4, len(events)),
        })
    return rows


def validation_construction_audit(dataset: Any) -> dict[str, Any]:
    missing_rows=[]; no_history=[]; fewer=[]; empty_lidar=[]; matched_rows=0;nonempty_cache={}
    missing=set(dataset.missing_sequences)
    for record in dataset.adapter.records:
        seq=record["sequence_id"]
        if seq in missing:
            missing_rows.append(record["sample_id"]); continue
        matched_rows += 1
        stream=dataset.streams[seq]
        # Metadata-only causal selection, avoiding point loads.
        timestamps=np.fromiter((x.timestamp for x in stream),dtype=np.float64,count=len(stream));end=int(np.searchsorted(timestamps,record["t0"],side="right"))
        count=min(dataset.max_events,end);selected=stream[max(0,end-dataset.max_events):end]
        if count==0:no_history.append(record["sample_id"])
        if count<dataset.max_events:fewer.append(record["sample_id"])
        has_points=False
        for event in selected:
            key=str(event.file_path)
            if key not in nonempty_cache:nonempty_cache[key]=len(load_released_xyz(event.file_path)[0])>0
            has_points=has_points or nonempty_cache[key]
        if not has_points:empty_lidar.append(record["sample_id"])
    return {
        "val_gt_rows": dataset.adapter.total_rows,
        "val_valid_gt_rows": len(dataset.adapter.records),
        "val_invalid_gt_rows": len(dataset.adapter.invalid_rows),
        "val_sequence_count": len({x["sequence_id"] for x in dataset.adapter.records}),
        "val_matched_sequence_count": len({x["sequence_id"] for x in dataset.adapter.records if x["sequence_id"] not in missing}),
        "val_sequence_matched_rows": matched_rows,
        "val_constructed_samples": len(dataset.adapter.records),
        "val_missing_sequence_count": len(missing_rows), "val_missing_sequence_examples": missing_rows[:10],
        "val_no_history_event_count": len(no_history), "val_no_history_event_examples": no_history[:10],
        "val_empty_lidar_count":len(empty_lidar),"val_empty_lidar_examples":empty_lidar[:10],
        "val_fewer_than_20_events_count": len(fewer), "val_fewer_than_20_events_examples": fewer[:10],
        "val_duplicate_timestamp_count": len(dataset.adapter.duplicate_keys),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists=path.exists()
    with path.open("a",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row));
        if not exists:writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(model, loader, criterion, selector, device, amp_enabled: bool, amp_dtype,
             progress_factory, description: str="Validating") -> tuple[dict[str, Any],list[dict[str, Any]],dict[str,float]]:
    model.eval();rows=[];tracker=EpochTracker();score_all=[];score_pos=[];score_neg=[]
    iterator=progress_factory(loader,desc=description); seen=hits10=top1_hits=0
    for raw in iterator:
        batch=move_batch(raw,device)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):outputs=model(batch)
        finite_or_raise("validation logits",outputs["logits"],description);finite_or_raise("validation pred_xyz",outputs["pred_xyz"],description)
        losses=criterion(outputs,batch); tracker.update(losses,len(outputs["logits"])); rows.extend(evaluate_batch(outputs,batch,selector,criterion))
        scores=torch.sigmoid(outputs["logits"].float());score_all.extend(scores.cpu().tolist())
        score_pos.extend(scores[losses["positive_mask"]].cpu().tolist());score_neg.extend(scores[losses["negative_mask"]].cpu().tolist())
        new_rows=rows[-len(batch["gt_xyz"]):]
        seen+=len(new_rows);hits10+=sum(bool(r["nms_distances"][:10]) and min(r["nms_distances"][:10])<=1 for r in new_rows)
        top1_hits+=sum(bool(r["nms_distances"]) and r["nms_distances"][0]<=1 for r in new_rows)
        if hasattr(iterator,"set_postfix_str"):
            iterator.set_postfix_str(f"GTs={seen} NMS_R10_1m={hits10/max(1,seen):.3f} Top1_1m={top1_hits/max(1,seen):.3f}")
    metrics=summarize_metrics(rows)
    top=[r["nms_distances"][0] if r["nms_distances"] else math.inf for r in rows]
    metrics["error_bins"]={
        "0_0p5m":sum(x<=.5 for x in top),"0p5_1m":sum(.5<x<=1 for x in top),
        "1_2m":sum(1<x<=2 for x in top),"2_5m":sum(2<x<=5 for x in top),">5m_or_empty":sum(x>5 for x in top)}
    def stats(values):
        a=np.asarray(values,dtype=np.float64);return {"mean":float(a.mean()) if len(a) else None,"std":float(a.std()) if len(a) else None,"max":float(a.max()) if len(a) else None}
    health={"loss":tracker.loss.value,"loss_cls":tracker.cls.value,"loss_reg":tracker.reg.value,
            "supervised_samples":tracker.supervised,"no_support_samples":tracker.no_support,
            "score":stats(score_all),"positive_score":stats(score_pos),"negative_score":stats(score_neg)}
    return metrics,rows,health


def metrics_csv_row(epoch:int,lr:float,train:EpochTracker,val:dict[str,Any],health:dict[str,float],
                    gpu_peak:float,train_time:float,val_time:float,is_best:bool)->dict[str,Any]:
    m=val["all"]
    return {"epoch":epoch,"lr":lr,"train_loss":train.loss.value,"train_cls_loss":train.cls.value,
            "train_reg_loss":train.reg.value,"train_pos_mean":train.pos.value,"train_neg_mean":train.neg.value,
            "train_l0_tokens_mean":train.l0.value,"train_supervised_samples":train.supervised,
            "train_no_support_samples":train.no_support,"gpu_peak_gb":gpu_peak,"train_time_s":train_time,"val_time_s":val_time,
            "val_loss":health["loss"],"val_cls_loss":health["loss_cls"],"val_reg_loss":health["loss_reg"],
            "val_raw_r1_05m":m["raw_recall_at_1_0.5m"],"val_raw_r1_1m":m["raw_recall_at_1_1m"],"val_raw_r1_2m":m["raw_recall_at_1_2m"],
            "val_raw_r5_1m":m["raw_recall_at_5_1m"],"val_raw_r10_1m":m["raw_recall_at_10_1m"],"val_raw_r20_1m":m["raw_recall_at_20_1m"],
            "val_nms_r1_1m":m["nms_recall_at_1_1m"],"val_nms_r5_1m":m["nms_recall_at_5_1m"],
            "val_nms_r10_1m":m["nms_recall_at_10_1m"],"val_nms_r20_1m":m["nms_recall_at_20_1m"],
            "val_top1_success_05m":m["nms_top1_success_0.5m"],"val_top1_success_1m":m["nms_top1_success_1m"],
            "val_top1_success_2m":m["nms_top1_success_2m"],"val_top1_mean_error":m["nms_top1_error_mean"],
            "val_top1_median_error":m["nms_top1_error_median"],"val_top1_p90_error":m["nms_top1_error_p90"],
            "val_top1_p95_error":m["nms_top1_error_p95"],"val_oracle_top10_error":m["nms_oracle_top10_error"],
            "val_coverage":m["nms_coverage"],"is_best":is_best}


def save_training_plots(metrics_csv:Path,plot_dir:Path)->None:
    if not metrics_csv.exists():return
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(metrics_csv.open()));
    if not rows:return
    plot_dir.mkdir(parents=True,exist_ok=True);x=[int(r["epoch"]) for r in rows]
    specs=[("loss_curve.png",[("train_loss","total"),("train_cls_loss","cls"),("train_reg_loss","reg")],"Loss"),
           ("recall_curve.png",[("val_nms_r1_1m","R@1"),("val_nms_r5_1m","R@5"),("val_nms_r10_1m","R@10"),("val_nms_r20_1m","R@20")],"NMS Recall @ 1m"),
           ("localization_error_curve.png",[("val_top1_median_error","Median"),("val_top1_p90_error","P90"),("val_top1_p95_error","P95")],"Top1 error (m)")]
    for name,curves,ylabel in specs:
        fig,ax=plt.subplots(figsize=(8,5))
        for key,label in curves:ax.plot(x,[float(r[key]) for r in rows],label=label)
        ax.set_xlabel("Epoch");ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(plot_dir/name,dpi=140);plt.close(fig)


@torch.no_grad()
def export_validation_candidates(model,loader,criterion,selector,device,checkpoint_path:Path,
                                 output_dir:Path,model_version:str,selector_version:str,
                                 coordinate_config_id:str,amp_enabled:bool,amp_dtype)->dict[str,Any]:
    """Run the best checkpoint once and export synchronized raw/NMS candidates and 128D features."""
    model.eval();metric_rows=[];prediction_rows=[];candidate_rows=[];features=[]
    checkpoint_id=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    for raw in tqdm(loader,desc="Best validation export",dynamic_ncols=True):
        batch=move_batch(raw,device)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):outputs=model(batch)
        finite_or_raise("export logits",outputs["logits"],"best validation export")
        metric_rows.extend(evaluate_batch(outputs,batch,selector,criterion));selected=selector(outputs)
        positive,_,_,_=criterion.labels(outputs,batch)
        for b,item in enumerate(selected):
            gt=batch["gt_xyz"][b];support="CURRENT_SUPPORT" if bool(torch.any(positive&(outputs["batch_index"]==b))) else "NO_CURRENT_SUPPORT"
            prediction_rows.append({"sequence_id":batch["sequence_id"][b],"sample_id":batch["sample_id"][b],"t0":float(batch["t0"][b]),
                "gt_x":float(gt[0]),"gt_y":float(gt[1]),"gt_z":float(gt[2]),"support_flag":support})
            for kind in ("raw","nms"):
                chosen=item[kind];dist=torch.linalg.vector_norm(chosen["xyz"]-gt,dim=1)
                for rank in range(len(chosen["xyz"])):
                    feature_index=len(features);features.append(chosen["feature"][rank].float().cpu().numpy())
                    candidate_rows.append({"sequence_id":batch["sequence_id"][b],"sample_id":batch["sample_id"][b],"t0":float(batch["t0"][b]),
                        "gt_x":float(gt[0]),"gt_y":float(gt[1]),"gt_z":float(gt[2]),"support_flag":support,"candidate_set":kind,
                        "rank":rank+1,"score":float(chosen["score"][rank]),"pred_x":float(chosen["xyz"][rank,0]),
                        "pred_y":float(chosen["xyz"][rank,1]),"pred_z":float(chosen["xyz"][rank,2]),"distance_to_gt":float(dist[rank]),
                        "source_token_id":int(chosen["source_token_id"][rank]),"feature_index":feature_index,
                        "checkpoint_id":checkpoint_id,"model_version":model_version,"selector_version":selector_version,
                        "coordinate_config_id":coordinate_config_id})
    metrics=summarize_metrics(metric_rows);output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(output_dir/"candidates.csv",candidate_rows);write_csv(output_dir.parent/"validation_predictions.csv",prediction_rows)
    np.savez_compressed(output_dir/"candidate_features.npz",features=np.stack(features).astype(np.float32) if features else np.empty((0,128),np.float32))
    (output_dir.parent/"best_validation_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=True))
    per_sequence=[]
    for sequence,values in metrics["per_sequence"].items():per_sequence.append({"sequence_id":sequence,**values})
    write_csv(output_dir.parent/"per_sequence_metrics.csv",per_sequence)
    return metrics
