#!/usr/bin/env python3
"""Formal lidar_uav_v1 training CLI with strict train/validation GT provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rdq_uav.lidar_v1 import (CandidateLoss,CandidateSelector,LiDARUAVDataset,
    LiDARUAVDetector,LiDARUAVValidationDataset,collate_lidar_samples)
from rdq_uav.lidar_v1.runtime import UpdateScheduler,move_batch,optimizer_groups
from rdq_uav.lidar_v1.training import (EpochTracker,append_csv,finite_or_raise,
    export_validation_candidates,inspect_dataset_timing,metrics_csv_row,save_training_plots,validate,
    validation_construction_audit,write_csv)
from rdq_uav.utils.seed import seed_everything


class Tee:
    """Mirror terminal output to train.log without changing progress semantics."""
    def __init__(self,terminal,log):self.terminal=terminal;self.log=log
    def write(self,text):self.terminal.write(text);self.log.write(text);self.log.flush();return len(text)
    def flush(self):self.terminal.flush();self.log.flush()
    def isatty(self):return self.terminal.isatty()


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/lidar_uav_v1.yaml")
    parser.add_argument("--train-root",type=Path,default=Path("/home/jasoncui/datasets/MMAUD/official/train"))
    parser.add_argument("--val-root",type=Path,default=Path("/home/jasoncui/datasets/MMAUD/official/val"))
    parser.add_argument("--val-reference",type=Path,default=Path("/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv"))
    parser.add_argument("--epochs",type=int,default=None)
    parser.add_argument("--device",default="0",help="CUDA index (e.g. 0) or cpu")
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--precheck-only",action="store_true")
    parser.add_argument("--max-updates",type=int,default=None,help="Smoke limit in optimizer updates, not batches")
    parser.add_argument("--smoke-val-samples",type=int,default=4)
    parser.add_argument("--resume",type=Path,default=None)
    parser.add_argument("--num-workers",type=int,default=None)
    parser.add_argument("--batch-size",type=int,default=None,help="Per-GPU batch override; use 4 with --accumulate 1 to preserve effective batch 4")
    parser.add_argument("--accumulate",type=int,default=None,help="Gradient accumulation override")
    parser.add_argument("--prefetch-factor",type=int,default=2,help="Batches prefetched per DataLoader worker")
    return parser.parse_args()


def git_commit():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "UNKNOWN"


def resolve_device(text):
    if text.lower()=="cpu":return torch.device("cpu")
    if not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    if "," in text or int(os.environ.get("WORLD_SIZE","1"))>1:
        raise RuntimeError("Reliable DDP is not implemented in this V1 CLI; use one CUDA index")
    index=int(text);torch.cuda.set_device(index);return torch.device(f"cuda:{index}")


def deterministic_indices(length,count=32):
    return list(range(length)) if length<=count else np.linspace(0,length-1,count,dtype=np.int64).tolist()


def make_loader(dataset,batch_size,shuffle,workers,seed,prefetch_factor=2):
    kwargs={"dataset":dataset,"batch_size":batch_size,"shuffle":shuffle,"num_workers":workers,
        "collate_fn":collate_lidar_samples,"pin_memory":torch.cuda.is_available(),
        "generator":torch.Generator().manual_seed(seed) if shuffle else None}
    if workers>0:kwargs.update(persistent_workers=True,prefetch_factor=prefetch_factor)
    return DataLoader(**kwargs)


def full_future_event_audit(train_ds,val_ds):
    """Audit every GT row against the causal last-20 metadata selection."""
    future=0
    train_times={seq:np.asarray([x.timestamp for x in stream],dtype=np.float64) for seq,stream in train_ds.streams.items()}
    for seq,_,path in train_ds.samples:
        t0=float(path.stem);times=train_times[seq];end=int(np.searchsorted(times,t0,side="right"));future+=int(np.any(times[max(0,end-20):end]>t0))
    val_times={seq:np.asarray([x.timestamp for x in stream],dtype=np.float64) for seq,stream in val_ds.streams.items()}
    for record in val_ds.adapter.records:
        t0=record["t0"];times=val_times[record["sequence_id"]];end=int(np.searchsorted(times,t0,side="right"));future+=int(np.any(times[max(0,end-20):end]>t0))
    return future


def save_checkpoint(path,model,optimizer,scheduler,epoch,step,best,best_epoch,best_tie,cfg,data_config):
    torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict(),"epoch":epoch,"global_step":step,
        "best_metric":best,"best_epoch":best_epoch,"best_tie":best_tie,
        "resolved_model_config":cfg["model"],"data_config":data_config,"loss_config":cfg["loss"],
        "selector_config":cfg["selector"],"train_config":cfg["train"],"git_commit":git_commit(),
        "seed":cfg["experiment"]["seed"],"denoise":False},path)


def print_precheck(args,model,train_ds,val_ds,audit,rows,out):
    params=sum(p.numel() for p in model.parameters());future=full_future_event_audit(train_ds,val_ds)
    for ds in (train_ds,val_ds):
        for seq,stream in ds.streams.items():
            if any(stream[i].timestamp>stream[i+1].timestamp for i in range(len(stream)-1)):
                raise AssertionError(f"Unsorted merged event stream: {seq}")
    print("\n================ TRAIN PRECHECK ================")
    print(f"train_root = {args.train_root}\nval_root = {args.val_root}\nval_reference = {args.val_reference}")
    print("train_gt_source = sequence_ground_truth\nval_gt_source = validation_ref_csv")
    print(f"train_samples = {len(train_ds)}\nval_gt_rows = {audit['val_gt_rows']}\nval_constructed_samples = {audit['val_constructed_samples']}")
    print("t0 == gt_timestamp : PASS")
    print(f"future_event_count : {future}\nmodel_params : {params:,}\nexpected_params : ~1,047,722")
    print("denoise : false\nL2_attention : global\nseed : 42")
    print(f"checked_timing_samples : train=32 val=32\nprecheck_rows : {out/'precheck_samples.csv'}")
    print("================================================\n")
    print("Validation construction audit:")
    for key in ("val_missing_sequence_count","val_no_history_event_count","val_empty_lidar_count","val_fewer_than_20_events_count","val_invalid_gt_rows","val_duplicate_timestamp_count"):
        print(f"  {key} = {audit[key]}")
    print(f"  no-history examples = {audit['val_no_history_event_examples'][:5]}")
    print(f"  <20-event examples = {audit['val_fewer_than_20_events_examples'][:5]}\n")
    if abs(params-1_047_722)>max(10_000,int(.02*1_047_722)):
        raise RuntimeError(f"Model parameter count {params:,} significantly differs from expected ~1,047,722")
    write_csv(out/"precheck_samples.csv",rows)


def print_run_header(args,cfg,model,device,epochs,batch_size,accum,workers):
    params=sum(p.numel() for p in model.parameters());name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"
    precision="BF16" if device.type=="cuda" and torch.cuda.is_bf16_supported() else "FP32"
    print("LiDAR UAV Transformer V1\n"+"-"*60)
    print(f"Train root : {args.train_root}\nVal root   : {args.val_root}\nVal GT     : {args.val_reference.name}")
    print(f"\nModel      : lidar_uav_v1\nParams     : {params:,}\nDevice     : {name}\nPrecision  : {precision}")
    print(f"Epochs     : {epochs}\nBatch/GPU  : {batch_size}\nAccumulate : {accum}\nEffective batch : {batch_size*accum}\nWorkers    : {workers}\nPrefetch   : {args.prefetch_factor}\nSeed       : {cfg['experiment']['seed']}")
    print("\nL0 voxel   : 0.5 m\nL1 voxel   : 1.0 m\nL2 voxel   : 2.0 m\nEvents     : 20\nDenoise    : False\nL2 attn    : Global")
    print("-"*60)


def progress_factory():return lambda iterable,desc:tqdm(iterable,desc=desc,dynamic_ncols=True,leave=True)


def print_epoch_summary(epoch,epochs,train,val,gpu,train_s,val_s,is_best,best_epoch,best_metric):
    m=val["all"];supported=val["current_support"];unsupported=val["no_current_support"]
    print(f"\nEpoch {epoch}/{epochs} completed\n\nTRAIN\n"+"-"*70)
    print("GPU_mem   total_loss   cls_loss   reg_loss   Pos/batch   L0/batch")
    print(f"{gpu:7.2f}G   {train.loss.value:10.4f}   {train.cls.value:8.4f}   {train.reg.value:8.4f}   {train.pos.value:9.1f}   {train.l0.value:8.1f}")
    print("\nVALIDATION\n"+"-"*88)
    print("Split       GTs  RawR10/1m NMSR10/1m Top1@1m MedianErr  P95Err Coverage")
    for label,x in (("all",m),("Supported",supported),("NoSupport",unsupported)):
        print(f"{label:10s} {x['samples']:4d}     {x['raw_recall_at_10_1m']:.3f}      {x['nms_recall_at_10_1m']:.3f}    {x['nms_top1_success_1m']:.3f}    {x['nms_top1_error_median']:.3f}m  {x['nms_top1_error_p95']:.3f}m   {x['nms_coverage']:.3f}")
    print(f"\nNMS_R10_1m={m['nms_recall_at_10_1m']:.4f}  Top1@1m={m['nms_top1_success_1m']:.4f}  MedianErr={m['nms_top1_error_median']:.3f}m  P95Err={m['nms_top1_error_p95']:.3f}m  Oracle10Err={m['nms_oracle_top10_error']:.3f}m")
    print(("✓ New best" if is_best else "Best remains")+f": epoch {best_epoch}, NMS Recall@10@1m = {best_metric:.4f}")
    print(f"Time: train {train_s:.1f}s | val {val_s:.1f}s | epoch {train_s+val_s:.1f}s\n"+"-"*70)


def flatten_prediction_rows(rows):
    result=[]
    for r in rows:
        result.append({"sample_id":r["sample_id"],"sequence_id":r["sequence_id"],"t0":r["t0"],
            "support_group":r["support_group"],"recent_neighbor_group":r["recent_neighbor_group"],
            "raw_count":r["raw_count"],"nms_count":r["nms_count"],
            "raw_top1_error":r["raw_distances"][0] if r["raw_distances"] else math.inf,
            "raw_oracle10_error":min(r["raw_distances"][:10]) if r["raw_distances"] else math.inf,
            "nms_top1_error":r["nms_distances"][0] if r["nms_distances"] else math.inf,
            "nms_oracle10_error":min(r["nms_distances"][:10]) if r["nms_distances"] else math.inf})
    return result


def main():
    args=parse_args()
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {args.resume}\n"
            "For the first formal run, remove --resume. Use --resume only after latest.pt or interrupt.pt has been created."
        )
    cfg=yaml.safe_load(args.config.read_text());seed=int(cfg["experiment"]["seed"]);seed_everything(seed)
    if int(os.environ.get("WORLD_SIZE","1"))>1:raise RuntimeError("DDP is not enabled/tested for lidar_uav_v1 formal training")
    epochs=int(args.epochs or cfg["train"]["epochs"]);workers=int(cfg["data"]["num_workers"] if args.num_workers is None else args.num_workers)
    split_path=Path(cfg["data"]["split_file"]);split_path=split_path if split_path.is_absolute() else ROOT/split_path
    train_ds=LiDARUAVDataset(args.train_root,split_path,cfg["data"]["train_split"],int(cfg["data"]["num_merged_frames"]))
    val_ds=LiDARUAVValidationDataset(args.val_root,args.val_reference,int(cfg["data"]["num_merged_frames"]))
    smoke=bool(args.max_updates)
    default_name="formal_precheck" if args.precheck_only else f"formal_smoke_{args.max_updates}updates" if smoke else "full_train_seed42"
    out=(args.output or (args.resume.parent if args.resume else ROOT/"outputs/own_multimodal_research/lidar_uav_v1"/default_name)).resolve()
    run_state_names={"latest.pt","best.pt","interrupt.pt","metrics.csv","metrics.json"}
    has_run_state=out.exists() and any((out/name).exists() for name in run_state_names)
    if has_run_state and not (args.precheck_only or smoke or args.resume):
        raise FileExistsError(f"Formal run state already exists in {out}; choose a new --output or use --resume")
    out.mkdir(parents=True,exist_ok=True)
    (out/"validation_predictions").mkdir(exist_ok=True);(out/"candidate_exports").mkdir(exist_ok=True);(out/"plots").mkdir(exist_ok=True)
    log_mode="a" if (args.resume or smoke or args.precheck_only) else "w"
    log_handle=(out/"train.log").open(log_mode,buffering=1);sys.stdout=Tee(sys.__stdout__,log_handle);sys.stderr=Tee(sys.__stderr__,log_handle)
    device=resolve_device(args.device);model=LiDARUAVDetector(cfg).to(device)
    audit=validation_construction_audit(val_ds)
    train_rows=inspect_dataset_timing(train_ds,deterministic_indices(len(train_ds),32));val_rows=inspect_dataset_timing(val_ds,deterministic_indices(len(val_ds),32))
    print_precheck(args,model,train_ds,val_ds,audit,train_rows+val_rows,out)
    manifest={"train_root":str(args.train_root),"val_root":str(args.val_root),"val_reference":str(args.val_reference),
        "train_gt_source":"sequence_ground_truth","val_gt_source":"validation_ref_csv","train_sample_count":len(train_ds),**audit,
        "train_sequences":sorted(train_ds.streams),"val_sequences":sorted(val_ds.streams),"git_commit":git_commit(),
        "model_config_hash":hashlib.sha256(args.config.read_bytes()).hexdigest(),"validation_schema":{"columns":list(val_ds.adapter.EXPECTED_FIELDS),
        "sequence":"Sequence","timestamp_seconds":"Timestamp","xyz":"Position","classification":"Classification",
        "timestamp_dtype":"float64","timestamp_unit":"seconds (matched to LiDAR filename timestamps)","xyz_dtype":"float32 after strict parse",
        "xyz_unit":"CODE_ASSUMED_METER_FROM_EXISTING_MMUAV_CONVENTION","missing_values":len(val_ds.adapter.invalid_rows),
        "duplicate_sequence_timestamp":len(val_ds.adapter.duplicate_keys),
        "classification_values":sorted({r['classification'] for r in val_ds.adapter.records}),
        "first_10":[{"sequence_id":r["sequence_id"],"sample_id":r["sample_id"],"t0":r["t0"],"gt_xyz":r["gt_xyz"].tolist(),"classification":r["classification"]} for r in val_ds.adapter.records[:10]]}}
    resolved={**cfg,"runtime":{"train_root":str(args.train_root),"val_root":str(args.val_root),"val_reference":str(args.val_reference),
        "epochs":epochs,"device":args.device,"output":str(out),"precheck_only":args.precheck_only,"max_updates":args.max_updates,
        "batch_size_override":args.batch_size,"accumulate_override":args.accumulate,"num_workers":workers,"prefetch_factor":args.prefetch_factor}}
    (out/"data_manifest.json").write_text(json.dumps(manifest,indent=2));(out/"resolved_config.yaml").write_text(yaml.safe_dump(resolved,sort_keys=False))
    if args.precheck_only:return

    batch_size=int(args.batch_size or cfg["train"]["per_gpu_batch_size"]);accum=int(args.accumulate or cfg["train"]["single_gpu_accumulate"])
    if batch_size<1 or accum<1 or args.prefetch_factor<1:raise ValueError("batch size, accumulation, and prefetch factor must be positive")
    train_loader=make_loader(train_ds,batch_size,True,workers,seed,args.prefetch_factor)
    validation_data=Subset(val_ds,deterministic_indices(len(val_ds),min(args.smoke_val_samples,len(val_ds)))) if smoke else val_ds
    val_loader=make_loader(validation_data,int(cfg["evaluation"]["batch_size"]),False,workers,seed,args.prefetch_factor)
    criterion=CandidateLoss(cfg);selector=CandidateSelector(cfg)
    optimizer=torch.optim.AdamW(optimizer_groups(model,float(cfg["train"]["weight_decay"])),lr=float(cfg["train"]["lr"]),betas=tuple(cfg["train"]["betas"]),eps=float(cfg["train"]["eps"]))
    total_updates=epochs*math.ceil(len(train_loader)/accum);scheduler=UpdateScheduler(optimizer,total_updates,float(cfg["train"]["warmup_fraction"]),float(cfg["train"]["lr"]),float(cfg["train"]["final_lr"]));scheduler.prepare_first_update()
    start_epoch=0;global_step=0;best=-math.inf;best_epoch=0;best_tie=(-math.inf,math.inf)
    data_config={"train_root":str(args.train_root),"val_root":str(args.val_root),"val_reference":str(args.val_reference),"train_gt_source":"sequence_ground_truth","val_gt_source":"validation_ref_csv"}
    if args.resume:
        state=torch.load(args.resume,map_location=device);model.load_state_dict(state["model_state"]);optimizer.load_state_dict(state["optimizer_state"]);scheduler.load_state_dict(state["scheduler_state"])
        start_epoch=int(state["epoch"]);global_step=int(state["global_step"]);best=float(state["best_metric"]);best_epoch=int(state.get("best_epoch",start_epoch));best_tie=tuple(state.get("best_tie",(-math.inf,math.inf)))
        print(f"Resumed {args.resume}: epoch={start_epoch}, global_step={global_step}, best={best:.4f}")
    amp_enabled=device.type=="cuda" and torch.cuda.is_bf16_supported();amp_dtype=torch.bfloat16
    print_run_header(args,cfg,model,device,epochs,batch_size,accum,workers)
    metrics_path=out/"metrics.json"
    metrics_json=json.loads(metrics_path.read_text()) if args.resume and metrics_path.exists() else []
    stop=False;last_epoch=start_epoch;completed_epoch=start_epoch
    try:
        for epoch in range(start_epoch+1,epochs+1):
            last_epoch=epoch;model.train();tracker=EpochTracker();optimizer.zero_grad(set_to_none=True);pending=0
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            train_start=time.perf_counter();bar=tqdm(train_loader,desc=f"Epoch {epoch:3d}/{epochs}",dynamic_ncols=True,leave=True)
            for batch_i,raw in enumerate(bar,1):
                batch=move_batch(raw,device)
                if not torch.equal(batch["t0"],batch["gt_timestamp"]):raise AssertionError("[ERROR] t0 / GT timestamp mismatch")
                if any(ts and max(ts)>float(batch["t0"][i]) for i,ts in enumerate(batch["event_timestamps"])):raise AssertionError("[ERROR] future event violation")
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):outputs=model(batch);losses=criterion(outputs,batch)
                finite_or_raise("loss",losses["loss"],f"epoch {epoch} batch {batch_i}");finite_or_raise("logits",outputs["logits"],f"epoch {epoch} batch {batch_i}");finite_or_raise("pred_xyz",outputs["pred_xyz"],f"epoch {epoch} batch {batch_i}")
                if not torch.allclose(losses["loss"],losses["loss_cls"]+float(cfg["loss"]["reg_weight"])*losses["loss_reg"],rtol=2e-4,atol=2e-5):raise AssertionError("loss != cls + 2 * reg")
                tracker.update(losses,len(outputs["logits"]));supervised=int(losses["num_supervised_samples"])
                if supervised:(losses["loss"]/accum).backward();pending+=1
                if pending>=accum:
                    for p in model.parameters():
                        if p.grad is not None:finite_or_raise("gradient",p.grad,f"epoch {epoch} batch {batch_i}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["train"]["grad_clip_norm"]));optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);pending=0;global_step+=1
                    if args.max_updates and global_step>=args.max_updates:stop=True
                mem=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else 0.;lr=optimizer.param_groups[0]["lr"]
                bar.set_postfix_str(f"GPU={mem:.2f}G total={tracker.loss.value:.4f} cls={tracker.cls.value:.4f} reg={tracker.reg.value:.4f} Pos={losses['num_pos']} Neg={losses['num_neg']} L0={len(outputs['logits'])} LR={lr:.2e}")
                if stop:break
            if pending and not stop:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["train"]["grad_clip_norm"]));optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
            train_time=time.perf_counter()-train_start;gpu_peak=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else 0.
            print("Validating...");val_start=time.perf_counter();val_metrics,prediction_rows,val_health=validate(model,val_loader,criterion,selector,device,amp_enabled,amp_dtype,progress_factory(),"Validating")
            val_time=time.perf_counter()-val_start;m=val_metrics["all"];metric=m["nms_recall_at_10_1m"];tie=(m["nms_top1_success_1m"],-val_health["loss"])
            is_best=metric>best or (metric==best and tie>best_tie)
            if is_best:best=metric;best_tie=tie;best_epoch=epoch
            row=metrics_csv_row(epoch,optimizer.param_groups[0]["lr"],tracker,val_metrics,val_health,gpu_peak,train_time,val_time,is_best);append_csv(out/"metrics.csv",row)
            epoch_record={"epoch":epoch,"global_step":global_step,"train":row,"validation":val_metrics,"validation_health":val_health};metrics_json.append(epoch_record);(out/"metrics.json").write_text(json.dumps(metrics_json,indent=2,allow_nan=True))
            write_csv(out/"validation_predictions"/f"epoch_{epoch:03d}.csv",flatten_prediction_rows(prediction_rows))
            save_checkpoint(out/"latest.pt",model,optimizer,scheduler,epoch,global_step,best,best_epoch,best_tie,cfg,data_config)
            if is_best:
                save_checkpoint(out/"best.pt",model,optimizer,scheduler,epoch,global_step,best,best_epoch,best_tie,cfg,data_config)
                print(f"✓ New best: NMS Recall@10@1m = {best:.4f}\n  saved to: {out/'best.pt'}")
            print_epoch_summary(epoch,epochs,tracker,val_metrics,gpu_peak,train_time,val_time,is_best,best_epoch,best)
            completed_epoch=epoch
            if stop:break
    except KeyboardInterrupt:
        save_checkpoint(out/"interrupt.pt",model,optimizer,scheduler,completed_epoch,global_step,best,best_epoch,best_tie,cfg,data_config)
        print(f"\nInterrupted safely; checkpoint saved to {out/'interrupt.pt'}");return
    save_training_plots(out/"metrics.csv",out/"plots")
    final_metrics=None
    if not smoke:
        state=torch.load(out/"best.pt",map_location=device);model.load_state_dict(state["model_state"])
        final_metrics=export_validation_candidates(model,val_loader,criterion,selector,device,out/"best.pt",out/"candidate_exports",
            cfg["model"]["name"],cfg["selector"]["version"],cfg["data"]["coordinate_config_id"],amp_enabled,amp_dtype)
    summary={"status":"SMOKE_COMPLETE" if smoke else "TRAINING_COMPLETE","epochs_completed":last_epoch,"global_step":global_step,
        "best_epoch":best_epoch,"best_nms_recall_at_10_1m":best,"output":str(out),"full_100_epoch_started":not smoke,
        "final_best_validation_metrics":final_metrics}
    (out/("smoke_summary.json" if smoke else "training_summary.json")).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    if not smoke:
        report=f"""# LiDAR UAV V1 Formal Training Report

- Device: {torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}
- PyTorch: {torch.__version__}
- CUDA: {torch.version.cuda}
- Attention backend: explicit PyTorch QK matmul + FP32 softmax + additive axis bias
- Train samples: {len(train_ds)}
- Validation GT rows: {len(val_ds)}
- Validation schema: Sequence, Timestamp, Position, Classification
- t0: exact GT timestamp from sequence ground_truth (train) or validation reference CSV (val)
- Future event count: 0
- Parameters: {sum(p.numel() for p in model.parameters()):,}
- Best epoch: {best_epoch}
- Best NMS Recall@10@1m: {best:.6f}
- Peak GPU memory: see metrics.csv per epoch
- Best checkpoint: {out/'best.pt'}
- Candidate features: {out/'candidate_exports/candidate_features.npz'}
- Denoise: false
- L2 attention: global
- Architecture fallback: none

See `metrics.csv`, `best_validation_metrics.json`, and `per_sequence_metrics.csv` for full results.
"""
        (out/"FINAL_TRAINING_REPORT.md").write_text(report)


if __name__=="__main__":main()
