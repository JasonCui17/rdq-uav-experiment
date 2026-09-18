#!/usr/bin/env python3
"""Real-data single-sample forward/backward smoke for V1."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rdq_uav.lidar_v1 import *
from rdq_uav.lidar_v1.runtime import move_batch
from train_lidar_uav_v1 import recent_supported
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--config",type=Path,default=ROOT/"configs/lidar_uav_v1.yaml");args=ap.parse_args();cfg=yaml.safe_load(args.config.read_text());device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 ds=LiDARUAVDataset(cfg["data"]["root"],ROOT/cfg["data"]["split_file"],cfg["data"]["train_split"]);sample=next(ds[i] for i in range(len(ds)) if recent_supported(ds[i]));batch=move_batch(collate_lidar_samples([sample]),device);model=LiDARUAVDetector(cfg).to(device);criterion=CandidateLoss(cfg)
 if device.type=="cuda":torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
 start=time.perf_counter();out=model(batch);loss=criterion(out,batch)
 if device.type=="cuda":torch.cuda.synchronize()
 forward=time.perf_counter()-start;start=time.perf_counter();loss["loss"].backward()
 if device.type=="cuda":torch.cuda.synchronize()
 backward=time.perf_counter()-start
 critical=("voxel_embed.proj.weight","merge01.parent.weight","encoder2.blocks.0.qkv.weight","up10.parent.weight","head.reg.2.weight");grads={n:bool(dict(model.named_parameters())[n].grad is not None and torch.isfinite(dict(model.named_parameters())[n].grad).all()) for n in critical}
 result={"device":str(device),"sample_id":sample["sample_id"],"event_count":sample["event_count"],"point_count":len(sample["points"]),"token_counts":out["aux_stats"]["token_counts"],"shapes":{k:list(v.shape) for k,v in out.items() if torch.is_tensor(v)},"loss":float(loss["loss"]),"loss_cls":float(loss["loss_cls"]),"loss_reg":float(loss["loss_reg"]),"num_pos":loss["num_pos"],"num_neg":loss["num_neg"],"num_ignore":loss["num_ignore"],"forward_seconds":forward,"backward_seconds":backward,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1e6 if device.type=="cuda" else 0,"parameters":sum(p.numel() for p in model.parameters()),"finite_gradients":grads,"attention_backend":out["aux_stats"]["attention_backend"],"future_leakage":any(t>sample["t0"] for t in sample["event_timestamps"])}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
