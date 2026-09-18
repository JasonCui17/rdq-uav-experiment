#!/usr/bin/env python3
"""Evaluate V2-base checkpoint and optionally export synchronized candidate features."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rdq_uav.lidar_v2 import *
from rdq_uav.lidar_v2.runtime import evaluate_batch,move_batch,summarize_metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument("checkpoint",type=Path);ap.add_argument("--config",type=Path,default=ROOT/"configs/lidar_uav_v2.yaml");ap.add_argument("--split",choices=["train_sub","validation_sub","heldout_test_sub"],default="validation_sub");ap.add_argument("--output",type=Path,required=True);ap.add_argument("--export-candidates",action="store_true");ap.add_argument("--max-samples",type=int);args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text());device=torch.device("cuda" if torch.cuda.is_available() else "cpu");ckpt=torch.load(args.checkpoint,map_location=device);model=LiDARUAVDetector(cfg).to(device);model.load_state_dict(ckpt["model_state"]);model.eval();criterion=CandidateLoss(cfg);selector=CandidateSelector(cfg)
    ds=LiDARUAVDataset(cfg["data"]["root"],ROOT/cfg["data"]["split_file"],args.split,cfg["data"]["num_merged_frames"]);ds=Subset(ds,range(min(len(ds),args.max_samples))) if args.max_samples else ds;loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=cfg["data"]["num_workers"],collate_fn=collate_lidar_samples)
    args.output.mkdir(parents=True,exist_ok=False);metric_rows=[];candidate_rows=[];features=[];checkpoint_id=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    with torch.no_grad():
        for raw in loader:
            batch=move_batch(raw,device);outputs=model(batch);metric_rows.extend(evaluate_batch(outputs,batch,selector,criterion));chosen=selector(outputs)[0];gt=batch["gt_xyz"][0]
            if args.export_candidates:
                support="CURRENT_SUPPORT" if criterion(outputs,batch)["num_pos"] else "NO_CURRENT_SUPPORT"
                for kind in ("raw","nms"):
                    item=chosen[kind];dist=torch.linalg.vector_norm(item["xyz"]-gt,dim=1)
                    for rank in range(len(item["xyz"])):
                        feature_index=len(features);features.append(item["feature"][rank].float().cpu().numpy())
                        candidate_rows.append({"sequence_id":raw["sequence_id"][0],"sample_id":raw["sample_id"][0],"t0":float(raw["t0"][0]),"gt_x":float(gt[0]),"gt_y":float(gt[1]),"gt_z":float(gt[2]),"support_flag":support,"candidate_set":kind,"rank":rank+1,"score":float(item["score"][rank]),"pred_x":float(item["xyz"][rank,0]),"pred_y":float(item["xyz"][rank,1]),"pred_z":float(item["xyz"][rank,2]),"distance_to_gt":float(dist[rank]),"source_token_id":int(item["source_token_id"][rank]),"feature_index":feature_index,"checkpoint_id":checkpoint_id,"model_version":cfg["model"]["name"],"selector_version":cfg["selector"]["version"],"coordinate_config_id":cfg["data"]["coordinate_config_id"]})
    (args.output/"metrics.json").write_text(json.dumps(summarize_metrics(metric_rows),indent=2,allow_nan=True))
    if args.export_candidates:
        fields=["sequence_id","sample_id","t0","gt_x","gt_y","gt_z","support_flag","candidate_set","rank","score","pred_x","pred_y","pred_z","distance_to_gt","source_token_id","feature_index","checkpoint_id","model_version","selector_version","coordinate_config_id"]
        with (args.output/"candidates.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(candidate_rows)
        np.savez_compressed(args.output/"candidate_features.npz",features=np.stack(features).astype(np.float32) if features else np.empty((0,128),np.float32))
if __name__=="__main__":main()
