#!/usr/bin/env python3
"""Real-sample B0/B1 identity gate; performs no optimizer update."""

from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
for p in (ROOT,ROOT/"src"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import torch,yaml
from torch.utils.data import DataLoader,Subset
from rdq_uav.runtime_paths import apply_runtime_path_overrides,resolve_project_path
from rdq_uav.multimodal_v2.data import build_datasets,collate_multimodal_v2,prepare_model_batch
from rdq_uav.multimodal_v2.training import build_runtime,synchronize_dino_device
from rdq_uav.multimodal_v1 import load_left_projection_context

def maxdiff(a,b):
    return 0.0 if not a.numel() else float((a.float()-b.float()).abs().max())

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--samples",type=int,default=2);args=p.parse_args()
    resolve=lambda x:resolve_project_path(x,ROOT);cfg=apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    _,_,dataset=build_datasets(cfg,ROOT);dataset=Subset(dataset,range(min(args.samples,len(dataset))))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_multimodal_v2)
    device=torch.device(args.device);runtime=build_runtime(cfg,ROOT,torch.device("cpu"));runtime.model.to(device).eval();synchronize_dino_device(runtime,device)
    runtime.projection_base=load_left_projection_context(runtime.camera_config,runtime.geometry_calibration,image_scale_xy=torch.ones((1,2),device=device),device=device)
    reports=[]
    with torch.no_grad():
      for batch in loader:
        lidar,images,masks,context,_,_=prepare_model_batch(batch,runtime.dino_detector,runtime.projection_base,device)
        runtime.model.interaction_enabled=False
        with torch.autocast(device_type=device.type,enabled=False):b0=runtime.model(lidar,images,masks,context)
        runtime.model.interaction_enabled=True
        with torch.autocast(device_type=device.type,enabled=False):b1=runtime.model(lidar,images,masks,context)
        reports.append({
          "sample_id":batch["sample_id"][0],"radar_xyz_max_diff":maxdiff(b0.radar_candidates.xyz_m,b1.radar_candidates.xyz_m),
          "radar_score_max_diff":maxdiff(b0.radar_candidates.score,b1.radar_candidates.score),
          "final_xyz_max_diff":maxdiff(b0.xyz_m,b1.xyz_m),"final_score_max_diff":maxdiff(b0.score_after,b1.score_after),
          "source_index_equal":bool(torch.equal(b0.radar_source_index,b1.radar_source_index)),
          "b1_nonzero_score_deltas":int(torch.count_nonzero(b1.score_delta_logit)),
          "b1_valid_evidence":int(b1.diagnostics["radar_evidence"].valid.sum()),
        })
    passed=all(r["radar_xyz_max_diff"]==0 and r["radar_score_max_diff"]==0 and r["final_xyz_max_diff"]==0 and r["final_score_max_diff"]==0 and r["source_index_equal"] and r["b1_nonzero_score_deltas"]==0 for r in reports)
    report={"status":"PASS" if passed else "FAIL","optimizer_steps":0,"samples":reports}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/"b0_b1_identity.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)
if __name__=="__main__":main()
