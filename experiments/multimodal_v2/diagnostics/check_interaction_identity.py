#!/usr/bin/env python3
"""Real-sample interaction-off/on identity gate; performs no optimizer update."""

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
from rdq_uav.multimodal_v2.geometry import load_left_projection_context

def maxdiff(a,b):
    return 0.0 if not a.numel() else float((a.float()-b.float()).abs().max())

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--samples",type=int,default=2);args=p.parse_args()
    resolve=lambda x:resolve_project_path(x,ROOT);cfg=apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    _,dataset=build_datasets(cfg,ROOT);dataset=Subset(dataset,range(min(args.samples,len(dataset))))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate_multimodal_v2)
    device=torch.device(args.device);runtime=build_runtime(cfg,ROOT,torch.device("cpu"));runtime.model.to(device).eval();synchronize_dino_device(runtime,device)
    runtime.projection_base=load_left_projection_context(runtime.camera_config,runtime.geometry_calibration,image_scale_xy=torch.ones((1,2),device=device),device=device)
    reports=[]
    with torch.no_grad():
      for batch in loader:
        lidar,images,masks,projection,_,_=prepare_model_batch(batch,runtime.dino_detector,runtime.projection_base,device)
        runtime.model.interaction_enabled=False
        with torch.autocast(device_type=device.type,enabled=False):off=runtime.model(lidar,images,masks,projection)
        runtime.model.interaction_enabled=True
        with torch.autocast(device_type=device.type,enabled=False):on=runtime.model(lidar,images,masks,projection)
        reports.append({
          "sample_id":batch["sample_id"][0],"radar_xyz_max_diff":maxdiff(off.radar_candidates.xyz_m,on.radar_candidates.xyz_m),
          "radar_score_max_diff":maxdiff(off.radar_candidates.score,on.radar_candidates.score),
          "final_xyz_max_diff":maxdiff(off.xyz_m,on.xyz_m),"final_box_max_diff":maxdiff(off.box_xyxy_px,on.box_xyxy_px),
          "score_3d_max_diff":maxdiff(off.score_3d_after,on.score_3d_after),
          "score_2d_max_diff":maxdiff(off.score_2d_after,on.score_2d_after),
          "source_index_equal":bool(torch.equal(off.radar_source_index,on.radar_source_index)),
          "top3d_equal":bool(torch.equal(off.top3d_indices(1)[0],on.top3d_indices(1)[0])),
          "top2d_equal":bool(torch.equal(off.top2d_indices(1)[0],on.top2d_indices(1)[0])),
          "on_nonzero_3d_deltas":int(torch.count_nonzero(on.delta_3d)),
          "on_nonzero_2d_deltas":int(torch.count_nonzero(on.delta_2d)),
          "on_valid_evidence":int(on.diagnostics["radar_evidence"].valid.sum()),
        })
    passed=all(r["radar_xyz_max_diff"]==0 and r["radar_score_max_diff"]==0 and r["final_xyz_max_diff"]==0 and r["final_box_max_diff"]==0 and r["score_3d_max_diff"]==0 and r["score_2d_max_diff"]==0 and r["source_index_equal"] and r["top3d_equal"] and r["top2d_equal"] and r["on_nonzero_3d_deltas"]==0 and r["on_nonzero_2d_deltas"]==0 for r in reports)
    report={"status":"PASS" if passed else "FAIL","optimizer_steps":0,"samples":reports}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/"interaction_identity.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)
if __name__=="__main__":main()
