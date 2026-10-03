#!/usr/bin/env python3
"""Evaluate the independent LiDAR V2 baseline on the annotated-20 val split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from rdq_uav.lidar_v2.data import LiDARUAVDataset, collate_lidar_samples
from rdq_uav.lidar_v2.loss import CandidateLoss
from rdq_uav.lidar_v2.model import LiDARUAVDetector
from rdq_uav.lidar_v2.runtime import evaluate_batch, summarize_metrics
from rdq_uav.lidar_v2.selector import CandidateSelector
from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--max-events", type=int, default=20)
    p.add_argument("--limit",type=int); p.add_argument("--num-workers",type=int,default=2); args=p.parse_args()
    resolve=lambda value: resolve_project_path(value,ROOT)
    cfg=apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text())); data=cfg["data"]
    lidar_cfg=yaml.safe_load(resolve(cfg["initialization"]["lidar_config"]).read_text())
    dataset=LiDARUAVDataset(resolve(data["root"]),resolve(data["split_file"]),data["val_split"],max_events=args.max_events)
    if args.limit is not None: dataset=Subset(dataset,range(min(args.limit,len(dataset))))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,collate_fn=collate_lidar_samples)
    device=torch.device(args.device); model=LiDARUAVDetector(lidar_cfg).to(device).eval()
    payload=torch.load(resolve(args.checkpoint),map_location="cpu",weights_only=False)
    model.load_state_dict(payload.get("model_state",payload),strict=True)
    # E0 is evaluated with its own frozen selector contract. The multimodal
    # Top-50 candidate pool is deliberately not substituted here.
    selector=CandidateSelector(lidar_cfg); criterion=CandidateLoss(lidar_cfg); rows=[]
    with torch.no_grad():
        for batch in loader:
            batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            with torch.autocast(device_type=device.type,enabled=False): output=model(batch)
            rows.extend(evaluate_batch(output,batch,selector,criterion))
    report={"baseline":"E0_independent_lidar_v2","queries":len(rows),"precision":"fp32",
            "checkpoint":str(resolve(args.checkpoint)),"metrics":summarize_metrics(rows)}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=True))
    with (args.output/"per_query.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row)+"\n")
    print(json.dumps(report,indent=2,allow_nan=True))


if __name__=="__main__": main()
