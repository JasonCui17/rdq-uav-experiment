#!/usr/bin/env python3
"""GT-conditioned module evaluator; not system-level detection metrics."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from rdq_uav.mmuav.center_regressor import regression_metrics
from build_mmuav_center_regression_dataset import write_csv, MODE


def accepted_rows(directory):
    with (directory / "metadata.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["accepted"] == "True"]


def arrays(rows):
    return (np.array([[float(r[f"geometric_{a}"]) for a in "xyz"] for r in rows]).reshape(-1,3),
            np.array([[float(r[f"gt_{a}"]) for a in "xyz"] for r in rows]).reshape(-1,3))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,default=ROOT / "outputs/mmuav_paper_reproduction/datasets/center_regression")
    args=p.parse_args()
    d=args.dataset_dir
    rows=accepted_rows(d)
    result={"evaluation_mode":MODE,"distance_grouping_status":"DISABLED_REFERENCE_FRAME_UNVERIFIED"}
    by_seq,by_count=[],[]
    for split in ("train_sub","validation_sub"):
        subset=[r for r in rows if r["split"]==split]
        result[split]=regression_metrics(*arrays(subset))
        for seq in sorted({r["sequence_id"] for r in subset}):
            selected=[r for r in subset if r["sequence_id"]==seq]
            by_seq.append({"split":split,"sequence_id":seq,**regression_metrics(*arrays(selected))})
        for low,high in ((1,5),(6,10),(11,20),(21,50),(51,100),(101,float("inf"))):
            selected=[r for r in subset if low<=int(r["point_count"])<=high]
            by_count.append({"split":split,"point_count_bin":f"{low}-{high}",**regression_metrics(*arrays(selected))})
    by_seq.sort(key=lambda r:r.get("MSE_3D",0),reverse=True)
    write_csv(d / "geometric_center_by_sequence.csv",by_seq)
    write_csv(d / "geometric_center_by_point_count.csv",by_count)
    (d / "evaluation_summary.json").write_text(json.dumps(result,indent=2))
    (d / "paper_reference.json").write_text(json.dumps({"paper_reported_center_mse_before":.27,
        "paper_reported_center_mse_after":.05,"metric_alignment":"unresolved"},indent=2))
    val=sorted([r for r in rows if r["split"]=="validation_sub"],key=lambda r:float(r["distance_to_gt"]),reverse=True)
    chosen=val[:10]
    low=val[len(val)//2:]
    if low:
        chosen += [low[i] for i in np.random.default_rng(42).choice(len(low),min(5,len(low)),replace=False)]
    write_csv(d / "high_error_samples.csv",chosen)
    figures=d / "figures"
    figures.mkdir(exist_ok=True)
    for i,r in enumerate(chosen):
        with np.load(r["shard_path"],allow_pickle=False) as shard:
            idx=int(r["shard_index"])
            points=shard["points"][shard["offsets"][idx]:shard["offsets"][idx+1]]
        geom,gt=arrays([r])
        fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(projection="3d")
        ax.scatter(*points.T,s=8,label="raw cluster")
        ax.scatter(*geom[0],s=80,label="geometric center")
        ax.scatter(*gt[0],s=80,marker="x",label="GT")
        ax.set_title(f'{r["sequence_id"]} {r["timestamp"]}\nN={r["point_count"]} error={float(r["distance_to_gt"]):.3f}m gap={float(r["time_gap_ms"]):.2f}ms')
        ax.legend(); fig.savefig(figures / f"sample_{i:02d}.png"); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
