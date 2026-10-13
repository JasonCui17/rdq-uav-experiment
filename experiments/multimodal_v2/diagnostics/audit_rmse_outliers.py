#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-query", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prediction-key", default="pred_xyz",
                   choices=("pred_xyz","before_pred_xyz","radar_pred_xyz"))
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args()

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def rmse(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a*a))) if len(a) else None

def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(x) for x in args.per_query.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for r in raw:
        if not r.get("has_gt3d"):
            continue
        gt, pred = r.get("gt_xyz"), r.get(args.prediction_key)
        if gt is None or pred is None:
            continue
        gt = np.asarray(gt, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if gt.shape != (3,) or pred.shape != (3,) or not np.isfinite(gt).all() or not np.isfinite(pred).all():
            continue
        d = pred - gt
        e = float(np.linalg.norm(d))
        rows.append({
            "sequence_id": r.get("sequence_id"),
            "sample_id": r.get("sample_id"),
            "query_time": r.get("query_time"),
            "image_path": r.get("image_path"),
            "gt_x": float(gt[0]), "gt_y": float(gt[1]), "gt_z": float(gt[2]),
            "pred_x": float(pred[0]), "pred_y": float(pred[1]), "pred_z": float(pred[2]),
            "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]),
            "abs_dx": float(abs(d[0])), "abs_dy": float(abs(d[1])), "abs_dz": float(abs(d[2])),
            "error_3d_m": e,
            "sq_x": float(d[0]**2), "sq_y": float(d[1]**2), "sq_z": float(d[2]**2),
            "sq_3d": float(np.dot(d,d)),
            "radar_candidate_count": r.get("radar_candidate_count"),
            "vision_candidate_count": r.get("vision_candidate_count"),
            "top_source_type": r.get("top_source_type"),
        })
    if not rows:
        raise SystemExit("No valid rows")

    dx = np.array([r["dx"] for r in rows]); dy = np.array([r["dy"] for r in rows]); dz = np.array([r["dz"] for r in rows])
    err = np.array([r["error_3d_m"] for r in rows])
    sx,sy,sz,s3 = np.sum(dx*dx),np.sum(dy*dy),np.sum(dz*dz),np.sum(err*err)
    for r in rows:
        r["share_x_sq_pct"] = 100*r["sq_x"]/sx if sx else 0
        r["share_y_sq_pct"] = 100*r["sq_y"]/sy if sy else 0
        r["share_z_sq_pct"] = 100*r["sq_z"]/sz if sz else 0
        r["share_3d_sq_pct"] = 100*r["sq_3d"]/s3 if s3 else 0

    n=min(args.top_n,len(rows))
    write_csv(args.output/"worst_3d.csv", sorted(rows,key=lambda r:r["error_3d_m"], reverse=True)[:n])
    write_csv(args.output/"worst_x.csv", sorted(rows,key=lambda r:r["abs_dx"], reverse=True)[:n])
    write_csv(args.output/"worst_y.csv", sorted(rows,key=lambda r:r["abs_dy"], reverse=True)[:n])
    write_csv(args.output/"worst_z.csv", sorted(rows,key=lambda r:r["abs_dz"], reverse=True)[:n])

    groups=defaultdict(list)
    for r in rows: groups[r["sequence_id"]].append(r)
    seq=[]
    for sid,g in sorted(groups.items()):
        ex=np.array([x["dx"] for x in g]); ey=np.array([x["dy"] for x in g]); ez=np.array([x["dz"] for x in g]); ee=np.array([x["error_3d_m"] for x in g])
        seq.append({
            "sequence_id":sid,"n":len(g),
            "rmse_x_m":rmse(ex),"rmse_y_m":rmse(ey),"rmse_z_m":rmse(ez),
            "rmse_3d_m":float(np.sqrt(np.mean(ee**2))),
            "median_error_m":float(np.median(ee)),
            "p95_error_m":float(np.percentile(ee,95)),
            "max_error_m":float(np.max(ee)),
            "n_over_2m":int(np.sum(ee>2)),"n_over_5m":int(np.sum(ee>5)),
            "n_over_10m":int(np.sum(ee>10)),"n_over_20m":int(np.sum(ee>20)),
        })
    write_csv(args.output/"sequence_summary.csv",seq)

    order=np.argsort(err)[::-1]
    trim=[]
    for k in [0,1,2,5,10,20,50,100]:
        if k>=len(rows): continue
        keep=order[k:]
        trim.append({
            "removed_worst_k":k,"remaining_n":len(keep),
            "rmse_x_m":rmse(dx[keep]),"rmse_y_m":rmse(dy[keep]),"rmse_z_m":rmse(dz[keep]),
            "rmse_3d_m":float(np.sqrt(np.mean(err[keep]**2))),
            "median_error_m":float(np.median(err[keep])),
            "success_0p5m":float(np.mean(err[keep]<=0.5)),
            "success_1m":float(np.mean(err[keep]<=1.0)),
            "success_2m":float(np.mean(err[keep]<=2.0)),
        })
    write_csv(args.output/"rmse_trim_curve.csv",trim)

    contrib=[]
    sorted_sq=(err[order]**2)
    for k in [1,2,5,10,20,50,100]:
        if k>len(rows): continue
        contrib.append({
            "worst_k":k,
            "fraction_of_queries_pct":100*k/len(rows),
            "share_of_total_squared_3d_error_pct":100*float(sorted_sq[:k].sum()/sorted_sq.sum())
        })
    write_csv(args.output/"outlier_contribution.csv",contrib)

    summary={
        "prediction_key":args.prediction_key,
        "valid_queries":len(rows),
        "rmse_x_m":rmse(dx),"rmse_y_m":rmse(dy),"rmse_z_m":rmse(dz),
        "rmse_3d_m":float(np.sqrt(np.mean(err**2))),
        "mean_error_m":float(np.mean(err)),"median_error_m":float(np.median(err)),
        "p90_error_m":float(np.percentile(err,90)),"p95_error_m":float(np.percentile(err,95)),
        "p99_error_m":float(np.percentile(err,99)),"max_error_m":float(np.max(err)),
        "counts":{
            "<=0.5m":int(np.sum(err<=0.5)),"<=1m":int(np.sum(err<=1)),
            "<=2m":int(np.sum(err<=2)),">2m":int(np.sum(err>2)),
            ">5m":int(np.sum(err>5)),">10m":int(np.sum(err>10)),">20m":int(np.sum(err>20))
        }
    }
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
