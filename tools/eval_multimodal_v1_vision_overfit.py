#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def box_iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1])
    bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    return inter / max(aa+bb-inter, 1e-12)


def center_error(a, b):
    ac=((a[0]+a[2])/2,(a[1]+a[3])/2)
    bc=((b[0]+b[2])/2,(b[1]+b[3])/2)
    return math.hypot(ac[0]-bc[0], ac[1]-bc[1])


def load_train_module():
    path=ROOT/"tools/train_multimodal_v1_vision_ssod.py"
    spec=importlib.util.spec_from_file_location("vision_ssod_train",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="manifests/multimodal_v1/vision_manual8_overfit.jsonl"
    )
    ap.add_argument(
        "--checkpoint",
        default="outputs/own_multimodal_research/vision_manual8_overfit300/burn_in.pt"
    )
    ap.add_argument(
        "--config",
        default="configs/multimodal_v1/vision_ssod.yaml"
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/own_multimodal_research/vision_manual8_overfit300/eval"
    )
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--oracle-topk",type=int,default=50)
    args=ap.parse_args()

    device=torch.device(args.device)
    train=load_train_module()

    cfg=yaml.safe_load(resolve(args.config).read_text())
    model_cfg=cfg["model"]
    data_cfg=cfg["data"]
    ssod_cfg=cfg["ssod"]

    detrex=resolve(model_cfg["detrex_root"])
    sys.path.insert(0,str(detrex))
    sys.path.insert(0,str(detrex/"detectron2"))
    sys.path.insert(0,str(ROOT/"src"))

    from detectron2.config import LazyConfig, instantiate
    from rdq_uav.multimodal_v1.vision.uav_dino import \
        adapt_dino_class_head_to_single_uav

    # Build exactly the same model topology as training.
    dino_cfg=LazyConfig.load(str(resolve(model_cfg["config"])))
    dino_cfg.model.device=str(device)
    model=instantiate(dino_cfg.model).to(device)
    adapt_dino_class_head_to_single_uav(model)

    ckpt=torch.load(resolve(args.checkpoint),map_location="cpu")
    model.load_state_dict(ckpt["student"],strict=True)
    model.eval()

    records=[
        json.loads(x)
        for x in resolve(args.manifest).read_text().splitlines()
        if x.strip()
    ]

    camera_cfg=yaml.safe_load(resolve(data_cfg["camera_config"]).read_text())
    source_wh=tuple(
        int(x) for x in camera_cfg["cameras"]["left"]["resolution"]
    )

    short_edges=ssod_cfg["weak_short_edges"]
    if isinstance(short_edges,(list,tuple)):
        short_edge=int(short_edges[0])
    else:
        short_edge=int(short_edges)

    max_size=int(ssod_cfg["max_size"])

    out_dir=resolve(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    rows=[]

    for idx,r in enumerate(records):
        source=train.load_source_rgb(r["image_path"],source_wh)

        view_wh=train.resize_wh(source_wh,short_edge,max_size)
        transform=SimpleNamespace(
            view_wh=view_wh,
            horizontal_flip=False
        )

        view=train.apply_geometry(source,transform)
        tensor=train.pil_tensor(view,device)

        with torch.no_grad():
            result=model([
                train.native_input(tensor,transform)
            ])[0]

        inst=result["instances"] if isinstance(result,dict) else result
        inst=inst.to("cpu")

        boxes=inst.pred_boxes.tensor
        scores=inst.scores

        order=torch.argsort(scores,descending=True)
        boxes=boxes[order]
        scores=scores[order]

        sx=view_wh[0]/source_wh[0]
        sy=view_wh[1]/source_wh[1]

        boxes_src=boxes.clone()
        boxes_src[:,[0,2]]/=sx
        boxes_src[:,[1,3]]/=sy

        gt=[float(x) for x in r["box_xyxy_px"]]

        if len(boxes_src)==0:
            top_box=[float("nan")]*4
            top_score=0.0
            top_iou=0.0
            top_center=float("inf")
            oracle_iou=0.0
        else:
            top_box=[float(x) for x in boxes_src[0].tolist()]
            top_score=float(scores[0])
            top_iou=box_iou(top_box,gt)
            top_center=center_error(top_box,gt)

            k=min(args.oracle_topk,len(boxes_src))
            oracle_iou=max(
                box_iou([float(x) for x in boxes_src[j].tolist()],gt)
                for j in range(k)
            )

        row={
            "sequence":r["sequence_id"],
            "image":Path(r["image_path"]).name,
            "gt_box":[round(x,2) for x in gt],
            "pred_box":[round(x,2) for x in top_box],
            "score":round(top_score,6),
            "iou":round(top_iou,6),
            "center_error_px":round(top_center,3),
            f"oracle_iou_top{args.oracle_topk}":round(oracle_iou,6),
        }
        rows.append(row)

        print(json.dumps(row,ensure_ascii=False))

        # Visualization: GT + Top-1 prediction.
        vis=source.copy()
        draw=ImageDraw.Draw(vis)
        draw.rectangle(gt,outline="lime",width=3)
        if len(boxes_src):
            draw.rectangle(top_box,outline="red",width=3)

        vis.save(
            out_dir /
            f"{idx:02d}_{r['sequence_id']}_{Path(r['image_path']).name}"
        )

    ious=[r["iou"] for r in rows]
    centers=[r["center_error_px"] for r in rows]
    oracle=[r[f"oracle_iou_top{args.oracle_topk}"] for r in rows]

    s=sorted(ious)
    n=len(s)
    median=(
        s[n//2]
        if n%2
        else (s[n//2-1]+s[n//2])/2
    )

    summary={
        "status":"PASS",
        "num_images":len(rows),
        "mean_iou":sum(ious)/len(ious),
        "median_iou":median,
        "recall_at_0.50":sum(x>=0.50 for x in ious)/len(ious),
        "recall_at_0.75":sum(x>=0.75 for x in ious)/len(ious),
        "mean_center_error_px":sum(centers)/len(centers),
        f"mean_oracle_iou_top{args.oracle_topk}":
            sum(oracle)/len(oracle),
        "checkpoint":str(resolve(args.checkpoint)),
        "visualizations":str(out_dir),
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary,indent=2))

    (out_dir/"metrics.json").write_text(
        json.dumps(
            {"per_image":rows,"summary":summary},
            indent=2
        )
    )


if __name__=="__main__":
    main()
