#!/usr/bin/env python3
"""Train the single-class UAV DINO with geometry-guided semi-supervision.

Pipeline:
  V0 burn-in: verified labeled_train boxes -> native DINO criterion.
  Calibration: labeled_calibration selects tau_high/tau_mid.
  V1 SSOD: 1 labeled + 3 unlabeled attempts per optimizer step.
    HIGH pseudo -> native DINO cls + L1 + GIoU.
    MEDIUM pseudo -> DINO matcher + classification loss only.
    IGNORE -> no positive or negative pseudo supervision.

The 3D GT projection is used only to filter teacher boxes. It is never converted
into a bounding-box target.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=ROOT/"configs/multimodal_v1/vision_ssod.yaml")
    p.add_argument("--device",default="cuda")
    p.add_argument("--tiny",action="store_true",help="20 burn-in + 20 SSOD steps for correctness/overfit diagnostics")
    p.add_argument("--burn-in-only",action="store_true")
    p.add_argument("--annotation-manifest",type=Path,help="Override data.annotation_manifest")
    p.add_argument("--burn-steps",type=int,help="Override burn-in steps; useful for tiny overfit")
    p.add_argument("--output-dir",type=Path)
    return p.parse_args()


def resolve(path_value: str | Path) -> Path:
    path=Path(path_value)
    return path if path.is_absolute() else ROOT/path


def resize_wh(source_wh: tuple[int,int],short_edge:int,max_size:int)->tuple[int,int]:
    width,height=source_wh
    scale=float(short_edge)/float(min(width,height))
    if float(max(width,height))*scale>float(max_size):
        scale=float(max_size)/float(max(width,height))
    return int(width*scale+0.5),int(height*scale+0.5)


def load_source_rgb(path: str | Path,source_wh:tuple[int,int])->Image.Image:
    image=Image.open(path).convert("RGB")
    width,height=source_wh
    if image.width<width or image.height<height:
        raise ValueError(f"image {image.size} smaller than calibrated {source_wh}")
    return image.crop((0,0,width,height))


def apply_geometry(image:Image.Image,transform:Any)->Image.Image:
    out=image.resize(transform.view_wh,Image.Resampling.BILINEAR)
    if transform.horizontal_flip:
        out=ImageOps.mirror(out)
    return out


def weak_photo(image:Image.Image,rng:random.Random)->Image.Image:
    out=ImageEnhance.Brightness(image).enhance(1.0+rng.uniform(-.05,.05))
    out=ImageEnhance.Contrast(out).enhance(1.0+rng.uniform(-.05,.05))
    out=ImageEnhance.Color(out).enhance(1.0+rng.uniform(-.05,.05))
    return out


def strong_photo(image:Image.Image,rng:random.Random)->Image.Image:
    out=ImageEnhance.Brightness(image).enhance(1.0+rng.uniform(-.30,.30))
    out=ImageEnhance.Contrast(out).enhance(1.0+rng.uniform(-.30,.30))
    out=ImageEnhance.Color(out).enhance(1.0+rng.uniform(-.30,.30))
    if rng.random()<.20:
        out=ImageOps.grayscale(out).convert("RGB")
    if rng.random()<.20:
        out=out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(.1,1.2)))
    return out


def pil_tensor(image:Image.Image,device:torch.device)->torch.Tensor:
    array=np.asarray(image,dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2,0,1).contiguous().to(device)


def make_transform(source_wh,short_edges,max_size,flip_p,rng,ViewTransform):
    edge=int(rng.choice(short_edges)); view_wh=resize_wh(source_wh,edge,max_size)
    return ViewTransform(source_wh,view_wh,rng.random()<flip_p)


def make_instances(box_source,transform,device,Instances,Boxes):
    box_view=transform.source_boxes_to_view(torch.as_tensor(box_source,dtype=torch.float32,device=device)).reshape(1,4)
    h,w=transform.view_wh[1],transform.view_wh[0]
    instances=Instances((h,w)); instances.gt_boxes=Boxes(box_view); instances.gt_classes=torch.zeros(1,dtype=torch.long,device=device)
    return instances


def native_input(tensor,transform,instances=None):
    item={"image":tensor,"height":transform.view_wh[1],"width":transform.view_wh[0]}
    if instances is not None:item["instances"]=instances
    return item


def sum_losses(loss_dict):
    values=[value if value.ndim==0 else value.sum() for value in loss_dict.values()]
    if not values:raise RuntimeError("empty DINO loss dictionary")
    return torch.stack(values).sum()


def build_optimizer(model,cfg):
    lr=float(cfg["lr"]); backbone_lr=float(cfg["backbone_lr"]); weight_decay=float(cfg["weight_decay"])
    backbone=[]; other=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        (backbone if name.startswith("backbone.") else other).append(param)
    return torch.optim.AdamW([
        {"params":backbone,"lr":backbone_lr},
        {"params":other,"lr":lr},
    ],weight_decay=weight_decay)


def build_train_pool(dataset,image_index,excluded_keys,excluded_images):
    pool=[]
    for record in dataset.records:
        key=(record["sequence_id"],record["query_uid"])
        if key in excluded_keys:continue
        match=image_index.match(record["sequence_id"],record["query_time"])
        if not match.valid or match.path is None:continue
        image_key=(record["sequence_id"],match.path.name)
        if image_key in excluded_images:continue
        pool.append({"record":record,"image_path":str(match.path)})
    if not pool:raise RuntimeError("unlabeled train pool is empty")
    return pool


def project_gt_source(record,projection,device,project_omni_radtan):
    xyz=np.load(record["target_path"],allow_pickle=False).reshape(3).astype(np.float32)
    point=torch.from_numpy(xyz).to(device).reshape(1,3)
    pixel,valid=project_omni_radtan(point,torch.zeros(1,dtype=torch.long,device=device),projection)
    if not bool(valid.item()):return None
    return pixel[0]


def teacher_raw(adapter,tensor,transform):
    output=adapter.forward_raw([native_input(tensor,transform)])
    return {"pred_logits":output["pred_logits"][0],"pred_boxes":output["pred_boxes"][0]}


def mine_one(teacher_adapter,image,point_source,cfg,rng,ViewTransform,PseudoLabelPolicy,mine_geometry_guided_pseudo,device):
    short_edges=tuple(int(x) for x in cfg["weak_short_edges"]); max_size=int(cfg["max_size"]); flip_p=float(cfg["horizontal_flip_p"])
    t1=make_transform(image.size,short_edges,max_size,flip_p,rng,ViewTransform)
    t2=make_transform(image.size,short_edges,max_size,flip_p,rng,ViewTransform)
    w1=pil_tensor(weak_photo(apply_geometry(image,t1),rng),device)
    w2=pil_tensor(weak_photo(apply_geometry(image,t2),rng),device)
    with torch.no_grad():
        o1=teacher_raw(teacher_adapter,w1,t1); o2=teacher_raw(teacher_adapter,w2,t2)
    policy=PseudoLabelPolicy(
        tau_high=float(cfg["tau_high"]),tau_mid=float(cfg["tau_mid"]),
        high_geometry_px=float(cfg["high_geometry_px"]),medium_geometry_px=float(cfg["medium_geometry_px"]),
        high_stability_iou=float(cfg["high_stability_iou"]),medium_stability_iou=float(cfg["medium_stability_iou"]),
    )
    pseudo=mine_geometry_guided_pseudo(o1,o2,transform1=t1,transform2=t2,projected_point_source=point_source,policy=policy)
    return pseudo,t1


def calibration_observations(records,teacher_adapter,projection,source_wh,ssod_cfg,rng,device,modules):
    ViewTransform,PseudoLabelPolicy,mine_geometry_guided_pseudo,box_iou_xyxy,project_omni_radtan=modules
    observations=[]
    permissive=dict(ssod_cfg); permissive["tau_high"]=0.0; permissive["tau_mid"]=0.0
    for index,record in enumerate(records):
        image=load_source_rgb(record.image_path,source_wh)
        # Calibration manifest stores GT XYZ directly; project without reading sequence files.
        xyz=torch.tensor(record.gt_xyz_m,dtype=torch.float32,device=device).reshape(1,3)
        point,valid=project_omni_radtan(xyz,torch.zeros(1,dtype=torch.long,device=device),projection)
        if not bool(valid.item()):continue
        pseudo,_=mine_one(teacher_adapter,image,point[0],permissive,rng,ViewTransform,PseudoLabelPolicy,mine_geometry_guided_pseudo,device)
        if not pseudo.valid:continue
        gt=torch.tensor(record.box_xyxy_px,dtype=torch.float32,device=device)
        observations.append(modules[5](pseudo.score,pseudo.geometry_px,pseudo.stability_iou,float(box_iou_xyxy(pseudo.box_xyxy_source,gt).item())))
    return observations


def save_checkpoint(path,student,teacher,optimizer,step,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"student":student.state_dict(),"teacher":None if teacher is None else teacher.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"metadata":metadata},path)


def main()->None:
    args=parse_args(); cfg=yaml.safe_load(args.config.read_text())
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    data_cfg=cfg["data"]; model_cfg=cfg["model"]; opt_cfg=cfg["optimizer"]; ssod_cfg=dict(cfg["ssod"])
    burn_steps=(20 if args.tiny else int(cfg["burn_in"]["steps"])) if args.burn_steps is None else int(args.burn_steps)
    if burn_steps <= 0: raise ValueError("burn-steps must be positive")
    ssod_steps=20 if args.tiny else int(ssod_cfg["steps"])
    output_dir=args.output_dir or resolve(cfg["output"]["directory"]); output_dir.mkdir(parents=True,exist_ok=True)
    rng=random.Random(int(data_cfg["seed"])); torch.manual_seed(int(data_cfg["seed"])); np.random.seed(int(data_cfg["seed"])%(2**32-1))

    detrex=resolve(model_cfg["detrex_root"]); sys.path.insert(0,str(detrex)); sys.path.insert(0,str(detrex/"detectron2")); sys.path.insert(0,str(ROOT/"src"))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig,instantiate
    from detectron2.structures import Boxes,Instances
    from rdq_uav.lidar_v2.data import LiDARUAVDataset
    from rdq_uav.multimodal_v1.data import LeftImageIndex
    from rdq_uav.multimodal_v1.interaction.geometry_local import project_omni_radtan
    from rdq_uav.multimodal_v1.projection import load_left_projection_context
    from rdq_uav.multimodal_v1.vision.dino_adapter import DINOAdapter
    from rdq_uav.multimodal_v1.vision.ssod import (
        CalibrationObservation,PseudoLabelPolicy,PseudoQuality,ViewTransform,box_iou_xyxy,
        calibrate_score_threshold,classification_only_pseudo_loss,ema_update,
        linear_unsup_weight,mine_geometry_guided_pseudo,source_xyxy_to_normalized_cxcywh,weighted_loss_sum,
    )
    from rdq_uav.multimodal_v1.vision.ssod_data import load_label_manifest
    from rdq_uav.multimodal_v1.vision.uav_dino import adapt_dino_class_head_to_single_uav

    manifest_path=resolve(args.annotation_manifest) if args.annotation_manifest is not None else resolve(data_cfg["annotation_manifest"])
    manifest=load_label_manifest(manifest_path,require_boxes=True)
    labeled=[r for r in manifest if r.role=="labeled_train"]; calibration=[r for r in manifest if r.role=="labeled_calibration"]
    if not labeled:raise RuntimeError("manifest needs at least one labeled_train verified box")
    if not args.burn_in_only and not calibration:
        raise RuntimeError("SSOD stage needs labeled_calibration verified boxes; burn-in-only does not")
    excluded={r.key for r in manifest}
    excluded_images={(r.sequence_id,Path(r.image_path).name) for r in manifest}

    camera_cfg=yaml.safe_load(resolve(data_cfg["camera_config"]).read_text()); source_wh=tuple(int(x) for x in camera_cfg["cameras"]["left"]["resolution"])
    geometry=json.loads(resolve(data_cfg["geometry_calibration"]).read_text())
    projection=load_left_projection_context(resolve(data_cfg["camera_config"]),resolve(data_cfg["geometry_calibration"]),image_scale_xy=torch.ones(1,2),device=device)
    unlabeled=[]
    if not args.burn_in_only:
        dataset=LiDARUAVDataset(resolve(data_cfg["root"]),resolve(data_cfg["split_file"]),data_cfg["train_split"])
        image_index=LeftImageIndex(resolve(data_cfg["root"]),time_offset_s=float(geometry["time_offset_s"]),max_abs_gap_s=float(data_cfg["max_image_gap_s"]))
        unlabeled=build_train_pool(dataset,image_index,excluded,excluded_images)

    dino_cfg=LazyConfig.load(str(resolve(model_cfg["config"]))); dino_cfg.model.device=str(device)
    student=instantiate(dino_cfg.model).to(device)
    DetectionCheckpointer(student).load(str(resolve(model_cfg["checkpoint"])))
    adapt_dino_class_head_to_single_uav(student)
    optimizer=build_optimizer(student,opt_cfg)
    amp=bool(opt_cfg.get("amp",True)) and device.type=="cuda"; scaler=torch.cuda.amp.GradScaler(enabled=amp)

    print(json.dumps({"manifest":str(manifest_path),"labeled_train":len(labeled),"labeled_calibration":len(calibration),"unlabeled":len(unlabeled),"burn_steps":burn_steps,"ssod_steps":0 if args.burn_in_only else ssod_steps,"device":str(device)},indent=2))

    # V0 supervised burn-in.
    student.train(); start=time.time()
    for step in range(burn_steps):
        record=labeled[step%len(labeled)]; image=load_source_rgb(record.image_path,source_wh)
        transform=make_transform(source_wh,ssod_cfg["weak_short_edges"],ssod_cfg["max_size"],ssod_cfg["horizontal_flip_p"],rng,ViewTransform)
        strong=pil_tensor(strong_photo(apply_geometry(image,transform),rng),device)
        instances=make_instances(record.box_xyxy_px,transform,device,Instances,Boxes)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            losses=student([native_input(strong,transform,instances)]); loss=sum_losses(losses)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(),float(opt_cfg["grad_clip_norm"]))
        scaler.step(optimizer); scaler.update()
        if (step+1)%int(cfg["output"]["log_every"])==0 or step==0:
            primary={k:float(v.detach()) for k,v in losses.items() if k in {"loss_class","loss_bbox","loss_giou"}}
            print(f"[burn-in] step={step+1}/{burn_steps} loss={float(loss.detach()):.5f} primary={json.dumps(primary,sort_keys=True)}")
    save_checkpoint(output_dir/"burn_in.pt",student,None,optimizer,burn_steps,{"stage":"burn_in","manifest":str(manifest_path)})
    if args.burn_in_only:
        print(json.dumps({"status":"PASS","stage":"burn_in_only","steps":burn_steps,"labeled_train":len(labeled),"checkpoint":str(output_dir/"burn_in.pt")},indent=2))
        return

    # Teacher begins as exact post-burn-in student.
    teacher=copy.deepcopy(student).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    teacher_adapter=DINOAdapter(teacher).eval(); student_adapter=DINOAdapter(student)

    modules=(ViewTransform,PseudoLabelPolicy,mine_geometry_guided_pseudo,box_iou_xyxy,project_omni_radtan,CalibrationObservation)
    observations=calibration_observations(calibration,teacher_adapter,projection,source_wh,ssod_cfg,rng,device,modules)
    tau_high=calibrate_score_threshold(observations,geometry_limit_px=float(ssod_cfg["high_geometry_px"]),stability_min_iou=float(ssod_cfg["high_stability_iou"]),success_iou_threshold=float(ssod_cfg["high_success_iou"]),minimum_precision=float(ssod_cfg["high_min_precision"]),min_count=int(ssod_cfg["calibration_min_count"]))
    tau_mid=calibrate_score_threshold(observations,geometry_limit_px=float(ssod_cfg["medium_geometry_px"]),stability_min_iou=float(ssod_cfg["medium_stability_iou"]),success_iou_threshold=float(ssod_cfg["medium_success_iou"]),minimum_precision=float(ssod_cfg["medium_min_precision"]),min_count=int(ssod_cfg["calibration_min_count"]))
    tau_mid=min(tau_mid,tau_high); ssod_cfg.update(tau_high=tau_high,tau_mid=tau_mid)
    calibration_report={"tau_high":tau_high,"tau_mid":tau_mid,"observations":len(observations),"medium_success_iou":float(ssod_cfg["medium_success_iou"])}
    (output_dir/"calibration.json").write_text(json.dumps(calibration_report,indent=2)+"\n")
    print("[calibration]",json.dumps(calibration_report))

    pseudo_counts={"high":0,"medium":0,"ignore":0,"invalid_projection":0}
    unlabeled_per_step=int(ssod_cfg["unlabeled_per_step"])
    for step in range(ssod_steps):
        optimizer.zero_grad(set_to_none=True); student.train(); teacher.eval()
        record=labeled[(burn_steps+step)%len(labeled)]; image=load_source_rgb(record.image_path,source_wh)
        transform=make_transform(source_wh,ssod_cfg["weak_short_edges"],ssod_cfg["max_size"],ssod_cfg["horizontal_flip_p"],rng,ViewTransform)
        tensor=pil_tensor(strong_photo(apply_geometry(image,transform),rng),device)
        instances=make_instances(record.box_xyxy_px,transform,device,Instances,Boxes)
        with torch.cuda.amp.autocast(enabled=amp):
            supervised=sum_losses(student([native_input(tensor,transform,instances)]))
        scaler.scale(supervised).backward()

        lambda_u=linear_unsup_weight(step,ssod_steps,target=float(ssod_cfg["lambda_unsup"]),ramp_fraction=float(ssod_cfg["ramp_fraction"]))
        unsup_value=0.0
        for offset in range(unlabeled_per_step):
            sample=unlabeled[(step*unlabeled_per_step+offset)%len(unlabeled)]; source=load_source_rgb(sample["image_path"],source_wh)
            point=project_gt_source(sample["record"],projection,device,project_omni_radtan)
            if point is None:
                pseudo_counts["invalid_projection"]+=1; continue
            pseudo,strong_transform=mine_one(teacher_adapter,source,point,ssod_cfg,rng,ViewTransform,PseudoLabelPolicy,mine_geometry_guided_pseudo,device)
            if pseudo.quality==PseudoQuality.IGNORE:
                pseudo_counts["ignore"]+=1; continue
            strong_tensor=pil_tensor(strong_photo(apply_geometry(source,strong_transform),rng),device)
            if pseudo.quality==PseudoQuality.HIGH:
                pseudo_counts["high"]+=1
                pseudo_instances=make_instances(pseudo.box_xyxy_source,strong_transform,device,Instances,Boxes)
                with torch.cuda.amp.autocast(enabled=amp):
                    u_loss=sum_losses(student([native_input(strong_tensor,strong_transform,pseudo_instances)]))
            else:
                pseudo_counts["medium"]+=1
                with torch.cuda.amp.autocast(enabled=amp):
                    raw=student_adapter.forward_raw([native_input(strong_tensor,strong_transform)],allow_training_candidate_path=True)
                    target_box=source_xyxy_to_normalized_cxcywh(pseudo.box_xyxy_source,strong_transform).reshape(1,4).to(device)
                    targets=[{"labels":torch.zeros(1,dtype=torch.long,device=device),"boxes":target_box}]
                    u_loss=weighted_loss_sum(classification_only_pseudo_loss(student.criterion,raw,targets))
            scale=lambda_u/max(1,unlabeled_per_step); scaler.scale(u_loss*scale).backward(); unsup_value+=float(u_loss.detach())*scale

        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(student.parameters(),float(opt_cfg["grad_clip_norm"]))
        scaler.step(optimizer); scaler.update(); ema_update(teacher,student,decay=float(ssod_cfg["ema_decay"]))
        if (step+1)%int(cfg["output"]["log_every"])==0 or step==0:
            print(f"[ssod] step={step+1}/{ssod_steps} sup={float(supervised.detach()):.5f} unsup={unsup_value:.5f} lambda_u={lambda_u:.3f} pseudo={pseudo_counts}")
        if (step+1)%int(cfg["output"]["save_every"])==0:
            save_checkpoint(output_dir/f"ssod_step_{step+1:06d}.pt",student,teacher,optimizer,step+1,{"stage":"ssod",**calibration_report,"pseudo_counts":dict(pseudo_counts)})

    save_checkpoint(output_dir/"vision_ssod_final.pt",student,teacher,optimizer,ssod_steps,{"stage":"ssod_final",**calibration_report,"pseudo_counts":pseudo_counts,"elapsed_s":time.time()-start})
    print(json.dumps({"status":"PASS","output":str(output_dir/"vision_ssod_final.pt"),"pseudo_counts":pseudo_counts,"calibration":calibration_report},indent=2))


if __name__=="__main__":
    main()
