#!/usr/bin/env python3
"""Export real A2/B2/B3 V2 forward computations, or serve their replay viewer."""
from __future__ import annotations

import argparse
import gc
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import numpy as np
from PIL import Image
import torch
import torchvision
import yaml

from rdq_uav.multimodal_v2.data import build_split_dataset, collate_multimodal_v2, prepare_model_batch
from rdq_uav.multimodal_v2.geometry import load_left_projection_context, project_omni_radtan
from rdq_uav.multimodal_v2.trace import TraceRecorder
from rdq_uav.multimodal_v2.training import build_runtime, synchronize_dino_device, _stage_batch
from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from experiments.multimodal_v2.evaluate import load_network

ASSETS = {
    "A2": ("experiments/multimodal_v2/configs/b0_a2_learned_sbe32.yaml",
           "outputs/radar_embedding_ab_seed42/a2_learned_sbe32/checkpoints/best.ckpt"),
    "B2": ("outputs/b2_b3_a2_coco_seed42/configs/b2.yaml",
           "outputs/b2_b3_a2_coco_seed42/b2/checkpoints/best.ckpt"),
    "B3": ("outputs/b2_b3_a2_coco_seed42/configs/b3.yaml",
           "outputs/b2_b3_a2_coco_seed42/b3/checkpoints/best.ckpt"),
}
PAGE = Path(__file__).with_name("trace_viewer.html")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_for(model: str, data_root: Path | None, *, require_checkpoint: bool = True):
    config_path = ROOT / ASSETS[model][0]
    checkpoint = ROOT / ASSETS[model][1]
    if not config_path.is_file(): raise FileNotFoundError(config_path)
    if require_checkpoint and not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    config = apply_runtime_path_overrides(yaml.safe_load(config_path.read_text()))
    overrides = {}
    if data_root is not None:
        config["data"]["root"] = str(resolve_project_path(data_root, ROOT))
        overrides["data.root"] = "--data-root"
    else:
        try:
            configured_root_exists = resolve_project_path(config["data"]["root"], ROOT).is_dir()
        except OSError:
            configured_root_exists = False
    if data_root is None and not configured_root_exists:
        local = ROOT / "data/mmaud_official_train"
        if not local.is_dir(): raise FileNotFoundError(f"configured data root and WSL fallback missing: {config['data']['root']}")
        config["data"]["root"] = str(local)
        overrides["data.root"] = "WSL fallback; configured server path absent"
    for variable in ("RDQ_DATA_ROOT", "RDQ_SPLIT_FILE", "RDQ_CAMERA_CONFIG", "RDQ_GEOMETRY_CALIBRATION"):
        if os.environ.get(variable): overrides[variable] = os.environ[variable]
    return config, config_path, checkpoint, overrides


def choose(dataset, sequence: str | None, query_time: float | None, index: int | None,
           tolerance: float):
    candidates = [(i, record) for i, record in enumerate(dataset.query_records)
                  if sequence is None or record.sequence_id == sequence]
    if query_time is not None:
        candidates = [(i, record) for i, record in candidates
                      if abs(record.query_time - query_time) <= tolerance]
        if len(candidates) != 1:
            raise ValueError(f"time {query_time} ± {tolerance} matched {len(candidates)} queries")
        return candidates[0]
    if index is None: raise ValueError("provide --query-time or --index")
    if index < 0 or index >= len(candidates): raise ValueError("dataset index out of range")
    return candidates[index]


def tensor_cpu(value): return value.detach().cpu()


def output_signature(output):
    fields = ("score_3d_before", "score_3d_after", "delta_3d", "score_2d_before",
              "score_2d_after", "xyz_m", "box_xyxy_px", "batch_index",
              "hypothesis_type", "radar_source_index", "vision_source_index",
              "association_valid", "association_d_box_px", "evidence_valid_3d",
              "evidence_valid_2d")
    return {name: tensor_cpu(getattr(output, name)) for name in fields}


def compare_outputs(reference, traced, atol=1e-5, rtol=1e-5):
    left, right = output_signature(reference), output_signature(traced)
    report = {}
    for name in left:
        a, b = left[name], right[name]
        exact = not a.dtype.is_floating_point
        equal = a.shape == b.shape and (torch.equal(a, b) if exact else torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
        report[name] = {"equal": bool(equal), "exact": exact,
                        "max_abs_error": float((a.float()-b.float()).abs().max()) if a.shape == b.shape and a.numel() else 0.0}
    for name in ("radar_candidates", "vision_candidates"):
        for field in ("source_index", "batch_index", "score", "xyz_m", "box_xyxy_px"):
            a, b = getattr(getattr(reference, name), field), getattr(getattr(traced, name), field)
            key = f"{name}.{field}"; exact = not a.dtype.is_floating_point
            report[key] = {"equal": bool(a.shape == b.shape and (torch.equal(a,b) if exact else torch.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))), "exact": exact}
    raw_a = reference.diagnostics.get("lidar_raw")
    raw_b = traced.diagnostics.get("lidar_raw")
    if raw_a is not None and raw_b is not None:
        for field in ("logits", "pred_xyz", "residual_xyz", "source_token_id"):
            a, b = tensor_cpu(raw_a[field]), tensor_cpu(raw_b[field])
            exact = not a.dtype.is_floating_point
            report[f"raw.{field}"] = {"equal": bool(a.shape == b.shape and
                (torch.equal(a,b) if exact else torch.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))),
                "exact": exact,
                "max_abs_error": float((a.float()-b.float()).abs().max()) if a.shape == b.shape and a.numel() else 0.0}
        def raw_order(raw):
            ids = tensor_cpu(raw["source_token_id"])
            logits = tensor_cpu(raw["logits"]).float()
            base = torch.argsort(ids, stable=True)
            return base[torch.argsort(logits[base], descending=True, stable=True)]
        report["raw.order"] = {"equal": bool(torch.equal(raw_order(raw_a), raw_order(raw_b))), "exact": True}
    for before in (True, False):
        a = reference.top3d_indices(1, before_interaction=before)[0]
        b = traced.top3d_indices(1, before_interaction=before)[0]
        report[f"top3d_before_{before}"] = {"equal": bool(torch.equal(a,b)), "exact": True}
    report["association_rows"] = {"equal": reference.diagnostics["association_rv_pairs"] == traced.diagnostics["association_rv_pairs"], "exact": True}
    return {"passed": all(item["equal"] for item in report.values()), "atol": atol, "rtol": rtol, "fields": report}


def save_sample(recorder, sample, images=None, masks=None):
    for name in ("points", "delta_t", "sensor_id", "target_xyz", "gt_box_xyxy_px"):
        recorder.save(f"sample.{name}", sample[name])
    if sample["m_V"]:
        image = sample["image_uint8"].permute(1, 2, 0).numpy()
        Image.fromarray(image).save(recorder.directory / "left_image.png")
    if images is not None:
        recorder.save("sample.preprocessed_left", images)
        recorder.save("sample.image_padding_mask", masks)
    return {"sequence_id": sample["sequence_id"], "sample_id": sample["sample_id"],
            "query_time": sample["query_time"], "m_R": sample["m_R"], "m_V": sample["m_V"],
            "gt3d_valid": sample["target_valid"], "gt2d_valid": sample["gt_2d_valid"],
            "event_timestamps": sample["event_timestamps"],
            "image_time": sample["image_time"], "image_path": sample["left_image_path"],
            "image_source_wh": sample["image_source_wh"].tolist() if sample["m_V"] else None,
            "image_view_wh": sample["image_view_wh"].tolist() if sample["m_V"] else None,
            "image_scale_xy": sample["image_scale_xy"].tolist() if sample["m_V"] else None,
            "processed_image_shape": list(images.shape) if images is not None else None,
            "image_padding_pixels": int(masks.sum()) if masks is not None else None,
            "gt_xyz_m": sample["target_xyz"].tolist() if sample["target_valid"] else None,
            "gt_box_xyxy_px": sample["gt_box_xyxy_px"].tolist() if sample["gt_2d_valid"] else None}


def save_hierarchy(recorder, raw):
    if raw is None: return None
    hierarchy = raw["layouts"]
    for name in ("point_to_l0", "parent_l0_to_l1", "parent_l1_to_l2"):
        recorder.save(f"mapping.{name}", getattr(hierarchy, name))
    for i, level in enumerate(hierarchy.levels):
        for name in ("coords", "centers", "batch_index", "point_count"):
            recorder.save(f"level{i}.{name}", getattr(level, name))
    for name in ("logits", "residual_xyz", "pred_xyz", "fine_features",
                 "voxel_centers", "source_token_id", "batch_index"):
        recorder.save(f"raw.{name}", raw[name])
    return raw


def save_candidates(recorder, output, gt):
    camera_meta = recorder.tensors.get("projection.camera_xyz", {})
    camera_xyz = (np.load(recorder.directory / camera_meta["file"], allow_pickle=False)
                  if camera_meta.get("status") == "saved" else None)
    for source, cand in (("radar", output.radar_candidates), ("vision", output.vision_candidates)):
        for name in ("feature", "score", "xyz_m", "box_xyxy_px", "batch_index",
                     "source_index", "projected_xy_px", "projection_valid"):
            recorder.save(f"candidate.{source}.{name}", getattr(cand, name))
    for name, value in output_signature(output).items(): recorder.save(f"output.{name}", value)
    for name in ("evidence_valid_3d", "evidence_valid_2d", "gate_3d", "gate_2d",
                 "token_count_3d", "token_count_2d", "association_d_center_normalized"):
        recorder.save(f"output.{name}", getattr(output, name))
    for source in ("radar", "vision"):
        evidence = output.diagnostics[f"{source}_evidence"]
        for name in ("feature", "valid", "token_count", "gate_weight", "projected_xy_px"):
            recorder.save(f"evidence.{source}.{name}", getattr(evidence, name))
    rows = []
    before_ids = output.top3d_indices(1, before_interaction=True)[0].tolist()
    after_ids = output.top3d_indices(1)[0].tolist()
    for i in range(len(output.score_3d_after)):
        row = {"row": i, "type": int(output.hypothesis_type[i]),
               "radar_source_id": int(output.radar_source_index[i]),
               "vision_source_id": int(output.vision_source_index[i]),
               "xyz": output.xyz_m[i].float().tolist() if bool(output.has_xyz[i]) else None,
               "box": output.box_xyxy_px[i].float().tolist() if bool(output.has_box[i]) else None,
               "score3d_before": float(output.score_3d_before[i]), "score3d_after": float(output.score_3d_after[i]),
               "score2d_before": float(output.score_2d_before[i]), "score2d_after": float(output.score_2d_after[i]),
               "delta3d": float(output.delta_3d[i]), "delta2d": float(output.delta_2d[i]),
               "association": bool(output.association_valid[i]),
               "evidence_valid_3d": bool(output.evidence_valid_3d[i]),
               "evidence_valid_2d": bool(output.evidence_valid_2d[i]),
               "gate3d": float(output.gate_3d[i]), "gate2d": float(output.gate_2d[i]),
               "token_count_3d": int(output.token_count_3d[i]),
               "token_count_2d": int(output.token_count_2d[i]),
               "projected_uv": None}
        if row["radar_source_id"] >= 0:
            match = torch.nonzero(output.radar_candidates.source_index == row["radar_source_id"]).flatten()
            if len(match):
                j = int(match[0]); row["projected_uv"] = output.radar_candidates.projected_xy_px[j].float().tolist()
                row["projection_valid"] = bool(output.radar_candidates.projection_valid[j])
                if camera_xyz is not None and j < len(camera_xyz):
                    row["camera_xyz"] = camera_xyz[j].tolist()
        if gt is not None and row["xyz"] is not None:
            row["gt_distance_m"] = float(torch.linalg.vector_norm(output.xyz_m[i].float().cpu()-gt))
        rows.append(row)
    before_rank = sorted((i for i,r in enumerate(rows) if r["xyz"] is not None),
                         key=lambda i:(-rows[i]["score3d_before"],i))
    after_rank = sorted((i for i,r in enumerate(rows) if r["xyz"] is not None),
                        key=lambda i:(-rows[i]["score3d_after"],i))
    for rank,i in enumerate(before_rank,1): rows[i]["rank_before"] = rank
    for rank,i in enumerate(after_rank,1): rows[i]["rank_after"] = rank
    (recorder.directory / "candidates.json").write_text(json.dumps(rows, indent=2))
    return rows, before_ids, after_ids


def save_selector_table(recorder, raw, gt):
    if raw is None:
        (recorder.directory / "selector.json").write_text("[]")
        return []
    idx = recorder.tensors
    def load(name):
        meta = idx.get(name)
        return np.load(recorder.directory / meta["file"], allow_pickle=False) if meta and meta["status"] == "saved" else None
    order = load("selector.batch0.order"); raw_top = load("selector.batch0.raw"); pool = load("selector.batch0.pool")
    final = load("selector.batch0.final"); suppressed = load("selector.batch0.suppressed")
    if order is None: return []
    pool_set=set(map(int,pool)); raw_set=set(map(int,raw_top)); final_set=set(map(int,final)); pool_pos={int(v):i for i,v in enumerate(pool)}
    selector_event=recorder.events["selector.batch0"]
    kept_local=selector_event["kept_local"]; evaluated_count=selector_event["evaluated_count"]
    logits=raw["logits"].detach().float().cpu().numpy(); xyz=raw["pred_xyz"].detach().float().cpu().numpy()
    rows=[]
    for rank, token in enumerate(order,1):
        token=int(token)
        reason=("selected" if token in final_set else
                "below_raw_topk" if token not in raw_set else
                "below_pre_nms_topk" if token not in pool_set else
                "not_evaluated_final_limit" if pool_pos[token]>=evaluated_count else "nms_radius")
        suppressor=None; distance=None
        if reason=="nms_radius" and suppressed is not None:
            position=pool_pos[token]
            for prior in kept_local:
                if suppressed[position,prior]:
                    suppressor=int(pool[prior]);distance=float(np.linalg.norm(xyz[token]-xyz[suppressor]));break
        row={"source_token_id": token, "rank": rank, "logit": float(logits[token]),
             "score": float(torch.sigmoid(torch.tensor(float(logits[token])))),
             "xyz": xyz[token].tolist(), "reason": reason,
             "suppressed_by": suppressor, "suppression_distance_m": distance}
        if gt is not None: row["gt_distance_m"] = float(np.linalg.norm(xyz[token]-gt.numpy()))
        rows.append(row)
    (recorder.directory / "selector.json").write_text(json.dumps(rows))
    return rows


def trace_one(model_name, config, config_path, checkpoint, overrides, dataset, dataset_index,
              output_root, device, focus_voxel, max_tensor_mb, display_points, split):
    sample=dataset[dataset_index]
    query_dir=output_root / sample["sample_id"] / model_name
    query_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("manifest.json", "browser.json", "summary.json", "parity.json"):
        (query_dir / stale).unlink(missing_ok=True)
    (query_dir / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    runtime=build_runtime(config, ROOT, torch.device("cpu"), initialize_from_baselines=False)
    checkpoint_info=load_network(runtime.model, checkpoint)
    checkpoint_info.update(missing_keys=[], unexpected_keys=[], strict=True)
    runtime.model.to(device).eval(); synchronize_dino_device(runtime, device)
    runtime.projection_base=load_left_projection_context(runtime.camera_config,runtime.geometry_calibration,
        image_scale_xy=torch.ones((1,2),device=device),device=device)
    batch=_stage_batch(collate_multimodal_v2([sample]), config["experiment"]["stage"])
    radar, images, masks, projection, targets, _ = prepare_model_batch(batch,runtime.dino_detector,runtime.projection_base,device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        reference=runtime.model(radar,images,masks,projection)
        with TraceRecorder(runtime.model,query_dir,focus_voxel=focus_voxel,max_tensor_mb=max_tensor_mb) as recorder:
            sample_info=save_sample(recorder,sample,images,masks)
            traced=runtime.model(radar,images,masks,projection)
            raw=save_hierarchy(recorder,traced.diagnostics["lidar_raw"])
            gt=sample["target_xyz"] if sample["target_valid"] else None
            candidates,before_ids,after_ids=save_candidates(recorder,traced,gt)
            selector=save_selector_table(recorder,raw,gt)
    parity=compare_outputs(reference,traced)
    (query_dir / "parity.json").write_text(json.dumps(parity,indent=2))
    if not parity["passed"]: raise RuntimeError(f"trace changed model output: {query_dir / 'parity.json'}")
    summary={"sample":sample_info,"model":model_name,"checkpoint_info":checkpoint_info,
             "tensor_shapes":recorder.tensors,"omitted_tensors":recorder.omitted,
             "final_top1_xyz":candidates[after_ids[0]]["xyz"] if after_ids else None,
             "before_top1_xyz":candidates[before_ids[0]]["xyz"] if before_ids else None,
             "final_top1_gt_error_m":candidates[after_ids[0]].get("gt_distance_m") if after_ids else None,
             "radar_raw_predictions":len(selector),"final_hypotheses":len(candidates),
             "association":traced.diagnostics["association_per_query"][0],
             "attention": {"radar_spatial":"actual selected window weights saved",
                           "cross_modal":"unavailable from nn.MultiheadAttention need_weights=False; inputs and outputs saved"},
             "parity_passed":parity["passed"]}
    (query_dir / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True))
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    manifest={"schema_version":1,"sample_id":sample["sample_id"],"sequence_id":sample["sequence_id"],
              "query_time":sample["query_time"],"model":model_name,"split":split,"code_commit":commit,
              "stages":sorted({name.split(".")[0] for name in recorder.tensors}),
              "config_source":str(config_path),"config_sha256":sha256(config_path),
              "effective_config_sha256":sha256(query_dir/"effective_config.yaml"),"overrides":overrides,
              "checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
              "checkpoint_load":checkpoint_info,
              "environment":{"python":platform.python_version(),"torch":torch.__version__,
                             "torchvision":torchvision.__version__,"torch_cuda":torch.version.cuda,
                             "cuda_available":torch.cuda.is_available(),
                             "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                             "device":str(device)},
              "calibration":{str(p):sha256(p) for p in (runtime.camera_config,runtime.geometry_calibration)},
              "complete":not recorder.omitted,"omitted_tensors":recorder.omitted,
              "display_points_limit":display_points,"attention_limits":summary["attention"]}
    (query_dir / "manifest.json").write_text(json.dumps(manifest,indent=2))
    points=sample["points"].numpy() if sample["m_R"] else np.empty((0,3),np.float32)
    step=max(1,int(np.ceil(len(points)/display_points)))
    ids=np.arange(0,len(points),step)
    browser={"sample":sample_info,"model":model_name,"points_display_only":True,
             "point_ids":ids.tolist(),"points":np.round(points[ids].astype(np.float64),3).tolist(),
             "sensor_id":sample["sensor_id"][ids].tolist() if sample["m_R"] else [],
             "delta_t":sample["delta_t"][ids].tolist() if sample["m_R"] else [],
             "point_count_full":len(points),"candidates":candidates,"selector_top":selector[:100],
             "stages":list(recorder.tensors),"tensor_stats":recorder.tensors,
             "attention":summary["attention"],"omitted":recorder.omitted}
    def recorded_array(name):
        meta=recorder.tensors.get(name,{})
        return np.load(query_dir/meta["file"],allow_pickle=False) if meta.get("status")=="saved" else None
    cells=recorded_array("visual_read.level0.cell_xy")
    visible=recorded_array("visual_read.level0.visible")
    feature_hw=recorded_array("visual_read.level0.feature_hw")
    padded_hw=recorded_array("visual_read.level0.padded_hw")
    if cells is not None and visible is not None and feature_hw is not None and padded_hw is not None:
        browser["visual_sampling"]={"radar_source_ids":traced.radar_candidates.source_index.tolist(),
                                    "level0_cells_xy":cells.tolist(),"level0_visible":visible.tolist(),
                                    "feature_hw":feature_hw.tolist(),"padded_hw":padded_hw.tolist()}
    if raw is not None:
        centers=raw["layouts"].levels[0].centers.detach().cpu().numpy()
        voxel_step=max(1,int(np.ceil(len(centers)/min(display_points,20000))))
        voxel_ids=np.arange(0,len(centers),voxel_step)
        browser["voxel_ids"]=voxel_ids.tolist()
        browser["voxel_centers"]=np.round(centers[voxel_ids].astype(np.float64),3).tolist()
        browser["voxel_count_full"]=len(centers)
    else:
        browser.update(voxel_ids=[],voxel_centers=[],voxel_count_full=0)
    gt_error=summary["final_top1_gt_error_m"]
    oracle=min((r.get("gt_distance_m",float("inf")) for r in candidates if r["xyz"] is not None),default=float("inf"))
    browser["flags"]={"top1_failed_1m":gt_error is not None and gt_error>1,
                      "final_top10_has_1m":oracle<=1,
                      "large_error_5m":gt_error is not None and gt_error>5,
                      "rescued":bool(before_ids and after_ids and candidates[before_ids[0]].get("gt_distance_m",float("inf"))>1 and gt_error is not None and gt_error<=1),
                      "demoted":bool(before_ids and after_ids and candidates[before_ids[0]].get("gt_distance_m",float("inf"))<=1 and gt_error is not None and gt_error>1)}
    (query_dir / "browser.json").write_text(json.dumps(browser,allow_nan=True))
    print(f"{model_name}: {sample['sample_id']} -> {query_dir}; parity=PASS; raw={len(selector)}",flush=True)
    del runtime,reference,traced,recorder
    gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    return query_dir


class ReplayHandler(SimpleHTTPRequestHandler):
    def __init__(self,*args,directory=None,**kwargs):
        super().__init__(*args,directory=directory,**kwargs)

    def do_GET(self):
        url=urlparse(self.path)
        if url.path == "/":
            content=PAGE.read_bytes();self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(content)));self.end_headers();self.wfile.write(content);return
        if url.path == "/api/index":
            rows=[]
            for path in sorted(Path(self.directory).glob("*/[AB][0-9]/manifest.json")):
                try: payload=json.loads(path.read_text())
                except (OSError,ValueError): continue
                rows.append({"query":path.parent.parent.name,"model":payload["model"],
                             "sequence":payload["sequence_id"],"query_time":payload["query_time"],
                             "split":payload["split"]})
            return self._json({"traces":rows})
        if url.path == "/api/project":
            query=parse_qs(url.query)
            key=query.get("query",[""])[0];model=query.get("model",[""])[0]
            if not key or key in {".", ".."} or Path(key).name!=key or model not in ASSETS:
                return self._json({"error":"invalid query/model"},400)
            base=Path(self.directory)/key/model
            if not (base/"manifest.json").is_file(): return self._json({"error":"trace not found"},404)
            try:
                point_id=None
                if "point" in query and "xyz" in query:
                    raise ValueError("provide point ID or XYZ, not both")
                if "point" in query:
                    point_id=int(query["point"][0]);path=base/"tensors"/"sample.points.npy"
                    if not path.is_file(): return self._json({"error":"saved point cloud unavailable"},404)
                    points=np.load(path,allow_pickle=False,mmap_mode="r")
                    if point_id<0 or point_id>=len(points): raise ValueError("point ID outside saved point cloud")
                    xyz=np.asarray(points[point_id],dtype=np.float32)
                elif "xyz" in query:
                    fields=query["xyz"][0].split(",")
                    if len(fields)!=3: raise ValueError("XYZ must contain exactly three comma-separated numbers")
                    xyz=np.asarray([float(value.strip()) for value in fields],dtype=np.float32)
                else: raise ValueError("provide point ID or XYZ")
                if xyz.shape!=(3,) or not np.isfinite(xyz).all(): raise ValueError("XYZ must be finite")
                if not (base/"left_image.png").is_file():
                    return self._json({"point_id":point_id,"xyz_m":xyz.tolist(),"valid":False,
                                       "reason":"selected query has no left image","uv_source_px":None})
                config=yaml.safe_load((base/"effective_config.yaml").read_text())
                projection=load_left_projection_context(
                    resolve_project_path(config["data"]["camera_config"],ROOT),
                    resolve_project_path(config["data"]["geometry_calibration"],ROOT),
                    image_scale_xy=torch.ones((1,2)),device="cpu")
                observed={}
                uv,valid=project_omni_radtan(torch.from_numpy(xyz.copy()).reshape(1,3),
                    torch.zeros(1,dtype=torch.long),projection,trace_observer=observed.update)
                inside=bool(valid[0]);pixel=uv[0].tolist()
                return self._json({"point_id":point_id,"xyz_m":xyz.tolist(),"valid":inside,
                                   "uv_source_px":pixel if np.isfinite(pixel).all() else None,
                                   "camera_xyz_m":observed["camera_xyz"][0].tolist(),
                                   "reason":"inside left image" if inside else "outside calibrated left image or invalid projection"})
            except (ValueError,OSError,KeyError,TypeError) as exc:
                return self._json({"error":str(exc)},400)
        if url.path in {"/api/level", "/api/row"}:
            query=parse_qs(url.query)
            key=query.get("query",[""])[0];model=query.get("model",[""])[0]
            if not key or key in {".", ".."} or Path(key).name!=key or model not in ASSETS:
                return self._json({"error":"invalid query/model"},400)
            base=Path(self.directory)/key/model
            if not (base/"manifest.json").is_file(): return self._json({"error":"trace not found"},404)
            if url.path == "/api/level":
                try: level=int(query.get("level",["0"])[0])
                except ValueError: return self._json({"error":"level must be 0, 1, or 2"},400)
                if level not in (0,1,2): return self._json({"error":"level must be 0, 1, or 2"},400)
                paths=[base/"tensors"/f"level{level}.{name}.npy" for name in ("centers","coords","point_count")]
                if not all(path.is_file() for path in paths): return self._json({"error":"level tensors unavailable"},404)
                centers,coords,counts=[np.load(path,allow_pickle=False,mmap_mode="r") for path in paths]
                total=len(centers);step=max(1,int(np.ceil(total/20000)));ids=np.arange(0,total,step)
                return self._json({"level":level,"voxel_size_m":(.5,1.,2.)[level],
                                   "full_count":total,"display_only":len(ids)<total,"ids":ids.tolist(),
                                   "centers":np.round(centers[ids].astype(np.float64),3).tolist(),
                                   "point_counts":counts[ids].tolist(),
                                   "statistics":{"input_points":int(counts.sum()),
                                                 "min_points":int(counts.min()) if total else 0,
                                                 "max_points":int(counts.max()) if total else 0,
                                                 "mean_points":float(counts.mean()) if total else 0.,
                                                 "median_points":float(np.median(counts)) if total else 0.,
                                                 "p90_points":float(np.percentile(counts,90)) if total else 0.}})
            name=query.get("name",[""])[0]
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name):
                return self._json({"error":"invalid tensor name"},400)
            path=base/"tensors"/(name+".npy")
            if not path.is_file(): return self._json({"error":"tensor unavailable or omitted"},404)
            data=np.load(path,allow_pickle=False,mmap_mode="r")
            try:
                raw_indices=query.get("indices",["0" if data.ndim else ""])[0].strip()
                indices=tuple(int(value.strip()) for value in raw_indices.split(",")) if raw_indices else ()
                if len(indices)>data.ndim or any(index<0 or index>=data.shape[axis] for axis,index in enumerate(indices)):
                    raise ValueError("index outside tensor shape")
                selected=np.asarray(data[indices])
                if selected.size>4096: raise ValueError("slice exceeds 4096 values; specify more indices")
            except ValueError as exc: return self._json({"error":str(exc)},400)
            def safe(value):
                if isinstance(value,list): return [safe(item) for item in value]
                if isinstance(value,float) and not np.isfinite(value): return str(value)
                return value
            return self._json({"name":name,"full_shape":list(data.shape),"indices":list(indices),
                               "shape":list(selected.shape),"dtype":str(data.dtype),
                               "values":safe(selected.tolist())})
        if url.path == "/api/details":
            query=parse_qs(url.query)
            key=query.get("query",[""])[0];model=query.get("model",[""])[0]
            if not key or Path(key).name!=key or model not in ASSETS:
                return self._json({"error":"invalid query/model"},400)
            base=Path(self.directory)/key/model
            if not (base/"manifest.json").is_file(): return self._json({"error":"trace not found"},404)
            def array(name):
                path=base/"tensors"/(name+".npy")
                return np.load(path,allow_pickle=False,mmap_mode="r") if path.is_file() else None
            try:
                if "point" in query:
                    point=int(query["point"][0]);mapping=array("mapping.point_to_l0")
                    if mapping is None or point<0 or point>=len(mapping): raise ValueError("point outside saved mapping")
                    voxel=int(mapping[point]);slot_map=array("frontend.point_to_slot")
                    result={"point_id":point,"l0_voxel":voxel,
                            "slot":int(slot_map[point]) if slot_map is not None else None}
                    for name,key in (("sample.points","xyz_m"),("sample.sensor_id","sensor_id"),
                                     ("sample.delta_t","delta_t_s"),("frontend.point_mlp_input","point_mlp_input"),
                                     ("frontend.point_mlp_output","point_mlp_output")):
                        values=array(name)
                        if values is not None: result[key]=np.asarray(values[point]).tolist()
                    center=array("level0.centers")
                    if center is not None: result["l0_center_m"]=center[voxel].tolist()
                else:
                    level=int(query.get("level",["0"])[0])
                    if level not in (0,1,2): raise ValueError("level must be 0, 1, or 2")
                    voxel=int(query.get("voxel",["-1"])[0]);centers=array(f"level{level}.centers")
                    if centers is None or voxel<0 or voxel>=len(centers): raise ValueError("voxel outside saved level")
                    p2v=array("mapping.point_to_l0");v01=array("mapping.parent_l0_to_l1");v12=array("mapping.parent_l1_to_l2")
                    l0_ids=(np.array([voxel]) if level==0 else
                            np.flatnonzero(v01==voxel) if level==1 and v01 is not None else
                            np.flatnonzero(v12[v01]==voxel) if v01 is not None and v12 is not None else np.empty(0,dtype=np.int64))
                    counts=array(f"level{level}.point_count");coords=array(f"level{level}.coords")
                    result={"level":level,"voxel_id":voxel,"center_m":centers[voxel].tolist(),
                            "coords":coords[voxel].tolist() if coords is not None else None,
                            "point_count":int(counts[voxel]) if counts is not None else None,
                            "point_ids":np.flatnonzero(np.isin(p2v,l0_ids)).tolist() if p2v is not None else None,
                            "l0_descendants":l0_ids.tolist(),
                            "parent_id":(int(v01[voxel]) if level==0 and v01 is not None else
                                         int(v12[voxel]) if level==1 and v12 is not None else None),
                            "child_ids":(np.flatnonzero(v01==voxel).tolist() if level==1 and v01 is not None else
                                         np.flatnonzero(v12==voxel).tolist() if level==2 and v12 is not None else None)}
                    group_ids=array(f"radar.encoder{level}.block0.groups.token_ids")
                    group_offsets=array(f"radar.encoder{level}.block0.groups.offsets")
                    if group_ids is not None and group_offsets is not None:
                        position=np.flatnonzero(group_ids==voxel)
                        if len(position):
                            group=int(np.searchsorted(group_offsets,position[0],side="right")-1)
                            result["window0_group"] = group
                            result["window0_neighbors"] = group_ids[group_offsets[group]:group_offsets[group+1]].tolist()
                    features=(("frontend.l0_tokens","radar.encoder0.block0.output","radar.final_norm.output") if level==0 else
                              ("radar.merge01.output","radar.encoder1.block0.output","radar.up21.output") if level==1 else
                              ("radar.merge12.output","radar.encoder2.block0.output"))
                    for name in features:
                        values=array(name)
                        if values is not None and voxel<len(values): result[name]=values[voxel].tolist()
                return self._json(result)
            except (ValueError,OSError) as exc: return self._json({"error":str(exc)},400)
        if url.path == "/api/tensor":
            query=parse_qs(url.query);key=query.get("query",[""])[0];model=query.get("model",[""])[0]
            name=query.get("name",[""])[0]
            if not key or Path(key).name!=key or model not in ASSETS or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name):
                return self._json({"error":"invalid tensor request"},400)
            path=Path(self.directory)/key/model/"tensors"/(name+".npy")
            if not path.is_file(): return self._json({"error":"tensor unavailable or omitted"},404)
            data=np.load(path,allow_pickle=False,mmap_mode="r")
            if data.size>50000: return self._json({"error":"tensor too large for JSON; use Python .npy"},413)
            return self._json({"shape":list(data.shape),"values":data.tolist()})
        return super().do_GET()

    def _json(self,value,status=200):
        content=json.dumps(value,allow_nan=True).encode();self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(content)));self.end_headers();self.wfile.write(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("list","trace","batch"):
        p=sub.add_parser(command)
        p.add_argument("--split",choices=("validation_sub","heldout_test_sub","train_sub"),default="validation_sub")
        p.add_argument("--final-heldout",action="store_true")
        p.add_argument("--data-root",type=Path)
        p.add_argument("--sequence")
        p.add_argument("--limit",type=int,default=20 if command=="list" else None)
        if command!="list":
            p.add_argument("--models",nargs="+",choices=tuple(ASSETS),default=["A2"])
            p.add_argument("--output",type=Path,default=Path("outputs/v2_trace_replay"))
            p.add_argument("--device",default="cuda:0")
            p.add_argument("--focus-voxel",type=int,default=0)
            p.add_argument("--max-tensor-mb",type=int,default=256)
            p.add_argument("--display-points",type=int,default=80000)
            p.add_argument("--tolerance",type=float,default=1e-5)
            p.add_argument("--query-time",type=float)
            p.add_argument("--index",type=int,help="zero-based index within --sequence, or whole split")
            if command=="batch": p.add_argument("--indices",type=int,nargs="+",required=True)
    serve=sub.add_parser("viewer")
    serve.add_argument("--output",type=Path,default=Path("outputs/v2_trace_replay"))
    serve.add_argument("--host",default="127.0.0.1");serve.add_argument("--port",type=int,default=8767)
    args=parser.parse_args()
    if args.command=="viewer":
        directory=resolve_project_path(args.output,ROOT)
        with ThreadingHTTPServer((args.host,args.port),lambda *a,**k:ReplayHandler(*a,directory=str(directory),**k)) as server:
            print(f"http://{args.host}:{args.port}",flush=True);server.serve_forever()
        return
    if args.split=="heldout_test_sub" and not args.final_heldout:
        parser.error("heldout test requires explicit --final-heldout; do not use it for tuning")
    config,_,_,_=config_for("A2",args.data_root,require_checkpoint=args.command!="list")
    dataset=build_split_dataset(config,ROOT,args.split)
    if args.command=="list":
        records=[(i,r) for i,r in enumerate(dataset.query_records) if args.sequence is None or r.sequence_id==args.sequence]
        for local,(i,r) in enumerate(records[:args.limit]):
            print(f"split_index={i} sequence_index={local} sample_id={r.sample_id} query_time={r.query_time:.7f}")
        print(f"matched_queries={len(records)} split={args.split}")
        return
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in WSL; choose --device cpu only if required operators work")
    if args.max_tensor_mb<1 or args.display_points<1 or args.tolerance<0: parser.error("limits/tolerance must be positive")
    if args.command=="batch":
        selected=[choose(dataset,args.sequence,None,i,args.tolerance) for i in args.indices]
    else:
        selected=[choose(dataset,args.sequence,args.query_time,args.index,args.tolerance)]
    output=resolve_project_path(args.output,ROOT)
    for _,record in selected:
        print(f"requested_time={args.query_time if args.command=='trace' else 'index'} actual_query={record.query_time:.7f} sequence={record.sequence_id} tolerance={args.tolerance}",flush=True)
        for model_name in args.models:
            cfg,cfg_path,checkpoint,overrides=config_for(model_name,args.data_root)
            model_dataset=build_split_dataset(cfg,ROOT,args.split)
            matches=[i for i,r in enumerate(model_dataset.query_records)
                     if r.sample_id==record.sample_id and r.sequence_id==record.sequence_id and r.query_time==record.query_time]
            if len(matches)!=1: raise RuntimeError(f"{model_name} query alignment failed: {record.sample_id}")
            trace_one(model_name,cfg,cfg_path,checkpoint,overrides,model_dataset,matches[0],output,device,
                      args.focus_voxel,args.max_tensor_mb,args.display_points,args.split)


if __name__=="__main__": main()
