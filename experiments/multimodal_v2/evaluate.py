#!/usr/bin/env python3
"""FP32 B0/B1/B2 evaluation with explicit no-output failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from rdq_uav.multimodal_v2.data import build_datasets, collate_multimodal_v2, prepare_model_batch
from rdq_uav.multimodal_v2.training import build_runtime, summarize_3d, synchronize_dino_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=("B0", "B1", "B2"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def resolve(value): return resolve_project_path(value, ROOT)


def load_network(model, path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if any(key.startswith("network.") for key in state):
        state = {key[len("network."):]: value for key, value in state.items() if key.startswith("network.")}
    model.load_state_dict(state, strict=True)
    return {"epoch": payload.get("epoch"), "global_step": payload.get("global_step")}


def metrics(values):
    values = [row for row in values if row["has_gt3d"]]
    rows = [{"top1_error": row["final_error"], "distances": row["final_distances"]} for row in values]
    final = summarize_3d(rows)
    before = summarize_3d([{"top1_error": row["before_error"], "distances": row["before_distances"]} for row in values])
    radar = summarize_3d([{"top1_error": row["radar_error"], "distances": row["radar_distances"]} for row in values])
    return {"final": final, "interaction_before": before, "lidar_candidates": radar}


@torch.no_grad()
def main():
    args = parse_args(); device = torch.device(args.device)
    if args.mode == "B2" and args.checkpoint is None:
        raise ValueError("B2 evaluation requires --checkpoint; an untrained zero head is B1")
    if args.mode in {"B0", "B1"} and args.checkpoint is not None:
        raise ValueError("B0/B1 identity gates do not accept a trained checkpoint")
    cfg = apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    cfg["model"]["interaction_enabled"] = args.mode != "B0"
    cfg["model"]["vision_reads_radar"] = args.mode == "B3"
    cfg["model"]["vision_scoring_enabled"] = args.mode == "B3"
    _, _, dataset = build_datasets(cfg, ROOT)
    if args.limit is not None: dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_multimodal_v2)
    runtime = build_runtime(cfg, ROOT, torch.device("cpu"))
    checkpoint_info = None
    if args.checkpoint:
        checkpoint_info = load_network(runtime.model, resolve(args.checkpoint))
    runtime.model.to(device).eval(); synchronize_dino_device(runtime, device)
    from rdq_uav.multimodal_v1 import load_left_projection_context
    runtime.projection_base = load_left_projection_context(
        runtime.camera_config, runtime.geometry_calibration,
        image_scale_xy=torch.ones((1,2),device=device), device=device,
    )
    rows, type_counts, top_type_counts = [], Counter(), Counter()
    loss_counts = Counter()
    for batch in loader:
        lidar, images, masks, context, targets, _ = prepare_model_batch(
            batch, runtime.dino_detector, runtime.projection_base, device
        )
        with torch.autocast(device_type=device.type, enabled=False):
            output = runtime.model(lidar, images, masks, context)
        loss_values = runtime.ranking_loss(output, targets)
        for key, value in loss_values.items():
            if key.startswith("n_"):
                loss_counts[key] += int(value)
        has_gt3d = bool(targets.has_xyz[0])
        gt = targets.xyz_m[0].float() if has_gt3d else None
        radar = output.radar_candidates
        rids = torch.nonzero(radar.batch_index == 0).flatten()
        rids = rids[
            torch.argsort(radar.source_index[rids], descending=False, stable=True)
        ]
        rids = rids[torch.argsort(radar.score[rids].float(), descending=True, stable=True)]
        rdist = (torch.linalg.vector_norm(radar.xyz_m[rids].float() - gt, dim=1).cpu().tolist()
                 if has_gt3d else [])
        before_ids = output.top3d_indices(1, before_interaction=True)[0]
        after_ids = output.top3d_indices(1)[0]
        before_dist = (torch.linalg.vector_norm(output.xyz_m[before_ids].float()-gt,dim=1).cpu().tolist()
                       if has_gt3d else [])
        after_dist = (torch.linalg.vector_norm(output.xyz_m[after_ids].float()-gt,dim=1).cpu().tolist()
                      if has_gt3d else [])
        all_ids = torch.nonzero(output.batch_index == 0).flatten()
        type_counts.update(map(int, output.hypothesis_type[all_ids].cpu().tolist()))
        if len(after_ids): top_type_counts[int(output.hypothesis_type[after_ids[0]])] += 1
        association = output.diagnostics["association_per_query"][0]
        rv_pairs = list(output.diagnostics["association_rv_pairs"])
        rows.append({
            "sequence_id": batch["sequence_id"][0], "sample_id": batch["sample_id"][0],
            "has_gt3d": has_gt3d,
            "radar_distances": rdist, "radar_error": rdist[0] if rdist else None,
            "radar_oracle_error": min(rdist) if rdist else None,
            "before_distances": before_dist, "before_error": before_dist[0] if before_dist else None,
            "final_distances": after_dist, "final_error": after_dist[0] if after_dist else None,
            "radar_candidate_count": len(rids), "vision_candidate_count": output.vision_candidates.n,
            "final_3d_count": len(after_ids), "v_only_excluded_from_3d": int(((output.hypothesis_type==2)&(output.batch_index==0)).sum()),
            "top_source_type": None if not len(after_ids) else int(output.hypothesis_type[after_ids[0]]),
            "before_top1_radar_source_index": None if not len(before_ids) else int(output.radar_source_index[before_ids[0]]),
            "after_top1_radar_source_index": None if not len(after_ids) else int(output.radar_source_index[after_ids[0]]),
            "score_3d_changed_count": int(torch.count_nonzero(output.delta_3d)),
            "score_2d_changed_count": int(torch.count_nonzero(output.delta_2d)),
            "association": association,
            "rv_pairs": rv_pairs,
        })
    metric_rows = [row for row in rows if row["has_gt3d"]]
    demoted = sum(
        row["before_error"] is not None and row["before_error"] <= 1.0
        and (row["final_error"] is None or row["final_error"] > 1.0)
        for row in metric_rows
    )
    oracle_lost = sum(
        row["radar_oracle_error"] is not None and row["radar_oracle_error"] <= 1.0
        and (row["final_error"] is None or row["final_error"] > 1.0)
        for row in metric_rows
    )
    before_correct = sum(row["before_error"] is not None and row["before_error"] <= 1.0
                         for row in metric_rows)
    lidar_oracle_success = sum(
        row["radar_oracle_error"] is not None and row["radar_oracle_error"] <= 1.0
        for row in metric_rows
    )
    report = {
        "mode": args.mode, "checkpoint": None if not args.checkpoint else str(resolve(args.checkpoint)),
        "checkpoint_info": checkpoint_info, "evaluation_precision": "fp32", "queries": len(rows),
        "evaluated_gt3d_queries": len(metric_rows),
        "metrics": metrics(rows), "hypothesis_counts": {"RV": type_counts[0], "R": type_counts[1], "V": type_counts[2]},
        "top_3d_hypothesis_counts": {"RV": top_type_counts[0], "R": top_type_counts[1], "V": top_type_counts[2]},
        "v_only_participates_in_3d_ranking": False,
        "visual_wrongly_overrode_correct_lidar": demoted,
        "radar_candidate_within_1m_but_final_failed": oracle_lost,
        "correct_lidar_top1_count": before_correct,
        "correct_lidar_top1_demoted_fraction": float(demoted / before_correct) if before_correct else 0.0,
        "lidar_oracle_recall_1m": float(lidar_oracle_success / len(metric_rows)) if metric_rows else 0.0,
        "loss_query_counts": dict(loss_counts),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    with (args.output / "per_query.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__": main()
