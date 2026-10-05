#!/usr/bin/env python3
"""Audit B0 validation failures on refactor-v2-standalone.

Audits:
1) queries with radar candidates but final Top-10 oracle error > 1 m;
2) queries with no radar candidate output.

For each Top-10 failure:
- GT-near input point support in the full causal radar history;
- every raw L0 candidate before Top-K/NMS;
- score rank / pre-NMS / NMS / final-TopK selection path;
- voxel-center error before regression and predicted XYZ error after regression;
- CandidateLoss positive/ignore/negative labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import yaml

from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
from rdq_uav.multimodal_v2.data import (
    build_split_dataset,
    collate_multimodal_v2,
    prepare_model_batch,
)
from rdq_uav.multimodal_v2.radar_data import load_released_xyz
from rdq_uav.multimodal_v2.training import (
    _stage_batch,
    build_runtime,
    synchronize_dino_device,
)
from rdq_uav.multimodal_v2.geometry import load_left_projection_context


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/multimodal_v2/configs/b0_standalone.yaml"),
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="validation_sub",
                   choices=("validation_sub", "heldout_test_sub"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--near-radius-m", type=float, default=2.0)
    p.add_argument("--dump-all-raw", action="store_true")
    p.add_argument("--expect-failed", type=int, default=214)
    p.add_argument("--expect-no-output", type=int, default=4)
    p.add_argument("--strict-expected-counts", action="store_true")
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    return resolve_project_path(path, ROOT)


def load_network(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if any(key.startswith("network.") for key in state):
        state = {
            key[len("network."):]: value
            for key, value in state.items()
            if key.startswith("network.")
        }
    model.load_state_dict(state, strict=True)
    return {"epoch": payload.get("epoch"), "global_step": payload.get("global_step")}


def stable_final_order(radar):
    ids = torch.arange(radar.n, device=radar.score.device, dtype=torch.long)
    if not len(ids):
        return ids
    ids = ids[torch.argsort(radar.source_index[ids], stable=True)]
    ids = ids[torch.argsort(radar.score[ids].float(), descending=True, stable=True)]
    return ids


def trace_selector(raw: dict[str, torch.Tensor], selector) -> dict[str, Any]:
    """Reproduce CandidateSelector exactly for batch_size=1."""
    ids = torch.nonzero(raw["batch_index"] == 0, as_tuple=False).flatten()
    if not len(ids):
        return {"order": ids, "pool": ids, "final": ids,
                "rank": {}, "suppressed_by": {}}

    order = ids[selector._order(raw["logits"][ids].float(), raw["source_token_id"][ids])]
    pool = order[:selector.pre]
    rank = {int(raw_idx): i + 1 for i, raw_idx in enumerate(order.tolist())}
    suppressed_by: dict[int, int] = {}
    kept_local: list[int] = []

    if len(pool):
        xyz = raw["pred_xyz"][pool].float()
        suppressed = (
            torch.linalg.vector_norm(xyz[:, None] - xyz[None, :], dim=2)
            <= selector.radius
        ).cpu()
        for local_idx in range(len(pool)):
            if len(kept_local) >= selector.final:
                break
            blockers = [k for k in kept_local if bool(suppressed[local_idx, k])]
            if blockers:
                suppressed_by[int(pool[local_idx])] = int(pool[blockers[0]])
            else:
                kept_local.append(local_idx)

    final = (
        pool[torch.tensor(kept_local, device=pool.device, dtype=torch.long)]
        if kept_local else pool[:0]
    )
    return {"order": order, "pool": pool, "final": final,
            "rank": rank, "suppressed_by": suppressed_by}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    if cfg["experiment"]["stage"] != "B0":
        raise ValueError("This audit requires a B0 config")

    dataset = build_split_dataset(cfg, ROOT, args.split)
    runtime = build_runtime(cfg, ROOT, torch.device("cpu"))
    checkpoint_info = load_network(runtime.model, resolve(args.checkpoint))
    runtime.model.to(device).eval()
    synchronize_dino_device(runtime, device)
    runtime.projection_base = load_left_projection_context(
        runtime.camera_config,
        runtime.geometry_calibration,
        image_scale_xy=torch.ones((1, 2), device=device),
        device=device,
    )

    selector = runtime.model.lidar.builder.selector
    failed_rows: list[dict[str, Any]] = []
    no_output_rows: list[dict[str, Any]] = []
    near_point_rows: list[dict[str, Any]] = []
    category_counts = Counter()
    sequence_counts = Counter()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_handle = None
    raw_writer = None
    if args.dump_all_raw:
        raw_handle = (args.output / "raw_candidates_failed.csv").open(
            "w", newline="", encoding="utf-8"
        )
        raw_fields = [
            "dataset_index", "sequence_id", "sample_id", "query_time",
            "source_token_id", "raw_rank", "score",
            "center_x", "center_y", "center_z",
            "pred_x", "pred_y", "pred_z",
            "residual_x", "residual_y", "residual_z",
            "center_error_m", "pred_error_m", "regression_delta_error_m",
            "positive_label", "ignore_label", "negative_label",
            "in_pre_nms_pool", "in_final_top10", "selector_status",
            "suppressed_by_source_token_id",
        ]
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields)
        raw_writer.writeheader()

    total_gt = 0
    total_with_output = 0
    total_top10_success = 0

    try:
        for dataset_index in range(len(dataset)):
            sample = dataset[dataset_index]
            batch = _stage_batch(collate_multimodal_v2([sample]), "B0")
            lidar, images, masks, projection, targets, _ = prepare_model_batch(
                batch, runtime.dino_detector, runtime.projection_base, device
            )
            with torch.autocast(device_type=device.type, enabled=False):
                output = runtime.model(lidar, images, masks, projection)

            if not bool(targets.has_xyz[0]):
                continue
            total_gt += 1
            gt = targets.xyz_m[0].float()
            radar = output.radar_candidates
            final_ids = stable_final_order(radar)
            final_dist = (
                torch.linalg.vector_norm(radar.xyz_m[final_ids].float() - gt, dim=1)
                if len(final_ids) else gt.new_empty(0)
            )

            if len(final_ids):
                total_with_output += 1
                if bool((final_dist[:10] <= 1.0).any()):
                    total_top10_success += 1
                    continue

            seq = sample["sequence_id"]
            sample_id = sample["sample_id"]
            query_time = float(sample["query_time"])

            # 4 no-output samples: inspect raw event files and valid XYZ counts.
            if not len(final_ids):
                events = dataset.select_radar_events(seq, query_time)
                raw_rows_total = valid_points_total = invalid_rows_total = 0
                avia_events = mid360_events = 0
                avia_valid_points = mid360_valid_points = 0
                for event in events:
                    xyz, raw_n, invalid_n = load_released_xyz(event.file_path)
                    raw_rows_total += raw_n
                    valid_points_total += len(xyz)
                    invalid_rows_total += invalid_n
                    if event.sensor_id == 0:
                        avia_events += 1
                        avia_valid_points += len(xyz)
                    else:
                        mid360_events += 1
                        mid360_valid_points += len(xyz)
                no_output_rows.append({
                    "dataset_index": dataset_index,
                    "sequence_id": seq,
                    "sample_id": sample_id,
                    "query_time": query_time,
                    "m_R": bool(sample["m_R"]),
                    "selected_event_count": len(events),
                    "event_time_min": min((e.timestamp for e in events), default=None),
                    "event_time_max": max((e.timestamp for e in events), default=None),
                    "raw_rows_total": raw_rows_total,
                    "valid_xyz_points_total": valid_points_total,
                    "invalid_or_zero_rows_total": invalid_rows_total,
                    "avia_event_count": avia_events,
                    "mid360_event_count": mid360_events,
                    "avia_valid_points": avia_valid_points,
                    "mid360_valid_points": mid360_valid_points,
                    "diagnosis": (
                        "NO_VALID_RADAR_POINTS" if valid_points_total == 0
                        else "HAS_VALID_POINTS_BUT_NO_CANDIDATE_OUTPUT"
                    ),
                })
                continue

            raw = output.diagnostics["lidar_raw"]
            lidar_batch = output.diagnostics["lidar_batch"]
            if raw is None or lidar_batch is None:
                raise AssertionError("Radar output exists but lidar_raw/lidar_batch is missing")

            trace = trace_selector(raw, selector)
            traced_source_ids = raw["source_token_id"][trace["final"]].long()
            actual_source_ids = radar.source_index[final_ids].long()
            if not torch.equal(traced_source_ids, actual_source_ids):
                raise AssertionError(f"selector trace mismatch: {sample_id}")

            # GT-near input point cloud.
            points = lidar_batch["points"].float()
            dt = lidar_batch["delta_t"].float()
            sensor = lidar_batch["sensor_id"].long()
            point_dist = torch.linalg.vector_norm(points - gt, dim=1)
            age = -dt
            near_mask = point_dist <= args.near_radius_m
            for i in torch.nonzero(near_mask, as_tuple=False).flatten().tolist():
                p = points[i]
                near_point_rows.append({
                    "dataset_index": dataset_index,
                    "sequence_id": seq,
                    "sample_id": sample_id,
                    "query_time": query_time,
                    "point_x": float(p[0]), "point_y": float(p[1]), "point_z": float(p[2]),
                    "distance_to_gt_m": float(point_dist[i]),
                    "delta_t_s": float(dt[i]), "age_s": float(age[i]),
                    "sensor_id": int(sensor[i]),
                    "sensor_name": "Avia" if int(sensor[i]) == 0 else "Mid360",
                })

            # Exact B0 supervision masks from CandidateLoss.
            pos, ign, neg, target_residual = runtime.lidar_loss.labels(raw, lidar_batch)

            pred = raw["pred_xyz"].float()
            center = raw["voxel_centers"].float()
            residual = raw["residual_xyz"].float()
            score = torch.sigmoid(raw["logits"].float())
            pred_dist = torch.linalg.vector_norm(pred - gt, dim=1)
            center_dist = torch.linalg.vector_norm(center - gt, dim=1)
            reg_l2 = torch.linalg.vector_norm(residual - target_residual.float(), dim=1)

            order = trace["order"]
            rank = torch.empty(len(pred), dtype=torch.long, device=device)
            rank[order] = torch.arange(1, len(order) + 1, device=device)

            raw_correct = pred_dist <= 1.0
            pre_correct = raw_correct[trace["pool"]]
            final_correct = raw_correct[trace["final"]]
            raw_correct_ids = torch.nonzero(raw_correct, as_tuple=False).flatten()
            best_correct_rank = int(rank[raw_correct_ids].min()) if len(raw_correct_ids) else None

            closest_pred_idx = int(torch.argmin(pred_dist))
            closest_center_idx = int(torch.argmin(center_dist))
            positive_ids = torch.nonzero(pos, as_tuple=False).flatten()
            positive_best_pred = float(pred_dist[positive_ids].min()) if len(positive_ids) else None
            positive_best_rank = int(rank[positive_ids].min()) if len(positive_ids) else None
            positive_best_reg = float(reg_l2[positive_ids].min()) if len(positive_ids) else None

            suppressor_source = suppressor_gt_error = suppressor_score = None
            if not bool(raw_correct.any()):
                category = (
                    "NO_TRAINING_SUPPORT_AND_NO_RAW_1M"
                    if int(pos.sum()) == 0
                    else "SUPPORT_EXISTS_BUT_REGRESSION_HAS_NO_RAW_1M"
                )
            elif not bool(pre_correct.any()):
                category = "RAW_1M_BELOW_PRE_NMS_TOPK"
            elif bool(final_correct.any()):
                category = "INTERNAL_MISMATCH_FINAL_HAS_1M"
            else:
                correct_pool_ids = trace["pool"][pre_correct]
                chosen = int(correct_pool_ids[torch.argmin(rank[correct_pool_ids])])
                if chosen in trace["suppressed_by"]:
                    category = "PRE_NMS_1M_SUPPRESSED_BY_NMS"
                    blocker = trace["suppressed_by"][chosen]
                    suppressor_source = int(raw["source_token_id"][blocker])
                    suppressor_gt_error = float(pred_dist[blocker])
                    suppressor_score = float(score[blocker])
                else:
                    category = "PRE_NMS_1M_LOST_TO_FINAL_TOPK_CUTOFF"

            category_counts[category] += 1
            sequence_counts[seq] += 1
            nearest_point_idx = int(torch.argmin(point_dist)) if len(point_dist) else None

            failed_rows.append({
                "dataset_index": dataset_index,
                "sequence_id": seq,
                "sample_id": sample_id,
                "query_time": query_time,
                "failure_category": category,
                "input_point_count": len(points),
                "event_count": int(sample["event_count"]),
                "point_min_gt_dist_m": float(point_dist.min()) if len(point_dist) else None,
                "points_within_0p5m": int((point_dist <= 0.5).sum()),
                "points_within_1m": int((point_dist <= 1.0).sum()),
                "points_within_2m": int((point_dist <= 2.0).sum()),
                "points_within_1m_age_le_0p1s": int(((point_dist <= 1.0) & (age <= 0.1)).sum()),
                "points_within_1m_age_le_0p2s": int(((point_dist <= 1.0) & (age <= 0.2)).sum()),
                "points_within_1m_age_le_0p5s": int(((point_dist <= 1.0) & (age <= 0.5)).sum()),
                "avia_points_total": int((sensor == 0).sum()),
                "mid360_points_total": int((sensor == 1).sum()),
                "avia_points_within_1m": int(((sensor == 0) & (point_dist <= 1.0)).sum()),
                "mid360_points_within_1m": int(((sensor == 1) & (point_dist <= 1.0)).sum()),
                "nearest_point_delta_t_s": float(dt[nearest_point_idx]) if nearest_point_idx is not None else None,
                "nearest_point_sensor_id": int(sensor[nearest_point_idx]) if nearest_point_idx is not None else None,
                "positive_voxel_count": int(pos.sum()),
                "ignore_voxel_count": int(ign.sum()),
                "negative_voxel_count": int(neg.sum()),
                "raw_candidate_count": len(pred),
                "raw_min_pred_error_m": float(pred_dist.min()),
                "raw_num_pred_within_1m": int(raw_correct.sum()),
                "raw_num_pred_within_2m": int((pred_dist <= 2.0).sum()),
                "best_raw_1m_score_rank": best_correct_rank,
                "pre_nms_topk": int(selector.pre),
                "pre_nms_has_1m": bool(pre_correct.any()),
                "final_topk": int(selector.final),
                "final_candidate_count": len(final_ids),
                "final_top1_error_m": float(final_dist[0]),
                "final_top10_oracle_error_m": float(final_dist[:10].min()),
                "closest_voxel_center_error_m": float(center_dist[closest_center_idx]),
                "closest_voxel_after_reg_error_m": float(pred_dist[closest_center_idx]),
                "closest_voxel_regression_delta_error_m": float(pred_dist[closest_center_idx] - center_dist[closest_center_idx]),
                "best_pred_candidate_error_m": float(pred_dist[closest_pred_idx]),
                "best_pred_candidate_center_error_m": float(center_dist[closest_pred_idx]),
                "best_pred_regression_delta_error_m": float(pred_dist[closest_pred_idx] - center_dist[closest_pred_idx]),
                "positive_best_pred_error_m": positive_best_pred,
                "positive_best_score_rank": positive_best_rank,
                "positive_best_reg_l2_error_m": positive_best_reg,
                "nms_suppressor_source_token_id": suppressor_source,
                "nms_suppressor_gt_error_m": suppressor_gt_error,
                "nms_suppressor_score": suppressor_score,
            })

            if raw_writer is not None:
                final_set = {int(x) for x in trace["final"].tolist()}
                pool_set = {int(x) for x in trace["pool"].tolist()}
                for raw_idx in range(len(pred)):
                    if raw_idx in final_set:
                        status, blocker = "kept_final", None
                    elif raw_idx in trace["suppressed_by"]:
                        status, blocker = "suppressed_by_nms", trace["suppressed_by"][raw_idx]
                    elif raw_idx in pool_set:
                        status, blocker = "final_topk_cutoff", None
                    else:
                        status, blocker = "below_pre_nms_topk", None
                    c = center[raw_idx]
                    p = pred[raw_idx]
                    r = residual[raw_idx]
                    raw_writer.writerow({
                        "dataset_index": dataset_index,
                        "sequence_id": seq,
                        "sample_id": sample_id,
                        "query_time": query_time,
                        "source_token_id": int(raw["source_token_id"][raw_idx]),
                        "raw_rank": int(rank[raw_idx]),
                        "score": float(score[raw_idx]),
                        "center_x": float(c[0]), "center_y": float(c[1]), "center_z": float(c[2]),
                        "pred_x": float(p[0]), "pred_y": float(p[1]), "pred_z": float(p[2]),
                        "residual_x": float(r[0]), "residual_y": float(r[1]), "residual_z": float(r[2]),
                        "center_error_m": float(center_dist[raw_idx]),
                        "pred_error_m": float(pred_dist[raw_idx]),
                        "regression_delta_error_m": float(pred_dist[raw_idx] - center_dist[raw_idx]),
                        "positive_label": bool(pos[raw_idx]),
                        "ignore_label": bool(ign[raw_idx]),
                        "negative_label": bool(neg[raw_idx]),
                        "in_pre_nms_pool": raw_idx in pool_set,
                        "in_final_top10": raw_idx in final_set,
                        "selector_status": status,
                        "suppressed_by_source_token_id": None if blocker is None else int(raw["source_token_id"][blocker]),
                    })
    finally:
        if raw_handle is not None:
            raw_handle.close()

    write_csv(args.output / "failed_queries.csv", failed_rows)
    write_csv(args.output / "no_output_queries.csv", no_output_rows)
    write_csv(args.output / "near_gt_points.csv", near_point_rows)

    summary = {
        "branch_contract": "refactor-v2-standalone",
        "config": str(resolve(args.config)),
        "checkpoint": str(resolve(args.checkpoint)),
        "checkpoint_info": checkpoint_info,
        "split": args.split,
        "threshold_m": 1.0,
        "queries_with_gt": total_gt,
        "queries_with_candidate_output": total_with_output,
        "top10_success_1m": total_top10_success,
        "top10_success_1m_rate": total_top10_success / total_gt if total_gt else 0.0,
        "top10_failed_with_candidates": len(failed_rows),
        "no_candidate_output": len(no_output_rows),
        "failed_category_counts": dict(category_counts),
        "failed_sequence_counts": dict(sequence_counts),
        "failed_with_any_point_within_1m": sum(int(r["points_within_1m"] > 0) for r in failed_rows),
        "failed_with_training_positive_voxel": sum(int(r["positive_voxel_count"] > 0) for r in failed_rows),
        "failed_with_any_raw_prediction_within_1m": sum(int(r["raw_num_pred_within_1m"] > 0) for r in failed_rows),
        "failed_with_pre_nms_topk_prediction_within_1m": sum(int(r["pre_nms_has_1m"]) for r in failed_rows),
        "failed_with_raw_prediction_within_2m": sum(int(r["raw_num_pred_within_2m"] > 0) for r in failed_rows),
        "expected": {"top10_failed_with_candidates": args.expect_failed,
                     "no_candidate_output": args.expect_no_output},
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))

    if args.strict_expected_counts:
        if len(failed_rows) != args.expect_failed or len(no_output_rows) != args.expect_no_output:
            raise AssertionError(
                f"Count mismatch: failed={len(failed_rows)} (expected {args.expect_failed}), "
                f"no_output={len(no_output_rows)} (expected {args.expect_no_output})"
            )


if __name__ == "__main__":
    main()
