"""Read-only B4 candidate evidence, preserving source IDs and task masks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .loss import box_iou_aligned


@torch.no_grad()
def candidate_evidence(output, targets, losses, criterion, *, batch_index=0):
    r, v = output.radar_candidates, output.vision_candidates
    rids = torch.nonzero(r.batch_index == batch_index).flatten()
    vids = torch.nonzero(v.batch_index == batch_index).flatten()
    rows = torch.nonzero(output.batch_index == batch_index).flatten()
    top3 = output.top3d_indices(len(targets.has_xyz))[batch_index].tolist()
    top2 = output.top2d_indices(len(targets.has_xyz))[batch_index].tolist()
    rank3 = {i: j + 1 for j, i in enumerate(top3)}
    rank2 = {i: j + 1 for j, i in enumerate(top2)}
    has_gt3 = bool(targets.has_xyz[batch_index])
    has_gt2 = bool(targets.has_box[batch_index])
    rs, vs = criterion.candidate_status(r, targets), criterion.candidate_status(v, targets)
    radar = []
    for i in rids.tolist():
        radar.append(dict(source_id=int(r.source_index[i]), original_score=float(r.score[i]),
                          xyz_m=r.xyz_m[i].float().cpu().tolist(),
                          projected_xy_px=r.projected_xy_px[i].float().cpu().tolist()
                          if bool(r.projection_valid[i]) else None,
                          projection_valid=bool(r.projection_valid[i]), train_status=int(rs[i]),
                          error_3d_m=float((r.xyz_m[i].float()-targets.xyz_m[batch_index]).norm())
                          if has_gt3 else None))
    vision = []
    for i in vids.tolist():
        vision.append(dict(source_id=int(v.source_index[i]), original_score=float(v.score[i]),
                           box_xyxy_px=v.box_xyxy_px[i].float().cpu().tolist(), train_status=int(vs[i]),
                           iou=float(box_iou_aligned(v.box_xyxy_px[i:i+1].float(),
                                     targets.box_xyxy_px[batch_index])[0]) if has_gt2 else None))
    pair_map = {(p["radar_source_index"], p["vision_source_index"]): p
                for p in output.diagnostics["association_rv_pairs"] if p["batch_index"] == batch_index}
    evidence = []
    for i in rows.tolist():
        paired = bool(output.association_valid[i])
        hr, hv = bool(output.has_xyz[i]), bool(output.has_box[i])
        rid, vid = int(output.radar_source_index[i]), int(output.vision_source_index[i])
        error = float((output.xyz_m[i].float()-targets.xyz_m[batch_index]).norm()) if hr and has_gt3 else None
        iou = float(box_iou_aligned(output.box_xyxy_px[i:i+1].float(),
                                   targets.box_xyxy_px[batch_index])[0]) if hv and has_gt2 else None
        pair = pair_map.get((rid, vid), {})
        evidence.append(dict(output_row=i, type="RV" if paired else ("R" if hr else "V"),
            radar_source_id=rid if hr else None, vision_source_id=vid if hv else None,
            rank_3d=rank3.get(i), rank_2d=rank2.get(i), joint_logit=float(output.joint_logit[i]),
            joint_score=float(output.joint_logit[i].sigmoid()), train_label=int(losses["joint_labels"][i]),
            xyz_m=output.xyz_m[i].float().cpu().tolist() if hr else None,
            box_xyxy_px=output.box_xyxy_px[i].float().cpu().tolist() if hv else None,
            error_3d_m=error, iou=iou,
            eval_joint_correct=bool(paired and error is not None and error <= 1. and iou is not None and iou >= .5),
            d_box_px=pair.get("d_box_px"), d_center_normalized=pair.get("d_center_normalized"),
            cosine=pair.get("similarity"), matching_cost=pair.get("cost")))
    correct = [p for p in evidence if p["eval_joint_correct"]]
    positive = [p["joint_logit"] for p in evidence if p["train_label"] == 1]
    negative = [p["joint_logit"] for p in evidence if p["train_label"] == 0]
    same = bool(top3 and top2 and top3[0] == top2[0])
    winner_good = bool(same and any(p["output_row"] == top3[0] for p in correct))
    reason = ("missing_joint_gt" if not (has_gt3 and has_gt2) else
              "no_correct_rv_pair" if not correct else
              "joint_success" if winner_good else
              "different_task_winners" if not same else "wrong_pair_ranked_first")
    return dict(radar_candidates=radar, vision_candidates=vision, evidence_pairs=evidence,
                failure_category=reason, best_positive_logit=max(positive) if positive else None,
                best_negative_logit=max(negative) if negative else None,
                positive_negative_margin=max(positive)-max(negative) if positive and negative else None,
                loss_components={name: float(losses[name]) for name in
                                 ("loss", "loss_joint_cls", "loss_align", "loss_joint_rank")},
                counts={k: int(value) for k, value in losses.items() if k.startswith("n_")})


def append_evidence(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


@torch.no_grad()
def save_feature_evidence(path, output, fusion):
    r, v = output.radar_candidates, output.vision_candidates
    if len(output.diagnostics["pair_alignment"]) != 1:
        raise ValueError("Feature export requires one query per inference batch")
    alignment = output.diagnostics["pair_alignment"][0]
    hr = fusion.radar_adapter(r.feature.float())
    hv = fusion.vision_adapter(v.feature.float())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cpu = lambda value: value.detach().cpu().numpy()
    np.savez_compressed(path, radar_features=cpu(r.feature), vision_features=cpu(v.feature),
        radar_mapped=cpu(hr), vision_mapped=cpu(hv),
        radar_source_ids=cpu(r.source_index), vision_source_ids=cpu(v.source_index),
        similarity=cpu(alignment["similarity"]), valid_edges=cpu(alignment["valid_edges"]),
        matching_cost=cpu(alignment["matching_cost"]), d_box_px=cpu(alignment["d_box_px"]),
        d_center_normalized=cpu(alignment["d_center_normalized"]),
        geometry_scale_px=cpu(alignment["geometry_scale_px"]),
        matching_feasible=cpu(alignment["matching_feasible"]),
        similarity_radar_source_ids=cpu(r.source_index[alignment["radar_index"]]),
        similarity_vision_source_ids=cpu(v.source_index[alignment["vision_index"]]))
