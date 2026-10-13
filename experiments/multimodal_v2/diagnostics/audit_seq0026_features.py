#!/usr/bin/env python3
"""Read-only, offline A2 feature audit of four real seq0026 trace exports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

TIMES = ("1706254922.6860642", "1706254922.7306657",
         "1706254923.0253723", "1706254923.5353887")
BACKGROUND_XYZ = np.array([-4.20, -1.64, .73])
BACKGROUND_CENTER = np.array([-3.75, -1.75, .50])
STAGES = (
    ("SBE token", "frontend.l0_tokens", 0),
    ("Encoder0 post position", "radar.encoder0.block0.input_0", 0),
    ("Encoder0 block0", "radar.encoder0.block0.output", 0),
    ("Encoder0 block1", "radar.encoder0.block1.output", 0),
    ("Merge01", "radar.merge01.output", 1),
    ("Encoder1 post position", "radar.encoder1.block0.input_0", 1),
    ("Encoder1 block0", "radar.encoder1.block0.output", 1),
    ("Encoder1 block1", "radar.encoder1.block1.output", 1),
    ("Merge12", "radar.merge12.output", 2),
    ("Encoder2 post position", "radar.encoder2.block0.input_0", 2),
    ("Encoder2 block0", "radar.encoder2.block0.output", 2),
    ("Encoder2 block1", "radar.encoder2.block1.output", 2),
    ("Up21", "radar.up21.output", 1),
    ("Up10", "radar.up10.output", 0),
    ("final norm / head input", "radar.head.input_0", 0),
    ("cls linear0", "radar.head.cls.linear0.output", 0),
    ("cls GELU", "radar.head.cls.gelu.output", 0),
)


class Trace:
    def __init__(self, root: Path, time: str):
        self.time = time
        self.path = root / f"seq0026_query_{time}" / "A2"
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.index = json.loads((self.path / "tensor_index.json").read_text())
        self.parity = json.loads((self.path / "parity.json").read_text())
        if self.manifest["model"] != "A2" or not self.parity["passed"]:
            raise ValueError(f"A2 / parity mismatch: {self.path}")
        self.cache = {}

    def a(self, name: str):
        if name not in self.cache:
            meta = self.index[name]
            if meta["status"] != "saved":
                raise ValueError(f"{self.path}: {name} {meta['status']}")
            self.cache[name] = np.load(self.path / meta["file"], mmap_mode="r", allow_pickle=False)
        return self.cache[name]


def csv_write(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def vector_metrics(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    delta = b - a
    top = np.argsort(np.abs(delta))[-5:][::-1]
    return {"norm_before": float(na), "norm_after": float(nb),
            "cosine": float(np.dot(a, b) / (na * nb)) if na and nb else None,
            "l2": float(np.linalg.norm(delta)),
            "relative_l2": float(np.linalg.norm(delta) / na) if na else None,
            "top_changed_dimensions": ";".join(f"{i}:{delta[i]:+.4f}" for i in top)}


def geometry(points):
    result = {"point_count": len(points), "xyz_mean": None, "xyz_std": None,
              "xyz_min": None, "xyz_max": None, "cov_eigenvalues": None,
              "linearity": None, "planarity": None, "scattering": None}
    if not len(points):
        return result
    result.update(xyz_mean=np.mean(points, axis=0).tolist(), xyz_std=np.std(points, axis=0).tolist(),
                  xyz_min=np.min(points, axis=0).tolist(), xyz_max=np.max(points, axis=0).tolist())
    if len(points) < 3:
        return result
    eigen = np.sort(np.linalg.eigvalsh(np.cov(points.astype(np.float64).T)))[::-1]
    result["cov_eigenvalues"] = eigen.tolist()
    if eigen[0] > 1e-12:
        result.update(linearity=float((eigen[0]-eigen[1])/eigen[0]),
                      planarity=float((eigen[1]-eigen[2])/eigen[0]),
                      scattering=float(eigen[2]/eigen[0]))
    return result


def choose_tokens(trace):
    pred = trace.a("raw.pred_xyz")
    gt = trace.a("sample.target_xyz")
    centers = trace.a("level0.centers")
    logits = trace.a("raw.logits")
    target = int(np.linalg.norm(pred-gt, axis=1).argmin())
    # Fixed physical neighborhood plus regressed position; choose the strongest
    # background token without assuming its source ID or z voxel is stable.
    pred_distance = np.linalg.norm(pred-BACKGROUND_XYZ, axis=1)
    center_distance = np.linalg.norm(centers-BACKGROUND_CENTER, axis=1)
    matches = np.flatnonzero((pred_distance <= .80) & (center_distance <= .55))
    background = int(matches[np.argmax(logits[matches])]) if len(matches) else None
    return {"T": target, "B": background}


def selector_reason(trace, token):
    order = trace.a("selector.batch0.order")
    rank = int(np.flatnonzero(order == token)[0]) + 1
    selected = set(map(int, trace.a("selector.batch0.final")))
    raw = set(map(int, trace.a("selector.batch0.raw")))
    pool = trace.a("selector.batch0.pool")
    event = json.loads((trace.path / "events.json").read_text())["selector.batch0"]
    if token in selected:
        return rank, "selected", None, None
    if token not in raw:
        return rank, "below_raw_topk", None, None
    if token not in set(map(int, pool)):
        return rank, "below_pre_nms_topk", None, None
    i = int(np.flatnonzero(pool == token)[0])
    if i >= event["evaluated_count"]:
        return rank, "not_evaluated_final_limit", None, None
    matrix = trace.a("selector.batch0.suppressed")
    for previous in event["kept_local"]:
        if matrix[i, previous]:
            suppressor = int(pool[previous])
            distance = float(np.linalg.norm(trace.a("raw.pred_xyz")[token]-trace.a("raw.pred_xyz")[suppressor]))
            return rank, "nms_radius", suppressor, distance
    raise AssertionError(f"no selector reason for {token}")


def context(trace, token, level, block):
    prefix = f"radar.encoder{level}.block{block}.groups"
    ids, offsets = trace.a(prefix+".token_ids"), trace.a(prefix+".offsets")
    position = np.flatnonzero(ids == token)
    if len(position) != 1:
        raise AssertionError(f"token {token} appears {len(position)} times in {prefix}")
    group = int(np.searchsorted(offsets, position[0], side="right")-1)
    members = np.asarray(ids[offsets[group]:offsets[group+1]], dtype=int)
    return {"level": level, "block": block, "group": group,
            "shift": json.loads((trace.path/"events.json").read_text())[prefix]["shift"],
            "morton_order_position": int(position[0]), "member_count": len(members),
            "member_token_ids": members.tolist(),
            "member_coords": trace.a(f"level{level}.coords")[members].tolist(),
            "member_point_counts": trace.a(f"level{level}.point_count")[members].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=Path("outputs/v2_anomaly_seq0026_feature_audit/traces"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v2_anomaly_seq0026_feature_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    traces = [Trace(args.trace_root, t) for t in TIMES]
    checkpoint_hashes = {t.manifest["checkpoint_sha256"] for t in traces}
    if len(checkpoint_hashes) != 1:
        raise ValueError("checkpoint differs across queries")
    checkpoint = torch.load(traces[0].manifest["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    weight = checkpoint["network.lidar.detector.head.cls.2.weight"].numpy().reshape(-1)
    bias = float(checkpoint["network.lidar.detector.head.cls.2.bias"].item())
    comparison, geom_rows, class_rows, windows, vectors, feature_rows = [], [], [], {}, {}, []
    by_time = {}
    for trace in traces:
        points, sensors, dt = (trace.a(k) for k in ("sample.points", "sample.sensor_id", "sample.delta_t"))
        point_map, parents01, parents12 = (trace.a(k) for k in ("mapping.point_to_l0", "mapping.parent_l0_to_l1", "mapping.parent_l1_to_l2"))
        coords = trace.a("level0.coords")
        if not np.array_equal(trace.a("level1.coords")[parents01], np.floor_divide(coords, 2)):
            raise AssertionError("L0->L1 parent coordinates mismatch")
        if not np.array_equal(trace.a("level2.coords")[parents12], np.floor_divide(trace.a("level1.coords"), 2)):
            raise AssertionError("L1->L2 parent coordinates mismatch")
        if not np.array_equal(trace.a("level0.point_count"), np.bincount(point_map, minlength=len(coords))):
            raise AssertionError("point-to-L0 counts mismatch")
        gt = np.asarray(trace.a("sample.target_xyz"))
        candidates = choose_tokens(trace)
        by_time[trace.time] = {}
        for kind, token in candidates.items():
            if token is None:
                comparison.append({"time": trace.time, "kind": kind, "status": "missing"})
                continue
            token = int(token)
            l1 = int(parents01[token])
            l2 = int(parents12[l1])
            pred = np.asarray(trace.a("raw.pred_xyz")[token]); center = np.asarray(trace.a("level0.centers")[token])
            logit = float(trace.a("raw.logits")[token]); score = float(1/(1+np.exp(-logit)))
            rank, reason, suppressor, suppression_distance = selector_reason(trace, token)
            own_points = np.flatnonzero(point_map == token)
            nearest_input = float(np.linalg.norm(points[own_points]-gt, axis=1).min())
            label = "positive" if nearest_input <= 1 else "ignore" if nearest_input <= 2 else "negative"
            row = {"time": trace.time, "kind": kind, "status": "matched", "source_token_id": token,
                   "l0_coord": json.dumps(coords[token].tolist()), "l0_center_xyz_m": json.dumps(center.tolist()),
                   "l1_token_id": l1, "l2_token_id": l2, "pred_xyz_m": json.dumps(pred.tolist()),
                   "gt_xyz_m": json.dumps(gt.tolist()), "gt_distance_m": float(np.linalg.norm(pred-gt)),
                   "background_pred_match_m": float(np.linalg.norm(pred-BACKGROUND_XYZ)) if kind == "B" else None,
                   "background_center_match_m": float(np.linalg.norm(center-BACKGROUND_CENTER)) if kind == "B" else None,
                   "raw_logit": logit, "raw_score": score, "raw_rank": rank,
                   "selector_reason": reason, "final_kept": reason == "selected",
                   "suppressed_by": suppressor, "suppression_distance_m": suppression_distance,
                   "nearest_input_point_gt_m": nearest_input, "training_label": label}
            comparison.append(row)
            by_time[trace.time][kind] = row
            own = np.asarray(points[own_points]); local = own-center
            slot_count = np.asarray(trace.a("frontend.slot_count")[token])
            if int(slot_count.sum()) != len(own_points):
                raise AssertionError("slot counts do not sum to voxel points")
            slot_ids = np.asarray(trace.a("frontend.point_to_slot")[own_points])
            if not np.array_equal(slot_count, np.bincount(slot_ids, minlength=8)):
                raise AssertionError("point-to-slot map mismatch")
            g = {"time": trace.time, "kind": kind, "source_token_id": token,
                 "voxel_points": len(own_points), "voxel_avia": int(np.count_nonzero(sensors[own_points] == 0)),
                 "voxel_mid360": int(np.count_nonzero(sensors[own_points] == 1)),
                 "slot_counts": json.dumps(slot_count.tolist()), "occupied_slots": int(np.count_nonzero(slot_count)),
                 "single_point_slots": int(np.count_nonzero(slot_count == 1)),
                 "singleton_fraction_occupied": float(np.count_nonzero(slot_count == 1)/np.count_nonzero(slot_count)),
                 "voxel_local_xyz_mean": json.dumps(local.mean(0).tolist()),
                 "voxel_local_xyz_std": json.dumps(local.std(0).tolist()),
                 "voxel_local_xyz_min": json.dumps(local.min(0).tolist()),
                 "voxel_local_xyz_max": json.dumps(local.max(0).tolist()),
                 "voxel_delta_t_mean": float(dt[own_points].mean()), "voxel_delta_t_std": float(dt[own_points].std()),
                 "voxel_delta_t_min": float(dt[own_points].min()), "voxel_delta_t_max": float(dt[own_points].max()),
                 "slot_mean_features": json.dumps(np.asarray(trace.a("frontend.slot_mean")[token]).tolist()),
                 "slot_max_features": json.dumps(np.asarray(trace.a("frontend.slot_max")[token]).tolist()),
                 "slot_log_counts": json.dumps(np.asarray(trace.a("frontend.slot_features")[token,:,64]).tolist()),
                 "slot_occupancy": json.dumps(np.asarray(trace.a("frontend.slot_features")[token,:,65]).tolist())}
            for radius in (.5, 1., 2.):
                nearby = np.flatnonzero(np.linalg.norm(points-center, axis=1) <= radius)
                shape = geometry(points[nearby]-center)
                prefix = f"r{radius:g}m_"
                g.update({prefix+"points": len(nearby), prefix+"avia": int(np.count_nonzero(sensors[nearby] == 0)),
                          prefix+"mid360": int(np.count_nonzero(sensors[nearby] == 1)),
                          prefix+"occupied_l0": len(np.unique(point_map[nearby]))})
                for key, value in shape.items():
                    if key != "point_count":
                        g[prefix+key] = json.dumps(value) if isinstance(value, list) else value
                vectors[f"{trace.time}_{kind}_neighborhood_{radius:g}m_xyz"] = np.asarray(points[nearby], dtype=np.float32)
            geom_rows.append(g)
            windows[f"{trace.time}_{kind}"] = {"l0": token, "l1": l1, "l2": l2,
                "encoder0": [context(trace, token, 0, b) for b in range(2)],
                "encoder1": [context(trace, l1, 1, b) for b in range(2)]}
            for name, array_name, level in STAGES:
                index = (token, l1, l2)[level]
                vectors[f"{trace.time}_{kind}_{name}"] = np.asarray(trace.a(array_name)[index]).copy()
            for name in ("frontend.slot_count", "frontend.slot_mean", "frontend.slot_max",
                         "frontend.slot_features", "frontend.dynamic_summary", "frontend.projection_input",
                         "radar.head.cls.linear1.output", "raw.residual_xyz"):
                vectors[f"{trace.time}_{kind}_{name}"] = np.asarray(trace.a(name)[token]).copy()
            vectors[f"{trace.time}_{kind}_point_ids"] = own_points.copy()
            vectors[f"{trace.time}_{kind}_point_slot_ids"] = slot_ids.copy()
            vectors[f"{trace.time}_{kind}_l0_token_id"] = np.array(token, dtype=np.int64)
            vectors[f"{trace.time}_{kind}_l1_token_id"] = np.array(l1, dtype=np.int64)
            vectors[f"{trace.time}_{kind}_l2_token_id"] = np.array(l2, dtype=np.int64)
            for level, index in enumerate((token, l1, l2)):
                vectors[f"{trace.time}_{kind}_l{level}_coord"] = np.asarray(trace.a(f"level{level}.coords")[index]).copy()
            for name in ("frontend.point_mlp_input", "frontend.point_mlp_output"):
                selected = np.asarray(trace.a(name)[own_points])
                vectors[f"{trace.time}_{kind}_{name}"] = selected.copy()
            for name in ("frontend.point_mlp_input", "frontend.point_mlp_output", "frontend.slot_features",
                         "frontend.dynamic_summary", "frontend.projection_input", "frontend.l0_tokens",
                         "radar.head.cls.linear0.output", "radar.head.cls.gelu.output"):
                x = np.asarray(trace.a(name)[own_points] if name.startswith("frontend.point_mlp") else trace.a(name)[token])
                class_rows.append({"time": trace.time, "kind": kind, "row_type": "stage_summary", "stage": name,
                                   "shape": json.dumps(list(x.shape)), "mean": float(x.mean()),
                                   "std": float(x.std()), "norm": float(np.linalg.norm(x)),
                                   "nonfinite": int(np.count_nonzero(~np.isfinite(x))),
                                   "raw_logit": logit, "raw_score": score,
                                   "hidden_top_abs_dimensions": json.dumps(np.argsort(np.abs(x))[-5:][::-1].tolist()) if x.ndim == 1 else None})
            pre = np.asarray(trace.a("radar.head.cls.linear0.output")[token])
            hidden = np.asarray(trace.a("radar.head.cls.gelu.output")[token])
            contributions = hidden.astype(np.float64)*weight.astype(np.float64)
            if abs(contributions.sum()+bias-logit) > 1e-5:
                raise AssertionError("checkpoint classification weights do not reproduce traced raw logit")
            vectors[f"{trace.time}_{kind}_cls_logit_contributions"] = contributions.copy()
            vectors["cls_final_weight"] = weight.copy()
            vectors["cls_final_bias"] = np.array(bias)
            for channel in range(len(hidden)):
                class_rows.append({"time": trace.time, "kind": kind, "row_type": "hidden_channel",
                                   "stage": "cls GELU -> final Linear", "channel": channel,
                                   "linear0_value": float(pre[channel]), "gelu_value": float(hidden[channel]),
                                   "final_weight": float(weight[channel]),
                                   "logit_contribution": float(contributions[channel]),
                                   "raw_logit": logit, "raw_score": score})
    for first, second in zip(traces, traces[1:]):
        for kind in ("T", "B"):
            if kind not in by_time[first.time] or kind not in by_time[second.time]:
                continue
            for name, _, _ in STAGES:
                a = vectors[f"{first.time}_{kind}_{name}"]
                b = vectors[f"{second.time}_{kind}_{name}"]
                feature_rows.append({"comparison": f"{first.time}->{second.time}", "kind": kind,
                                     "stage": name, **vector_metrics(a, b)})
    for trace in traces:
        if not {"T", "B"} <= by_time[trace.time].keys():
            continue
        for name, _, _ in STAGES:
            feature_rows.append({"comparison": trace.time, "kind": "T_vs_B", "stage": name,
                                 **vector_metrics(vectors[f"{trace.time}_T_{name}"],
                                                  vectors[f"{trace.time}_B_{name}"])})
    for name, value in vectors.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite selected feature: {name}")
    csv_write(args.output/"candidate_comparison.csv", comparison)
    csv_write(args.output/"raw_geometry_stats.csv", geom_rows)
    csv_write(args.output/"layer_feature_metrics.csv", feature_rows)
    csv_write(args.output/"classification_trace.csv", class_rows)
    np.savez_compressed(args.output/"feature_vectors.npz", **vectors)
    (args.output/"context_windows.json").write_text(json.dumps(windows, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = [t[-6:] for t in TIMES]
    fig, ax = plt.subplots(figsize=(8, 4))
    for kind in ("T", "B"):
        ax.plot(short, [by_time[t][kind]["raw_score"] for t in TIMES], marker="o", label=kind)
    ax.set(xlabel="Query time suffix", ylabel="Raw sigmoid score", title="A2 raw classification")
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(args.output/"candidate_scores.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=True)
    for ax, (first, second) in zip(axes, zip(TIMES, TIMES[1:])):
        rows = [r for r in feature_rows if r["comparison"] == f"{first}->{second}"]
        for kind in ("T", "B"):
            ax.plot(range(len(STAGES)), [r["relative_l2"] for r in rows if r["kind"] == kind],
                    marker=".", label=kind)
        ax.set_xticks(range(len(STAGES)), [s[0] for s in STAGES], rotation=90)
        ax.set_title(f"{first[-6:]} -> {second[-6:]}"); ax.grid(alpha=.3); ax.legend()
    axes[0].set_ylabel("Cross-query relative L2 (within each stage)")
    fig.tight_layout(); fig.savefig(args.output/"layer_changes.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for ax, time in zip(axes, TIMES):
        for kind, shift in (("T", -.18), ("B", .18)):
            row = next(r for r in geom_rows if r["time"] == time and r["kind"] == kind)
            ax.bar(np.arange(8)+shift, json.loads(row["slot_counts"]), width=.36, label=kind)
        ax.set_ylabel("Points"); ax.set_title(time); ax.legend()
    axes[-1].set_xticks(range(8)); axes[-1].set_xlabel("Ordered slot ID")
    fig.tight_layout(); fig.savefig(args.output/"slot_counts.png", dpi=160); plt.close(fig)
    fig = plt.figure(figsize=(12, 12))
    for i, time in enumerate(TIMES):
        for j, kind in enumerate(("T", "B")):
            ax = fig.add_subplot(4, 2, i*2+j+1, projection="3d")
            row = by_time[time][kind]
            xyz = vectors[f"{time}_{kind}_neighborhood_2m_xyz"]-np.array(json.loads(row["l0_center_xyz_m"]))
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1 if kind == "B" else 12, alpha=.35)
            ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2), title=f"{time[-6:]} {kind} / {len(xyz)} points")
    fig.tight_layout(); fig.savefig(args.output/"local_clouds_2m.png", dpi=160); plt.close(fig)
    print(f"wrote {args.output}; candidates={len(comparison)} vectors={len(vectors)} parity=PASS x4")


if __name__ == "__main__":
    main()
