#!/usr/bin/env python3
"""M2 GT-conditioned module dataset. GT never enters candidate generation."""
import argparse
import csv
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdq_uav.baselines.mmuav_preprocess import (
    _accumulate_lidar_360_blocks, _dbscan_labels, extract_feature_set_predict,
    process_lidar_livox, process_fusion, read_lidar_files)
from rdq_uav.mmuav.attention_lstm import AttentionLSTMClassifier
from build_mmuav_cluster_dataset import timestamp_files, load_xyz, load_gt, nearest_gt

MODE = "gt_conditioned_module_level"


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            **{f"P{p}": float(np.percentile(values, p)) for p in (10, 90, 95)},
            "max": float(np.max(values))}


def logits_only(output):
    return output[0] if isinstance(output, tuple) else output


def annotate_source(row):
    """Read-only sidecar: source fusion merges timestamps, never mixes a frame.

    At equal keys {**avia, **mid360} retains Mid360, including empty inputs.
    This records existing merge semantics; it does not change fusion/points.
    """
    work = Path(row["shard_path"]).parent
    timestamp = row["sample_id"].rsplit(":",2)[1]
    mid = (work / "lidar_360_processed" / f"{timestamp}.npy").exists()
    avia = (work / "livox_avia_processed" / f"{timestamp}.npy").exists()
    if not (mid or avia):
        raise FileNotFoundError("Cannot verify fusion timestamp source")
    row.update(source_identity="MID360" if mid else "LIVOX_AVIA",
               source_basis="PUBLIC_TIMESTAMP_MERGE_MID360_OVERWRITES_AVIA",
               mid360_point_count=int(row["point_count"]) if mid else 0,
               livox_point_count=0 if mid else int(row["point_count"]),
               has_mid360=mid,has_livox=not mid)
    return row


def preprocess(sequence, output, model):
    """Adapter for M1 logits; original preprocessing module is untouched."""
    mid = {p.stem: load_xyz(p) for p in timestamp_files(sequence / "lidar_360")}
    avia = {p.stem: load_xyz(p) for p in timestamp_files(sequence / "livox_avia")}
    if not mid or not avia:
        raise FileNotFoundError("Missing LiDAR sensor frames")
    folder = output / "lidar_360_processed"
    folder.mkdir(parents=True, exist_ok=True)
    blocks, times = _accumulate_lidar_360_blocks(mid)
    for key, block in blocks.items():
        per_frame = {t: [] for t in times[key]}
        if block.size:
            labels = _dbscan_labels(block[:, 1:], 2, 10)
            features, ids = extract_feature_set_predict(block[:, 1:], labels, block[:, 0])
            if len(features):
                with torch.no_grad():
                    pred = logits_only(model(torch.tensor(features, dtype=torch.float32))).argmax(1).numpy()
                positive = ids.reshape(-1)[pred == 1]
                for frame, t in enumerate(times[key], 1):
                    per_frame[t] = block[(block[:, 0] == frame) & np.isin(labels, positive), 1:]
        for t, points in per_frame.items():
            np.save(folder / f"{t}.npy", np.asarray(points).reshape(-1, 3))
    process_lidar_livox(avia, output / "livox_avia_processed")
    audit = process_fusion(read_lidar_files(output / "livox_avia_processed"),
                           read_lidar_files(folder), output / "lidar_fusion")
    if audit["cannot_process"]:
        raise RuntimeError("PUBLIC_FUSION_CANNOT_PROCESS_50000_POINTS")
    return audit


def freeze_config(rows, spatial):
    """Only training rows may choose timestamp tolerance."""
    train = [r for r in rows if r["split"] == "train_sub"]
    if not train:
        raise ValueError("No successful train timestamps")
    gaps = [r["time_gap_ms"] for r in train]
    tolerance = float(np.percentile(gaps, 95)) + 1e-6
    return {"evaluation_mode": MODE, "timestamp_tolerance_ms": tolerance,
            "association_threshold_m": spatial,
            "threshold_selection_basis": "Explicit training-pair quality ceiling, not paper MSE matching; timestamp tolerance=train P95",
            "distance_grouping_status": "DISABLED_REFERENCE_FRAME_UNVERIFIED",
            "preprocessing": {"mid360_dbscan_eps": 2, "mid360_min_samples": 10,
              "M1_training_dbscan_eps": 1, "window_size": 20, "window_stride": 20,
              "final_window": "extra final 20 frames; source overwrite semantics",
              "temporal_accumulation": "non-overlap blocks plus final20, no compensation",
              "fusion_eps": 1, "fusion_min_samples": 10, "candidate_eps": 1,
              "candidate_min_samples": 1}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/official/train"))
    p.add_argument("--splits", type=Path, default=ROOT / "outputs/mmuav_paper_reproduction/splits/splits.json")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/mmuav_paper_reproduction/classification/attention_9d/best_val_loss.pth")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mmuav_paper_reproduction/datasets/center_regression")
    p.add_argument("--association-threshold-m", type=float, required=True,
                   help="Train-only pairing quality decision; never chosen from validation MSE")
    p.add_argument("--limit-sequences", type=int)
    p.add_argument("--finalize-only", action="store_true",
                   help="Re-export existing per-sequence records only; no DBSCAN/LSTM/GT reads")
    args = p.parse_args()
    torch.set_num_threads(1)  # Avoid CPU oversubscription in tiny LSTM inference.
    if args.association_threshold_m <= 0:
        p.error("Association threshold must be positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dataset_summary.json").exists() and not args.finalize_only:
        raise FileExistsError("Completed dataset is frozen; use a new directory")
    model = AttentionLSTMClassifier()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()
    splits = json.loads(args.splits.read_text())
    if set(splits["train_sub"]) & set(splits["validation_sub"]):
        raise ValueError("Split overlap")
    if (set(splits["train_sub"]) | set(splits["validation_sub"])) & set(splits["heldout_test_sub"]):
        raise ValueError("Heldout split overlap")
    rows, candidates, failures = [], [], []
    np.random.seed(42)
    for split in ("train_sub", "validation_sub"):
        for seq in tqdm(splits[split][:args.limit_sequences], desc=split):
            print(f"BUILD {split}/{seq}", flush=True)
            work = out / "sequences" / seq
            work.mkdir(parents=True, exist_ok=True)
            try:
                cached = work / "sequence_records.json"
                checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                if cached.exists():
                    record = json.loads(cached.read_text())
                    if record["checkpoint_sha256"] != checkpoint_hash or record["split"] != split:
                        raise ValueError("Cached sequence provenance mismatch")
                    rows.extend(record["timestamps"])
                    candidates.extend(record["candidates"])
                    if args.finalize_only:
                        continue
                    state = record.get("numpy_rng_state")
                    if state is None:
                        # Source NumPy FPS consumes exactly one randint for each
                        # Avia frame exceeding 100 points. Replay RNG only, not FPS/DBSCAN.
                        for path in timestamp_files(args.data_root / seq / "livox_avia"):
                            count = len(load_xyz(path))
                            if count > 100:
                                np.random.randint(0,count)
                    else:
                        np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))
                    continue
                if args.finalize_only:
                    raise FileNotFoundError(f"Missing finalized sequence records for {seq}")
                audit = preprocess(args.data_root / seq, work, model)
                gt_times, gt_xyz = load_gt(args.data_root / seq)  # after preprocessing
                buffer, offsets = [], [0]
                local_rows, local_candidates = [], []
                fusion = read_lidar_files(work / "lidar_fusion")
                # Include successful empty raw timestamps rather than losing their denominator.
                timestamps = sorted(set(read_lidar_files(work / "lidar_360_processed")) |
                                    set(read_lidar_files(work / "livox_avia_processed")), key=float)
                for t in timestamps:
                    gt_t, gt, gap = nearest_gt(float(t), gt_times, gt_xyz)
                    points = np.asarray(fusion.get(t, np.empty((0, 3)))).reshape(-1, 3)
                    labels = _dbscan_labels(points, 1, 1) if len(points) else np.empty(0)
                    frame_candidates = []
                    for cid in sorted(set(labels.astype(int)) - {-1}):
                        cid = int(cid)
                        cluster = points[labels == cid]
                        center = cluster.mean(0)  # full points BEFORE sampling
                        idx = len(offsets) - 1
                        buffer.append(cluster)
                        offsets.append(offsets[-1] + len(cluster))
                        r = {"sample_id": f"{seq}:{t}:{cid}", "split": split,
                             "sequence_id": seq, "timestamp": float(t), "gt_timestamp": gt_t,
                             "time_gap_ms": gap, "cluster_id": cid, "point_count": len(cluster),
                             "shard_index": idx, "shard_path": str(work / "candidates.npz"),
                             "source_identity": "PUBLIC_FUSION_SOURCE_IDENTITY_LOST",
                             **{f"geometric_{a}": float(center[i]) for i,a in enumerate("xyz")},
                             **{f"gt_{a}": float(gt[i]) for i,a in enumerate("xyz")},
                             "distance_to_gt": float(np.linalg.norm(center - gt))}
                        frame_candidates.append(r)
                    best = min(frame_candidates, key=lambda r:r["distance_to_gt"]) if frame_candidates else None
                    for r in frame_candidates:
                        r["oracle_selected"] = r is best
                        local_candidates.append(r)
                    local_rows.append({"split": split, "sequence_id": seq, "timestamp": float(t),
                                       "time_gap_ms": gap, "candidate_count": len(frame_candidates),
                                       "nearest_sample_id": best["sample_id"] if best else "",
                                       "association_distance": best["distance_to_gt"] if best else None})
                np.savez(work / "candidates.npz", points=np.concatenate(buffer) if buffer else np.empty((0,3)),
                         offsets=np.array(offsets, dtype=np.int64))
                rows.extend(local_rows)
                candidates.extend(local_candidates)
                (work / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2))
                state = np.random.get_state()
                cached.write_text(json.dumps({"checkpoint_sha256":checkpoint_hash,"split":split,
                    "numpy_rng_state":[state[0],state[1].tolist(),state[2],state[3],state[4]],
                    "timestamps":local_rows,"candidates":local_candidates},indent=2))
            except Exception as exc:
                failures.append({"split": split, "sequence_id": seq,
                                 "status": "PROCESSING_FAILURE", "error": repr(exc),
                                 "kind": "missing_input" if isinstance(exc,FileNotFoundError) else "processing"})
                write_csv(out / "processing_failures.csv", failures)
                raise  # never turn a processing exception into a valid empty sequence
    config = freeze_config(rows, args.association_threshold_m)
    for r in candidates:
        annotate_source(r)
        reason = ("not_oracle_selected" if not r["oracle_selected"] else
                  "rejected_timestamp_gap" if r["time_gap_ms"] > config["timestamp_tolerance_ms"] else
                  "rejected_spatial_distance" if r["distance_to_gt"] > args.association_threshold_m else "")
        r.update(accepted=not reason, reject_reason=reason)
    accepted = [r for r in candidates if r["accepted"]]
    write_csv(out / "all_candidates.csv", candidates)
    write_csv(out / "metadata.csv", candidates)
    write_csv(out / "timestamps.csv", rows)
    write_csv(out / "evaluation_sample_ids.csv", [r for r in accepted if r["split"] == "validation_sub"])
    summary = {**config, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "source_root": str(args.data_root.resolve()), "splits_path": str(args.splits.resolve()),
               "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
               "seed": 42, "valid_empty_sequences": [], "missing_input_sequences": [],
               "model_failure_sequences": [], "processing_failure_sequences": failures}
    thresholds = []
    for threshold in (.5,1.,2.,3.):
        row = {"threshold_m": threshold}
        for split, name in (("train_sub","train"),("validation_sub","validation")):
            eligible = [r for r in candidates if r["split"] == split and r["oracle_selected"] and
                        r["time_gap_ms"] <= config["timestamp_tolerance_ms"] and r["distance_to_gt"] <= threshold]
            denom = sum(r["split"] == split for r in rows)
            mse = np.mean([r["distance_to_gt"]**2 for r in eligible]) if eligible else None
            row.update({f"{name}_accepted":len(eligible), f"{name}_coverage":len(eligible)/denom if denom else 0,
                        f"{name}_MSE_3D":mse, f"{name}_MSE_coord":mse/3 if mse is not None else None})
        thresholds.append(row)
    write_csv(out / "association_threshold_audit.csv", thresholds)
    for split in ("train_sub","validation_sub"):
        ts = [r for r in rows if r["split"] == split]
        ac = [r for r in accepted if r["split"] == split]
        stats = {"timestamps_processed":len(ts), "timestamps_with_candidates":sum(r["candidate_count"]>0 for r in ts),
                 "accepted_samples":len(ac), "time_gap":distribution([r["time_gap_ms"] for r in ts]),
                 "association_distance":distribution([r["association_distance"] for r in ts if r["association_distance"] is not None]),
                 "point_count":distribution([r["point_count"] for r in ac])}
        stats["rejected_no_candidate"] = sum(not r["candidate_count"] for r in ts)
        stats["rejected_timestamp_gap"] = sum(r["candidate_count"]>0 and r["time_gap_ms"]>config["timestamp_tolerance_ms"] for r in ts)
        stats["rejected_spatial_distance"] = sum(r["candidate_count"]>0 and r["time_gap_ms"]<=config["timestamp_tolerance_ms"] and r["association_distance"]>args.association_threshold_m for r in ts)
        summary[split] = stats
    summary["valid_empty_sequences"] = sorted({r["sequence_id"] for r in rows} - {r["sequence_id"] for r in candidates})
    summary["sequences_with_accepted_samples"] = sorted({r["sequence_id"] for r in accepted})
    (out / "frozen_config.json").write_text(json.dumps(config,indent=2))
    (out / "dataset_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
