#!/usr/bin/env python3
"""Collect existing logs, actual train/val samples and replayed B4 evidence without training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if keys: writer.writeheader(); writer.writerows(rows)


def epoch_rows(paths):
    result = []
    for path in paths:
        epochs = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                if not row.get("epoch"): continue
                values = epochs.setdefault(int(float(row["epoch"])), {})
                values.update({k: v for k, v in row.items() if v != ""})
        for epoch, values in sorted(epochs.items()):
            result.append(dict(csv_file=str(path), epoch=epoch, **{k:v for k,v in values.items() if k != "epoch"}))
    return result


def choose_indices(size, limit):
    if limit <= 0 or limit >= size: return list(range(size))
    return sorted(set(np.linspace(0, size-1, limit, dtype=int).tolist()))


def copy_provenance(run, output, config_path, checkpoint):
    import torch
    import yaml
    from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path
    snapshot = output / "provenance"; snapshot.mkdir()
    csvs = sorted((run / "logs/csv").glob("**/metrics.csv"))
    if not csvs: raise FileNotFoundError(f"No metrics.csv under {run / 'logs/csv'}")
    for path in csvs:
        dest = snapshot / "logs" / path.relative_to(run / "logs")
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    write_csv(output / "epoch_metrics.csv", epoch_rows(csvs))
    for name in ("effective_config.yaml", "effective_config.json"):
        if (run / name).is_file(): shutil.copy2(run / name, snapshot / name)
    shutil.copy2(config_path, snapshot / "diagnostic_config.yaml")
    cfg = apply_runtime_path_overrides(yaml.safe_load(config_path.read_text()))
    if cfg["experiment"]["stage"] != "B4": raise ValueError("Expected a B4 configuration")
    refs = [("initialization", "lidar_config"), ("initialization", "candidate_config"),
            ("data", "split_file"), ("data", "camera_config"), ("data", "geometry_calibration")]
    config_sources = {}
    for section, key in refs:
        path = resolve_project_path(cfg[section][key], ROOT)
        dest = snapshot / f"{section}_{key}{path.suffix}"
        shutil.copy2(path, dest); config_sources[f"{section}.{key}"] = str(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = {"path": str(checkpoint), "sha256": file_hash(checkpoint), "epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
            "size_bytes": checkpoint.stat().st_size,
            "callbacks": [{"key": str(key), "monitor": state.get("monitor"),
                           "best_model_score": float(state["best_model_score"]) if state.get("best_model_score") is not None else None,
                           "best_model_path": state.get("best_model_path")}
                          for key, state in payload.get("callbacks", {}).items() if isinstance(state, dict)]}
    stored = payload.get("hyper_parameters", {}).get("config")
    if stored is not None: (snapshot / "checkpoint_config.yaml").write_text(yaml.safe_dump(stored, sort_keys=False))
    branch = {key: value.detach().float().cpu().numpy() for key, value in payload.get("state_dict", {}).items()
              if key.startswith(("network.pair_fusion.", "network.vision.builder.feature_proj."))}
    if branch: np.savez_compressed(snapshot / "joint_branch_weights.npz", **branch)
    del payload
    inventory = [{"name": p.name, "size_bytes": p.stat().st_size} for p in sorted((run / "checkpoints").glob("*.ckpt")) if p.is_file()]
    write_json(snapshot / "checkpoint_metadata.json", meta)
    write_json(snapshot / "checkpoint_inventory.json", inventory)
    command = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    versions = {}
    for name in ("torch", "lightning", "torchvision", "scipy", "timm"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    write_json(snapshot / "environment.json", dict(code_commit=command.stdout.strip(), versions=versions,
               config_sources=config_sources, checkpoint=meta,
               existing_checkpoint_names=[p["name"] for p in inventory]))
    diff = subprocess.run(["git", "diff", "--", "src/rdq_uav/multimodal_v2", "experiments/multimodal_v2"],
                          cwd=ROOT, capture_output=True, text=True)
    (snapshot / "uncommitted_source.patch").write_text(diff.stdout)
    return cfg, meta["sha256"]


def sample_manifest(dataset, split, path):
    import torch
    from rdq_uav.multimodal_v2.data import load_3d_target, load_yolo_box
    from rdq_uav.multimodal_v2.radar_data import load_released_xyz
    @lru_cache(maxsize=64)
    def points(path): return load_released_xyz(path)[0].astype(np.float32)
    stats = []
    with path.open("w") as stream:
        for index, record in enumerate(dataset.query_records):
            target = load_3d_target(dataset.target_3d_index.get(record.query_time))
            match = dataset.image_index.match(record.sequence_id, record.query_time)
            label = dataset.root / record.sequence_id / dataset.label_directory / f"{match.path.stem}.txt" if match.path else None
            box = load_yolo_box(label, dataset.camera_wh) if label else None
            events = dataset.select_radar_events(record.sequence_id, record.query_time)
            counts = Counter(); minimum = None; near = 0
            for event in events:
                array = points(event.file_path); counts[event.sensor_name] += len(array)
                if target.valid and len(array):
                    distance = np.linalg.norm(array-target.xyz_m.numpy(), axis=1)
                    minimum = min(minimum, float(distance.min())) if minimum is not None else float(distance.min())
                    near += int((distance <= 1.).sum())
            row = dict(split=split, dataset_index=index, sample_id=record.sample_id, sequence_id=record.sequence_id,
                       query_time=record.query_time, image_time=match.image_time,
                       image_gap_s=-match.delta_t if match.valid else None,
                       image_path=str(match.path) if match.valid else None,
                       label_path=str(label) if label else None, point_count=sum(counts.values()),
                       sensor_point_counts=dict(counts), event_count=len(events),
                       event_files=[str(e.file_path) for e in events],
                       has_gt3d=target.valid, has_image=match.valid, has_gt2d=box is not None,
                       gt_xyz=target.xyz_m.tolist() if target.valid else None,
                       gt_range_m=float(torch.linalg.vector_norm(target.xyz_m)) if target.valid else None,
                       gt_box_xyxy_px=box.tolist() if box is not None else None,
                       gt_box_width_px=float(box[2]-box[0]) if box is not None else None,
                       gt_box_height_px=float(box[3]-box[1]) if box is not None else None,
                       input_point_min_gt_distance_m=minimum, input_points_within_1m=near)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stats.append({k:v for k,v in row.items() if k not in ("event_files", "gt_xyz", "gt_box_xyxy_px")})
            if (index+1) % 500 == 0: print(f"Manifest {split}: {index+1}/{len(dataset)}", flush=True)
    sequence_rows = []
    grouped = defaultdict(list)
    for row in stats: grouped[row["sequence_id"]].append(row)
    for sequence, rows in grouped.items():
        ranges = [r["gt_range_m"] for r in rows if r["gt_range_m"] is not None]
        widths = [r["gt_box_width_px"] for r in rows if r["gt_box_width_px"] is not None]
        gaps = [r["image_gap_s"] for r in rows if r["image_gap_s"] is not None]
        sequence_rows.append(dict(split=split, sequence_id=sequence, queries=len(rows),
            queries_with_gt3d=sum(r["has_gt3d"] for r in rows),
            queries_with_gt2d=sum(r["has_gt2d"] for r in rows),
            dual_gt_queries=sum(r["has_gt3d"] and r["has_gt2d"] for r in rows),
            unique_supervised_images=len({r["image_path"] for r in rows if r["has_gt2d"]}),
            queries_with_input_point_near_gt=sum(r["input_points_within_1m"] > 0 for r in rows),
            median_point_count=float(np.median([r["point_count"] for r in rows])),
            median_gt_range_m=float(np.median(ranges)) if ranges else None,
            median_box_width_px=float(np.median(widths)) if widths else None,
            median_image_gap_s=float(np.median(gaps)) if gaps else None))
    return sequence_rows


def summarize_results(directory):
    rows = [json.loads(line) for line in (directory / "per_query.jsonl").read_text().splitlines()]
    candidates = [json.loads(line) for line in (directory / "b4_candidates.jsonl").read_text().splitlines()]
    by_id = {r["sample_id"]: r for r in rows}
    failures = []
    for record in candidates:
        row = by_id[record["sample_id"]]
        top = next((p for p in record["evidence_pairs"] if p["rank_3d"] == 1), {})
        failures.append(dict(dataset_index=row["dataset_index"], sample_id=row["sample_id"],
            sequence_id=row["sequence_id"], image_path=row["image_path"],
            failure_category=record["failure_category"],
            final_error_m=row["final_error"], radar_error_m=row["radar_error"],
            radar_oracle_error_m=row["radar_oracle_error"],
            top3_pair_iou=top.get("iou"), top3_matching_cost=top.get("matching_cost"),
            positive_negative_margin=record["positive_negative_margin"],
            gt_range_m=float(np.linalg.norm(row["gt_xyz"])) if row["has_gt3d"] else None,
            correct_rv_pair_count=sum(p["eval_joint_correct"] for p in record["evidence_pairs"])))
    order = lambda row: row["final_error_m"] if row["final_error_m"] is not None else -1.
    write_csv(directory / "failures.csv", sorted(failures, key=order, reverse=True))
    groups = defaultdict(list)
    for row in failures: groups[row["sequence_id"]].append(row)
    stats = []
    for sequence, group in sorted(groups.items()):
        valid = [r for r in group if r["gt_range_m"] is not None]
        errors = [r["final_error_m"] for r in valid if r["final_error_m"] is not None]
        stats.append(dict(sequence_id=sequence, queries=len(group), gt3d_queries=len(valid),
            final_success_1m=sum(r["final_error_m"] is not None and r["final_error_m"] <= 1. for r in valid)/len(valid) if valid else None,
            rmse_3d_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            counts=dict(Counter(r["failure_category"] for r in group))))
    write_json(directory / "sequence_metrics.json", stats)
    write_json(directory / "failure_counts.json", dict(Counter(r["failure_category"] for r in failures)))
    return failures


def save_examples(dataset, indices, directory):
    from PIL import Image
    directory.mkdir()
    for index in indices:
        sample = dataset[index]
        dest = directory / f"query_{index:06d}"; dest.mkdir()
        points = sample["points"].numpy() if sample["m_R"] else np.empty((0, 3), dtype=np.float32)
        # Full points are retained; these files are never training inputs.
        np.savez_compressed(dest / "input_points.npz", points=points,
                            delta_t=sample["delta_t"].numpy() if sample["m_R"] else np.empty(0),
                            sensor_id=sample["sensor_id"].numpy() if sample["m_R"] else np.empty(0, dtype=int))
        if sample["m_V"]:
            with Image.open(sample["left_image_path"]) as image:
                w, h = sample["image_source_wh"].tolist()
                image.convert("RGB").crop((0, 0, w, h)).save(dest / "left_image.jpg", quality=95)
        write_json(dest / "sample.json", {"dataset_index": index, "sample_id": sample["sample_id"],
            "sequence_id": sample["sequence_id"], "query_time": sample["query_time"],
            "image_time": sample["image_time"], "image_path": sample["left_image_path"],
            "gt_xyz": sample["target_xyz"].tolist() if sample["target_valid"] else None,
            "gt_box_xyxy_px": sample["gt_box_xyxy_px"].tolist() if sample["gt_2d_valid"] else None,
            "point_count": len(points), "point_cloud_downsampled": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/b4_learnsbe_mtce_bs4_seed42"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true", help="Copy logs/config/checkpoint metadata without loading data or running inference")
    parser.add_argument("--train-limit", type=int, default=512, help="Evenly spaced train diagnosis subset; 0 evaluates all train queries")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    run, output = resolve(args.run), resolve(args.output)
    checkpoint = resolve(args.checkpoint) if args.checkpoint else run / "checkpoints/best_joint.ckpt"
    config_path = resolve(args.config) if args.config else run / "effective_config.yaml"
    if output.exists(): raise FileExistsError(f"Choose a new diagnostic directory: {output}")
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    if not config_path.is_file(): raise FileNotFoundError(config_path)
    if args.train_limit < 0 or args.examples < 0 or args.cpu_threads < 1: parser.error("invalid limits")
    import torch
    torch.set_num_threads(args.cpu_threads)
    output.mkdir(parents=True)
    cfg, checkpoint_hash = copy_provenance(run, output, config_path, checkpoint)
    write_json(output / "collection_status.json", {"complete": False, "checkpoint": str(checkpoint), "metadata_only": args.metadata_only})
    if not args.metadata_only:
        from rdq_uav.multimodal_v2.data import build_datasets
        train, val = build_datasets(cfg, ROOT)
        datasets = [(cfg["data"]["train_split"], train), (cfg["data"]["val_split"], val)]
        sequence_rows = []
        for split, dataset in datasets:
            sequence_rows.extend(sample_manifest(dataset, split, output / f"{split}_samples.jsonl"))
        write_csv(output / "dataset_sequences.csv", sequence_rows)
        for split, dataset in datasets:
            indices = choose_indices(len(dataset), args.train_limit) if split == cfg["data"]["train_split"] else list(range(len(dataset)))
            write_json(output / f"{split}_evaluation_indices.json", indices)
            command = [sys.executable, str(ROOT / "experiments/multimodal_v2/evaluate.py"),
                       "--config", str(config_path), "--mode", "B4", "--split", split,
                       "--checkpoint", str(checkpoint), "--output", str(output / split),
                       "--device", args.device, "--num-workers", str(args.num_workers), "--dump-candidates"]
            if len(indices) != len(dataset): command += ["--indices", *map(str, indices)]
            print("Evaluating", split, len(indices), "queries", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            failures = summarize_results(output / split)
            if split == cfg["data"]["val_split"] and args.examples:
                selected = []
                # Include both large 3D errors and oracle-present ranking failures.
                for category in ("wrong_pair_ranked_first", "different_task_winners", "no_correct_rv_pair", "joint_success"):
                    pool = sorted((r for r in failures if r["failure_category"] == category),
                                  key=lambda r: r["final_error_m"] if r["final_error_m"] is not None else -1., reverse=True)
                    selected += [r["dataset_index"] for r in pool[:max(1, args.examples // 4)]]
                selected = list(dict.fromkeys(selected))[:args.examples]
                if selected:
                    write_json(output / "example_indices.json", selected)
                    save_examples(dataset, selected, output / "examples")
                    subprocess.run([sys.executable, str(ROOT / "experiments/multimodal_v2/evaluate.py"),
                        "--config", str(config_path), "--mode", "B4", "--split", split,
                        "--checkpoint", str(checkpoint), "--output", str(output / "feature_examples"),
                        "--device", args.device, "--num-workers", "0", "--dump-features",
                        "--indices", *map(str, selected)], cwd=ROOT, check=True)
    if file_hash(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during collection; stop training and collect from a stable backup")
    write_json(output / "collection_status.json", {"complete": True, "checkpoint": str(checkpoint),
               "metadata_only": args.metadata_only, "train_subset_limit": args.train_limit,
               "heldout_test_evaluated": False, "full_detector_weights_included": False,
               "joint_branch_weights_included": (output / "provenance/joint_branch_weights.npz").exists()})
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists(): raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as f: f.add(output, arcname=output.name)
    print("Diagnostic package:", archive, "bytes:", archive.stat().st_size, flush=True)


if __name__ == "__main__": main()
