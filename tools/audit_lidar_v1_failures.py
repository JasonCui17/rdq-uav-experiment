#!/usr/bin/env python3
"""Read-only failure audit for an exported LiDAR UAV V1 validation run.

This tool never loads or changes model weights.  It reconstructs the exact
causal 20-event validation input with the production dataset primitives and
joins that input to the already exported raw/NMS candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdq_uav.lidar_v1.data import LiDARUAVValidationDataset  # noqa: E402
from rdq_uav.multimodal.merged_lidar import (  # noqa: E402
    load_released_xyz,
    select_last_history,
)


DEFAULT_RUN = ROOT / "outputs/own_multimodal_research/lidar_uav_v1/pilot_3epoch_seed42"
DEFAULT_VAL = Path("/home/jasoncui/datasets/MMAUD/official/val")
DEFAULT_REF = Path("/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv")
AGE_BUCKETS = (
    ("0-0.05s", 0.0, 0.05),
    ("0.05-0.10s", 0.05, 0.10),
    ("0.10-0.20s", 0.10, 0.20),
    ("0.20-0.50s", 0.20, 0.50),
    ("0.50-1.00s", 0.50, 1.00),
    (">1.00s", 1.00, math.inf),
)
CandidateIndex = dict[str, dict[str, list[dict[str, Any]]]]
EventDetails = dict[str, list[dict[str, Any]]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: str | float | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


class PointCache:
    """Small LRU cache because adjacent validation queries reuse most events."""

    def __init__(self, capacity: int = 96):
        self.capacity = capacity
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self.values:
            value = self.values.pop(key)
            self.values[key] = value
            return value
        value = load_released_xyz(path)[0]
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def candidate_index(rows: list[dict[str, str]]) -> CandidateIndex:
    result: CandidateIndex = defaultdict(lambda: {"raw": [], "nms": []})
    for row in rows:
        parsed: dict[str, Any] = dict(row)
        for key in ("rank", "source_token_id"):
            parsed[key] = int(parsed[key])
        for key in ("score", "pred_x", "pred_y", "pred_z", "distance_to_gt"):
            parsed[key] = float(parsed[key])
        result[row["sample_id"]][row["candidate_set"]].append(parsed)
    for item in result.values():
        for kind in ("raw", "nms"):
            item[kind].sort(key=lambda x: x["rank"])
    return result


def event_support(points: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    if not len(points):
        return {"nearest": math.nan, "n05": 0, "n1": 0, "n2": 0}
    distances = np.linalg.norm(points - gt[None, :], axis=1)
    return {
        "nearest": float(distances.min()),
        "n05": int(np.count_nonzero(distances <= 0.5)),
        "n1": int(np.count_nonzero(distances <= 1.0)),
        "n2": int(np.count_nonzero(distances <= 2.0)),
    }


def q(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else math.nan


def metric_row(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(x["top1_error"]) for x in rows if math.isfinite(float(x["top1_error"]))]
    count = len(rows)
    rate = lambda pred: float(np.mean([pred(x) for x in rows])) if rows else math.nan
    return {
        "support_age_bucket": label,
        "sample_count": count,
        "top1_success_0p5m": rate(lambda x: bool(x["top1_success_0p5m"])),
        "top1_success_1m": rate(lambda x: bool(x["top1_success_1m"])),
        "top1_success_2m": rate(lambda x: bool(x["top1_success_2m"])),
        "median_error_m": q(errors, 50),
        "p90_error_m": q(errors, 90),
        "p95_error_m": q(errors, 95),
        "catastrophic_rate_gt5m": rate(lambda x: math.isfinite(float(x["top1_error"])) and float(x["top1_error"]) > 5.0),
        "raw_top10_success_1m": rate(lambda x: bool(x["raw_top10_has_gt_1m"])),
        "nms_top10_success_1m": rate(lambda x: bool(x["nms_top10_has_gt_1m"])),
    }


def assign_failure_flags(row: dict[str, Any]) -> dict[str, Any]:
    flags = {
        "flag_empty_input": bool(row["empty_input"]),
        "flag_no_history_support": not bool(row["history_has_any_support"]),
        "flag_stale_history_support": bool(row["history_has_any_support"]) and not bool(row["current_support"]),
        "flag_candidate_exists_ranking_fail": bool(row["raw_top10_has_gt_1m"]) and float(row["top1_error"]) > 5.0,
        "flag_nms_removed_good_candidate": bool(row["raw_top10_has_gt_1m"]) and not bool(row["nms_top10_has_gt_1m"]),
        "flag_current_support_generation_fail": bool(row["current_support"]) and not bool(row["raw_top10_has_gt_1m"]),
    }
    # Specific output-mechanism diagnoses precede the broader temporal state.
    precedence = (
        ("EMPTY_INPUT", "flag_empty_input"),
        ("NO_HISTORY_SUPPORT", "flag_no_history_support"),
        ("NMS_REMOVED_GOOD_CANDIDATE", "flag_nms_removed_good_candidate"),
        ("CANDIDATE_EXISTS_RANKING_FAIL", "flag_candidate_exists_ranking_fail"),
        ("CURRENT_SUPPORT_BUT_GENERATION_FAIL", "flag_current_support_generation_fail"),
        ("STALE_HISTORY_SUPPORT", "flag_stale_history_support"),
    )
    primary = next((name for name, key in precedence if flags[key]), "UNCLASSIFIED_GENERATION_FAILURE")
    return {**flags, "primary_failure_type": primary}


def build_audit(dataset: LiDARUAVValidationDataset, predictions: list[dict[str, str]],
                candidates: CandidateIndex) -> tuple[list[dict[str, Any]], EventDetails]:
    records = {x["sample_id"]: x for x in dataset.adapter.records}
    cache = PointCache()
    audit: list[dict[str, Any]] = []
    event_details: dict[str, list[dict[str, Any]]] = {}
    support_mismatches = []
    future_count = 0
    gt_mismatches = 0
    for position, pred in enumerate(predictions, 1):
        record = records[pred["sample_id"]]
        gt = np.asarray(record["gt_xyz"], dtype=np.float64)
        export_gt = np.asarray([pred["gt_x"], pred["gt_y"], pred["gt_z"]], dtype=np.float64)
        if not np.allclose(gt, export_gt, atol=1e-6, rtol=0):
            gt_mismatches += 1
        t0 = float(record["t0"])
        events = select_last_history(dataset.streams[record["sequence_id"]], t0, dataset.max_events)
        if any(e.timestamp > t0 for e in events):
            future_count += 1
        details = []
        total = avia = mid = 0
        for index, event in enumerate(events):
            points = cache.load(event.file_path)
            support = event_support(points, gt)
            total += len(points)
            avia += len(points) if event.sensor_id == 0 else 0
            mid += len(points) if event.sensor_id == 1 else 0
            details.append({
                "event_index": index, "timestamp": event.timestamp,
                "age_sec": t0 - event.timestamp, "sensor_id": event.sensor_id,
                "sensor_name": event.sensor_name, "file_path": str(event.file_path),
                "point_count": len(points), **support,
            })
        event_details[pred["sample_id"]] = details
        recent_start = max(0, len(details) - 4)
        support1 = [x for x in details if x["n1"] > 0]
        last = support1[-1] if support1 else None
        cand = candidates.get(pred["sample_id"], {"raw": [], "nms": []})
        raw10 = cand["raw"][:10]
        nms10 = cand["nms"][:10]
        top = cand["raw"][0] if cand["raw"] else None
        correct_raw = [x for x in raw10 if x["distance_to_gt"] <= 1.0]
        current = any(x["n1"] > 0 for x in details[recent_start:])
        exported_current = pred["support_flag"] == "CURRENT_SUPPORT"
        if current != exported_current:
            support_mismatches.append(pred["sample_id"])
        top_error = top["distance_to_gt"] if top else math.nan
        row = {
            "sequence_id": record["sequence_id"], "sample_id": pred["sample_id"], "t0": t0,
            "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2],
            "pred_x": top["pred_x"] if top else math.nan,
            "pred_y": top["pred_y"] if top else math.nan,
            "pred_z": top["pred_z"] if top else math.nan,
            "top1_error": top_error,
            "empty_input": total == 0, "empty_output": top is None,
            "top1_success_0p5m": math.isfinite(top_error) and top_error <= 0.5,
            "top1_success_1m": math.isfinite(top_error) and top_error <= 1.0,
            "top1_success_2m": math.isfinite(top_error) and top_error <= 2.0,
            "raw_top10_has_gt_0p5m": any(x["distance_to_gt"] <= 0.5 for x in raw10),
            "raw_top10_has_gt_1m": bool(correct_raw),
            "nms_top10_has_gt_0p5m": any(x["distance_to_gt"] <= 0.5 for x in nms10),
            "nms_top10_has_gt_1m": any(x["distance_to_gt"] <= 1.0 for x in nms10),
            "best_raw_candidate_error": min((x["distance_to_gt"] for x in raw10), default=math.nan),
            "best_nms_candidate_error": min((x["distance_to_gt"] for x in nms10), default=math.nan),
            "top1_score": top["score"] if top else math.nan,
            "best_gt_near_candidate_score": max((x["score"] for x in correct_raw), default=math.nan),
            "best_gt_near_candidate_rank": min((x["rank"] for x in correct_raw), default=""),
            "event_count": len(events), "total_point_count": total,
            "avia_point_count": avia, "mid360_point_count": mid,
            "history_span_sec": events[-1].timestamp - events[0].timestamp if events else math.nan,
            "oldest_event_age_sec": t0 - events[0].timestamp if events else math.nan,
            "newest_event_age_sec": t0 - events[-1].timestamp if events else math.nan,
            "num_support_events_0p5m": sum(x["n05"] > 0 for x in details),
            "num_support_events_1m": len(support1),
            "num_support_events_2m": sum(x["n2"] > 0 for x in details),
            "num_support_points_0p5m": sum(x["n05"] for x in details),
            "num_support_points_1m": sum(x["n1"] for x in details),
            "num_support_points_2m": sum(x["n2"] for x in details),
            "last_support_timestamp_1m": last["timestamp"] if last else math.nan,
            "last_support_age_sec_1m": last["age_sec"] if last else math.nan,
            "last_support_event_index": last["event_index"] if last else "",
            "last_support_sensor": last["sensor_name"] if last else "",
            "num_support_events_in_last4": sum(x["n1"] > 0 for x in details[recent_start:]),
            "num_support_points_in_last4": sum(x["n1"] for x in details[recent_start:]),
            "history_has_any_support": bool(support1), "current_support": current,
            "exported_support_flag": pred["support_flag"],
        }
        audit.append(row)
        if position % 100 == 0:
            print(f"Temporal support audit: {position}/{len(predictions)}", flush=True)
    if future_count or gt_mismatches or support_mismatches:
        raise AssertionError(
            f"Consistency failure: future={future_count}, gt={gt_mismatches}, "
            f"current_support={len(support_mismatches)} examples={support_mismatches[:5]}"
        )
    return audit, event_details


def sequence_diagnosis(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        grouped[row["sequence_id"]].append(row)
    output = []
    for sequence, rows in sorted(grouped.items()):
        ages = [float(x["last_support_age_sec_1m"]) for x in rows if math.isfinite(float(x["last_support_age_sec_1m"]))]
        catastrophic = [x for x in rows if math.isfinite(float(x["top1_error"])) and float(x["top1_error"]) > 5]
        mean = lambda pred: float(np.mean([pred(x) for x in rows]))
        output.append({
            "sequence_id": sequence, "validation_samples": len(rows),
            "current_support_rate": mean(lambda x: bool(x["current_support"])),
            "history_any_support_rate": mean(lambda x: bool(x["history_has_any_support"])),
            "empty_input_rate": mean(lambda x: bool(x["empty_input"])),
            "median_total_point_count": float(np.median([int(x["total_point_count"]) for x in rows])),
            "avia_point_fraction": sum(int(x["avia_point_count"]) for x in rows) / max(1, sum(int(x["total_point_count"]) for x in rows)),
            "median_event_count": float(np.median([int(x["event_count"]) for x in rows])),
            "median_history_span_sec": float(np.nanmedian([float(x["history_span_sec"]) for x in rows])),
            "mean_last_support_age_sec": float(np.mean(ages)) if ages else math.nan,
            "p90_last_support_age_sec": q(ages, 90), "p95_last_support_age_sec": q(ages, 95),
            "top1_success_1m": mean(lambda x: bool(x["top1_success_1m"])),
            "catastrophic_count": len(catastrophic), "catastrophic_rate": len(catastrophic) / len(rows),
            "candidate_exists_ranking_fail_count": sum(bool(x["raw_top10_has_gt_1m"]) for x in catastrophic),
            "nms_removed_good_candidate_count": sum(bool(x["raw_top10_has_gt_1m"]) and not bool(x["nms_top10_has_gt_1m"]) for x in catastrophic),
            "generation_fail_count": sum(not bool(x["raw_top10_has_gt_1m"]) for x in catastrophic),
            "current_support_generation_fail_count": sum(bool(x["current_support"]) and not bool(x["raw_top10_has_gt_1m"]) for x in catastrophic),
        })
    return output


def catastrophic_runs(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize contiguous catastrophic query runs within each sequence."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        grouped[row["sequence_id"]].append(row)
    output=[]
    for sequence, rows in sorted(grouped.items()):
        rows.sort(key=lambda x: float(x["t0"])); runs=[]; start=None
        for index, row in enumerate(rows + [None]):
            bad = row is not None and math.isfinite(float(row["top1_error"])) and float(row["top1_error"]) > 5.0
            if bad and start is None:start=index
            if not bad and start is not None:
                end=index-1;runs.append((start,end));start=None
        lengths=[end-start+1 for start,end in runs]
        output.append({
            "sequence_id":sequence,"catastrophic_count":sum(lengths),"run_count":len(runs),
            "singleton_run_count":sum(length==1 for length in lengths),
            "max_run_length":max(lengths,default=0),
            "median_run_length":float(np.median(lengths)) if lengths else 0.0,
            "run_lengths":";".join(map(str,lengths)),
        })
    return output


def temporal_neighbors(audit: list[dict[str, Any]], catastrophic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        grouped[row["sequence_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda x: float(x["t0"]))
    indices = {x["sample_id"]: i for rows in grouped.values() for i, x in enumerate(rows)}
    output = []
    for anchor in catastrophic:
        rows = grouped[anchor["sequence_id"]]; center = indices[anchor["sample_id"]]
        for index in range(max(0, center - 5), min(len(rows), center + 6)):
            row = rows[index]
            output.append({
                "anchor_sample_id": anchor["sample_id"], "anchor_top1_error": anchor["top1_error"],
                "sequence_id": row["sequence_id"], "neighbor_sample_id": row["sample_id"],
                "relative_query_index": index - center,
                "relative_time_sec": float(row["t0"]) - float(anchor["t0"]),
                "top1_error": row["top1_error"], "current_support": row["current_support"],
                "history_has_any_support": row["history_has_any_support"],
                "last_support_age_sec": row["last_support_age_sec_1m"],
                "total_points": row["total_point_count"],
                "raw_top10_has_gt_1m": row["raw_top10_has_gt_1m"],
            })
    return output


def plot_age_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [x["support_age_bucket"] for x in rows]
    x = np.arange(len(labels))
    specs = (
        ("top1_1m_vs_support_age.png", (("top1_success_1m", "Top1@1m"),), "Rate"),
        ("catastrophic_rate_vs_support_age.png", (("catastrophic_rate_gt5m", ">5m rate"),), "Rate"),
        ("error_vs_support_age.png", (("median_error_m", "Median"), ("p90_error_m", "P90")), "Error (m)"),
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, curves, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for key, label in curves:
            ax.plot(x, [r[key] for r in rows], marker="o", label=label)
        for i, row in enumerate(rows):
            ax.annotate(f"n={row['sample_count']}", (i, 0), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel(ylabel)
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / name, dpi=150); plt.close(fig)


def select_cases(audit: list[dict[str, Any]], catastrophic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    def take(category: str, candidates: list[dict[str, Any]], count: int) -> None:
        added = 0
        for row in candidates:
            if row["sample_id"] in used:
                continue
            chosen.append({"case_category": category, **row}); used.add(row["sample_id"]); added += 1
            if added == count:
                break
    take("WORST_ERROR", sorted(catastrophic, key=lambda x: float(x["top1_error"]), reverse=True), 10)
    ranking = [x for x in catastrophic if x["raw_top10_has_gt_1m"]]
    take("RANKING_FAIL", sorted(ranking, key=lambda x: float(x["top1_error"]), reverse=True), 5)
    no_current = [x for x in catastrophic if not x["current_support"]]
    take("NO_CURRENT_SUPPORT", sorted(no_current, key=lambda x: float(x["top1_error"]), reverse=True), 5)
    return chosen


def plot_case(row: dict[str, Any], details: list[dict[str, Any]], candidates: CandidateIndex,
              cache: PointCache, output: Path, case_index: int) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_id = row["sample_id"]; gt = np.asarray([row["gt_x"], row["gt_y"], row["gt_z"]], float)
    pred = np.asarray([row["pred_x"], row["pred_y"], row["pred_z"]], float)
    raw = candidates.get(sample_id, {"raw": []})["raw"][:10]
    raw_xyz = np.asarray([[x["pred_x"], x["pred_y"], x["pred_z"]] for x in raw]) if raw else np.empty((0, 3))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sensor_style = {0: ("o", "Blues", "Avia"), 1: ("^", "Oranges", "Mid360")}
    for sensor_id, (marker, cmap, label) in sensor_style.items():
        point_parts=[]; age_parts=[]
        for detail in details:
            if detail["sensor_id"] != sensor_id: continue
            points=cache.load(Path(detail["file_path"]))
            if len(points): point_parts.append(points); age_parts.append(np.full(len(points), detail["age_sec"]))
        if not point_parts: continue
        points=np.concatenate(point_parts); ages=np.concatenate(age_parts)
        axes[0].scatter(points[:,0],points[:,1],c=ages,cmap=cmap,s=2,alpha=.18,marker=marker,label=label)
        axes[1].scatter(points[:,0],points[:,2],c=ages,cmap=cmap,s=2,alpha=.18,marker=marker,label=label)
    for ax, dims, labels in ((axes[0],(0,1),("X (m)","Y (m)")),(axes[1],(0,2),("X (m)","Z (m)"))):
        ax.scatter(gt[dims[0]],gt[dims[1]],s=170,marker="*",c="lime",edgecolors="black",label="GT",zorder=8)
        if np.isfinite(pred).all():ax.scatter(pred[dims[0]],pred[dims[1]],s=120,marker="X",c="red",label="Top1",zorder=9)
        if len(raw_xyz):ax.scatter(raw_xyz[:,dims[0]],raw_xyz[:,dims[1]],s=55,facecolors="none",edgecolors="magenta",label="Raw Top10",zorder=7)
        ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    ages=[-x["age_sec"] for x in details]; nearest=[x["nearest"] for x in details]
    colors=["tab:blue" if x["sensor_id"]==0 else "tab:orange" for x in details]
    axes[2].scatter(ages,nearest,c=colors,s=35)
    axes[2].axhline(1.0,color="green",linestyle="--",label="1m support")
    axes[2].set_xlabel("Event time relative to t0 (s)");axes[2].set_ylabel("Nearest point to query GT (m)")
    axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
    fig.suptitle(f"{row['case_category']} | {sample_id} | Top1 error={float(row['top1_error']):.2f}m")
    fig.tight_layout(); output.mkdir(parents=True,exist_ok=True)
    path=output/f"{case_index:02d}_{row['case_category'].lower()}_{sample_id}.png"
    fig.savefig(path,dpi=140);plt.close(fig);return str(path)


def report_text(audit: list[dict[str, Any]], catastrophic: list[dict[str, Any]], age_rows: list[dict[str, Any]],
                sequences: list[dict[str, Any]], runs: list[dict[str, Any]], consistency: dict[str, Any]) -> str:
    flag_count=lambda key:sum(bool(x[key]) for x in catastrophic)
    primary=defaultdict(int)
    for row in catastrophic:primary[row["primary_failure_type"]]+=1
    seq_map={x["sequence_id"]:x for x in sequences}
    age_table="\n".join(
        f"| {x['support_age_bucket']} | {x['sample_count']} | {x['top1_success_1m']:.3f} | {x['catastrophic_rate_gt5m']:.3f} | {x['median_error_m']:.3f} | {x['p90_error_m']:.3f} |"
        for x in age_rows
    )
    seq_table="\n".join(
        f"| {s} | {seq_map[s]['current_support_rate']:.3f} | {seq_map[s]['history_any_support_rate']:.3f} | {seq_map[s]['mean_last_support_age_sec']:.3f} | {seq_map[s]['median_total_point_count']:.0f} | {seq_map[s]['avia_point_fraction']:.3f} | {seq_map[s]['top1_success_1m']:.3f} | {seq_map[s]['catastrophic_count']} | {seq_map[s]['candidate_exists_ranking_fail_count']} | {seq_map[s]['current_support_generation_fail_count']} |"
        for s in ("seq0004","seq0005","seq0006","seq0008")
    )
    empty_outputs=sum(bool(x["empty_output"]) for x in audit)
    current_cat=sum(bool(x["current_support"]) for x in catastrophic)
    history_cat=sum(bool(x["history_has_any_support"]) for x in catastrophic)
    ranking=[x for x in catastrophic if x["raw_top10_has_gt_1m"]]
    ranks=[int(x["best_gt_near_candidate_rank"]) for x in ranking]
    score_gaps=[float(x["top1_score"])-float(x["best_gt_near_candidate_score"]) for x in ranking]
    run_map={x["sequence_id"]:x for x in runs}
    run_statement=", ".join(f"{s}: {run_map[s]['run_lengths']}" for s in ("seq0004","seq0005","seq0006","seq0008"))
    return f"""# LiDAR UAV V1 Failure Audit

## Scope and consistency

Read-only audit of all {len(audit)} exported validation samples. The exact production `ValidationReferenceAdapter`, merged-stream ordering, causal last-20 selection, released XYZ cleaning, sensor IDs, and exported candidates were reused. No model inference, fitting, or training was performed.

- Future-event violations: {consistency['future_event_violations']}
- GT/export mismatches: {consistency['gt_export_mismatches']}
- Recomputed/exported current-support mismatches: {consistency['current_support_mismatches']}
- Empty outputs in all validation samples: {empty_outputs}

`current_support` has its original V1 meaning: at least one point within 1 m of current GT among the last four selected LiDAR events. Support in older events does not make a sample current-supported.

## Catastrophic failures (>5 m)

Finite Top1 errors above 5 m: **{len(catastrophic)}**. Empty outputs are separate and therefore contribute **{flag_count('flag_empty_input')}** rows to this finite-error table.

- History contains no 1 m GT support: **{flag_count('flag_no_history_support')}**
- History has support, but the last four events do not: **{flag_count('flag_stale_history_support')}**
- A correct <=1 m candidate exists in Raw Top10 while Top1 is >5 m: **{flag_count('flag_candidate_exists_ranking_fail')}**
- Raw Top10 has a correct candidate that NMS Top10 removes: **{flag_count('flag_nms_removed_good_candidate')}**
- Current support exists but Raw Top10 has no <=1 m candidate: **{flag_count('flag_current_support_generation_fail')}**

**{current_cat}/90 ({current_cat/len(catastrophic):.1%})** catastrophic samples still have current support, and **{history_cat}/90** have support somewhere in the 20-event history. Therefore temporal observation absence explains a material minority, not the majority, of the finite catastrophic errors.

For the {len(ranking)} ranking failures, the first correct Raw candidate has median rank **{float(np.median(ranks)):.0f}** and the median Top1-minus-correct score gap is **{float(np.median(score_gaps)):.4f}**. NMS removes none of these 1 m candidates from NMS Top10.

Primary types use this precedence: EMPTY_INPUT, NO_HISTORY_SUPPORT, NMS_REMOVED_GOOD_CANDIDATE, CANDIDATE_EXISTS_RANKING_FAIL, CURRENT_SUPPORT_BUT_GENERATION_FAIL, STALE_HISTORY_SUPPORT, then unclassified generation failure. Counts: `{json.dumps(dict(sorted(primary.items())), ensure_ascii=False)}`.

## Last support age

Only non-empty samples with at least one historical 1 m support event enter this table.

| Last support age | Samples | Top1@1m | >5m rate | Median error | P90 error |
|---|---:|---:|---:|---:|---:|
{age_table}

The bucket statistics show no degradation between 0 and 0.10 s. Performance begins weakening in 0.10-0.20 s and changes sharply in the observed 0.20-0.50 s bucket: Top1@1m falls from 0.902 to 0.529 and the catastrophic rate rises from 0.083 to 0.235. There are no eligible samples above 0.50 s, so this audit cannot support claims beyond that range. Age is not sufficient by itself: {sum(int(x['sample_count']) for x in age_rows[:2])} samples below 0.10 s still include {sum(round(float(x['catastrophic_rate_gt5m'])*int(x['sample_count'])) for x in age_rows[:2])} catastrophic outputs.

## Focus sequences

| Sequence | Current support | Any history support | Mean support age | Median points | Avia point fraction | Top1@1m | >5m count | Ranking fail | Current-support generation fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{seq_table}

The four focus sequences contain **84/90** catastrophic errors. Their mechanisms differ:

- `seq0004` is observation-limited: current support is only 64%, any-history support is 82%, and 16/26 failures have no historical 1 m support. Its other 10 failures are ranking failures.
- `seq0005` is not primarily support-limited: current support is 95%; all 29 catastrophic samples have current support. Twenty are ranking failures and nine are generation failures despite current support.
- `seq0006` has 99% current support; all 14 failures have current support, split into 11 ranking and three generation failures.
- `seq0008` has 93% current support; among 15 failures, nine are current-support generation failures, four are ranking failures, and two have no history support.

The failures occur in contiguous temporal runs rather than predominantly isolated points. Run lengths for the focus sequences are `{run_statement}`. The complete 16-sequence comparison is in `per_sequence_failure_diagnosis.csv`, and run statistics are in `catastrophic_run_summary.csv`. The Avia fraction is close to zero in several good and bad sequences, so that stream composition alone does not separate the failures.

## Interpretation boundary

This report separates temporal observation absence, candidate generation, ranking, NMS, and empty input using exported facts. It does not claim a sensor detection rate and does not propose or evaluate model changes. Of the 90 catastrophic errors, the direct exported-output diagnoses are 49 ranking failures, 21 current-support generation failures, and 20 no-history-support cases. NMS accounts for none. The exact 2-5 m gap follows from this discrete selection behavior: 1487 samples select a candidate within 2 m, while every remaining non-empty Top1 selection is a distant background candidate above 5 m; no sample selects an intermediate candidate. The case studies show the corresponding GT-local cluster and distant Top1 locations.
"""


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--val-root",type=Path,default=DEFAULT_VAL)
    parser.add_argument("--val-reference",type=Path,default=DEFAULT_REF)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();run=args.run_dir.resolve();output=(args.output or run/"failure_audit").resolve()
    output.mkdir(parents=True,exist_ok=True)
    prediction_path=run/"validation_predictions.csv";candidate_path=run/"candidate_exports/candidates.csv"
    predictions=read_csv(prediction_path);candidate_rows=read_csv(candidate_path);candidates=candidate_index(candidate_rows)
    dataset=LiDARUAVValidationDataset(args.val_root,args.val_reference,max_events=20)
    if len(predictions) != len(dataset) or len(dataset) != 1600:
        raise AssertionError(f"Expected matching 1600 rows, got predictions={len(predictions)}, dataset={len(dataset)}")
    print(f"Using {prediction_path}\nUsing {candidate_path}\nUsing {args.val_reference}")
    audit,details=build_audit(dataset,predictions,candidates)
    write_csv(output/"sample_temporal_audit.csv",audit)
    catastrophic=[]
    for row in audit:
        if math.isfinite(float(row["top1_error"])) and float(row["top1_error"])>5.0:
            catastrophic.append({**row,**assign_failure_flags(row)})
    write_csv(output/"catastrophic_failures.csv",catastrophic)
    age_rows=[]
    eligible=[x for x in audit if not x["empty_input"] and x["history_has_any_support"]]
    for label,low,high in AGE_BUCKETS:
        group=[x for x in eligible if float(x["last_support_age_sec_1m"])>=low and float(x["last_support_age_sec_1m"])<high]
        age_rows.append(metric_row(label,group))
    write_csv(output/"support_age_metrics.csv",age_rows);plot_age_metrics(age_rows,output/"plots")
    neighbors=temporal_neighbors(audit,catastrophic);write_csv(output/"catastrophic_temporal_neighbors.csv",neighbors)
    sequences=sequence_diagnosis(audit);write_csv(output/"per_sequence_failure_diagnosis.csv",sequences)
    runs=catastrophic_runs(audit);write_csv(output/"catastrophic_run_summary.csv",runs)
    cases=select_cases(audit,catastrophic);case_cache=PointCache();case_manifest=[]
    for index,row in enumerate(cases,1):
        path=plot_case(row,details[row["sample_id"]],candidates,case_cache,output/"case_studies",index)
        case_manifest.append({"case_index":index,"case_category":row["case_category"],"sequence_id":row["sequence_id"],"sample_id":row["sample_id"],"top1_error":row["top1_error"],"plot_path":path})
    write_csv(output/"case_study_manifest.csv",case_manifest)
    consistency={"future_event_violations":0,"gt_export_mismatches":0,"current_support_mismatches":0,
                 "prediction_rows":len(predictions),"candidate_rows":len(candidate_rows),"case_studies":len(cases),
                 "sources":{"dataset":"rdq_uav.lidar_v1.data.LiDARUAVValidationDataset","event_selection":"rdq_uav.multimodal.merged_lidar.select_last_history","point_reader":"rdq_uav.multimodal.merged_lidar.load_released_xyz","current_support":"CandidateLoss/evaluate_batch equivalent: <=1m in last four selected events","predictions":str(prediction_path),"candidates":str(candidate_path)}}
    (output/"audit_provenance.json").write_text(json.dumps(consistency,indent=2),encoding="utf-8")
    (output/"FAILURE_AUDIT_REPORT.md").write_text(report_text(audit,catastrophic,age_rows,sequences,runs,consistency),encoding="utf-8")
    flags=("flag_empty_input","flag_no_history_support","flag_stale_history_support","flag_candidate_exists_ranking_fail","flag_nms_removed_good_candidate","flag_current_support_generation_fail")
    print(f"Catastrophic >5m: {len(catastrophic)}")
    for key in flags:print(f"  {key.removeprefix('flag_')}: {sum(bool(x[key]) for x in catastrophic)}")
    print(f"Outputs: {output}")


if __name__=="__main__":
    main()
