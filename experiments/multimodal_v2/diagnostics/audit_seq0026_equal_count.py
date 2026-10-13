#!/usr/bin/env python3
"""Compare real T centroid rates with 100 equal-bin-count B resamples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_seq0026_centroid_motion import (
    TIMES, SavedQuery, bin_indices, one_query, rows, write_csv,
)


def original_value(row, field):
    value = row[field]
    return None if value == "" else float(value)


def same_original(actual, saved, fields):
    for field in fields:
        a = actual[field]
        b = saved[field]
        if a == "" or b == "":
            if a != b:
                raise AssertionError(f"prior centroid field changed: {field}: {a!r}, {b!r}")
        elif not np.isclose(float(a), float(b), atol=1e-10, rtol=1e-10):
            raise AssertionError(f"prior centroid field changed: {field}: {a!r}, {b!r}")


def rates(centroids, mean_times):
    values = []
    for k in range(4):
        a, b = centroids[k], centroids[k+1]
        if a is None or b is None:
            values.append(None)
            continue
        gap = mean_times[k+1]-mean_times[k]
        if gap <= 0:
            raise AssertionError("nonpositive actual mean-time gap")
        value = float(np.linalg.norm(b-a)/gap)
        if not np.isfinite(value) or value < 0:
            raise AssertionError("invalid sampled centroid rate")
        values.append(value)
    return values


def distribution(values):
    finite = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(finite):
        return {"valid_replicates": 0, "mean": "", "std": "", "p05": "", "median": "", "p95": ""}
    return {"valid_replicates": len(finite), "mean": float(finite.mean()),
            "std": float(finite.std()), "p05": float(np.quantile(finite, .05)),
            "median": float(np.median(finite)), "p95": float(np.quantile(finite, .95))}


def analyze(audit_root, motion_root, repeats, seed):
    candidates = {(r["time"], r["kind"]): r for r in rows(audit_root / "candidate_comparison.csv")}
    prior_geometry = {(r["time"], r["kind"]): r for r in rows(audit_root / "raw_geometry_stats.csv")}
    prior_bins = {(r["query_time"], r["candidate_type"], r["layer"], int(r["bin_index"])): r
                  for r in rows(motion_root / "centroid_by_bin.csv")}
    prior_velocity = {(r["query_time"], r["candidate_type"], r["layer"]): r
                      for r in rows(motion_root / "centroid_velocity.csv")}
    with np.load(audit_root / "feature_vectors.npz", allow_pickle=False) as archive:
        identities = {name: archive[name] for name in archive.files if
                      name.endswith(("_point_ids", "_token_id", "_coord"))}
    repeat_rows, comparison = [], []
    for query_index, time in enumerate(TIMES):
        # Recompute the archived physical statistics before sampling. This also
        # validates source tokens, parents, bin coverage and point set inclusion.
        current_bins, current_velocity, _ = one_query(audit_root, time, candidates, prior_geometry, identities)
        for row in current_bins:
            old = prior_bins[(time, row["candidate_type"], row["layer"], row["bin_index"])]
            same_original(row, old, ("point_count", "avia_count", "mid360_count",
                                     "centroid_x", "centroid_y", "centroid_z", "mean_delta_t_s"))
        for row in current_velocity:
            old = prior_velocity[(time, row["candidate_type"], row["layer"])]
            same_original(row, old, ("total_point_count", "valid_time_bin_count",
                                     "valid_adjacent_pair_count", "mean_velocity_mps", "max_velocity_mps"))
            for k in range(4):
                same_original(row, old, (f"velocity_bin{k}_to_bin{k+1}_mps",))
        query = SavedQuery(audit_root, time)
        points = query.a("sample.points")
        delta_t = query.a("sample.delta_t")
        bins = bin_indices(delta_t)
        point_l0 = query.a("mapping.point_to_l0")
        parent01 = query.a("mapping.parent_l0_to_l1")
        parent12 = query.a("mapping.parent_l1_to_l2")
        layer_ids = (point_l0, parent01[point_l0], parent12[parent01[point_l0]])
        for level in range(3):
            layer = f"L{level}"
            t = candidates[(time, "T")]
            b = candidates[(time, "B")]
            if t["status"] != "matched" or b["status"] != "matched":
                raise ValueError("missing original T/B identity")
            token_t = int(t["source_token_id"])
            token_b = int(b["source_token_id"])
            target_token = (token_t, int(parent01[token_t]), int(parent12[parent01[token_t]]))[level]
            background_token = (token_b, int(parent01[token_b]), int(parent12[parent01[token_b]]))[level]
            t_ids = [np.flatnonzero((layer_ids[level] == target_token) & (bins == k)) for k in range(5)]
            b_ids = [np.flatnonzero((layer_ids[level] == background_token) & (bins == k)) for k in range(5)]
            t_counts = [len(ids) for ids in t_ids]
            b_counts = [len(ids) for ids in b_ids]
            if any(len(t_ids[k]) != int(prior_bins[(time, "T", layer, k)]["point_count"]) or
                   len(b_ids[k]) != int(prior_bins[(time, "B", layer, k)]["point_count"]) for k in range(5)):
                raise AssertionError("saved T/B per-bin point counts changed")
            if any(t_count > b_count for t_count, b_count in zip(t_counts, b_counts)):
                raise ValueError(f"B has fewer points than T in {time} {layer}; cannot sample without replacement")
            random = np.random.default_rng(np.random.SeedSequence([seed, query_index, level]))
            for replicate in range(repeats):
                chosen = [random.choice(b_ids[k], size=t_counts[k], replace=False) if t_counts[k] else np.empty(0, dtype=np.int64)
                          for k in range(5)]
                centroids = [points[ids].astype(np.float64).mean(axis=0) if len(ids) else None for ids in chosen]
                mean_times = [float(delta_t[ids].astype(np.float64).mean()) if len(ids) else None for ids in chosen]
                for k, ids in enumerate(chosen):
                    if len(ids) != t_counts[k] or len(np.unique(ids)) != len(ids) or not np.isin(ids, b_ids[k]).all():
                        raise AssertionError("sampled B IDs violate count, replacement or bin identity")
                velocity = rates(centroids, mean_times)
                valid = [v for v in velocity if v is not None]
                original_t = prior_velocity[(time, "T", layer)]
                if len(valid) != int(original_t["valid_adjacent_pair_count"]):
                    raise AssertionError("equal-count B has different valid adjacent pairs than T")
                row = {"query_time": time, "layer": layer,
                       "t_source_l0_token_id": token_t, "b_source_l0_token_id": token_b,
                       "t_layer_token_id": target_token, "b_layer_token_id": background_token,
                       "seed": seed, "replicate": replicate,
                       "t_bin_counts": json.dumps(t_counts), "original_b_bin_counts": json.dumps(b_counts),
                       "valid_adjacent_pair_count": len(valid),
                       "mean_velocity_mps": "" if not valid else float(np.mean(valid)),
                       "max_velocity_mps": "" if not valid else float(np.max(valid))}
                for k in range(5):
                    row[f"sampled_b_point_ids_bin{k}"] = json.dumps(chosen[k].tolist())
                    row[f"sampled_b_mean_delta_t_bin{k}_s"] = "" if mean_times[k] is None else mean_times[k]
                    row[f"sampled_b_centroid_bin{k}_xyz"] = "" if centroids[k] is None else json.dumps(centroids[k].tolist())
                for k, value in enumerate(velocity):
                    row[f"velocity_bin{k}_to_bin{k+1}_mps"] = "" if value is None else value
                repeat_rows.append(row)
            subset = [r for r in repeat_rows if r["query_time"] == time and r["layer"] == layer]
            t_velocity = prior_velocity[(time, "T", layer)]
            b_velocity = prior_velocity[(time, "B", layer)]
            mean_dist = distribution([original_value(r, "mean_velocity_mps") for r in subset])
            t_mean = original_value(t_velocity, "mean_velocity_mps")
            values = np.asarray([float(r["mean_velocity_mps"]) for r in subset if r["mean_velocity_mps"] != ""])
            result = {"query_time": time, "layer": layer,
                      "t_source_l0_token_id": token_t, "b_source_l0_token_id": token_b,
                      "t_bin_counts": json.dumps(t_counts), "original_b_bin_counts": json.dumps(b_counts),
                      "t_valid_adjacent_pairs": int(t_velocity["valid_adjacent_pair_count"]),
                      "t_original_mean_mps": "" if t_mean is None else t_mean,
                      "b_original_mean_mps": b_velocity["mean_velocity_mps"],
                      "b_equal_count_valid_replicates": mean_dist["valid_replicates"],
                      "b_equal_count_mean_mps": mean_dist["mean"],
                      "b_equal_count_std_mps": mean_dist["std"],
                      "b_equal_count_p05_mps": mean_dist["p05"],
                      "b_equal_count_median_mps": mean_dist["median"],
                      "b_equal_count_p95_mps": mean_dist["p95"],
                      "b_equal_count_fraction_ge_t": "" if t_mean is None else float(np.mean(values >= t_mean)),
                      "b_equal_count_fraction_le_t": "" if t_mean is None else float(np.mean(values <= t_mean))}
            for k in range(4):
                pair = distribution([original_value(r, f"velocity_bin{k}_to_bin{k+1}_mps") for r in subset])
                result[f"t_velocity_bin{k}_to_bin{k+1}_mps"] = t_velocity[f"velocity_bin{k}_to_bin{k+1}_mps"]
                result[f"b_equal_count_pair{k}_valid_replicates"] = pair["valid_replicates"]
                result[f"b_equal_count_pair{k}_p05_mps"] = pair["p05"]
                result[f"b_equal_count_pair{k}_median_mps"] = pair["median"]
                result[f"b_equal_count_pair{k}_p95_mps"] = pair["p95"]
            comparison.append(result)
    return repeat_rows, comparison


def plot_distribution(output, repeat_rows, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, time in enumerate(TIMES):
        for level in range(3):
            ax = axes[i, level]
            name = f"L{level}"
            item = next(r for r in comparison if r["query_time"] == time and r["layer"] == name)
            values = [float(r["mean_velocity_mps"]) for r in repeat_rows if r["query_time"] == time and
                      r["layer"] == name and r["mean_velocity_mps"] != ""]
            if values:
                ax.hist(values, bins=20, color="#6fa5c9", edgecolor="white", label="B equal count")
                ax.axvline(item["t_original_mean_mps"], color="#c34754", lw=2, label="T observed")
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "N/A: no adjacent T bins", transform=ax.transAxes,
                        horizontalalignment="center", verticalalignment="center")
            ax.set_title(f"{time[-8:]} {name}")
            ax.set_xlabel("Mean centroid rate (m/s)")
            ax.set_ylabel("Replicates")
    fig.tight_layout()
    fig.savefig(output / "equal_count_distribution.png", dpi=160)
    plt.close(fig)


def report(output, comparison, repeats, seed):
    fmt = lambda v: "N/A" if v == "" else f"{v:.3f}"
    lines = ["# seq0026 B 等点数下采样与 T 真实质心变化率", "",
             f"复用原 A2 特征审计锁定的 T/B token 和固定体素点集。每个 query×层级×Bin 从 B 无放回抽取与 T 相同的点数，独立重复 {repeats} 次；seed={seed}。T 原始点、质心与变化率未改变。",
             "每次抽样以抽中 B 点的真实 `delta_t` 计算平均观测时间；只在相邻非空 Bin 间求距离/实际时间差。", "",
             "| Query | 层级 | T真实 (m/s) | B全点 (m/s) | B等点数中位数 [5%,95%] (m/s) | 有效重复 | B≥T比例 |",
             "|---|---|---:|---:|---|---:|---:|"]
    for item in comparison:
        interval = ("N/A" if item["b_equal_count_valid_replicates"] == 0 else
                    f"{fmt(item['b_equal_count_median_mps'])} [{fmt(item['b_equal_count_p05_mps'])}, {fmt(item['b_equal_count_p95_mps'])}]")
        lines.append(f"| {item['query_time']} | {item['layer']} | {fmt(item['t_original_mean_mps'])} | "
                     f"{fmt(float(item['b_original_mean_mps']))} | {interval} | "
                     f"{item['b_equal_count_valid_replicates']} | {fmt(item['b_equal_count_fraction_ge_t'])} |")
    comparable = [item for item in comparison if item["t_original_mean_mps"] != ""]
    covered = sum(item["b_equal_count_p05_mps"] <= item["t_original_mean_mps"] <=
                  item["b_equal_count_p95_mps"] for item in comparable)
    lines += ["", "## 解释", "",
              f"可比较的 {len(comparable)} 个 query×层级中，T 真实均值有 {covered} 个落在 B 等点数抽样的 5%–95% 区间内。"
              "因此原先用全点 B 与少点 T 得出的变化率差异，在等点数条件下没有保持清晰分离。",
              "例如首次错误帧 L2：B 全点均值 0.220 m/s，等点数抽样中位数 1.496 m/s；T 真实值 1.417 m/s。"
              "该对比说明点数会强烈影响这项质心统计，但不能证明全点差异只由点数造成。",
              "T 只有一个有效 Bin 的层级没有相邻速度；B 按相同每 Bin 点数抽样后同样不可计算，分布标为 N/A。",
              "B≥T 比例是本次 100 次随机抽样中的经验比例，不是检验静态/运动的 p 值。",
              "全点 B 与等点数 B 的变化率差异体现采样量对质心稳定性的影响；不同传感器、视角、遮挡和帧时间戳精度仍未控制。",
              "同一帧所有点共享文件时间戳，缺乏逐点扫描时间。质心变化率不是物体真实速度，不足以直接判断动静态。", "",
              "`b_equal_count_repeats.csv` 保存每次抽中点的原始 ID、各 Bin 质心/平均时间和四个相邻变化率；`equal_count_comparison.csv` 保存均值分布及每个相邻对的分位数。", "",
              "复现命令：", "", "```bash",
              "MPLCONFIGDIR=/tmp/mpl-v2-equal XDG_CACHE_HOME=/tmp/v2-equal-cache \\",
              "  /home/jasoncui/miniconda3/envs/rdq/bin/python \\",
              "  experiments/multimodal_v2/diagnostics/audit_seq0026_equal_count.py --repeats 100 --seed 42",
              "```", ""]
    (output / "equal_count_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/v2_anomaly_seq0026_feature_audit"))
    parser.add_argument("--motion-root", type=Path, default=Path("outputs/v2_anomaly_seq0026_centroid_motion"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v2_anomaly_seq0026_centroid_downsample"))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1 or args.seed < 0:
        parser.error("--repeats must be positive and --seed nonnegative")
    repeat_rows, comparison = analyze(args.audit_root, args.motion_root, args.repeats, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "b_equal_count_repeats.csv", repeat_rows)
    write_csv(args.output / "equal_count_comparison.csv", comparison)
    plot_distribution(args.output, repeat_rows, comparison)
    report(args.output, comparison, args.repeats, args.seed)
    print(f"output={args.output} sampled_replicates={len(repeat_rows)} comparisons={len(comparison)}")


if __name__ == "__main__":
    main()
