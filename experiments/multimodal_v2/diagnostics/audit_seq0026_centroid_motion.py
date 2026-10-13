#!/usr/bin/env python3
"""Offline physical centroid statistics for saved seq0026 A2 T/B voxels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

TIMES = ("1706254922.6860642", "1706254922.7306657",
         "1706254923.0253723", "1706254923.5353887")
EDGES = np.array([-1., -.8, -.6, -.4, -.2, 0.], dtype=np.float64)
SCALES = (.5, 1., 2.)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    if not data:
        raise ValueError(f"no data for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(data[0]))
        writer.writeheader()
        writer.writerows(data)


class SavedQuery:
    def __init__(self, audit_root, time):
        self.time = time
        self.directory = audit_root / "traces" / f"seq0026_query_{time}" / "A2"
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.index = json.loads((self.directory / "tensor_index.json").read_text())
        if self.manifest["model"] != "A2" or self.manifest["split"] != "validation_sub":
            raise ValueError(f"unexpected trace identity: {self.directory}")
        if self.manifest["sequence_id"] != "seq0026" or abs(self.manifest["query_time"]-float(time)) > 1e-6:
            raise ValueError(f"query identity mismatch: {self.directory}")
        config = yaml.safe_load((self.directory / "effective_config.yaml").read_text())
        if float(config["data"]["radar_history_s"]) != 1.0:
            raise ValueError("expected the original 1-second radar history")
        if not json.loads((self.directory / "parity.json").read_text())["passed"]:
            raise ValueError(f"trace parity failed: {self.directory}")

    def a(self, name):
        entry = self.index[name]
        if entry["status"] != "saved":
            raise ValueError(f"full saved tensor required: {name}: {entry['status']}")
        return np.load(self.directory / entry["file"], mmap_mode="r", allow_pickle=False)


def bin_indices(delta_t):
    # Half-open bins except that the final bin includes exactly 0.0 s.
    dt = np.asarray(delta_t, dtype=np.float64)
    if not np.isfinite(dt).all() or (dt < -1.).any() or (dt > 0.).any():
        raise ValueError("delta_t outside inclusive [-1,0] window")
    index = np.searchsorted(EDGES, dt, side="right") - 1
    index[dt == 0.] = 4
    if ((index < 0) | (index > 4)).any():
        raise AssertionError("time bin omitted input point")
    for k in range(5):
        mask = index == k
        valid = (dt[mask] >= EDGES[k]) & ((dt[mask] < EDGES[k+1]) if k < 4 else (dt[mask] <= 0.))
        if not valid.all():
            raise AssertionError("time bins overlap or violate boundary rule")
    if sum(np.count_nonzero(index == k) for k in range(5)) != len(dt):
        raise AssertionError("time bins do not cover all points")
    return index


def one_query(audit_root, time, candidates, prior_geometry, identities):
    query = SavedQuery(audit_root, time)
    points, delta_t, sensor = (query.a(name) for name in
                               ("sample.points", "sample.delta_t", "sample.sensor_id"))
    point_l0, parent01, parent12 = (query.a(name) for name in
                                    ("mapping.point_to_l0", "mapping.parent_l0_to_l1", "mapping.parent_l1_to_l2"))
    if len(points) != len(delta_t) or len(points) != len(sensor) or len(points) != len(point_l0):
        raise AssertionError("packed point fields have different lengths")
    if not np.isin(sensor, [0, 1]).all():
        raise AssertionError("unexpected sensor ID")
    bins = bin_indices(delta_t)
    point_l1 = parent01[point_l0]
    point_l2 = parent12[point_l1]
    if not np.array_equal(query.a("level1.coords")[parent01],
                          np.floor_divide(query.a("level0.coords"), 2)):
        raise AssertionError("L0->L1 coordinates disagree with saved parent map")
    if not np.array_equal(query.a("level2.coords")[parent12],
                          np.floor_divide(query.a("level1.coords"), 2)):
        raise AssertionError("L1->L2 coordinates disagree with saved parent map")
    bin_rows, velocity_rows, trajectories = [], [], {}
    for kind in ("T", "B"):
        record = candidates[(time, kind)]
        if record["status"] != "matched":
            raise ValueError(f"prior T/B identity missing: {time} {kind}")
        token = int(record["source_token_id"])
        tokens = (token, int(parent01[token]), int(parent12[parent01[token]]))
        if tokens[1:] != (int(record["l1_token_id"]), int(record["l2_token_id"])):
            raise AssertionError("prior candidate parent IDs changed")
        if tokens != tuple(int(identities[f"{time}_{kind}_l{level}_token_id"]) for level in range(3)):
            raise AssertionError("feature NPZ candidate IDs disagree with CSV")
        point_masks = (point_l0 == tokens[0], point_l1 == tokens[1], point_l2 == tokens[2])
        if not np.all(~point_masks[0] | point_masks[1]) or not np.all(~point_masks[1] | point_masks[2]):
            raise AssertionError("L0/L1/L2 point set inclusion failed")
        original_count = int(prior_geometry[(time, kind)]["voxel_points"])
        if int(point_masks[0].sum()) != original_count:
            raise AssertionError("L0 point count differs from prior feature audit")
        if not np.array_equal(np.flatnonzero(point_masks[0]), identities[f"{time}_{kind}_point_ids"]):
            raise AssertionError("L0 point IDs differ from prior feature audit")
        for level, (token_id, point_mask) in enumerate(zip(tokens, point_masks)):
            coord = np.asarray(query.a(f"level{level}.coords")[token_id])
            center = np.asarray(query.a(f"level{level}.centers")[token_id])
            if not np.array_equal(coord, identities[f"{time}_{kind}_l{level}_coord"]):
                raise AssertionError("saved hierarchy coordinate changed")
            if level == 0:
                if not np.array_equal(coord, json.loads(record["l0_coord"])) or not np.allclose(center, json.loads(record["l0_center_xyz_m"]), atol=0, rtol=0):
                    raise AssertionError("prior L0 identity/center changed")
            if not np.allclose(center, (coord + .5)*SCALES[level], atol=0, rtol=0):
                raise AssertionError("voxel center/scale mismatch")
            if int(point_mask.sum()) != int(query.a(f"level{level}.point_count")[token_id]):
                raise AssertionError("bin source point count differs from hierarchy")
            centroids, mean_times, counts = [], [], []
            for k in range(5):
                point_ids = np.flatnonzero(point_mask & (bins == k))
                count = len(point_ids)
                avia = int(np.count_nonzero(sensor[point_ids] == 0))
                mid360 = int(np.count_nonzero(sensor[point_ids] == 1))
                if count != avia + mid360:
                    raise AssertionError("sensor counts do not sum to time-bin points")
                centroid = np.mean(points[point_ids].astype(np.float64), axis=0) if count else None
                mean_time = float(np.mean(delta_t[point_ids].astype(np.float64))) if count else None
                if (centroid is None) != (mean_time is None):
                    raise AssertionError("empty-bin centroid/time contract failed")
                centroids.append(centroid); mean_times.append(mean_time); counts.append(count)
                bin_rows.append({"query_time": time, "candidate_type": kind, "layer": f"L{level}",
                                 "source_l0_token_id": token, "layer_token_id": token_id,
                                 "voxel_coord_xyz": json.dumps(coord.tolist()),
                                 "voxel_center_xyz": json.dumps(center.tolist()),
                                 "bin_index": k, "bin_start_s": float(EDGES[k]), "bin_end_s": float(EDGES[k+1]),
                                 "point_count": count, "avia_count": avia, "mid360_count": mid360,
                                 "centroid_x": "" if centroid is None else float(centroid[0]),
                                 "centroid_y": "" if centroid is None else float(centroid[1]),
                                 "centroid_z": "" if centroid is None else float(centroid[2]),
                                 "mean_delta_t_s": "" if mean_time is None else mean_time})
            if sum(counts) != int(point_mask.sum()):
                raise AssertionError("five bins do not sum to original voxel count")
            speeds, reasons = [], []
            for k in range(4):
                if not counts[k] or not counts[k+1]:
                    speeds.append(None); reasons.append("empty_adjacent_bin")
                elif mean_times[k+1] <= mean_times[k]:
                    speeds.append(None); reasons.append("nonpositive_mean_time_gap")
                else:
                    speed = float(np.linalg.norm(centroids[k+1]-centroids[k]) /
                                  (mean_times[k+1]-mean_times[k]))
                    if not np.isfinite(speed) or speed < 0:
                        raise AssertionError("invalid centroid change rate")
                    speeds.append(speed); reasons.append("valid")
            valid = [speed for speed in speeds if speed is not None]
            row = {"query_time": time, "candidate_type": kind, "layer": f"L{level}",
                   "source_l0_token_id": token, "layer_token_id": token_id,
                   "voxel_coord_xyz": json.dumps(coord.tolist()),
                   "voxel_center_xyz": json.dumps(center.tolist()),
                   "total_point_count": sum(counts), "valid_time_bin_count": sum(count > 0 for count in counts),
                   "valid_adjacent_pair_count": len(valid)}
            for k in range(4):
                row[f"velocity_bin{k}_to_bin{k+1}_mps"] = "" if speeds[k] is None else speeds[k]
                row[f"reason_bin{k}_to_bin{k+1}"] = reasons[k]
                row[f"actual_mean_time_gap_bin{k}_to_bin{k+1}_s"] = (
                    "" if speeds[k] is None else mean_times[k+1]-mean_times[k])
            row["mean_velocity_mps"] = "" if not valid else float(np.mean(valid))
            row["max_velocity_mps"] = "" if not valid else float(np.max(valid))
            velocity_rows.append(row)
            trajectories[(time, kind, level)] = {"centroids": centroids, "center": center}
    return bin_rows, velocity_rows, trajectories


def plots(output, times, velocity_rows, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4*len(times)))
    colors = ("#3469bd", "#d98222", "#34a889")
    for i, time in enumerate(times):
        for j, kind in enumerate(("T", "B")):
            ax = fig.add_subplot(len(times), 2, 2*i+j+1, projection="3d")
            for level in range(3):
                trajectory = trajectories[(time, kind, level)]
                center = trajectory["center"]
                values = trajectory["centroids"]
                for k, centroid in enumerate(values):
                    if centroid is None:
                        continue
                    xyz = centroid-center
                    ax.scatter(*xyz, color=colors[level], s=28)
                    ax.text(*xyz, str(k), fontsize=7)
                    if k and values[k-1] is not None:
                        previous = values[k-1]-center
                        ax.plot([previous[0], xyz[0]], [previous[1], xyz[1]],
                                [previous[2], xyz[2]], color=colors[level], lw=1.5)
                ax.plot([], [], [], color=colors[level], label=f"L{level}")
            ax.set(xlabel="x-center (m)", ylabel="y-center (m)", zlabel="z-center (m)",
                   title=f"{time} {kind}; labels=bin")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "centroid_trajectories.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(times), 1, figsize=(9, 2.7*len(times)), squeeze=False)
    for ax, time in zip(axes[:, 0], times):
        for j, kind in enumerate(("T", "B")):
            for level in range(3):
                row = next(r for r in velocity_rows if r["query_time"] == time and
                           r["candidate_type"] == kind and r["layer"] == f"L{level}")
                x = j + (level-1)*.23
                if row["mean_velocity_mps"] == "":
                    ax.text(x, .02, "N/A", ha="center", rotation=90, fontsize=8)
                else:
                    ax.bar(x, row["mean_velocity_mps"], width=.21, color=colors[level])
        ax.set_xticks((0, 1), ("T", "B")); ax.set_ylabel("Mean rate (m/s)")
        ax.set_title(time); ax.grid(axis="y", alpha=.25)
        ax.set_xlim(-.5, 1.5); ax.set_ylim(bottom=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"L{i}") for i, c in enumerate(colors)]
    fig.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "centroid_velocity_comparison.png", dpi=150)
    plt.close(fig)


def summary(output, times, velocity_rows, bin_rows):
    def cell(row):
        return "N/A" if row["mean_velocity_mps"] == "" else f"{row['mean_velocity_mps']:.3f}"
    lines = ["# seq0026 T/B 多尺度体素质心变化率", "",
             "本统计使用已保存的四帧 A2 真实 Dataset 点云、`delta_t` 和原 feature audit 的 T/B 身份。",
             "每行固定一个当前 query 的 L0/L1/L2 体素；点集取其真实父子映射。",
             "五个 Bin 为 [-1,-0.8)、[-0.8,-0.6)、[-0.6,-0.4)、[-0.4,-0.2)、[-0.2,0] 秒。",
             "只以**相邻且非空** Bin 的真实平均观测时间差计算质心距离/时间；N/A 不视为零。", "",
             "| Query | 候选 | L0 平均 (m/s) | L1 平均 (m/s) | L2 平均 (m/s) | 有效 Bin 数 L0/L1/L2 | 有效相邻对数 L0/L1/L2 |",
             "|---|---|---:|---:|---:|---|---|"]
    for time in times:
        for kind in ("T", "B"):
            r = [next(x for x in velocity_rows if x["query_time"] == time and
                      x["candidate_type"] == kind and x["layer"] == f"L{i}") for i in range(3)]
            lines.append(f"| {time} | {kind} | {cell(r[0])} | {cell(r[1])} | {cell(r[2])} | "
                         f"{'/'.join(str(x['valid_time_bin_count']) for x in r)} | "
                         f"{'/'.join(str(x['valid_adjacent_pair_count']) for x in r)} |")
    if "1706254922.7306657" in times:
        time = "1706254922.7306657"
        lines += ["", "## 首次错误帧 L2 的逐 Bin 实际质心", "",
                  "两候选在同一 2 m 尺度比较；坐标为原始 XYZ（m），不是 GT 位置。", "",
                  "| Bin | T 点数 | T 平均 Δt (s) | T 质心 XYZ (m) | B 点数 | B 平均 Δt (s) | B 质心 XYZ (m) |",
                  "|---:|---:|---:|---|---:|---:|---|"]
        for k in range(5):
            pair = [next(r for r in bin_rows if r["query_time"] == time and
                         r["candidate_type"] == kind and r["layer"] == "L2" and r["bin_index"] == k)
                    for kind in ("T", "B")]
            def xyz(row):
                return "N/A" if not row["point_count"] else ", ".join(
                    f"{row['centroid_'+axis]:.3f}" for axis in "xyz")
            lines.append(f"| {k} | {pair[0]['point_count']} | {pair[0]['mean_delta_t_s']:.3f} | {xyz(pair[0])} | "
                         f"{pair[1]['point_count']} | {pair[1]['mean_delta_t_s']:.3f} | {xyz(pair[1])} |")
        lines += ["", "在该帧 L2，T 的 5 个 Bin 均有点（每 Bin 1–3 个），平均变化率 "
                  f"{next(r for r in velocity_rows if r['query_time']==time and r['candidate_type']=='T' and r['layer']=='L2')['mean_velocity_mps']:.3f} m/s；"
                  "B 每 Bin 有 54–67 个点，平均变化率 "
                  f"{next(r for r in velocity_rows if r['query_time']==time and r['candidate_type']=='B' and r['layer']=='L2')['mean_velocity_mps']:.3f} m/s。"
                  "T 的小样本质心更容易受扫描采样与体素边界影响。"]
    lines += ["", "## 可计算性与来源", "",
              "- 每个 query/候选/层级的四个速度、实际平均时间差、缺失原因和最大值见 `centroid_velocity.csv`。空 Bin 的质心和平均时间在 `centroid_by_bin.csv` 中留空。",
              "- `centroid_trajectories.png` 的点标注是 Bin 编号，坐标为原始质心减当前体素中心；只有相邻有效 Bin 连线。绝对质心 XYZ 见 CSV。",
              "- Avia/Mid360 点数仅用于来源核对。Dataset 将**整帧所有点赋予同一个文件时间戳**，没有逐点扫描时间；因此结果只能称作固定空间体素内的观测质心变化率，不等同目标真实速度。",
              "- T/B 在不同 query 使用此前审计锁定的各自 token；不能将跨 query 的 token ID 当作同一物理体素。", "",
              "## 逐层有效性", ""]
    for time in times:
        for kind in ("T", "B"):
            lines.append(f"- `{time}` {kind}：" + "；".join(
                f"L{i} {next(x for x in velocity_rows if x['query_time']==time and x['candidate_type']==kind and x['layer']==f'L{i}')['valid_time_bin_count']} 个有效 Bin、"
                f"{next(x for x in velocity_rows if x['query_time']==time and x['candidate_type']==kind and x['layer']==f'L{i}')['valid_adjacent_pair_count']} 对可计算速度"
                for i in range(3)) + "。")
    lines += ["", "## 解释边界", "",
              "同一固定体素内不同时间 Bin 的观测点可能来自不同扫描位置和表面片段；点数很少时，质心对单点极其敏感。即便 B 的变化率较低或较高，也不能仅凭该统计判定动静态。GT 轨迹没有参与任何质心计算。", "",
              "复现命令（在仓库根目录执行）：", "",
              "```bash", "MPLCONFIGDIR=/tmp/mpl-v2-centroid XDG_CACHE_HOME=/tmp/v2-centroid-cache \\",
              "  /home/jasoncui/miniconda3/envs/rdq/bin/python \\",
              "  experiments/multimodal_v2/diagnostics/audit_seq0026_centroid_motion.py", "```", ""]
    (output / "centroid_motion_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/v2_anomaly_seq0026_feature_audit"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v2_anomaly_seq0026_centroid_motion"))
    parser.add_argument("--query-time", choices=TIMES, help="run one query first for validation")
    args = parser.parse_args()
    selected = (args.query_time,) if args.query_time else TIMES
    candidates = {(r["time"], r["kind"]): r for r in rows(args.audit_root / "candidate_comparison.csv")}
    geometry = {(r["time"], r["kind"]): r for r in rows(args.audit_root / "raw_geometry_stats.csv")}
    with np.load(args.audit_root / "feature_vectors.npz", allow_pickle=False) as identity_file:
        identities = {name: identity_file[name] for name in identity_file.files if
                      name.endswith(("_point_ids", "_token_id", "_coord"))}
    all_bins, all_velocities, all_trajectories = [], [], {}
    for time in selected:
        bins, velocities, trajectories = one_query(args.audit_root, time, candidates, geometry, identities)
        all_bins.extend(bins); all_velocities.extend(velocities); all_trajectories.update(trajectories)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "centroid_by_bin.csv", all_bins)
    write_csv(args.output / "centroid_velocity.csv", all_velocities)
    plots(args.output, selected, all_velocities, all_trajectories)
    summary(args.output, selected, all_velocities, all_bins)
    print(f"queries={len(selected)} bin_rows={len(all_bins)} velocity_rows={len(all_velocities)} output={args.output}")


if __name__ == "__main__":
    main()
