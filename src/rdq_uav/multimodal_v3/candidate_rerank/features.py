"""Causal scan and voxel features from the archived candidate ranking experiment."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class ScanIndex:
    def __init__(self, sequence, data_root):
        root = Path(data_root)
        self.paths = [
            sorted(
                (root / sequence / sensor).glob("*.npy"), key=lambda p: float(p.stem)
            )
            for sensor in ("livox_avia", "lidar_360")
        ]
        self.times = [np.array([float(p.stem) for p in paths]) for paths in self.paths]

    @lru_cache(maxsize=48)
    def scan(self, sensor, idx):
        points = np.load(self.paths[sensor][idx], allow_pickle=False)[:, :3]
        points = points[np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)]
        return points, cKDTree(points)

    def close(self):
        self.scan.cache_clear()


def summarize(observations, now, anchor):
    """Return compact support and a bounded centroid extrapolation."""
    options = []
    for sensor in range(2):
        obs = [r for r in observations if r[0] == sensor]
        if len(obs) < 2:
            continue
        t = np.array([r[1] - now for r in obs])
        p = np.array([r[2] for r in obs])
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        if np.any(step > 0.50) or t[-1] < -0.15:
            continue
        center = p[-1]
        if len(obs) >= 3 and np.ptp(t) >= 0.09:
            tc = t - t.mean()
            v = (tc[:, None] * p).sum(axis=0) / (tc @ tc)
            fit = p.mean(axis=0) + tc[:, None] * v
            if (
                np.linalg.norm(v) <= 6
                and np.sqrt(np.mean(np.sum((p - fit) ** 2, axis=1))) <= 0.12
            ):
                center = p.mean(axis=0) - t.mean() * v
        if np.linalg.norm(center - anchor) > 0.65:
            continue
        options.append((len(obs), float(t[-1]), center))
    if not options:
        return 0.0, anchor
    if len(options) == 2 and np.linalg.norm(options[0][2] - options[1][2]) > 0.4:
        return 0.0, anchor
    n, _, center = max(options, key=lambda r: (r[0], r[1]))
    return n / 3.0, center


def fit_prior(history, now):
    """Use a supported past trajectory, tolerating discontinuous top-one peaks."""
    bins = {}
    for t, p, s in reversed(history):
        age = now - t
        if age <= 0 or age > 0.9 or s < 0.25:
            continue
        slot = int(age / 0.10)
        if slot not in bins or s > bins[slot][2]:
            bins[slot] = (t, p, s)
    if len(bins) < 3:
        return None
    rows = sorted(bins.values(), key=lambda r: r[0])
    t = np.array([r[0] - now for r in rows])
    p = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows])
    i, j = np.triu_indices(len(t), 1)
    keep = t[j] - t[i] >= 0.17
    i, j = i[keep], j[keep]
    if not len(i):
        return None
    velocities = (p[j] - p[i]) / (t[j] - t[i])[:, None]
    hypotheses = p[i] - velocities * t[i, None]
    err = np.linalg.norm(
        hypotheses[:, None] + velocities[:, None] * t[None, :, None] - p[None], axis=2
    )
    weights = s * np.exp(t / 0.6)
    good = err <= 0.45
    quality = good @ weights
    quality[np.linalg.norm(velocities, axis=1) > 12] = -1
    best = int(quality.argmax())
    support = good[best]
    if quality[best] < 0 or support.sum() < 3 or t[support][-1] < -0.32:
        return None
    ts, pts = t[support], p[support]
    if np.ptp(ts) < 0.2:
        return None
    tc = ts - ts.mean()
    v = (tc[:, None] * pts).sum(axis=0) / (tc @ tc)
    origin = pts.mean(axis=0) - v * ts.mean()
    rms = np.sqrt(np.mean(np.sum((pts - origin - ts[:, None] * v) ** 2, axis=1)))
    if rms > 0.35 or np.linalg.norm(v) > 12:
        return None
    return origin, float(np.linalg.norm(v)), float(rms)


BIAS = 1 << 19


STRIDE = 2 * BIAS


def voxel_keys(points):
    grid = np.floor(points / 0.5).astype(np.int64) + BIAS
    if (grid < 0).any() or (grid >= STRIDE).any():
        raise ValueError("Voxel coordinates exceed exact packing bounds")
    return (grid[:, 0] * STRIDE + grid[:, 1]) * STRIDE + grid[:, 2]


class SourceVoxels:
    def __init__(self, scans, history_s):
        self.scans = scans
        self.history_s = history_s

    @lru_cache(maxsize=40)
    def scan(self, sensor, idx):
        points = self.scans.scan(sensor, idx)[0].astype(np.float32, copy=False)
        keys = voxel_keys(points)
        return np.unique(keys, return_counts=True)

    def centers(self, now, selected, expected_counts, token_ids):
        parts = []
        for sensor, times in enumerate(self.scans.times):
            first = np.searchsorted(times, now - self.history_s, side="left")
            last = np.searchsorted(times, now, side="right")
            parts.extend(self.scan(sensor, idx) for idx in range(first, last))
        if not parts or not np.array_equal(token_ids, np.arange(len(token_ids))):
            return None
        keys, inverse = np.unique(
            np.concatenate([p[0] for p in parts]), return_inverse=True
        )
        counts = np.zeros(len(keys), dtype=np.int64)
        np.add.at(counts, inverse, np.concatenate([p[1] for p in parts]))
        if not np.array_equal(counts, expected_counts):
            return None
        picked = keys[selected]
        cells = np.stack(
            (picked // STRIDE**2, (picked // STRIDE) % STRIDE, picked % STRIDE), axis=1
        )
        return (cells - BIAS + 0.5) * 0.5


BASE_FEATURES = (
    "logit",
    "rank",
    "delta_logit",
    "voxel_points_log",
    "offset_m",
    "offset_valid",
    "range_m",
    "from_top1_m",
)


LANDING_FEATURES = (
    "recent_points_avia",
    "recent_points_mid",
    "recent_points_total",
    "recent_support_avia",
    "recent_support_mid",
    "recent_support_total",
    "recent_available_avia",
    "recent_available_mid",
    "local_spread_m",
)


HISTORY_FEATURES = (
    "history_distance_m",
    "history_available",
    "old_repeat_fraction",
    "old_comparable_points",
    "echo_support",
    "motion_valid",
    "motion_dynamic",
    "motion_speed",
    "motion_reliability",
)


FEATURES = BASE_FEATURES + LANDING_FEATURES + HISTORY_FEATURES


def landing(scans, now, anchors):
    n = len(anchors)
    counts = np.zeros((n, 2), dtype=int)
    supports = np.zeros((n, 2), dtype=int)
    available = np.zeros(2, dtype=int)
    repeats, comparable_counts = np.zeros((2, n), dtype=int)
    spread = [[] for _ in anchors]
    echoes = [[] for _ in anchors]
    for sensor, times in enumerate(scans.times):
        end = np.searchsorted(times, now, side="right")
        refs = sorted(
            {
                int(np.searchsorted(times, now - age, side="right")) - 1
                for age in (0.7, 1.1, 1.5)
            }
        )
        refs = [i for i in refs if i >= 0 and now - times[i] <= 1.8]
        trees = [scans.scan(sensor, i)[1] for i in refs]
        can_compare = len(trees) == 3 and all(t.n for t in trees)
        for i in range(max(0, end - 3), end):
            if now - times[i] > 0.24:
                continue
            assert times[i] <= now
            available[sensor] += 1
            points, tree = scans.scan(sensor, i)
            neighborhoods = tree.query_ball_point(anchors, 0.65)
            occupied = [
                (j, np.asarray(idx)) for j, idx in enumerate(neighborhoods) if idx
            ]
            matches = None
            if can_compare and occupied:
                unique = np.unique(np.concatenate([idx for _, idx in occupied]))
                matches = np.zeros(len(points), dtype=np.uint8)
                matches[unique] = sum(
                    t.query(points[unique], distance_upper_bound=0.18)[0] < np.inf
                    for t in trees
                )
            for j, idx in occupied:
                counts[j, sensor] += len(idx)
                supports[j, sensor] += 1
                local = points[idx]
                center = np.median(local, axis=0)
                spread[j].append(
                    float(np.median(np.linalg.norm(local - center, axis=1)))
                )
                if matches is not None:
                    repeated = matches[idx] >= 2
                    repeats[j] += int(repeated.sum())
                    comparable_counts[j] += len(local)
                    foreground = local[~repeated]
                    if len(foreground):
                        center = np.median(foreground, axis=0)
                        compact = foreground[
                            np.linalg.norm(foreground - center, axis=1) <= 0.23
                        ]
                        if len(compact) >= max(1, 0.6 * len(foreground)):
                            echoes[j].append(
                                (
                                    sensor,
                                    float(times[i]),
                                    compact.mean(axis=0),
                                    len(compact),
                                )
                            )
    old_fraction = np.divide(
        repeats, comparable_counts, where=comparable_counts > 0, out=np.full(n, np.nan)
    )
    width = [np.median(x) if x else np.nan for x in spread]
    echo_score = [
        summarize(obs, now, anchor)[0] for obs, anchor in zip(echoes, anchors)
    ]
    return np.column_stack(
        (
            counts,
            counts.sum(axis=1),
            supports,
            supports.sum(axis=1),
            np.tile(available, (n, 1)),
            width,
        )
    ), np.column_stack((old_fraction, comparable_counts, echo_score))


RADII = np.array([0.18, 0.30, 0.45])


SENSOR_FIELDS = (
    "latest_nearest_m",
    "best_nearest_m",
    "median_nearest_m",
    "count_r018",
    "count_r030",
    "count_r045",
    "frames_r030",
    "inner_fraction",
    "gaussian_mass",
    "centroid_gap_m",
    "weighted_rms_m",
)


FIELDS = tuple(
    f"{sensor}_{field}" for sensor in ("avia", "mid") for field in SENSOR_FIELDS
)


def support(scans, t, anchors):
    n = len(anchors)
    result = []
    for sensor, times in enumerate(scans.times):
        end = int(np.searchsorted(times, t, side="right"))
        close = [i for i in range(max(0, end - 3), end) if t - times[i] <= 0.24]
        accum = np.zeros((n, len(RADII)), dtype=int)
        frames = np.zeros(n, dtype=int)
        count = np.zeros(n, dtype=int)
        weight = np.zeros(n)
        second = np.zeros(n)
        moment = np.zeros((n, 3))
        nearest = np.full((len(close), n), 0.65)
        for slot, i in enumerate(close):
            assert times[i] <= t
            points, tree = scans.scan(sensor, i)
            neighbors = tree.query_ball_point(anchors, 0.65)
            for j, indices in enumerate(neighbors):
                if not indices:
                    continue
                delta = points[indices] - anchors[j]
                squared = np.einsum("ij,ij->i", delta, delta)
                near = (squared[:, None] <= RADII**2).sum(axis=0)
                local_weight = np.exp(-squared / (2 * 0.2**2))
                nearest[slot, j] = np.sqrt(squared.min())
                accum[j] += near
                frames[j] += near[1] > 0
                count[j] += len(indices)
                weight[j] += local_weight.sum()
                moment[j] += local_weight @ delta
                second[j] += local_weight @ squared
        if not close:
            result.append(np.full((n, len(SENSOR_FIELDS)), np.nan))
            continue
        fraction = np.divide(accum[:, 1], count, where=count > 0, out=np.zeros(n))
        centroid_gap = np.divide(
            np.linalg.norm(moment, axis=1),
            weight,
            where=weight > 0,
            out=np.full(n, np.nan),
        )
        rms = np.sqrt(
            np.divide(second, weight, where=weight > 0, out=np.full(n, np.nan))
        )
        result.append(
            np.column_stack(
                (
                    nearest[-1],
                    nearest.min(axis=0),
                    np.median(nearest, axis=0),
                    accum,
                    frames,
                    fraction,
                    weight,
                    centroid_gap,
                    rms,
                )
            )
        )
    return np.column_stack(result)


METHODS = ("raw_tail", "source_points", "recent", "foreground")


QUOTAS = (0, 1, 3, 5)


CAPACITY = QUOTAS[-1]


DIVERSITY = 0.35


class Evidence:
    def __init__(self, scans):
        self.scans = scans

    @lru_cache(maxsize=48)
    def foreground_tree(self, sensor, index, refs):
        points, _ = self.scans.scan(sensor, index)
        matches = sum(
            self.scans.scan(sensor, j)[1].query(points, distance_upper_bound=0.18)[0]
            < np.inf
            for j in refs
        )
        return cKDTree(points[matches < 2])

    def count(self, now, xyz):
        recent = np.zeros((len(xyz), 3))
        changed = np.zeros_like(recent)
        for sensor, times in enumerate(self.scans.times):
            end = np.searchsorted(times, now, side="right")
            refs = tuple(
                sorted(
                    {
                        int(np.searchsorted(times, now - age, side="right")) - 1
                        for age in (0.7, 1.1, 1.5)
                    }
                )
            )
            refs = tuple(j for j in refs if j >= 0 and now - times[j] <= 1.8)
            comparable = len(refs) == 3 and all(
                self.scans.scan(sensor, j)[1].n for j in refs
            )
            for i in range(max(0, end - 3), end):
                if now - times[i] > 0.24:
                    continue
                assert times[i] <= now
                tree = self.scans.scan(sensor, i)[1]
                add_counts(recent, tree, xyz)
                if comparable:
                    assert all(times[j] < times[i] for j in refs)
                    add_counts(changed, self.foreground_tree(sensor, i, refs), xyz)
        return recent, changed


def add_counts(table, tree, xyz):
    core = tree.query_ball_point(xyz, 0.35, return_length=True)
    shell = tree.query_ball_point(xyz, 0.65, return_length=True)
    table += np.column_stack((core, shell, core > 0))


def strength(counts):
    return np.log1p(counts[:, 0]) + 0.5 * np.log1p(counts[:, 1]) + 0.5 * counts[:, 2]


def shortlist(evidence, now, xyz, logits, src_counts):
    candidates = xyz[50:]
    empty = np.full((len(METHODS), CAPACITY), -1, dtype=np.int64)
    if not len(candidates):
        return empty
    far = cKDTree(xyz[:50]).query(candidates)[0] >= DIVERSITY
    eligible = np.flatnonzero(far)
    if not len(eligible):
        return empty
    xyz = candidates[eligible]
    logits, src_counts = logits[50:][eligible], src_counts[50:][eligible]
    recent, changed = evidence.count(now, xyz)
    scores = (
        logits,
        logits + np.log1p(src_counts),
        np.where(recent[:, 1] > 0, strength(recent) + 0.25 * logits, -np.inf),
        np.where(changed[:, 1] > 0, strength(changed) + 0.25 * logits, -np.inf),
    )
    for mode, key in enumerate(scores):
        order = np.argsort(-key, kind="stable")
        chosen = []
        for j in order[np.isfinite(key[order])]:
            if (
                chosen
                and (np.linalg.norm(xyz[chosen] - xyz[j], axis=1) < DIVERSITY).any()
            ):
                continue
            chosen.append(j)
            if len(chosen) == CAPACITY:
                break
        empty[mode, : len(chosen)] = eligible[chosen] + 50
    return empty


def full_features(
    scans,
    voxels,
    now,
    selected,
    dense,
    xyz,
    logits,
    ids,
    counts,
    head,
    head_logit,
    history,
    rank,
):
    source = voxels.centers(now, selected, counts, ids)
    offset = (
        np.linalg.norm(xyz - source, axis=1)
        if source is not None
        else np.full(len(selected), np.nan)
    )
    prior = fit_prior(history, now)
    distance = (
        np.linalg.norm(xyz - prior[0], axis=1)
        if prior is not None
        else np.full(len(selected), np.nan)
    )
    base = np.column_stack(
        (
            logits,
            rank,
            np.subtract(logits, head_logit),
            np.log1p(counts[selected]),
            offset,
            np.isfinite(offset),
            np.linalg.norm(xyz, axis=1),
            np.linalg.norm(xyz - head, axis=1),
        )
    )
    local, old = landing(scans, now, xyz)
    motion = dense["dense_motion_valid"][selected]
    motion_data = [
        np.where(motion, dense[k][selected], np.nan)
        for k in (
            "dense_motion_dynamic",
            "dense_motion_speed_mps",
            "dense_motion_reliability",
        )
    ]
    history_x = np.column_stack(
        (distance, np.full(len(selected), prior is not None), old, motion, *motion_data)
    )
    return np.column_stack((base, local, history_x, support(scans, now, xyz)))
