"""Archived quality fusion, candidate admission and spatial suppression."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logit

from .features import (
    FEATURES,
    FIELDS,
    METHODS,
    QUOTAS,
    Evidence,
    ScanIndex,
    SourceVoxels,
    full_features,
    shortlist,
)


MODEL_WIDTHS = (len(FEATURES), len(FEATURES) + len(FIELDS))


def nms(xyz, ranked):
    keep = []
    for index in ranked:
        if len(keep) == 10:
            break
        if not np.any(np.linalg.norm(xyz[keep] - xyz[index], axis=1) <= 1.0):
            keep.append(index)
    return np.asarray(keep, dtype=int)


GUARDS = ("off", "fewer1", "fewer3", "fewer5", "fewer1_farther")


POINTS = FEATURES.index("recent_points_total")


NEAREST = [
    len(FEATURES) + FIELDS.index(sensor + "_best_nearest_m")
    for sensor in ("avia", "mid")
]


def order(d, q, gate):
    alpha, tau = gate
    return score_selection(d, rank_scores(d, q, alpha), tau)


def evidence(d, base, fine):
    row = np.arange(len(d.x))
    b, f = base[:, 0], fine[:, 0]
    points = d.x[row, b, POINTS] - d.x[row, f, POINTS]
    nearest = np.fmin(d.x[:, :, NEAREST[0]], d.x[:, :, NEAREST[1]])
    farther = nearest[row, f] - nearest[row, b]
    return b != f, points, farther


def masks(values):
    changed, points, farther = values
    return {
        "off": np.zeros_like(changed),
        "fewer1": changed & (points >= 1),
        "fewer3": changed & (points >= 3),
        "fewer5": changed & (points >= 5),
        "fewer1_farther": changed & (points >= 1) & (farther >= 0.03),
    }


def quality(model, group, width):
    valid = group.valid
    p = np.zeros(valid.shape)
    if valid.any():
        p[valid] = model.predict(group.x[valid, :width])
    return logit(np.clip(p, 1e-4, 1 - 1e-4))


def rank_scores(group, q, alpha):
    return np.where(group.valid, q + alpha * np.nan_to_num(group.x[:, :, 2]), -np.inf)


def proposal(scores, valid):
    best = scores.argmax(axis=1)
    margin = np.full(len(scores), -np.inf)
    exists = valid[:, 0]
    margin[exists] = scores[np.flatnonzero(exists), best[exists]] - scores[exists, 0]
    return best, margin


def score_selection(group, scores, tau):
    order = np.broadcast_to(np.arange(50), group.valid.shape).copy()
    pick, margin = proposal(scores, group.valid)
    change = (margin > tau) & (pick != 0)
    order[change] = np.argsort(-scores[change], axis=1, kind="stable")
    return order


COUNT_THRESHOLDS = (1, 3, 5)


OFF = (0, -1.0)


def proposals(d, base_order, fine_order, guarded_order, fused):
    row = np.arange(len(fused))
    base, fine, guarded = base_order[:, 0], fine_order[:, 0], guarded_order[:, 0]
    cols = np.arange(fused.shape[1])[None]
    distance = np.linalg.norm(d.xyz - d.xyz[row, fine, None], axis=2)
    local = (
        d.valid
        & (cols != base[:, None])
        & (cols != fine[:, None])
        & (distance > 0)
        & (distance <= 1)
    )
    counts = d.x[:, :, POINTS]
    improvement = counts - np.maximum(counts[row, base], counts[row, fine])[:, None]
    result = {}
    for threshold in COUNT_THRESHOLDS:
        scores = np.where(local & (improvement >= threshold), fused, -np.inf)
        best = scores.argmax(axis=1)
        slack = np.full(len(fused), np.inf)
        available = np.isfinite(scores[row, best])
        slack[available] = (
            fused[row[available], guarded[available]]
            - scores[row[available], best[available]]
        )
        result[threshold] = (best, slack)
    return result


def promote(d, base_order, fine_order, guarded_order, fused, policy):
    if policy == OFF:
        return guarded_order
    threshold, slack = policy
    best, margin = proposals(d, base_order, fine_order, guarded_order, fused)[threshold]
    selected = guarded_order.copy()
    for r in np.flatnonzero(margin <= slack):
        winner = best[r]
        selected[r] = np.r_[winner, selected[r, selected[r] != winner]]
    return selected


ECHO = FEATURES.index("echo_support")


@dataclass(frozen=True)
class CandidatePolicy:
    gates: tuple
    guard: str
    local_promotion: tuple
    admission: tuple
    echo_margin: float

    def __post_init__(self):
        method, quota = self.admission
        if self.guard not in GUARDS:
            raise ValueError(f"Unknown point support guard: {self.guard}")
        if self.admission != ("off", 0) and (
            method not in METHODS or quota not in QUOTAS[1:]
        ):
            raise ValueError(f"Invalid admission policy: {self.admission}")


@dataclass
class CandidatePool:
    x: np.ndarray
    xyz: np.ndarray
    token: np.ndarray

    @property
    def valid(self):
        return self.token >= 0


@dataclass(frozen=True)
class CandidateResult:
    pre_nms_token_ids: np.ndarray
    admitted_token_ids: np.ndarray
    token_ids: np.ndarray
    xyz: np.ndarray
    scores: np.ndarray


def quality_order(pool, models, gates, guard, local_promotion):
    scores = [quality(model, pool, width) for model, width in zip(models, MODEL_WIDTHS)]
    base, fine = [order(pool, q, gate) for q, gate in zip(scores, gates)]
    revert = masks(evidence(pool, base, fine))[guard]
    guarded = np.where(revert[:, None], base, fine)
    fused = rank_scores(pool, scores[1], gates[1][0])
    return promote(pool, base, fine, guarded, fused, local_promotion), scores[1]


def admission_option(pool, added, ranked, fine_quality, mode):
    allowed = added if mode == "quality" else added & (pool.x[:, :, ECHO] > 0)
    row = np.arange(len(ranked))
    best = np.where(allowed, fine_quality, -np.inf).argmax(axis=1)
    gap = fine_quality[row, best] - fine_quality[row, ranked[:, 0]]
    return best, np.where(allowed[row, best], gap, -np.inf)


def promote_admitted(ranked, option, threshold):
    best, gap = option
    out = ranked.copy()
    for row in np.flatnonzero(gap > threshold):
        winner = best[row]
        out[row] = np.r_[winner, ranked[row, ranked[row] != winner]]
    return out


def rank_prepared(pool, added, models, policy):
    """Rank padded 50-slot rows, leaving final NMS to the caller."""
    ranked, fine = quality_order(
        pool, models, policy.gates, policy.guard, policy.local_promotion
    )
    option = admission_option(pool, added, ranked, fine, "echo")
    return promote_admitted(ranked, option, policy.echo_margin)


class RadarCandidateReranker:
    """One stream per instance. Feed strictly increasing scan query times."""

    def __init__(self, sequence, data_root, models, policy, history_s=1.0):
        self.scans = ScanIndex(sequence, data_root)
        self.evidence = Evidence(self.scans)
        self.voxels = SourceVoxels(self.scans, history_s)
        self.models = tuple(models)
        if len(self.models) != len(MODEL_WIDTHS):
            raise ValueError(
                "Expected the fitted G3 and multiscale models in that order"
            )
        self.policy = policy
        self.history = []
        self.last_time = -np.inf

    def _check_time(self, now):
        if not np.isfinite(now) or now <= self.last_time:
            raise ValueError("Query times must be finite and strictly increasing")

    def observe_original(self, now, xyz=None, score=None):
        """Advance an unscored query using its original first candidate, if present."""
        self._check_time(now)
        if xyz is not None:
            xyz = np.asarray(xyz)
            if (
                xyz.shape != (3,)
                or not np.isfinite(xyz).all()
                or score is None
                or not np.isfinite(score)
            ):
                raise ValueError("Expected an original XYZ point and finite confidence")
        self.history = [r for r in self.history if now - r[0] <= 0.9]
        if xyz is not None:
            self.history.append((now, xyz.copy(), float(score)))
        self.last_time = now

    def step(self, now, dense):
        """Read one dense frame with the same field names as the inference cache."""
        self._check_time(now)
        xyz, logits, ids, counts, confidence = (
            np.asarray(dense[k])
            for k in (
                "dense_xyz",
                "dense_rank",
                "dense_token_id",
                "dense_point_count",
                "dense_score",
            )
        )
        n = len(ids)
        if xyz.shape != (n, 3) or any(
            a.shape != (n,) for a in (logits, ids, counts, confidence)
        ):
            raise ValueError("Dense candidate arrays have incompatible shapes")
        if not all(np.isfinite(a).all() for a in (xyz, logits, counts, confidence)):
            raise ValueError(
                "Dense XYZ, rank, point count and confidence must be finite"
            )
        if ids.dtype.kind not in "iu" or (ids < 0).any() or len(np.unique(ids)) != n:
            raise ValueError("Dense token IDs must be unique nonnegative integers")
        if (counts < 0).any():
            raise ValueError("Dense point counts must be nonnegative")
        if not n:
            self.observe_original(now)
            return CandidateResult(ids, ids, ids, xyz, confidence)

        dense_order = np.lexsort((ids, -logits))
        head = dense_order[:50]
        first = head[0]
        added = dense_order[:0]
        method, quota = self.policy.admission
        if quota and n > 50:
            proposals = shortlist(
                self.evidence,
                now,
                xyz[dense_order],
                logits[dense_order],
                counts[dense_order],
            )
            ranks = proposals[METHODS.index(method), :quota]
            added = dense_order[ranks[ranks >= 0]]
        selected = np.r_[head[: len(head) - len(added)], added]
        ranks = np.empty(n, dtype=int)
        ranks[dense_order] = np.arange(1, n + 1)
        features = full_features(
            self.scans,
            self.voxels,
            now,
            selected,
            dense,
            xyz[selected],
            logits[selected],
            ids,
            counts,
            xyz[first],
            logits[first],
            self.history,
            ranks[selected],
        )
        pool = CandidatePool(
            np.full((1, 50, len(FEATURES) + len(FIELDS)), np.nan, dtype=np.float32),
            np.full((1, 50, 3), np.nan, dtype=xyz.dtype),
            np.full((1, 50), -1, dtype=np.int64),
        )
        size = len(selected)
        pool.x[0, :size], pool.xyz[0, :size], pool.token[0, :size] = (
            features,
            xyz[selected],
            ids[selected],
        )
        is_added = np.zeros((1, 50), dtype=bool)
        is_added[0, size - len(added) : size] = True
        ranked = rank_prepared(pool, is_added, self.models, self.policy)[0]
        ranked = selected[ranked[ranked < size]]
        kept = nms(xyz, ranked)
        self.observe_original(now, xyz[first], confidence[first])
        return CandidateResult(
            ids[ranked], ids[added], ids[kept], xyz[kept], confidence[kept]
        )

    def close(self):
        self.evidence.foreground_tree.cache_clear()
        self.voxels.scan.cache_clear()
        self.scans.close()
