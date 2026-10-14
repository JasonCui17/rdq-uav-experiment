"""Causal window scoring and hysteretic choice among confirmed tracks.

A static-background hint is a bounded, optional heuristic. Missing or invalid
motion evidence is unknown, never treated as a background observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    window_s: float = 1.
    score_weight: float = 1.
    support_weight: float = .4
    recency_weight: float = .2
    innovation_weight: float = .2
    motion_weight: float = .3
    background_weight: float = .4
    min_selection_score: float = .2
    switch_margin: float = .08
    switch_consecutive: int = 2
    innovation_scale_m: float = 2.
    background_enabled: bool = True
    background_min_points: int = 32
    background_min_reliability: float = .5
    background_speed_scale_mps: float = .2

    def __post_init__(self):
        positive = (self.window_s, self.innovation_scale_m,
                    self.background_speed_scale_mps)
        nonnegative = (self.score_weight, self.support_weight, self.recency_weight,
                       self.innovation_weight, self.motion_weight, self.background_weight,
                       self.switch_margin)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Invalid positive selection scale")
        if any(not math.isfinite(x) or x < 0 for x in nonnegative):
            raise ValueError("Invalid selection weight")
        if not math.isfinite(self.min_selection_score) or self.switch_consecutive < 1:
            raise ValueError("Invalid selection threshold")
        if self.background_min_points < 1 or not 0 <= self.background_min_reliability <= 1:
            raise ValueError("Invalid background support requirement")


class TrackSelector:
    def __init__(self, config: SelectionConfig, *, use_evidence: bool):
        self.cfg = config
        self.use_evidence = use_evidence
        self.current = None
        self.challenger = None
        self.challenger_streak = 0
        self.query_times = []

    def release(self, reason: str):
        self.current = None
        self.challenger = None
        self.challenger_streak = 0
        if reason == "time_gap":
            self.query_times.clear()

    def _score(self, now, history, query_count):
        cfg = self.cfg
        recent = [row for row in history if now - cfg.window_s <= row["timestamp"] <= now]
        updates = [row for row in recent if row["output_kind"] == "updated"]
        if not updates:
            return None
        last = updates[-1]
        age = now - last["timestamp"]
        if age > cfg.window_s:
            return None
        target = float(np.mean([row["score"] for row in updates]))
        support = min(1., len({row["timestamp"] for row in updates}) / max(1, query_count))
        recency = math.exp(-age / cfg.window_s)
        innovations = [row["innovation_m"] for row in updates if row["innovation_m"] is not None]
        innovation = (math.exp(-float(np.mean(innovations)) / cfg.innovation_scale_m)
                      if innovations else None)
        dynamic_values = []
        static_values = []
        if self.use_evidence:
            for row in updates:
                evidence = row["evidence"]
                if not evidence or not evidence.get("motion_valid", False):
                    continue
                reliability = float(evidence["motion_reliability"])
                if evidence.get("motion_dynamic", False):
                    dynamic_values.append(reliability)
                if (cfg.background_enabled and reliability >= cfg.background_min_reliability
                        and evidence["point_count"] >= cfg.background_min_points):
                    # High point support AND a valid, reliable slow fit are
                    # required. This is not a static/identity classifier.
                    slow = max(0., 1. - float(evidence["motion_speed_mps"]) /
                               cfg.background_speed_scale_mps)
                    static_values.append(reliability * slow)
        motion = float(np.mean(dynamic_values)) if dynamic_values else None
        background = float(np.mean(static_values)) if static_values else None
        total = (cfg.score_weight * target + cfg.support_weight * support +
                 cfg.recency_weight * recency +
                 cfg.innovation_weight * (innovation if innovation is not None else 0.))
        if self.use_evidence:
            total += cfg.motion_weight * (motion if motion is not None else 0.)
            # A static hint can change the ranking, but cannot by itself
            # suppress an otherwise eligible confirmed hovering track.
            penalty = cfg.background_weight * (background if background is not None else 0.)
            total -= min(penalty, max(0., total - cfg.min_selection_score))
        return {"total": float(total), "target_score": target,
                "observation_support": support, "recency": recency,
                "innovation_consistency": innovation,
                "reliable_motion_evidence": motion,
                "static_background_hint": background,
                "background_status": "unknown" if background is None else "heuristic_hint",
                "last_observation_time": last["timestamp"],
                "observation_age_s": age, "window_observations": len(updates),
                "window_queries": query_count}

    def choose(self, now: float, histories: dict[int, list[dict]],
               eligible_ids: set[int] | None = None):
        cfg = self.cfg
        self.query_times.append(now)
        self.query_times = [t for t in self.query_times if t >= now - cfg.window_s]
        scores = {number: score for number, history in histories.items()
                  if (score := self._score(now, history, len(self.query_times))) is not None}
        eligible = set(histories) if eligible_ids is None else eligible_ids
        for number, score in scores.items():
            score["eligible_for_output"] = number in eligible
        trusted = {number: score for number, score in scores.items()
                   if number in eligible and score["total"] >= cfg.min_selection_score}
        lost_previous = self.current is not None and self.current not in trusted
        if self.current not in trusted:
            self.release("selected_track_lost")
        if not trusted:
            return None, scores, "no_trusted_track"
        winner = max(trusted, key=lambda number: (trusted[number]["total"], -number))
        if self.current is None:
            self.current = winner
            return winner, scores, "switch_after_loss" if lost_previous else "initial_select"
        if winner == self.current:
            self.challenger = None
            self.challenger_streak = 0
            return self.current, scores, "retain"
        if trusted[winner]["total"] < trusted[self.current]["total"] + cfg.switch_margin:
            self.challenger = None
            self.challenger_streak = 0
            return self.current, scores, "retain_margin"
        if self.challenger != winner:
            self.challenger = winner
            self.challenger_streak = 0
        self.challenger_streak += 1
        if self.challenger_streak < cfg.switch_consecutive:
            return self.current, scores, "wait_switch"
        self.current = winner
        self.challenger = None
        self.challenger_streak = 0
        return winner, scores, "switch_sustained_advantage"
