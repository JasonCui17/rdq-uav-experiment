"""Stone Soup multi-target adapter around frozen dense radar candidates.

No GT enters this module. Stone Soup owns the CV filter, global one-to-one
association, multi-measurement confirmation, and time-based deletion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

import numpy as np

from .candidate_adapter import CandidateFrame
from .tracking import suppress
from .track_selection import SelectionConfig, TrackSelector

try:
    from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
    from stonesoup.dataassociator.tree import DetectionKDTreeGNN2D
    from stonesoup.deleter.time import UpdateTimeDeleter, UpdateTimeStepsDeleter
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.initiator.simple import MultiMeasurementInitiator, SimpleMeasurementInitiator
    from stonesoup.measures import Mahalanobis
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.updater.kalman import KalmanUpdater
    from stonesoup.tracker.simple import MultiTargetTracker
    from stonesoup.types.detection import Detection
    from stonesoup.types.hypothesis import SingleDistanceHypothesis
    from stonesoup.types.multihypothesis import MultipleHypothesis
    from stonesoup.types.state import GaussianState
    from stonesoup.types.update import Update
    import stonesoup
except ImportError as exc:
    raise ImportError("V3 multi-target methods require stonesoup==1.9.1; install requirements-tracking.txt") from exc

if stonesoup.__version__ != "1.9.1":
    raise RuntimeError(f"V3 multi-target API is pinned to stonesoup==1.9.1, found {stonesoup.__version__}")


@dataclass(frozen=True)
class MultiFilterConfig:
    measurement_std_m: float = .5
    process_noise_density: float = 4.
    initial_position_std_m: float = .75
    initial_velocity_std_mps: float = 5.
    mahalanobis_gate_sq: float = 11.34
    max_innovation_m: float = 4.
    max_speed_mps: float = 20.
    quality_weight: float = .5
    missed_cost: float = 8.
    local_radius_min_m: float = 1.
    local_radius_max_m: float = 4.
    local_pre_topk: int = 100
    local_final_topk: int = 20
    local_nms_radius_m: float = .3
    birth_pre_topk: int = 100
    birth_max_per_step: int = 10
    birth_nms_radius_m: float = 1.
    # ab3dmot uses cumulative observation hits and missed QUERY counts.
    # This is lifecycle adaptation, not the original 3D-box AB3DMOT tracker.
    use_kdtree: bool = False
    lifecycle_mode: str = "stone_soup"
    min_hits: int = 3
    max_age: int = 24
    capacity_pruning_enabled: bool = True
    max_active_tracks: int = 10
    max_tentative_tracks: int = 12
    duplicate_radius_m: float = .5
    confirm_hits: int = 3
    max_prediction_age_s: float = .5
    max_tentative_age_s: float = .35
    reset_gap_s: float = 2.

    def __post_init__(self):
        if not isinstance(self.use_kdtree, bool):
            raise ValueError("use_kdtree must be a boolean")
        if self.lifecycle_mode not in ("stone_soup", "ab3dmot"):
            raise ValueError("Unknown lifecycle_mode")
        if any(type(value) is not int or value < 1 for value in (self.min_hits, self.max_age)):
            raise ValueError("min_hits and max_age must be positive integers")
        if self.lifecycle_mode == "ab3dmot" and self.capacity_pruning_enabled:
            raise ValueError("AB3DMOT lifecycle requires capacity_pruning_enabled=false")
        if not isinstance(self.capacity_pruning_enabled, bool):
            raise ValueError("capacity_pruning_enabled must be a boolean")
        positive = (self.measurement_std_m, self.process_noise_density,
                    self.initial_position_std_m, self.initial_velocity_std_mps,
                    self.mahalanobis_gate_sq, self.max_innovation_m, self.max_speed_mps,
                    self.missed_cost, self.local_radius_min_m, self.local_radius_max_m,
                    self.max_prediction_age_s, self.max_tentative_age_s, self.reset_gap_s)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Multi-filter scales must be finite and positive")
        if self.local_radius_min_m > self.local_radius_max_m:
            raise ValueError("Invalid local radius bounds")
        if any(x < 0 or not math.isfinite(x) for x in
               (self.quality_weight, self.local_nms_radius_m, self.birth_nms_radius_m,
                self.duplicate_radius_m)):
            raise ValueError("Invalid nonnegative filter parameter")
        if min(self.local_pre_topk, self.local_final_topk, self.birth_pre_topk,
               self.birth_max_per_step, self.max_active_tracks,
               self.max_tentative_tracks, self.confirm_hits) < 1:
            raise ValueError("Multi-filter counts must be positive")


class GatedQualityHypothesiser(DistanceHypothesiser):
    """Stone Soup hypotheses with explicit Euclidean, speed and score costs."""
    def __init__(self, *args, config: MultiFilterConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_config = config or MultiFilterConfig()

    def hypothesise(self, track, detections, timestamp, **kwargs):
        cfg = self.filter_config
        base = super().hypothesise(track, detections, timestamp, **kwargs)
        kept = []
        for hyp in base:
            if not hyp:
                kept.append(SingleDistanceHypothesis(hyp.prediction, hyp.measurement,
                                                     cfg.missed_cost))
                continue
            predicted_xyz = np.asarray(hyp.prediction.state_vector).reshape(-1)[[0, 2, 4]]
            observation_xyz = np.asarray(hyp.measurement.state_vector).reshape(-1)
            if np.linalg.norm(observation_xyz - predicted_xyz) > cfg.max_innovation_m:
                continue
            posterior = self.updater.update(hyp)
            velocity = np.asarray(posterior.state_vector).reshape(-1)[[1, 3, 5]]
            if np.linalg.norm(velocity) > cfg.max_speed_mps:
                continue
            score = float(hyp.measurement.metadata["score"])
            cost = .5 * float(hyp.distance) ** 2 - cfg.quality_weight * math.log(max(score, 1e-8))
            if cost < cfg.missed_cost:
                kept.append(SingleDistanceHypothesis(hyp.prediction, hyp.measurement,
                                                     cost, hyp.measurement_prediction))
        return MultipleHypothesis(sorted(kept, reverse=True))


class BirthOnlyInitiator(SimpleMeasurementInitiator):
    def initiate(self, detections, timestamp, **kwargs):
        eligible = {d for d in detections if d.metadata.get("birth_eligible", False)}
        return super().initiate(eligible, timestamp, **kwargs)


def _stamp(timestamp: float) -> datetime:
    if not math.isfinite(timestamp):
        raise ValueError("Nonfinite query time")
    return datetime.fromtimestamp(timestamp, timezone.utc)


def _state_xyz(state):
    vector = np.asarray(state.state_vector).reshape(-1)
    return vector[[0, 2, 4]], vector[[1, 3, 5]]


def _state_observation(state):
    if not isinstance(state, Update) or not state.hypothesis:
        return None, None
    detection = state.hypothesis.measurement
    predicted, _ = _state_xyz(state.hypothesis.prediction) if state.hypothesis.prediction is not None else (None, None)
    innovation = (None if predicted is None else
                  float(np.linalg.norm(np.asarray(detection.state_vector).reshape(-1) - predicted)))
    return detection.metadata, innovation


class MultiCandidateTracker:
    """One sequence; emits a complete record for each confirmed/tentative track."""
    def __init__(self, config: MultiFilterConfig, selection: SelectionConfig, *, select_evidence: bool):
        self.config = config
        self.selector = TrackSelector(selection, use_evidence=select_evidence)
        self.last_time = None
        self.track_numbers: dict[str, int] = {}
        self.history: dict[int, list[dict]] = {}
        self.last_recorded_length: dict[str, int] = {}
        self.next_track_number = 1
        self.ever_selected = False
        self.switch_count = 0
        self._build()

    def _build(self):
        cfg = self.config
        transition = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(cfg.process_noise_density) for _ in range(3)])
        measurement = LinearGaussian(ndim_state=6, mapping=(0, 2, 4),
                                     noise_covar=np.eye(3) * cfg.measurement_std_m ** 2)
        self.predictor = KalmanPredictor(transition)
        updater = KalmanUpdater(measurement)
        hypo = GatedQualityHypothesiser(self.predictor, updater, Mahalanobis(),
                                        missed_distance=math.sqrt(cfg.mahalanobis_gate_sq),
                                        config=cfg)
        assoc = (DetectionKDTreeGNN2D(
            hypothesiser=hypo, predictor=self.predictor, updater=updater,
            number_of_neighbours=None,
            # Query a conservative superset at the boundary; the original
            # exact Euclidean gate remains in GatedQualityHypothesiser.
            max_distance=np.nextafter(cfg.max_innovation_m, np.inf),
            max_distance_covariance_multiplier=None)
            if cfg.use_kdtree else GNNWith2DAssignment(hypo))
        covariance = np.diag([cfg.initial_position_std_m ** 2,
                              cfg.initial_velocity_std_mps ** 2] * 3)
        prior = GaussianState(np.zeros((6, 1)), covariance)
        if cfg.lifecycle_mode == "ab3dmot":
            # AB3DMOT births all eligible unmatched observations immediately.
            # Immature tracks participate in the SAME association as mature
            # tracks; min_hits gates output, not admission into the filter bank.
            self.initiator = BirthOnlyInitiator(prior, measurement)
            self.initiator.holding_tracks = set()
        elif cfg.confirm_hits == 1:
            self.initiator = BirthOnlyInitiator(prior, measurement)
            self.initiator.holding_tracks = set()
        else:
            self.initiator = MultiMeasurementInitiator(
                prior_state=prior, measurement_model=measurement,
                deleter=UpdateTimeDeleter(timedelta(seconds=cfg.max_tentative_age_s)),
                data_associator=assoc, updater=updater, min_points=cfg.confirm_hits,
                updates_only=True, initiator=BirthOnlyInitiator(prior, measurement))
        self.tracker = MultiTargetTracker(
            initiator=self.initiator,
            deleter=(UpdateTimeStepsDeleter(cfg.max_age)
                     if cfg.lifecycle_mode == "ab3dmot" else
                     UpdateTimeDeleter(timedelta(seconds=cfg.max_prediction_age_s))),
            detector=None, data_associator=assoc, updater=updater)
        self.measurement = measurement

    def _candidate_pool(self, time, frame: CandidateFrame):
        cfg = self.config
        dense, global_pool = frame.dense, frame.global_pool
        selected = set(int(token) for token in global_pool.token_id)
        births = suppress(dense, cfg.birth_pre_topk, cfg.birth_max_per_step,
                          cfg.birth_nms_radius_m)
        birth_tokens = set(map(int, births.token_id))
        selected.update(birth_tokens)
        for track in self.tracker.tracks | self.initiator.holding_tracks:
            predicted = self.predictor.predict(track, timestamp=time)
            center = np.asarray(predicted.state_vector).reshape(-1)[[0, 2, 4]]
            position_covar = np.asarray(predicted.covar)[np.ix_([0, 2, 4], [0, 2, 4])]
            radius = float(np.clip(math.sqrt(cfg.mahalanobis_gate_sq *
                                             np.linalg.eigvalsh(position_covar +
                                                 np.eye(3) * cfg.measurement_std_m**2).max()),
                                   cfg.local_radius_min_m, cfg.local_radius_max_m))
            local_ids = np.flatnonzero(np.linalg.norm(dense.xyz - center, axis=1) <= radius)
            local = suppress(dense.take(local_ids), cfg.local_pre_topk,
                             cfg.local_final_topk, cfg.local_nms_radius_m)
            selected.update(map(int, local.token_id))
        detections = set()
        for i, token in enumerate(dense.token_id):
            token = int(token)
            if token not in selected:
                continue
            metadata = {"token_id": token, "score": float(dense.score[i]),
                        "birth_eligible": token in birth_tokens,
                        **frame.evidence.get(token, {})}
            detections.add(Detection(dense.xyz[i].reshape(3, 1), timestamp=time,
                                     measurement_model=self.measurement, metadata=metadata))
        return detections, len(birth_tokens)

    def _number(self, track):
        if track.id not in self.track_numbers:
            self.track_numbers[track.id] = self.next_track_number
            self.history[self.next_track_number] = []
            self.next_track_number += 1
        return self.track_numbers[track.id]

    def _confirmation_hits(self):
        return (self.config.min_hits if self.config.lifecycle_mode == "ab3dmot"
                else self.config.confirm_hits)

    @staticmethod
    def _lifecycle_counts(states):
        # Derive the AB3DMOT counters from Stone Soup's actual states, avoiding
        # a second mutable counter implementation alongside the filter bank.
        hits = sum(isinstance(state, Update) and bool(state.hypothesis)
                   for state in states)
        time_since_update = 0
        for state in reversed(states):
            if isinstance(state, Update) and state.hypothesis:
                break
            time_since_update += 1
        return hits, time_since_update

    def _record(self, tracks, active, tentative, removal_reasons):
        for track in sorted(tracks, key=lambda tr: (tr[0].timestamp, int(tr[0].hypothesis.measurement.metadata["token_id"]))):
            number = self._number(track)
            start = self.last_recorded_length.get(track.id, 0)
            for state_index in range(start, len(track)):
                state = track[state_index]
                xyz, velocity = _state_xyz(state)
                metadata, innovation = _state_observation(state)
                hits, missed_queries = self._lifecycle_counts(track[:state_index + 1])
                confirmed_at_state = hits >= self._confirmation_hits()
                self.history[number].append({
                    "track_id": number, "timestamp": state.timestamp.timestamp(),
                    "xyz": xyz.tolist(), "velocity_mps": velocity.tolist(),
                    "observation_xyz": (None if metadata is None else
                        np.asarray(state.hypothesis.measurement.state_vector).reshape(-1).tolist()),
                    "hits": hits,
                    "time_since_update": missed_queries,
                    "birth_time": track[0].timestamp.timestamp(),
                    "birth_token_id": int(track[0].hypothesis.measurement.metadata["token_id"]),
                    "output_kind": "updated" if metadata is not None else "predicted",
                    "selected_token_id": None if metadata is None else int(metadata["token_id"]),
                    "score": None if metadata is None else float(metadata["score"]),
                    "innovation_m": innovation,
                    "status": "confirmed" if confirmed_at_state else "tentative",
                    "evidence": {} if metadata is None else
                        {k: metadata[k] for k in ("point_count", "motion_valid", "motion_dynamic",
                                               "motion_reliability", "motion_speed_mps") if k in metadata},
                })
            self.last_recorded_length[track.id] = len(track)
            if self.history[number]:
                self.history[number][-1]["status"] = (
                    "confirmed" if track in active else "tentative" if track in tentative else "deleted")
                if track in removal_reasons:
                    self.history[number][-1]["removal_reason"] = removal_reasons[track]

    def _prune(self, active, tentative):
        cfg = self.config
        removed = {}
        ranked = sorted(active | tentative,
                        key=lambda tr: (tr not in active, -sum(isinstance(s, Update) for s in tr),
                                        -float(tr.metadata.get("score", 0)), tr[0].timestamp,
                                        int(tr[0].hypothesis.measurement.metadata["token_id"])))
        keep = []
        for track in ranked:
            xyz, _ = _state_xyz(track.state)
            if any(np.linalg.norm(xyz - _state_xyz(other.state)[0]) <= cfg.duplicate_radius_m
                   for other in keep):
                removed[track] = "duplicate"
            else:
                keep.append(track)
        # Keep duplicate handling identical to the legacy experiment. Only
        # capacity eviction is optional; initiation and timeout deletion remain
        # owned by Stone Soup and do not depend on available track slots.
        if cfg.capacity_pruning_enabled:
            for group, cap in ((active, cfg.max_active_tracks), (tentative, cfg.max_tentative_tracks)):
                survivors = [tr for tr in keep if tr in group]
                for track in survivors[cap:]:
                    removed[track] = "capacity"
        for track in removed:
            active.discard(track)
            tentative.discard(track)
        return removed

    def _select_tracks(self, t, live_ids, active_ids):
        """Independent output view over one physical filter bank."""
        previous_choice = self.selector.current
        chosen, scores, reason = self.selector.choose(
            t, {number: self.history[number] for number in live_ids}, active_ids)
        latest = None if chosen is None else self.history[chosen][-1]
        output_kind = "none" if latest is None else latest["output_kind"]
        if reason == "switch_sustained_advantage":
            self.switch_count += 1
        reacquisition = (chosen is not None and self.ever_selected and
                         (previous_choice is None or reason == "switch_after_loss"))
        if chosen is not None:
            self.ever_selected = True
        return {
            "timestamp": t, "track_id": chosen,
            "lifecycle_mode": self.config.lifecycle_mode,
            "hits": None if latest is None else latest["hits"],
            "time_since_update": None if latest is None else latest["time_since_update"],
            "status": ("confirmed" if chosen else "tentative" if live_ids - active_ids else
                       "no_trusted_track" if active_ids else
                       "lost" if self.ever_selected else "uninitialized"),
            "event": reason, "output_kind": output_kind,
            "xyz": None if latest is None else latest["xyz"],
            "velocity_mps": None if latest is None else latest["velocity_mps"],
            "selected_token_id": None if latest is None or output_kind == "predicted" else latest["selected_token_id"],
            "selected_score": None if latest is None or output_kind == "predicted" else latest["score"],
            "observation_xyz": None if latest is None else latest["observation_xyz"],
            "innovation_m": None if latest is None or output_kind == "predicted" else latest["innovation_m"],
            "track_scores": scores,
            "selection_score": None if chosen is None else scores[chosen]["total"],
            "switch_reason": reason, "switch_count": self.switch_count,
            "accepted_observation": output_kind == "updated", "reacquisition": reacquisition,
            "measurement_age_s": None if chosen is None else t - max(
                row["timestamp"] for row in self.history[chosen] if row["output_kind"] == "updated"),
        }

    def step(self, timestamp: float, frame: CandidateFrame) -> dict:
        t = float(timestamp)
        time = _stamp(t)
        if self.last_time is not None and t <= self.last_time:
            raise ValueError("Queries must be strictly chronological within sequence")
        before = self.tracker.tracks | self.initiator.holding_tracks
        reset = self.last_time is not None and t - self.last_time > self.config.reset_gap_s
        removal_reasons = {}
        if reset:
            removal_reasons.update({track: "time_gap" for track in before})
            self.tracker.tracks.clear()
            self.initiator.holding_tracks.clear()
            self.selector.release("time_gap")
        self.last_time = t
        detections, birth_count = self._candidate_pool(time, frame)
        self.tracker.update_tracker(time, detections)
        if self.config.lifecycle_mode == "ab3dmot":
            # Stone Soup owns every live track. Partition only for output and
            # logging; do not withhold immature tracks from association.
            active = {track for track in self.tracker.tracks
                      if self._lifecycle_counts(track)[0] >= self.config.min_hits}
            tentative = self.tracker.tracks - active
        else:
            active, tentative = self.tracker.tracks, self.initiator.holding_tracks
        after = active | tentative
        for track in before - after:
            removal_reasons.setdefault(track, "max_age" if self.config.lifecycle_mode == "ab3dmot"
                                       else "expired")
        removal_reasons.update(self._prune(active, tentative))
        after = active | tentative
        if self.config.lifecycle_mode == "ab3dmot":
            # _prune acts on the partition copies; synchronize duplicate
            # removals back into Stone Soup's bank before the next query.
            self.tracker.tracks.intersection_update(after)
        self._record(before | after | set(removal_reasons), active, tentative, removal_reasons)
        result = {
            **self._select_tracks(t, {self._number(track) for track in after},
                                  {self._number(track) for track in active}),
            "active_track_count": len(active), "tentative_track_count": len(tentative),
            "birth_candidate_count": birth_count, "detection_count": len(detections),
            "birth_candidate_token_ids": sorted(int(d.metadata["token_id"]) for d in detections
                                                 if d.metadata["birth_eligible"]),
            "new_track_ids": sorted(self._number(track) for track in after - before),
            "track_states": [{"track_id": self._number(track),
                              "status": "confirmed" if track in active else "tentative",
                              **self.history[self._number(track)][-1]}
                             for track in sorted(after, key=self._number)],
            "removed_tracks": [{"track_id": self._number(track), "reason": why}
                               for track, why in sorted(removal_reasons.items(), key=lambda x: self._number(x[0]))],
        }
        self.last_step_result = result
        return result


class SequenceMultiTrackers:
    def __init__(self, filter_config: MultiFilterConfig, selection: SelectionConfig, *, select_evidence: bool, shared_bank=None):
        self.shared_bank = shared_bank
        self.filter_config = filter_config
        self.selection = selection
        self.select_evidence = select_evidence
        self.states = {}

    def step(self, sequence_id: str, timestamp: float, frame: CandidateFrame):
        if self.shared_bank is not None:
            source = self.shared_bank.states.get(sequence_id)
            if source is None or source.last_time != timestamp:
                raise ValueError("Shared bank must be stepped once before its output view")
            if sequence_id not in self.states:
                # A shallow view shares filter history, but only calls selection.
                # It never builds, predicts, updates, prunes or records tracks.
                view = copy.copy(source)
                view.selector = TrackSelector(self.selection, use_evidence=self.select_evidence)
                view.ever_selected = False
                view.switch_count = 0
                self.states[sequence_id] = view
            view = self.states[sequence_id]
            common = source.last_step_result
            if view.last_time is not None and timestamp - view.last_time > self.filter_config.reset_gap_s:
                view.selector.release("time_gap")
            view.last_time = timestamp
            live_ids = {row["track_id"] for row in common["track_states"]}
            active_ids = {row["track_id"] for row in common["track_states"]
                          if row["status"] == "confirmed"}
            return {**common, **view._select_tracks(timestamp, live_ids, active_ids)}
        # dict.setdefault evaluates its default even when the key exists;
        # constructing unused Stone Soup component graphs every query is wasteful.
        if sequence_id not in self.states:
            self.states[sequence_id] = MultiCandidateTracker(
                self.filter_config, self.selection, select_evidence=self.select_evidence)
        state = self.states[sequence_id]
        return state.step(timestamp, frame)
