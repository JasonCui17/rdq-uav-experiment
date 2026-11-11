"""Use the reranked order among candidates that pass the original birth gates."""

from dataclasses import replace

import numpy as np

from ..candidate_adapter import EVIDENCE_FIELDS
from ..tracking import stable_order
from .ranking import RadarCandidateReranker


class BirthReranker:
    def __init__(self, frames, manifest, bundle):
        if manifest.get("schema_version") != 2:
            raise ValueError("Candidate reranking requires a schema 2 cache")
        for key, expected in bundle.cache_identity.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Candidate reranker expects a different cache {key}")
        self.sequences = {frame["query_time"]: frame["sequence_id"] for frame in frames}
        self.data_root = manifest["data_root"]
        self.bundle = bundle
        self.streams = {}
        self.scored_queries = 0
        self.preferred_queries = 0

    def __call__(self, timestamp, frame, eligible):
        sequence = self.sequences[timestamp]
        if sequence not in self.streams:
            self.streams[sequence] = RadarCandidateReranker(
                sequence, self.data_root, self.bundle.models, self.bundle.policy
            )
        stream = self.streams[sequence]
        obs = frame.dense
        original_order = stable_order(obs)
        if eligible is None or not eligible.any():
            if len(original_order):
                first = original_order[0]
                stream.observe_original(
                    timestamp, obs.xyz[first].astype(np.float32), obs.score[first]
                )
            else:
                stream.observe_original(timestamp)
            return frame

        if len(frame.evidence) != len(obs.xyz):
            raise ValueError("Candidate reranking requires complete dense evidence")
        dense = {
            "dense_xyz": obs.xyz.astype(np.float32),
            "dense_rank": obs.rank_value.astype(np.float32),
            "dense_score": obs.score.astype(np.float32),
            "dense_token_id": obs.token_id,
        }
        for name in EVIDENCE_FIELDS:
            dense[f"dense_{name}"] = np.array(
                [frame.evidence[int(i)][name] for i in obs.token_id]
            )
        result = stream.step(timestamp, dense)
        self.scored_queries += 1

        eligible_indices = {int(obs.token_id[i]): i for i in np.flatnonzero(eligible)}
        preferred = [
            eligible_indices[int(token)]
            for token in result.token_ids
            if int(token) in eligible_indices
        ]
        if not preferred:
            return frame
        self.preferred_queries += 1
        # Keep all candidates. Only their ordering at this birth can change.
        remaining = original_order[~np.isin(original_order, preferred)]
        order = np.concatenate((preferred, remaining))
        rank = np.empty_like(obs.rank_value)
        rank[order] = -np.arange(len(order))
        return replace(frame, dense=replace(obs, rank_value=rank))

    def close(self):
        for stream in self.streams.values():
            stream.close()
