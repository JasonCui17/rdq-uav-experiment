"""Inference for the numeric histogram trees exported from the fitted models."""

from dataclasses import dataclass

import numpy as np

from .ranking import CandidatePolicy, MODEL_WIDTHS


@dataclass(frozen=True)
class NumericGBDT:
    baseline: float
    trees: tuple

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        rows = np.arange(len(features))
        prediction = np.full(len(features), self.baseline, dtype=np.float64)
        for tree in self.trees:
            node = np.zeros(len(features), dtype=int)
            active = rows
            while len(active):
                active = active[~tree["is_leaf"][node[active]]]
                current = tree[node[active]]
                value = features[active, current["feature_idx"]]
                left = np.where(
                    np.isnan(value),
                    current["missing_go_to_left"],
                    value <= current["num_threshold"],
                )
                node[active] = np.where(left, current["left"], current["right"])
            prediction += tree["value"][node]
        return prediction


@dataclass(frozen=True)
class RankingBundle:
    models: tuple
    policy: CandidatePolicy
    cache_identity: dict
    training_sequences: tuple
    source_sha256: str


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        if data["version"].item() != 1 or tuple(data["widths"]) != MODEL_WIDTHS:
            raise ValueError(
                "Candidate reranker bundle format or feature width mismatch"
            )
        models = tuple(
            NumericGBDT(
                float(bias), tuple(np.split(data[f"trees_{i}"], data[f"cuts_{i}"]))
            )
            for i, bias in enumerate(data["bias"])
        )
        policy = CandidatePolicy(
            gates=tuple(map(tuple, data["gates"])),
            guard=data["guard"].item(),
            local_promotion=(
                data["min_extra_points"].item(),
                data["max_quality_gap"].item(),
            ),
            admission=(data["admission_method"].item(), data["admission_quota"].item()),
            echo_margin=data["echo_margin"].item(),
        )
        identity = {
            key: data[key].item()
            for key in (
                "checkpoint_sha256",
                "lidar_config_sha256",
                "candidate_config_sha256",
            )
        }
        return RankingBundle(
            models,
            policy,
            identity,
            tuple(data["training_sequences"]),
            data["source_sha256"].item(),
        )
