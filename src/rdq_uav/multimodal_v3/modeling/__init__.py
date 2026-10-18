"""Candidate-level geometry constrained Multimodal V2."""

from .contracts import CandidateBatch, CrossModalEvidence, MultimodalOutput
from .interaction import CandidateCrossAttention
from .lidar import LiDARCandidateModel
from .loss import CandidateRankingLoss, MultimodalTargets
from .model import MultimodalV2
from .scoring import CandidateScoring, EvidenceScoreHead
from .vision import VisionCandidateModel

__all__ = [
    "CandidateBatch", "CrossModalEvidence", "MultimodalOutput",
    "CandidateCrossAttention", "LiDARCandidateModel", "VisionCandidateModel",
    "CandidateScoring", "EvidenceScoreHead", "CandidateRankingLoss",
    "MultimodalTargets", "MultimodalV2",
]
