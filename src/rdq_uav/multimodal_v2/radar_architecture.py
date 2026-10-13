"""B4's required Learned-SBE + trained three-scale MTCE radar contract."""
from __future__ import annotations

import torch

from .lidar_learned_sbe import LearnedSBEVoxelEmbed

MOTION_KEYS = ("voxel_embed.motion_proj.weight", "merge01.motion_proj.weight",
               "merge12.motion_proj.weight")


def validate_b4_radar_config(radar_config, radar_history_s):
    model = radar_config["model"]
    voxel = model["voxel"]
    if voxel.get("embedding") != "learned_sbe":
        raise ValueError("B4 requires learned_sbe; sbe_lite is deprecated for B4. "
                         "Use prepare_b4_learnsbe_mtce.py with a trained MTCE B0 checkpoint.")
    motion = voxel.get("motion_centroid", {})
    if not motion.get("enabled", False):
        raise ValueError("B4 requires enabled three-scale MTCE (voxel.motion_centroid.enabled=true)")
    if model["dim"] != 128:
        raise ValueError("B4 Learned-SBE + MTCE requires radar dim=128")
    if float(radar_history_s) != 1. or float(motion.get("history_s", 1.)) != 1.:
        raise ValueError("B4 MTCE requires the causal 1-second radar history")
    if int(motion.get("num_bins", 5)) != 5:
        raise ValueError("B4 MTCE requires five time bins")
    if motion.get("reliability_only", False):
        raise ValueError("B4 requires the MTCE speed feature, not reliability_only")
    return {"embedding": "learned_sbe", "point_dim": int(voxel.get("learned_sbe", {}).get("point_dim", 32)),
            "mtce_enabled": True, "mtce_levels": ["L0", "L1", "L2"],
            "blocks_per_level": list(model["transformer"]["blocks_per_level"])}


def audit_b4_radar_detector(detector):
    """Check loaded weights, not just a YAML flag on a zero-initialized module."""
    if not isinstance(detector.voxel_embed, LearnedSBEVoxelEmbed):
        raise ValueError("B4 radar is not LearnedSBEVoxelEmbed")
    if not detector.motion_enabled or detector.temporal_centroid is None:
        raise ValueError("B4 radar MTCE is not active in the forward path")
    if detector.temporal_centroid.reliability_only:
        raise ValueError("B4 MTCE speed feature must be enabled")
    state = detector.state_dict()
    norms = {}
    for key in MOTION_KEYS:
        value = state.get(key)
        if value is None or value.shape != (128, 3):
            raise ValueError(f"B4 requires the MTCE projection {key} with shape [128,3]")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite trained MTCE weights: {key}")
        if not torch.count_nonzero(value):
            raise ValueError(f"{key} is all zero: frozen B4 would not use this MTCE contribution. "
                             "Load a trained LearnSBE + MTCE B0 checkpoint.")
        norms[key] = float(value.float().norm())
    return {"embedding_class": type(detector.voxel_embed).__name__,
            "mtce_enabled": True, "motion_projection_norms": norms,
            "blocks_per_level": [len(getattr(detector, f"encoder{i}").blocks) for i in range(3)]}


def confirm_b4_radar(runtime):
    if runtime.initialization.get("stage") != "B4":
        return None
    report = audit_b4_radar_detector(runtime.lidar_detector)
    runtime.initialization["radar_architecture"] = report
    print("B4 radar: LearnedSBEVoxelEmbed + MTCE L0/L1/L2; loaded projection norms:",
          report["motion_projection_norms"])
    return report
