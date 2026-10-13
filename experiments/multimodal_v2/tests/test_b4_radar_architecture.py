"""B4 uses the trained LearnSBE/MTCE path, never a frozen zero MTCE placeholder."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
from torch import nn
import yaml

from rdq_uav.multimodal_v2.radar_architecture import (
    MOTION_KEYS, validate_b4_radar_config, audit_b4_radar_detector,
)
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector, build_voxel_embedding
from rdq_uav.multimodal_v2.lidar_learned_sbe import LearnedSBEVoxelEmbed
from rdq_uav.multimodal_v2.radar_loss import CandidateLoss
from rdq_uav.multimodal_v2.radar_temporal_centroid import set_radar_train_mode
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.candidate_builders import RadarCandidateBuilder, RGBCandidateBuilder
from rdq_uav.multimodal_v2.model import MultimodalV2
from rdq_uav.multimodal_v2.pair_fusion import CandidatePairFusion
from rdq_uav.multimodal_v2.pair_loss import JointPairLoss
from rdq_uav.multimodal_v2.loss import MultimodalTargets
from rdq_uav.multimodal_v2.training import build_runtime, freeze_for_stage
from test_radar_mtce import config as radar_config, batch
from test_v2_pair_fusion import candidate

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prepare_b4", ROOT / "experiments/multimodal_v2/diagnostics/prepare_b4_learnsbe_mtce.py")
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


def b4_config():
    return yaml.safe_load((ROOT / "experiments/multimodal_v2/configs/b4_learnsbe_mtce.yaml").read_text())


def trained_radar():
    torch.manual_seed(42)
    detector = LiDARUAVDetector(radar_config())
    set_radar_train_mode(detector, "motion_only")
    optimizer = torch.optim.AdamW([p for p in detector.parameters() if p.requires_grad], lr=.0003)
    criterion = CandidateLoss(radar_config())
    for _ in range(2):
        optimizer.zero_grad()
        criterion(detector(batch()), batch())["loss"].backward()
        optimizer.step()
    detector.zero_grad(set_to_none=True)
    return detector.eval()


@pytest.mark.parametrize("case", ["sbe_lite", "no_motion", "reliability_only", "bad_history", "bad_bins"])
def test_b4_rejects_deprecated_or_disabled_radar(case):
    cfg = radar_config()
    voxel = cfg["model"]["voxel"]
    history = 1.
    if case == "sbe_lite": voxel["embedding"] = "sbe_lite"
    if case == "no_motion": voxel["motion_centroid"]["enabled"] = False
    if case == "reliability_only": voxel["motion_centroid"]["reliability_only"] = True
    if case == "bad_history": history = .5
    if case == "bad_bins": voxel["motion_centroid"]["num_bins"] = 4
    with pytest.raises(ValueError): validate_b4_radar_config(cfg, history)


def test_b4_legacy_config_fails_before_loading_detectors():
    cfg = b4_config()
    cfg["initialization"]["lidar_config"] = "experiments/multimodal_v2/configs/radar.yaml"
    with pytest.raises(ValueError, match="sbe_lite is deprecated"):
        build_runtime(cfg, ROOT, torch.device("cpu"))


def test_missing_embedding_defaults_to_learned_sbe():
    cfg = radar_config()["model"]
    cfg["voxel"].pop("embedding")
    assert isinstance(build_voxel_embedding(cfg), LearnedSBEVoxelEmbed)


def test_enabled_but_zero_frozen_mtce_is_rejected():
    with pytest.raises(ValueError, match="all zero"):
        audit_b4_radar_detector(LiDARUAVDetector(radar_config()))


def test_prepare_preserves_server_data_and_b1_and_checks_radar_architecture(tmp_path):
    radar_path = tmp_path / "radar.yaml"
    radar_path.write_text(yaml.safe_dump(radar_config()))
    detector = trained_radar()
    checkpoint = tmp_path / "b0.ckpt"
    payload = {"state_dict": {"network.lidar.detector." + key: value for key, value in detector.state_dict().items()},
               "hyper_parameters": {"config": {"experiment": {"stage": "B0"},
                                    "initialization": {"lidar_config": str(radar_path)}}}}
    torch.save(payload, checkpoint)
    cfg = b4_config()
    cfg["initialization"]["b1_checkpoint"] = str(tmp_path / "b1.ckpt")
    (tmp_path / "b1.ckpt").touch()
    cfg["data"]["root"] = "/server/actual/data"
    original = deepcopy(cfg)
    changed, report = prepare_module.prepare(cfg, checkpoint, tmp_path)
    assert cfg == original
    for key in ("data", "model", "loss", "training", "lightning"):
        assert changed[key] == cfg[key]
    assert changed["initialization"]["b1_checkpoint"] == cfg["initialization"]["b1_checkpoint"]
    assert changed["initialization"]["b0_checkpoint"] == "b0.ckpt"
    assert report["learned_sbe_weights"] and report["mtce_enabled"]
    assert all(value > 0 for value in report["motion_projection_norms"].values())
    # A deep/non-deep checkpoint must never be silently accepted by loose loading.
    mismatch = radar_config()
    mismatch["model"]["transformer"]["blocks_per_level"] = [4, 2, 2]
    radar_path.write_text(yaml.safe_dump(mismatch))
    with pytest.raises(RuntimeError, match="Missing key"):
        prepare_module.prepare(cfg, checkpoint, tmp_path)


def test_actual_mtce_enters_all_scales_and_stays_frozen_during_b4_updates():
    radar = trained_radar()
    report = audit_b4_radar_detector(radar)
    assert report["blocks_per_level"] == [2, 2, 2]
    value = batch()
    with torch.no_grad():
        raw = radar(value)
    assert len(raw["motion_features"]) == len(raw["temporal_stats"]) == 3
    assert all(stat["valid_pair_count"] > 0 for stat in raw["temporal_stats"])
    # Removing each individual projection changes actual output features.
    saved = {key: radar.state_dict()[key].clone() for key in MOTION_KEYS}
    for key in MOTION_KEYS:
        state = radar.state_dict()
        state[key].zero_()
        with torch.no_grad(): without = radar(value)["fine_features"]
        assert not torch.equal(raw["fine_features"], without), key
        state[key].copy_(saved[key])
    vision = nn.Module()
    vision.builder = RGBCandidateBuilder()
    model = MultimodalV2(radar, vision, nn.Identity(), nn.Identity(), pair_fusion=CandidatePairFusion())
    freeze_for_stage(SimpleNamespace(model=model), "B4")
    original = {key: value.clone() for key, value in radar.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    builder = RadarCandidateBuilder(CandidateSelector({"raw_topk": 50, "pre_nms_topk": 50, "final_topk": 50, "nms_radius_m": 1.}, raw_only=True), use_raw=True)
    target = MultimodalTargets(value["target_xyz"], value["target_valid"],
                              torch.tensor([[40., 40., 60., 60.]] * 2), torch.ones(2, dtype=torch.bool))
    for _ in range(2):
        optimizer.zero_grad()
        with torch.no_grad(): rc = builder(radar(value))
        rc = replace(rc, projected_xy_px=torch.tensor([[50., 50.]] * rc.n), projection_valid=torch.ones(rc.n, dtype=torch.bool))
        vc = replace(candidate("V", 2), batch_index=torch.tensor([0, 1]),
                     feature=vision.builder.feature_proj(torch.randn(2, 256)))
        loss = JointPairLoss()(model.pair_fusion(rc, vc, num_samples=2), target)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pair_fusion.parameters())
        assert all(p.grad is None for p in radar.parameters())
        optimizer.step()
    assert all(torch.equal(tensor, original[key]) for key, tensor in radar.state_dict().items())
