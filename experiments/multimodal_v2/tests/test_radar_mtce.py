"""MTCE statistics, A2 compatibility, and frozen-backbone gradient gates."""
from pathlib import Path

import pytest
import torch
import yaml

from rdq_uav.multimodal_v2.radar_geometry import HierarchyBuilder
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_loss import CandidateLoss
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.radar_temporal_centroid import (
    TemporalCentroidFeatures, load_a2_warmstart, set_radar_train_mode,
)

ROOT = Path(__file__).resolve().parents[3]


def config(motion=True):
    cfg = yaml.safe_load((ROOT / "experiments/multimodal_v2/configs/radar_a2_mtce3.yaml").read_text())
    cfg["model"]["voxel"]["motion_centroid"]["enabled"] = motion
    return cfg


def batch():
    # Same coordinates in different samples must never share a voxel statistic.
    times = [-.9, -.7, -.5, -.3, -.1]
    points, dt, sample = [], [], []
    for b in range(2):
        for i, time in enumerate(times):
            for j in range(3):
                points.append([.05 + .04 * i if b == 0 else .2, .05 + .01*j, .1])
                dt.append(time)
                sample.append(b)
    return {"points": torch.tensor(points), "delta_t": torch.tensor(dt),
            "point_batch_index": torch.tensor(sample),
            "sensor_id": torch.arange(len(points)) % 2, "num_samples": 2,
            "target_xyz": torch.tensor([[.13,.06,.1],[.2,.06,.1]]),
            "target_valid": torch.ones(2,dtype=torch.bool)}


def warmstart_file(tmp_path, detector):
    path = tmp_path / "a2.ckpt"
    torch.save({"state_dict": {"network.lidar.detector."+k:v for k,v in detector.state_dict().items()}}, path)
    return path


def test_statistics_known_motion_and_sample_isolation():
    b = batch(); h = HierarchyBuilder()(b["points"],b["point_batch_index"])
    features, debug = TemporalCentroidFeatures({})(b,h,True)
    for level, f, d in zip(h.levels,features,debug):
        assert f.shape == (len(level.coords),3)
        torch.testing.assert_close(f[0],torch.tensor([torch.log1p(torch.tensor(.2)),1.,torch.log1p(torch.tensor(3.))]))
        assert f[1,0].abs() < 1e-6
        assert torch.equal(d["bin_count"].sum(1), level.point_count)
        assert d["valid_pair_count"] == 8


def test_boundaries_missing_pairs_and_small_gap():
    b = batch()
    b["delta_t"][:] = -.2
    features,_ = TemporalCentroidFeatures({})(b,HierarchyBuilder()(b["points"],b["point_batch_index"]))
    assert all(torch.count_nonzero(f) == 0 for f in features)
    b["delta_t"] = torch.tensor([-1.,-.8,-.6,-.4,-.2,0.]*5)
    _,debug=TemporalCentroidFeatures({})(b,HierarchyBuilder()(b["points"],b["point_batch_index"]),True)
    assert debug[0]["bin_count"].sum() == 30
    b["delta_t"] = torch.tensor([-.801,-.799]*15)
    features,debug=TemporalCentroidFeatures({})(b,HierarchyBuilder()(b["points"],b["point_batch_index"]),True)
    assert all(torch.count_nonzero(f) == 0 for f in features)
    assert debug[0]["rejected_small_gap_count"] == 2


@pytest.mark.parametrize("dt",[.01,-1.01,float("nan")])
def test_invalid_time_rejected(dt):
    b=batch(); b["delta_t"][0]=dt
    h=HierarchyBuilder()(b["points"],b["point_batch_index"])
    with pytest.raises(ValueError): TemporalCentroidFeatures({})(b,h)


def test_empty_hierarchy_and_no_cross_empty_bin():
    b=batch()
    for key in ("points","delta_t","point_batch_index","sensor_id"): b[key]=b[key][:0]
    h=HierarchyBuilder()(b["points"],b["point_batch_index"])
    features,_=TemporalCentroidFeatures({})(b,h)
    assert [f.shape for f in features] == [torch.Size([0,3])]*3
    b=batch(); b["delta_t"]=torch.tensor([-.9,-.1]*15)
    features,_=TemporalCentroidFeatures({})(b,HierarchyBuilder()(b["points"],b["point_batch_index"]))
    assert all(torch.count_nonzero(f) == 0 for f in features)


def test_parent_centroids_use_raw_point_weights():
    b=batch(); b["points"][0,0]=.7
    h=HierarchyBuilder()(b["points"],b["point_batch_index"])
    _,debug=TemporalCentroidFeatures({})(b,h,True)
    torch.testing.assert_close(debug[1]["bin_centroid"][0,0], b["points"][:3].mean(0))
    assert debug[1]["bin_count"][0,0] == 3


def test_zero_initialization_exact_forward_selector_and_strict_load(tmp_path):
    torch.manual_seed(42)
    original=LiDARUAVDetector(config(False)).eval()
    changed=LiDARUAVDetector(config()).eval()
    path=warmstart_file(tmp_path,original)
    report=load_a2_warmstart(changed,path)
    assert len(report["new_zero_initialized_keys"]) == 3
    b=batch()
    with torch.no_grad(): before,after=original(b),changed(b)
    for key in ("logits","residual_xyz","pred_xyz","fine_features"):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    selector=CandidateSelector({"raw_topk":10,"pre_nms_topk":50,"nms_radius_m":1.,"final_topk":10})
    for a,c in zip(selector(before),selector(after)):
        for kind in ("raw","nms"):
            for key in a[kind]: torch.testing.assert_close(a[kind][key],c[kind][key],rtol=0,atol=0)
    state=torch.load(path,weights_only=False); state["state_dict"].pop(next(iter(state["state_dict"])))
    torch.save(state,path)
    with pytest.raises(ValueError,match="mismatch"): load_a2_warmstart(changed,path)


def test_two_updates_only_motion_changes_and_reload(tmp_path):
    original=LiDARUAVDetector(config(False))
    model=LiDARUAVDetector(config()).train()
    load_a2_warmstart(model,warmstart_file(tmp_path,original))
    set_radar_train_mode(model,"motion_only")
    named={k:p for k,p in model.named_parameters() if p.requires_grad}
    assert set(named)=={"voxel_embed.motion_proj.weight","merge01.motion_proj.weight","merge12.motion_proj.weight"}
    assert sum(p.numel() for p in named.values()) == 1152
    initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(named.values(),lr=.0003)
    loss_fn=CandidateLoss(config()); b=batch()
    for _ in range(2):
        optimizer.zero_grad(); loss=loss_fn(model(b),b)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for p in named.values():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
        optimizer.step()
    for k,v in model.state_dict().items():
        if k in named: assert not torch.equal(v,initial[k])
        else: assert torch.equal(v,initial[k])
    path=tmp_path/"mtce.pt"; torch.save(model.state_dict(),path)
    restored=LiDARUAVDetector(config());restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(model(b)["pred_xyz"],restored(b)["pred_xyz"],rtol=0,atol=0)
    zero=torch.zeros(3,3)
    for module in (model.voxel_embed,model.merge01,model.merge12):
        assert torch.count_nonzero(module.motion_proj(zero)) == 0


def test_reliability_only_preserves_support():
    b=batch();h=HierarchyBuilder()(b["points"],b["point_batch_index"])
    full,_=TemporalCentroidFeatures({})(b,h)
    reliability,_=TemporalCentroidFeatures({"reliability_only":True})(b,h)
    for a,c in zip(full,reliability):
        assert torch.count_nonzero(c[:,0]) == 0
        torch.testing.assert_close(a[:,1:],c[:,1:])


def test_runtime_warmstart_freezing_and_optimizer_integration(tmp_path):
    from rdq_uav.multimodal_v2.training import build_runtime, trainable_parameter_groups
    cfg=yaml.safe_load((ROOT/"experiments/multimodal_v2/configs/b0_a2_mtce3.yaml").read_text())
    cfg["initialization"]["b0_warmstart_checkpoint"]=str(warmstart_file(tmp_path,LiDARUAVDetector(config(False))))
    runtime=build_runtime(cfg,ROOT,torch.device("cpu"))
    groups=trainable_parameter_groups(runtime,.0003,.0001)
    assert sum(p.numel() for group in groups for p in group["params"]) == 1152
    assert len([p for p in runtime.model.parameters() if p.requires_grad]) == 3
