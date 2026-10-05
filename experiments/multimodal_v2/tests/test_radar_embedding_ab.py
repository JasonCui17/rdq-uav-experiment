"""Frontend ablation contracts: actual tensor tests, runnable on WSL/server."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_geometry import HierarchyBuilder
from rdq_uav.multimodal_v2.lidar_legacy import LegacyVoxelEmbed
from rdq_uav.multimodal_v2.lidar_learned_sbe import LearnedSBEVoxelEmbed

ROOT = Path(__file__).resolve().parents[3]

def cfg(name):
    value = yaml.safe_load((ROOT / "experiments/multimodal_v2/configs/radar.yaml").read_text())
    value["model"]["voxel"].update(embedding=name, matched_frontend_init=True,
                                  frontend_seed=42, learned_sbe={"point_dim": 32})
    return value

def batch():
    points = torch.tensor([[.01,.02,.03],[.12,.13,.14],[.3,.3,.3],
                           [-.1,-.1,-.1],[.01,.02,.03],[.4,.4,.4]])
    return {"points": points, "sensor_id": torch.tensor([0,1,0,1,1,0]),
            "delta_t": torch.tensor([0.,-.2,-.4,-.8,-.1,-.9]),
            "point_batch_index": torch.tensor([0,0,0,0,1,1]), "num_samples": 2}

@pytest.mark.parametrize("name", ["sbe_lite", "legacy", "learned_sbe"])
def test_frontend_shape_and_full_network_backward(name):
    torch.manual_seed(42)
    model = LiDARUAVDetector(cfg(name))
    value=batch(); hierarchy=model.hierarchy(value["points"],value["point_batch_index"])
    token=model.voxel_embed(value,hierarchy)
    assert token.shape==(len(hierarchy.levels[0].coords),128)
    assert torch.isfinite(token).all()
    result=model(value)
    loss=result["logits"].sum()+result["residual_xyz"].square().sum()
    loss.backward()
    for module in (model.voxel_embed,model.encoder0,model.encoder1,model.encoder2,model.head):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)

def test_matched_seed_keeps_every_downstream_weight_identical():
    models=[]
    for name in ("sbe_lite","legacy","learned_sbe"):
        torch.manual_seed(42)
        models.append(LiDARUAVDetector(cfg(name)))
    states=[{k:v for k,v in m.state_dict().items() if not k.startswith("voxel_embed.")} for m in models]
    assert states[0].keys()==states[1].keys()==states[2].keys()
    for key in states[0]:
        assert torch.equal(states[0][key],states[1][key]),key
        assert torch.equal(states[0][key],states[2][key]),key

@pytest.mark.parametrize("width", [16,32])
def test_learned_slots_empty_mask_counts_and_permutation(width):
    torch.manual_seed(42)
    model=LearnedSBEVoxelEmbed(point_dim=width)
    value=batch(); h=HierarchyBuilder()(value["points"],value["point_batch_index"])
    token,debug=model(value,h,return_debug=True)
    assert debug["slot_features"].shape==(len(h.levels[0].coords),8,2*width+2)
    assert debug["slot_count"].sum()==len(value["points"])
    empty=debug["slot_count"]==0
    assert (debug["slot_features"][empty]==0).all()
    order=torch.tensor([5,3,1,4,2,0])
    other={k:(v[order] if torch.is_tensor(v) else v) for k,v in value.items()}
    other_h=HierarchyBuilder()(other["points"],other["point_batch_index"])
    assert torch.allclose(token,model(other,other_h),atol=1e-5,rtol=1e-5)
    # Batch identity is included in the voxel key, even at identical XYZ.
    assert len(h.levels[0].coords)==3
    assert debug["slot_count"].sum(1).tolist()==h.levels[0].point_count.tolist()

def test_legacy_matches_archived_mathematical_formula():
    model=LegacyVoxelEmbed(); value=batch()
    h=HierarchyBuilder()(value["points"],value["point_batch_index"])
    level=h.levels[0]; inv=h.point_to_l0
    pf=torch.cat(((value["points"]-level.centers[inv])/.5,
                  model.sensor_embedding(value["sensor_id"]),value["delta_t"][:,None]),1)
    features=model.point_mlp(pf)
    rows=[]
    for i in range(len(level.coords)):
        selected=features[inv==i]
        rows.append(torch.cat((selected.max(0).values,selected.mean(0),
                               torch.log1p(level.point_count[i].float()).reshape(1))))
    expected=model.norm(model.proj(torch.stack(rows)))
    assert torch.allclose(model(value,h),expected,atol=1e-6,rtol=1e-6)

def test_unknown_embedding_fails_and_a0_default_remains_sbe():
    with pytest.raises(ValueError,match="Unknown voxel embedding"):
        LiDARUAVDetector(cfg("unknown"))
    from rdq_uav.multimodal_v2.radar_sbe import SBELiteVoxelEmbed
    baseline=yaml.safe_load((ROOT/"experiments/multimodal_v2/configs/radar.yaml").read_text())
    assert isinstance(LiDARUAVDetector(baseline).voxel_embed,SBELiteVoxelEmbed)


def test_ab_configs_only_change_frontend_and_share_training_protocol():
    names=("a0_sbe","a1_legacy","a2_learned_sbe16","a2_learned_sbe32")
    configs=[yaml.safe_load((ROOT/f"experiments/multimodal_v2/configs/b0_{n}.yaml").read_text()) for n in names]
    for c in configs[1:]:
        for key in ("data","training","loss","model","lightning"):
            assert c[key]==configs[0][key],key
    radars=[yaml.safe_load((ROOT/f"experiments/multimodal_v2/configs/radar_{n}.yaml").read_text()) for n in names]
    for r in radars:
        v=r["model"]["voxel"]
        for key in ("embedding","learned_sbe"):v.pop(key)
    assert all(r==radars[0] for r in radars)
