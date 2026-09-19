from __future__ import annotations
import sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/"src"))
from rdq_uav.lidar_v2 import CandidateLoss,CandidateSelector,LiDARUAVDetector
from rdq_uav.lidar_v2.geometry import HierarchyBuilder
from rdq_uav.lidar_v2.model import RelativeBias,SpatialBlock,local_groups,morton_order

CFG=yaml.safe_load((ROOT/"configs/lidar_uav_v2.yaml").read_text())
def batch(points,batch_index=None,gt=None,recent=None):
    n=len(points);batch_index=torch.zeros(n,dtype=torch.long) if batch_index is None else batch_index; b=int(batch_index.max())+1 if n else 1
    return {"points":points.float(),"sensor_id":torch.arange(n)%2,"delta_t":-torch.linspace(0,1,n) if n else torch.empty(0),"recent_mask":torch.ones(n,dtype=torch.bool) if recent is None else recent,"point_batch_index":batch_index,"num_samples":b,"target_xyz":torch.zeros((b,3)) if gt is None else gt,
            "target_valid":torch.ones(b,dtype=torch.bool),"query_valid_mask":torch.ones((b,1),dtype=torch.bool),
            "query_time_clip":torch.zeros((b,1),dtype=torch.float64)}

def test_voxel_floor_parent_maps_child_positions_and_counts():
    p=torch.tensor([[-.1,0,0],[.1,0,0],[1.1,0,0],[2.1,0,0]])
    h=HierarchyBuilder()(p,torch.zeros(4,dtype=torch.long));assert tuple(h.levels[0].coords[0])==(-1,0,0);assert len(h.point_to_l0)==4
    rel=h.levels[0].coords.float()-2*h.levels[1].coords[h.parent_l0_to_l1].float()-.5
    assert set(rel.flatten().tolist())<={-.5,.5};assert torch.equal(h.levels[1].point_count,torch.zeros(len(h.levels[1].coords),dtype=torch.long).index_add_(0,h.parent_l0_to_l1,h.levels[0].point_count))

def test_single_and_many_point_voxel_forward_without_cap():
    model=LiDARUAVDetector(CFG);one=batch(torch.tensor([[.1,.1,.1]]));assert model(one)["logits"].shape==(1,)
    many=batch(torch.zeros((10001,3))+.1);out=model(many);assert out["logits"].shape==(1,) and out["layouts"].levels[0].point_count.item()==10001

def test_batch_and_global_attention_never_mix_samples():
    points=torch.tensor([[.1,0,0],[.2,0,0],[100,.1,0],[100.2,0,0]]);bi=torch.tensor([0,0,1,1]);out=LiDARUAVDetector(CFG)(batch(points,bi,torch.tensor([[0.,0,0],[100.,0,0]])))
    assert set(out["batch_index"].tolist())=={0,1};l2=out["layouts"].levels[2]
    assert all(torch.unique(l2.batch_index[g]).numel()==1 for g in [torch.nonzero(l2.batch_index==b).flatten() for b in (0,1)])

def test_morton_identity_and_shifted_windows_no_wrap():
    coords=torch.stack((torch.arange(130),torch.zeros(130),torch.zeros(130)),1);level=type("L",(),{"coords":coords,"batch_index":torch.zeros(130,dtype=torch.long)})()
    order=morton_order(coords);assert torch.equal(torch.sort(order).values,torch.arange(130))
    groups=local_groups(level,64,32);assert sum(len(g) for g in groups)==130
    assert not any(0 in g.tolist() and 129 in g.tolist() for g in groups)

def test_batched_local_attention_matches_per_window_reference():
    torch.manual_seed(7);block=SpatialBlock(128,4,256);bias=RelativeBias(4,32)
    x=torch.randn(11,128);coords=torch.randint(-5,6,(11,3));groups=[torch.arange(0,4),torch.arange(4,11)]
    reference=block(x,coords,groups,bias,batched_local=False);batched=block(x,coords,groups,bias,batched_local=True)
    assert torch.allclose(reference,batched,rtol=2e-5,atol=2e-6)

def test_loss_ignore_no_support_decode_and_finite_backward():
    model=LiDARUAVDetector(CFG);criterion=CandidateLoss(CFG);p=torch.tensor([[0.,0,0],[1.5,0,0],[4.,0,0]])
    recent=torch.tensor([True,False,False]);b=batch(p,gt=torch.tensor([[0.,0,0.]]),recent=recent);out=model(b);loss=criterion(out,b)
    assert loss["num_pos"]>=1 and loss["num_ignore"]>=1 and torch.isfinite(loss["loss"]);loss["loss"].backward()
    for name in ("voxel_embed.proj.weight","merge01.parent.weight","encoder2.blocks.0.qkv.weight","up10.parent.weight","head.reg.2.weight"):
        grad=dict(model.named_parameters())[name].grad;assert grad is not None and torch.isfinite(grad).all()
    decoded=out["voxel_centers"]+(b["target_xyz"][out["batch_index"]]-out["voxel_centers"]);assert torch.allclose(decoded,b["target_xyz"][out["batch_index"]])
    b2=batch(torch.tensor([[10.,0,0]]),gt=torch.zeros((1,3)));l2=criterion(model(b2),b2);assert l2["num_no_current_support"]==1 and l2["num_supervised_samples"]==0

def test_selector_keeps_xyz_score_feature_identity_and_stable_tie():
    o={"logits":torch.zeros(3),"pred_xyz":torch.tensor([[0.,0,0],[3.,0,0],[6.,0,0]]),"fine_features":torch.arange(384).reshape(3,128).float(),"source_token_id":torch.tensor([2,0,1]),"batch_index":torch.zeros(3,dtype=torch.long)}
    result=CandidateSelector(CFG)(o)[0]["raw"];assert result["source_token_id"].tolist()==[0,1,2]
    for i,source_index in enumerate((1,2,0)):assert torch.equal(result["feature"][i],o["fine_features"][source_index]) and torch.equal(result["xyz"][i],o["pred_xyz"][source_index])

def test_empty_input_returns_empty_prediction():
    out=LiDARUAVDetector(CFG)(batch(torch.empty((0,3))));assert out["logits"].shape==(0,) and out["pred_xyz"].shape==(0,3)
    selected=CandidateSelector(CFG)(out);assert len(selected)==1 and selected[0]["nms"]["xyz"].shape==(0,3)

def test_architecture_contract_and_initialization():
    model=LiDARUAVDetector(CFG);assert model.encoder2.global_attention and not model.encoder0.global_attention
    assert not hasattr(model.voxel_embed,'sensor_embedding') if CFG['model']['voxel'].get('embedding')=='sbe_lite' else torch.equal(model.voxel_embed.sensor_embedding.weight.detach().flatten(),torch.tensor([0.,1.]));assert torch.all(model.encoder0.bias.tables==0)
    source=(ROOT/"src/rdq_uav/lidar_v2/model.py").read_text().lower();assert "attentionpool" not in source and "dbscan" not in source and "occupancy_mask" not in source
