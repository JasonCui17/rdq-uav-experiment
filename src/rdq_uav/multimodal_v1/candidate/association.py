from __future__ import annotations

import torch

from .candidate_set import CandidateSet
from .hypothesis import HypothesisSet,HYP_RV,HYP_R,HYP_V


def point_to_box_distance(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise shortest source-pixel distance [R,V]; zero for points inside."""
    if points.ndim!=2 or points.shape[1]!=2 or boxes.ndim!=2 or boxes.shape[1]!=4:
        raise ValueError('points [R,2], boxes [V,4] required')
    if len(points)==0 or len(boxes)==0:
        return points.new_empty((len(points),len(boxes)))
    px=points[:,0,None]; py=points[:,1,None]
    x1,y1,x2,y2=[boxes[:,i][None,:] for i in range(4)]
    dx=torch.maximum(torch.maximum(x1-px,px-x2),torch.zeros_like(px+x1))
    dy=torch.maximum(torch.maximum(y1-py,py-y2),torch.zeros_like(py+y1))
    return torch.sqrt(dx.square()+dy.square())


def _append(rows: dict[str,list[torch.Tensor]], *, r: CandidateSet, v: CandidateSet,
            ri: int|None, vi: int|None, batch: int, dgeo: torch.Tensor|None=None, cosine: torch.Tensor|None=None, tau: float=16.) -> None:
    device=r.feature.device if r.n else v.feature.device
    dtype=r.feature.dtype if r.n else v.feature.dtype
    has_r=ri is not None; has_v=vi is not None
    rf=r.feature[ri] if has_r else torch.zeros(128,device=device,dtype=dtype)
    vf=v.feature[vi] if has_v else torch.zeros(128,device=device,dtype=dtype)
    rs=r.score[ri] if has_r else torch.zeros((),device=device,dtype=dtype)
    vs=v.score[vi] if has_v else torch.zeros((),device=device,dtype=dtype)
    xyz=r.xyz[ri] if has_r else torch.zeros(3,device=device,dtype=dtype)
    box=v.box_xyxy_px[vi] if has_v else torch.zeros(4,device=device,dtype=dtype)
    typ=HYP_RV if has_r and has_v else (HYP_R if has_r else HYP_V)
    dg=torch.zeros((),device=device,dtype=dtype) if dgeo is None else torch.clamp(dgeo.to(dtype=dtype)/tau,max=2.)
    cs=torch.zeros((),device=device,dtype=dtype) if cosine is None else cosine.to(dtype=dtype)
    assoc=torch.stack((dg,cs,rs,vs))
    values={
      'radar_feature':rf,'vision_feature':vf,'radar_score':rs,'vision_score':vs,
      'radar_xyz':xyz,'radar_xyz_valid':torch.tensor(has_r,device=device,dtype=torch.bool),
      'vision_box_xyxy_px':box,'vision_box_valid':torch.tensor(has_v,device=device,dtype=torch.bool),
      'association_info':assoc,'m_R':torch.tensor(has_r,device=device,dtype=torch.bool),
      'm_V':torch.tensor(has_v,device=device,dtype=torch.bool),'hypothesis_type':torch.tensor(typ,device=device),
      'batch_index':torch.tensor(batch,device=device),'radar_source_index':r.source_index[ri] if has_r else torch.tensor(-1,device=device),
      'vision_source_index':v.source_index[vi] if has_v else torch.tensor(-1,device=device),
    }
    for k,val in values.items(): rows[k].append(val)


def associate_candidates(
    radar: CandidateSet,
    vision: CandidateSet,
    *,
    geometry_gate_px: float=16.,
    projected_radar_xy: torch.Tensor | None=None,
    projection=None,
) -> HypothesisSet:
    if radar.source!='radar' or vision.source!='rgb':
        raise ValueError('expected radar and rgb CandidateSets')
    if radar.feature.device != vision.feature.device:
        raise ValueError('candidate modalities must share device')
    if projected_radar_xy is None:
        if projection is None:
            raise ValueError('projection or projected_radar_xy is required')
        from ..interaction.geometry_local import project_omni_radtan
        projected_radar_xy, valid=project_omni_radtan(radar.xyz,radar.batch_index,projection)
    else:
        if projected_radar_xy.shape!=(radar.n,2): raise ValueError('projected_radar_xy must be [Nr,2]')
        valid=torch.isfinite(projected_radar_xy).all(1)
    rows={k:[] for k in ('radar_feature','vision_feature','radar_score','vision_score','radar_xyz','radar_xyz_valid',
                          'vision_box_xyxy_px','vision_box_valid','association_info','m_R','m_V','hypothesis_type','batch_index',
                          'radar_source_index','vision_source_index')}
    batches=sorted(set(radar.batch_index.tolist()) | set(vision.batch_index.tolist()))
    for b in batches:
        rids=torch.nonzero(radar.batch_index==b).flatten(); vids=torch.nonzero(vision.batch_index==b).flatten()
        matched_r=set(); matched_v=set()
        if len(rids) and len(vids):
            # Candidate association is a discrete, non-neural assignment step.
            # Keep its geometry/cosine cost in FP32 under AMP: the 1e6
            # infeasible sentinel exceeds FP16's finite range (65504), and
            # Hungarian ranking benefits from full precision. The selected
            # differentiable cosine values are cast back by ``_append``.
            with torch.autocast(device_type=radar.feature.device.type, enabled=False):
                d=point_to_box_distance(
                    projected_radar_xy[rids].float(),vision.box_xyxy_px[vids].float()
                )
                feasible=(d<=geometry_gate_px) & valid[rids,None]
                rf=torch.nn.functional.normalize(radar.feature[rids].float(),dim=1)
                vf=torch.nn.functional.normalize(vision.feature[vids].float(),dim=1)
                cosine=rf@vf.T
                cost=1.-cosine
                cost=torch.where(feasible,cost,torch.full_like(cost,1e6))
            from scipy.optimize import linear_sum_assignment
            rr,cc=linear_sum_assignment(cost.detach().float().cpu().numpy())
            for lr,lv in zip(rr.tolist(),cc.tolist()):
                if bool(feasible[lr,lv]):
                    ri=int(rids[lr]); vi=int(vids[lv]); matched_r.add(ri); matched_v.add(vi)
                    _append(rows,r=radar,v=vision,ri=ri,vi=vi,batch=b,dgeo=d[lr,lv],cosine=cosine[lr,lv],tau=geometry_gate_px)
        for ri in rids.tolist():
            if ri not in matched_r: _append(rows,r=radar,v=vision,ri=ri,vi=None,batch=b,tau=geometry_gate_px)
        for vi in vids.tolist():
            if vi not in matched_v: _append(rows,r=radar,v=vision,ri=None,vi=vi,batch=b,tau=geometry_gate_px)
    if not rows['batch_index']:
        device=radar.feature.device if radar.feature.numel() else vision.feature.device
        dtype=radar.feature.dtype if radar.feature.numel() else vision.feature.dtype
        return HypothesisSet.empty(device=device,dtype=dtype)
    stacked={k:torch.stack(v) for k,v in rows.items()}
    return HypothesisSet(**stacked)
