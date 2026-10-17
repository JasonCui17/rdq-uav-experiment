"""Single-target 3D candidate association + constant-velocity KF. No GT input."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    measurement_std_m: float = 0.5
    process_noise_density: float = 4.0  # white-acceleration PSD, m^2/s^3
    initial_position_std_m: float = 0.75
    initial_velocity_std_mps: float = 5.0
    mahalanobis_gate_sq: float = 11.34
    max_innovation_m: float = 4.0
    max_speed_mps: float = 20.0
    quality_weight: float = 0.5
    min_score: float = 0.0  # detector scores are not calibrated probabilities
    local_radius_min_m: float = 1.0
    local_radius_max_m: float = 4.0
    local_pre_topk: int = 100
    local_final_topk: int = 20
    local_nms_radius_m: float = 0.3
    confirm_hits: int = 3
    confirmation_window_s: float = 1.0
    max_prediction_age_s: float = 0.5
    max_tentative_age_s: float = 0.35
    reset_gap_s: float = 2.0

    def __post_init__(self):
        positive = ('measurement_std_m','process_noise_density','initial_position_std_m',
                    'initial_velocity_std_mps','mahalanobis_gate_sq','max_innovation_m',
                    'max_speed_mps','local_radius_min_m','local_radius_max_m',
                    'confirmation_window_s','max_prediction_age_s','max_tentative_age_s','reset_gap_s')
        for name in positive:
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError(name)
        if not 0<=self.min_score<=1 or self.quality_weight<0: raise ValueError('invalid quality settings')
        if self.local_radius_min_m>self.local_radius_max_m: raise ValueError('local radius')
        if min(self.local_pre_topk,self.local_final_topk,self.confirm_hits)<1: raise ValueError('invalid count')
        if self.local_nms_radius_m<0: raise ValueError('NMS radius')


@dataclass
class Observations:
    xyz: np.ndarray
    score: np.ndarray
    rank_value: np.ndarray
    token_id: np.ndarray

    def __post_init__(self):
        self.xyz=np.asarray(self.xyz,dtype=np.float64).reshape(-1,3)
        self.score=np.asarray(self.score,dtype=np.float64).reshape(-1)
        self.rank_value=np.asarray(self.rank_value,dtype=np.float64).reshape(-1)
        self.token_id=np.asarray(self.token_id,dtype=np.int64).reshape(-1)
        n=len(self.xyz)
        if not all(len(x)==n for x in (self.score,self.rank_value,self.token_id)): raise ValueError('observation shapes')
        if not all(np.isfinite(x).all() for x in (self.xyz,self.score,self.rank_value)): raise ValueError('nonfinite observations')
        if np.any((self.score<0)|(self.score>1)): raise ValueError('scores must lie in [0,1]')
        if len(np.unique(self.token_id))!=n: raise ValueError('duplicate candidate token IDs')

    @classmethod
    def empty(cls): return cls(np.empty((0,3)),np.empty(0),np.empty(0),np.empty(0,dtype=np.int64))

    def take(self, idx):
        idx=np.asarray(idx,dtype=np.int64).reshape(-1)
        return Observations(self.xyz[idx],self.score[idx],self.rank_value[idx],self.token_id[idx])


def stable_order(obs):
    return np.lexsort((obs.token_id,-obs.rank_value))


def suppress(obs, pre, final, radius):
    order=stable_order(obs)[:pre];keep=[]
    for idx in order:
        if not keep or not np.any(np.linalg.norm(obs.xyz[keep]-obs.xyz[idx],axis=1)<=radius): keep.append(int(idx))
        if len(keep)>=final:break
    return obs.take(keep)


class CVKalman:
    """State [px,py,pz,vx,vy,vz], Joseph update, float64 covariance."""
    def __init__(self, xyz, cfg):
        self.cfg=cfg
        self.x=np.r_[np.asarray(xyz,dtype=np.float64),np.zeros(3)]
        self.P=np.diag([cfg.initial_position_std_m**2]*3+[cfg.initial_velocity_std_mps**2]*3)
        self.H=np.c_[np.eye(3),np.zeros((3,3))]
        self.R=np.eye(3)*cfg.measurement_std_m**2

    def predict(self, dt):
        if not math.isfinite(dt) or dt<=0:raise ValueError('dt must be positive')
        I=np.eye(3);F=np.eye(6);F[:3,3:]=dt*I
        Q=self.cfg.process_noise_density*np.block([[dt**3/3*I,dt**2/2*I],[dt**2/2*I,dt*I]])
        self.x=F@self.x;self.P=F@self.P@F.T+Q;self.P=(self.P+self.P.T)/2

    def innovations(self, xyz):
        delta=np.asarray(xyz,dtype=np.float64)-self.x[:3]
        S=self.P[:3,:3]+self.R
        L=np.linalg.cholesky(S)
        whitened=np.linalg.solve(L,delta.T)
        return delta,np.sum(whitened**2,axis=0)

    def posterior(self, z):
        y=np.asarray(z)-self.x[:3]
        S=self.P[:3,:3]+self.R
        K=np.linalg.solve(S,(self.P@self.H.T).T).T
        x=self.x+K@y
        A=np.eye(6)-K@self.H
        P=A@self.P@A.T+K@self.R@K.T
        return x,(P+P.T)/2

    def update(self,z):self.x,self.P=self.posterior(z)


class CandidateTracker:
    """One sequence/one target. Candidate association occurs BEFORE KF update."""
    def __init__(self,cfg=None,mode='candidate'):
        if mode not in ('candidate','top1'):raise ValueError(mode)
        self.cfg=cfg or TrackerConfig();self.mode=mode
        self.kf=None;self.last_time=None;self.last_measurement_time=None;self.birth_time=None
        self.status='uninitialized';self.hits=0;self.track_id=0;self.ever_confirmed=False

    def _drop(self):
        self.kf=None;self.status='lost';self.hits=0;self.birth_time=None;self.last_measurement_time=None

    def step(self,timestamp,dense,global_pool,baseline_top1):
        t=float(timestamp);cfg=self.cfg
        if not math.isfinite(t):raise ValueError('invalid timestamp')
        dt=None if self.last_time is None else t-self.last_time
        if dt is not None and dt<=0:raise ValueError('queries must be strictly chronological within sequence')
        event='none';reset_reason=None
        if dt is not None and dt>cfg.reset_gap_s:
            self._drop();event='gap_reset';reset_reason='time_gap'
        if self.kf is not None and t-self.last_measurement_time>cfg.max_prediction_age_s:
            self._drop();event='lost';reset_reason='measurement_timeout'
        if self.kf is not None:
            self.kf.predict(dt)
            if self.status=='tentative' and (t-self.birth_time>cfg.confirmation_window_s or t-self.last_measurement_time>cfg.max_tentative_age_s):
                self._drop();event='tentative_expired';reset_reason='tentative_timeout'
        self.last_time=t
        prior=self.kf.x[:3].copy() if self.kf is not None else None
        age=t-self.last_measurement_time if self.last_measurement_time is not None else None
        debug=dict(timestamp=t,mode=self.mode,track_id=self.track_id,status=self.status,event=event,reset_reason=reset_reason,
                   prior_xyz=None if prior is None else prior.tolist(),local_pool_count=0,global_pool_count=len(global_pool.xyz),
                   gated_candidates=0,selected_token_id=None,selected_score=None,observation_xyz=None,
                   innovation_m=None,innovation_mahalanobis_sq=None,association_cost=None,
                   accepted_observation=False,measurement_age_s=age,output_kind='none',xyz=None,velocity_mps=None,
                   position_std_m=None,hits=self.hits,reacquisition=False,rejected_speed_candidates=0,
                   local_radius_m=None)
        if self.kf is None:
            pool=baseline_top1 if self.mode=='top1' else global_pool
            if len(pool.xyz):
                eligible=[i for i in stable_order(pool) if pool.score[i]>=cfg.min_score]
                if eligible:
                    i=int(eligible[0]);self.kf=CVKalman(pool.xyz[i],cfg)
                    self.track_id+=1;self.birth_time=t;self.last_measurement_time=t;self.hits=1;self.status='tentative'
                    debug.update(event='birth',selected_token_id=int(pool.token_id[i]),selected_score=float(pool.score[i]),
                                 observation_xyz=pool.xyz[i].tolist(),accepted_observation=True)
                    if cfg.confirm_hits==1:
                        self.status='confirmed';debug['event']='confirmed';debug['reacquisition']=self.ever_confirmed;self.ever_confirmed=True
        else:
            if self.mode=='candidate':
                radius=float(np.clip(math.sqrt(cfg.mahalanobis_gate_sq*np.linalg.eigvalsh(self.kf.P[:3,:3]+self.kf.R).max()),cfg.local_radius_min_m,cfg.local_radius_max_m))
                local_ids=np.flatnonzero(np.linalg.norm(dense.xyz-self.kf.x[:3],axis=1)<=radius)
                local=suppress(dense.take(local_ids),cfg.local_pre_topk,cfg.local_final_topk,cfg.local_nms_radius_m)
                # Reserve local candidates BEFORE global ranking; deduplicate token IDs.
                combined={int(token):(local,i) for i,token in enumerate(local.token_id)}
                for i,token in enumerate(global_pool.token_id):combined.setdefault(int(token),(global_pool,i))
                values=list(combined.values())
                pool=Observations([o.xyz[i] for o,i in values],[o.score[i] for o,i in values],
                                  [o.rank_value[i] for o,i in values],[o.token_id[i] for o,i in values]) if values else Observations.empty()
                debug.update(local_pool_count=len(local.xyz),local_radius_m=radius)
            else:pool=baseline_top1
            chosen=None
            if len(pool.xyz):
                delta,d2=self.kf.innovations(pool.xyz)
                distance=np.linalg.norm(delta,axis=1)
                allowed=np.flatnonzero((d2<=cfg.mahalanobis_gate_sq)&(distance<=cfg.max_innovation_m)&(pool.score>=cfg.min_score))
                debug['gated_candidates']=len(allowed)
                cost=.5*d2-cfg.quality_weight*np.log(np.clip(pool.score,1e-8,1.))
                order=allowed[np.lexsort((pool.token_id[allowed],cost[allowed]))]
                for i in order:
                    posterior,_=self.kf.posterior(pool.xyz[i])
                    if np.linalg.norm(posterior[3:])<=cfg.max_speed_mps:
                        chosen=int(i);break
                    debug['rejected_speed_candidates']+=1
                if chosen is not None:
                    i=chosen;self.kf.update(pool.xyz[i]);self.last_measurement_time=t;self.hits+=1
                    debug.update(event='update',selected_token_id=int(pool.token_id[i]),selected_score=float(pool.score[i]),
                                 observation_xyz=pool.xyz[i].tolist(),innovation_m=float(distance[i]),
                                 innovation_mahalanobis_sq=float(d2[i]),association_cost=float(cost[i]),accepted_observation=True)
                    if self.status=='tentative' and self.hits>=cfg.confirm_hits:
                        self.status='confirmed';debug['event']='confirmed';debug['reacquisition']=self.ever_confirmed;self.ever_confirmed=True
            if chosen is None:
                debug['event']='predict_only' if self.status=='confirmed' else 'tentative_miss'
        debug.update(status=self.status,track_id=self.track_id,hits=self.hits,
                     measurement_age_s=t-self.last_measurement_time if self.last_measurement_time is not None else None)
        if self.kf is not None:
            debug.update(velocity_mps=self.kf.x[3:].tolist(),position_std_m=np.sqrt(np.diag(self.kf.P)[:3]).tolist())
            if self.status=='confirmed':
                debug.update(xyz=self.kf.x[:3].tolist(),output_kind='updated' if debug['accepted_observation'] else 'predicted')
        return debug


class SequenceTrackers:
    def __init__(self,cfg,mode):self.cfg=cfg;self.mode=mode;self.states={}
    def step(self,sequence_id,timestamp,dense,global_pool,baseline_top1):
        tracker=self.states.setdefault(sequence_id,CandidateTracker(self.cfg,self.mode))
        return tracker.step(timestamp,dense,global_pool,baseline_top1)
