"""RECONSTRUCTED_ROBUSTNESS_DESIGN: shared recovery, no identity decisions."""
from dataclasses import dataclass
import numpy as np
from .observation_path import build_bounded_output_path


@dataclass(frozen=True)
class RecoveryConfig:
    internal_gap_limit: float = 1.0
    tail_prediction_limit: float = 0.1


def recover_bounded_missing(observed_states,query_timestamps,config=RecoveryConfig()):
    """Reuse the previously fixed interpolation/tail API, with input isolation.

    Recovered states must never be reused as observations. Only explicit observed
    or measurement-update states are accepted. Extra identity/GT columns ignored.
    """
    selected=[]
    for r in observed_states:
        kind=r.get('state_type')
        update=r.get('measurement_update') is True or r.get('measurement_update')=='True'
        if kind in ('predicted','missing') or (kind!='observed' and not update):
            raise ValueError('Recovery accepts observed states only')
        clean={k:float(r[k]) for k in ('timestamp','x','y','z','vx','vy','vz')}
        if not np.isfinite(list(clean.values())).all():raise ValueError('Invalid observation')
        selected.append(clean)
    selected.sort(key=lambda r:r['timestamp'])
    if any(a['timestamp']>=b['timestamp'] for a,b in zip(selected,selected[1:])):raise ValueError('Duplicate observation time')
    query=np.array(query_timestamps,dtype=float)
    if not np.isfinite(query).all() or any(a>=b for a,b in zip(query,query[1:])):raise ValueError('Query must increase')
    result=build_bounded_output_path(selected,query,max_gap=config.internal_gap_limit,terminal_horizon=config.tail_prediction_limit)
    t=np.array([r['timestamp'] for r in selected])
    for r in result:
        r['recovery_source']=r['state_type'];r['recovery_duration']=0.
        if r['state_type']=='predicted':
            if r['timestamp']>t[-1]:
                r['recovery_source']='tail_prediction';r['recovery_duration']=r['timestamp']-t[-1]
            else:
                j=np.searchsorted(t,r['timestamp']);r['recovery_source']='internal_interpolation'
                r['recovery_duration']=t[j]-t[j-1]
    return result
