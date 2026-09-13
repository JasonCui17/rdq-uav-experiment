"""POST-HOC robustness design. No GT, model inference, or Kalman changes."""
import numpy as np


def supported(r):
    return r['measurement_update'] is True or r['measurement_update']=='True'


def select_track_v2(rows, maximum_gap=1.0, spatial_gate=3.0):
    """Measurement-count ranking, then supported duration, then tail penalty.

    Stitch forward measurement-supported segments using existing 1s/3m scales.
    Discard prediction-only states rather than represent them as observations.
    Position continuity uses endpoint velocity; no GT or sequence-specific rule.
    This fixed heuristic is RECONSTRUCTED_ROBUSTNESS_DESIGN, not paper code.
    """
    groups={}
    for r in rows: groups.setdefault(r['track_id'],[]).append(r)
    valid=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:float(r['timestamp']))
        updates=[r for r in group if supported(r)]
        if len(updates)<2:continue
        tail=float(group[-1]['timestamp'])-float(updates[-1]['timestamp'])
        rank=(len(updates),float(updates[-1]['timestamp'])-float(updates[0]['timestamp']),-tail)
        valid.append((rank,updates))
    if not valid:return []
    _,chosen=max(valid,key=lambda item:(item[0],tuple(float(item[1][0][a]) for a in 'xyz')))
    result=list(chosen);used={chosen[0]['track_id']}
    while True:
        endpoint=result[-1];t=float(endpoint['timestamp']);p=np.array([float(endpoint[a]) for a in 'xyz'])
        v=np.array([float(endpoint['v'+a]) for a in 'xyz']);options=[]
        for _,updates in valid:
            if updates[0]['track_id'] in used:continue
            future=[r for r in updates if float(r['timestamp'])>t]
            if len(future)<2:continue
            dt=float(future[0]['timestamp'])-t
            distance=np.linalg.norm(np.array([float(future[0][a]) for a in 'xyz'])-(p+dt*v))
            if dt<=maximum_gap and distance<=spatial_gate:
                options.append(((len(future),float(future[-1]['timestamp'])-float(future[0]['timestamp']),-distance),future))
        if not options:break
        _,future=max(options,key=lambda item:item[0]);result.extend(future);used.add(future[0]['track_id'])
    return result
