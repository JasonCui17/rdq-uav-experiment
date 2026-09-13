"""Public StoneSoup settings; minimal reconstructed selection/completion. No GT."""
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import interp1d, splrep, splev


def track_candidates(frames):
    from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.predictor.kalman import ExtendedKalmanPredictor
    from stonesoup.updater.kalman import ExtendedKalmanUpdater
    from stonesoup.types.state import GaussianState
    from stonesoup.types.detection import Detection
    from stonesoup.types.update import GaussianStateUpdate
    from stonesoup.initiator.simple import MultiMeasurementInitiator
    from stonesoup.deleter.error import CovarianceBasedDeleter
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.dataassociator.neighbour import NearestNeighbour
    from stonesoup.measures import Euclidean
    nonempty=[(t,p) for t,p in frames if len(p)]
    if not nonempty: return []
    t,p=nonempty[0]
    stamp=lambda t:datetime.fromtimestamp(float(t),tz=timezone.utc)
    # SOURCE-FAITHFUL defaults from fusion_tracking.py; six-state XYZ/velocity.
    prior=GaussianState([[p[0,0]],[.001],[p[0,1]],[.001],[p[0,2]],[.001]],
                        np.diag([.01,.1,.01,.1,.01,.1]),timestamp=stamp(t))
    measurement=LinearGaussian(ndim_state=6,mapping=(0,2,4),noise_covar=np.eye(3)*.001)
    transition=CombinedLinearGaussianTransitionModel([ConstantVelocity(.15) for _ in range(3)])
    predictor=ExtendedKalmanPredictor(transition)
    updater=ExtendedKalmanUpdater(measurement_model=measurement)
    deleter=CovarianceBasedDeleter(covar_trace_thresh=30.)
    associator=NearestNeighbour(DistanceHypothesiser(predictor,updater,Euclidean(),missed_distance=3.))
    initiator=MultiMeasurementInitiator(prior_state=prior,measurement_model=measurement,
        deleter=deleter,data_associator=associator,updater=updater,min_points=1)
    tracks=set();archive={}
    for t,points in nonempty:  # Public tracker skips empty measurement frames.
        time=stamp(t)
        detections={Detection(p,timestamp=time,measurement_model=measurement) for p in points}
        hypotheses=associator.associate(tracks,detections,time);used=set()
        for track in tracks.copy():
            h=hypotheses[track]
            track.append(updater.update(h) if h.measurement else h.prediction)
            if h.measurement: used.add(h.measurement)
        tracks-=deleter.delete_tracks(tracks)
        tracks|=initiator.initiate(detections-used,time)
        for track in tracks: archive[track.id]=track
    # DATA-ADAPTER CHANGE: preserve deleted tracks and prevent timestamp-file overwrite.
    rows=[]
    for tid,track in archive.items():
        for state in track:
            v=np.asarray(state.state_vector).reshape(6)
            rows.append(dict(timestamp=state.timestamp.timestamp(),track_id=tid,
                x=float(v[0]),y=float(v[2]),z=float(v[4]),vx=float(v[1]),vy=float(v[3]),vz=float(v[5]),
                measurement_update=isinstance(state,GaussianStateUpdate)))
    return sorted(rows,key=lambda r:(r['timestamp'],r['track_id']))


def select_track(rows):
    """RECONSTRUCTED_DESIGN: duration, then update count; no GT tie breaking."""
    groups={}
    for r in rows: groups.setdefault(r['track_id'],[]).append(r)
    valid=[sorted(g,key=lambda r:r['timestamp']) for g in groups.values() if len(g)>=2]
    if not valid: return []
    # Coordinate tie-breaker only for exact ties; independent of random track UUID.
    return max(valid,key=lambda g:(g[-1]['timestamp']-g[0]['timestamp'],
        sum(r['measurement_update'] for r in g),tuple(g[0][a] for a in 'xyz')))


def ar_complete(times,points,grid,fit=True):
    """AR3 per-axis OLS on selected observed track, never GT. Manual formal fit only.

    Fixed 0.1s grid, interior gaps only, maximum completion gap 1s. Coefficients
    reconstructed; fallback constant-velocity extrapolation [2,-1,0] if <8 states.
    """
    coeff=np.tile([2.,-1.,0.,0.],(3,1))
    if fit and len(points)>=8:
        # Uniformize observations before AR fitting; no GT coordinates/times.
        regular=np.arange(times[0],times[-1]+1e-7,.1)
        obs=np.column_stack([np.interp(regular,times,points[:,a]) for a in range(3)])
        if len(obs)>=8:
            for a in range(3):
                X=np.column_stack([obs[2:-1,a],obs[1:-2,a],obs[:-3,a],np.ones(len(obs)-3)])
                coeff[a]=np.linalg.lstsq(X,obs[3:,a],rcond=None)[0]
    out_t=list(times);out_p=list(points)
    for t in grid:
        if t<=times[0] or t>=times[-1] or np.min(np.abs(times-t))<=.05: continue
        j=np.searchsorted(times,t)
        if times[j]-times[j-1]>1.: continue
        history=np.array(out_p)[np.argsort(out_t)]
        history_times=np.sort(out_t)
        previous=history[history_times<t]
        if len(previous)<3: continue
        new=np.sum(coeff[:,:3]*previous[-3:][::-1].T,axis=1)+coeff[:,3]
        if not np.isfinite(new).all(): raise ValueError('Nonfinite AR completion')
        out_t.append(t);out_p.append(new)
    order=np.argsort(out_t)
    return np.array(out_t)[order],np.array(out_p)[order],coeff


def resample(times,points,query,smooth=False):
    """Public linear/splrep(s=.5) math, bounded support (no unlimited extrapolation)."""
    if len(times)<2: return np.full((len(query),3),np.nan)
    t,indices=np.unique(times,return_index=True);p=points[indices]
    result=np.column_stack([interp1d(t,p[:,a],fill_value=np.nan,bounds_error=False)(query) for a in range(3)])
    if smooth and len(t)>=4:
        result=np.column_stack([splev(query,splrep(t,p[:,a],s=.5)) for a in range(3)])
    result[(query<t[0])|(query>t[-1])]=np.nan
    j=np.clip(np.searchsorted(t,query),1,len(t)-1)
    result[(t[j]-t[j-1]>1.) & (np.minimum(abs(query-t[j]),abs(query-t[j-1]))>.05)]=np.nan
    return result
