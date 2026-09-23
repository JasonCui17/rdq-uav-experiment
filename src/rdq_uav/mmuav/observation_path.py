"""RECONSTRUCTED_ROBUSTNESS_DESIGN: pure observation-state DAG selection.

Posterior measurement-update states are NOT assumed to be raw measurements.
Provisional scalar costs are transparent and are never tuned using case GT.
"""
from dataclasses import dataclass,asdict
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class ObservationNode:
    timestamp: float
    xyz: tuple
    velocity: tuple
    track_id: str
    source_index: int
    measurement_id: str | None = None

    def row(self):
        return dict(timestamp=self.timestamp,**dict(zip('xyz',self.xyz)),
            **dict(zip(('vx','vy','vz'),self.velocity)),track_id=self.track_id,
            source_state_index=self.source_index,measurement_id=self.measurement_id,state_type='observed')


@dataclass(frozen=True)
class PathConfig:
    node_reward: float = 1.0
    lambda_d: float = 0.1
    lambda_g: float = 0.1
    d0: float = 3.0
    tau0: float = 1.0
    max_dt: float = 1.0
    max_distance: float = 3.0

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):raise ValueError('Nonfinite config')
        if min(self.node_reward,self.d0,self.tau0,self.max_dt)<=0 or min(self.lambda_d,self.lambda_g,self.max_distance)<0:
            raise ValueError('Invalid config')


def extract_observation_nodes(rows):
    nodes=[]
    for index,r in enumerate(rows):
        if not (r['measurement_update'] is True or r['measurement_update']=='True'):continue
        t=float(r['timestamp']);p=tuple(float(r[k]) for k in 'xyz');v=tuple(float(r[k]) for k in ('vx','vy','vz'))
        if not np.isfinite([t,*p,*v]).all():raise ValueError('Nonfinite observed state')
        mid=r.get('measurement_id') or r.get('detection_id') or None
        nodes.append(ObservationNode(t,p,v,str(r['track_id']),index,str(mid) if mid is not None else None))
    return sorted(nodes,key=lambda n:(n.timestamp,n.track_id,n.xyz,n.velocity,n.source_index))


def deduplicate_observation_nodes(nodes,tolerance=1e-8):
    """Actual ID dedup if present; otherwise same-time tiny-XYZ approximate dedup.

    ID-bearing distinct detections are not spatially merged with each other.
    A canonical posterior state is chosen by deterministic sorting, not GT.
    """
    if tolerance<0:raise ValueError('Negative tolerance')
    kept=[];ids=set();duplicates=0
    for n in sorted(nodes,key=lambda n:(n.timestamp,n.track_id,n.xyz,n.velocity,n.source_index)):
        duplicate=n.measurement_id is not None and n.measurement_id in ids
        if n.measurement_id is None:
            duplicate=any(k.timestamp==n.timestamp and k.measurement_id is None and
                np.max(np.abs(np.array(k.xyz)-n.xyz))<=tolerance for k in kept)
        if duplicate:duplicates+=1;continue
        kept.append(n)
        if n.measurement_id is not None:ids.add(n.measurement_id)
    return kept,dict(duplicates_removed=duplicates,tolerance=tolerance,
        exact_measurement_deduplication='available for supplied IDs only' if ids else 'exact measurement deduplication unavailable')


def score_edge(dt,distance,config):
    return config.lambda_d*(distance/config.d0)**2+config.lambda_g*dt/config.tau0


def build_observation_dag(nodes,config=PathConfig()):
    config.validate();edges=[];audit=[];rejected=Counter()
    if any(a.timestamp>b.timestamp for a,b in zip(nodes,nodes[1:])):
        raise ValueError('Observation nodes must be sorted by timestamp')
    for i,a in enumerate(nodes):
        for j in range(i+1,len(nodes)):
            b=nodes[j];dt=b.timestamp-a.timestamp
            if dt<=0:rejected['NON_INCREASING_TIME']+=1;continue
            if dt>config.max_dt:
                rejected['DT_GATE']+=len(nodes)-j;break
            distance=float(np.linalg.norm(np.array(a.xyz)+np.array(a.velocity)*dt-np.array(b.xyz)))
            accepted=distance<=config.max_distance
            record=dict(source_node=i,target_node=j,source_track=a.track_id,target_track=b.track_id,
                source_timestamp=a.timestamp,target_timestamp=b.timestamp,dt=dt,predicted_distance=distance,
                same_track=a.track_id==b.track_id,accepted=accepted,rejection_reason='' if accepted else 'DISTANCE_GATE',
                edge_cost=score_edge(dt,distance,config))
            audit.append(record)
            if accepted:edges.append(record)
            else:rejected['DISTANCE_GATE']+=1
    return dict(nodes=nodes,edges=edges,edge_audit=audit,rejection_counts=dict(rejected))


def audit_reachability(graph):
    """All-edge graph evidence plus transitive reachability from earliest nodes."""
    nodes=graph['nodes'];edges=graph['edges'];cross=[e for e in edges if not e['same_track']]
    reachable=set(i for i,n in enumerate(nodes) if nodes and n.timestamp==nodes[0].timestamp)
    for e in sorted(edges,key=lambda e:(e['source_node'],e['target_node'])):
        if e['source_node'] in reachable:reachable.add(e['target_node'])
    handoffs=[e for e in cross if e['source_node'] in reachable]
    return dict(has_cross_track_reachable_path=bool(cross),number_of_nodes=len(nodes),number_of_edges=len(edges),
        number_of_cross_track_edges=len(cross),earliest_cross_track_edge=min(cross,key=lambda e:(e['source_timestamp'],e['target_timestamp'],e['source_track'],e['target_track'])) if cross else None,
        minimum_cross_track_distance=min((e['predicted_distance'] for e in cross),default=None),
        earliest_observation_timestamp=nodes[0].timestamp if nodes else None,
        reachable_node_count_from_earliest=len(reachable),
        has_cross_track_path_from_earliest=bool(handoffs),
        latest_reachable_timestamp=max((nodes[i].timestamp for i in reachable),default=None))


def solve_best_observation_path(graph,config=PathConfig()):
    config.validate();nodes=graph['nodes'];incoming=[[] for _ in nodes]
    for e in graph['edges']:incoming[e['target_node']].append(e)
    best=[]
    def rank(candidate):
        score,count,start,cost,path=candidate
        span=nodes[path[-1]].timestamp-start
        return (-score,-count,-span,cost,path)
    for j,n in enumerate(nodes):
        options=[(config.node_reward,1,n.timestamp,0.,(j,))]
        for e in incoming[j]:
            s,c,t,cost,path=best[e['source_node']]
            options.append((s+config.node_reward-e['edge_cost'],c+1,t,cost+e['edge_cost'],path+(j,)))
        best.append(min(options,key=rank))
    if not best:return dict(selected=[],transitions=[],selected_node_ids=[],score=0.,cumulative_motion_cost=0.)
    s,count,t,cost,path=min(best,key=rank);lookup={(e['source_node'],e['target_node']):e for e in graph['edges']}
    return dict(selected=[nodes[i].row() for i in path],transitions=[lookup[(a,b)] for a,b in zip(path,path[1:])],
        selected_node_ids=list(path),score=s,cumulative_motion_cost=cost)


def select_observation_path(rows,config=PathConfig()):
    nodes,dedup=deduplicate_observation_nodes(extract_observation_nodes(rows))
    graph=build_observation_dag(nodes,config)
    result=solve_best_observation_path(graph,config)
    chosen=set(zip(result['selected_node_ids'],result['selected_node_ids'][1:]));sources=set(result['selected_node_ids'])
    result.update(graph=graph,deduplication=dedup,reachability=audit_reachability(graph),
        alternatives=[dict(e,on_selected_path=(e['source_node'],e['target_node']) in chosen)
            for e in graph['edges'] if e['source_node'] in sources],config=asdict(config))
    return result


def build_bounded_output_path(selected,query,max_gap=1.,terminal_horizon=.1):
    """Optional diagnostic only: observed, internal linear prediction, bounded tail.

    Query coordinates never affect DAG scoring. No AR/spline fitting. Tail<=1s.
    """
    if max_gap<=0 or not 0<=terminal_horizon<=1.:raise ValueError('Invalid bounds')
    t=np.array([r['timestamp'] for r in selected]);out=[]
    for q in query:
        p=np.full(3,np.nan);kind='missing'
        if len(t):
            j=int(np.searchsorted(t,q))
            if j<len(t) and t[j]==q:p=np.array([selected[j][a] for a in 'xyz']);kind='observed'
            elif 0<j<len(t) and t[j]-t[j-1]<=max_gap:
                a,b=selected[j-1],selected[j];alpha=(q-t[j-1])/(t[j]-t[j-1])
                p=(1-alpha)*np.array([a[k] for k in 'xyz'])+alpha*np.array([b[k] for k in 'xyz']);kind='predicted'
            elif j==len(t) and 0<q-t[-1]<=terminal_horizon:
                a=selected[-1];p=np.array([a[k] for k in 'xyz'])+(q-t[-1])*np.array([a[k] for k in ('vx','vy','vz')]);kind='predicted'
        out.append(dict(timestamp=float(q),**dict(zip('xyz',p)),state_type=kind))
    return out
