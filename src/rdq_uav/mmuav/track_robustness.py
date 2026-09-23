"""POST-HOC robustness design. No GT, model inference, or Kalman changes."""
import numpy as np
from .observation_path import (ObservationNode, PathConfig, extract_observation_nodes,
    deduplicate_observation_nodes, build_observation_dag, audit_reachability,
    score_edge, solve_best_observation_path, select_observation_path, build_bounded_output_path)


def supported(r):
    return r['measurement_update'] is True or r['measurement_update']=='True'


def select_track_v1(rows, maximum_gap=1.0, spatial_gate=3.0):
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


FIELDS=('timestamp','track_id','x','y','z','vx','vy','vz','measurement_update')


def support_statistics(rows):
    """Validate tracker-only fields and calculate original-track support statistics."""
    groups={}
    for source in rows:
        r={key:source[key] for key in FIELDS}  # Deliberately ignore all extra/oracle fields.
        r['track_id']=str(r['track_id']);r['measurement_update']=supported(r)
        for key in ('timestamp','x','y','z','vx','vy','vz'):
            r[key]=float(r[key])
            if not np.isfinite(r[key]):raise ValueError('Nonfinite tracker state: '+key)
        groups.setdefault(r['track_id'],[]).append(r)
    stats=[]
    for tid,group in sorted(groups.items()):
        input_count=len(group)
        # Deterministic duplicate policy: supported state first, then numeric state.
        group.sort(key=lambda r:(r['timestamp'],not r['measurement_update'],tuple(r[k] for k in ('x','y','z','vx','vy','vz'))))
        unique={}
        for r in group:unique.setdefault(r['timestamp'],r)
        group=list(unique.values());groups[tid]=group
        updates=[r for r in group if supported(r)]
        first=updates[0]['timestamp'] if updates else None
        last=updates[-1]['timestamp'] if updates else None
        stats.append(dict(track_id=tid,raw_state_count=len(group),input_state_count=input_count,
            measurement_count=len(updates),first_supported_timestamp=first,last_supported_timestamp=last,
            raw_duration=group[-1]['timestamp']-group[0]['timestamp'],
            supported_duration=last-first if updates else 0.,prediction_only_tail=group[-1]['timestamp']-last if updates else None,
            measurement_ratio=len(updates)/len(group)))
    return groups,stats


def stitch_tracks(rows, maximum_gap=1.0, spatial_gate=3.0, min_measurements=2):
    """RECONSTRUCTED_ROBUSTNESS_DESIGN: endpoint-gated disjoint fragment chains.

    Supported fragments are maximal consecutive measurement-update runs; also
    split when supported timestamps differ by >maximum_gap. Original tracks need
    >=min_measurements in total. Single-state runs may join a chain; final chains
    need >=min_measurements. No trimming overlapping fragments or extrapolated
    states. Global edge priority is distance, supported continuity, deterministic
    tracker-only identifiers. Each fragment has <=1 predecessor and successor.
    """
    if maximum_gap<=0 or spatial_gate<0 or min_measurements<2:raise ValueError('Invalid robustness settings')
    groups,stats=support_statistics(rows);fragments=[]
    for tid,group in sorted(groups.items()):
        if sum(supported(r) for r in group)<min_measurements:continue
        runs=[];run=[]
        for r in group:
            if not supported(r):
                if run:runs.append(run);run=[]
                continue
            if run and r['timestamp']-run[-1]['timestamp']>maximum_gap:
                runs.append(run);run=[]
            run.append(r)
        if run:runs.append(run)
        for index,run in enumerate(runs):
            fragments.append(dict(fragment_id=f'{tid}:{index:04d}',track_id=tid,states=run))
    decisions=[];eligible=[]
    for a in fragments:
        end=a['states'][-1];p=np.array([end[k] for k in 'xyz']);v=np.array([end['v'+k] for k in 'xyz'])
        for b in fragments:
            if a is b:continue
            start=b['states'][0];dt=start['timestamp']-end['timestamp']
            if dt<=0:continue  # No reverse/overlapping endpoints enter the graph.
            distance=float(np.linalg.norm(p+v*dt-np.array([start[k] for k in 'xyz'])))
            reason='GAP_TOO_LARGE' if dt>maximum_gap else 'DISTANCE_TOO_LARGE' if distance>spatial_gate else 'ELIGIBLE'
            decision=dict(source_track=a['track_id'],target_track=b['track_id'],source_fragment=a['fragment_id'],
                target_fragment=b['fragment_id'],dt=dt,predicted_distance=distance,accepted=False,reason=reason)
            decisions.append(decision)
            if reason=='ELIGIBLE':
                duration=b['states'][-1]['timestamp']-start['timestamp']
                eligible.append(((distance,-len(b['states']),-duration,dt,a['fragment_id'],b['fragment_id']),decision))
    successor={};predecessor={}
    for _,d in sorted(eligible,key=lambda item:item[0]):
        a,b=d['source_fragment'],d['target_fragment']
        if a in successor or b in predecessor:d['reason']='FRAGMENT_ALREADY_LINKED';continue
        successor[a]=b;predecessor[b]=a;d.update(accepted=True,reason='ACCEPTED')
    lookup={f['fragment_id']:f for f in fragments};chains=[]
    for root in sorted(set(lookup)-set(predecessor)):
        ids=[];states=[];current=root
        while current is not None:
            ids.append(current);states.extend(lookup[current]['states']);current=successor.get(current)
        assert all(a['timestamp']<b['timestamp'] for a,b in zip(states,states[1:]))
        tracks=list(dict.fromkeys(r['track_id'] for r in states));raw_start=min(groups[k][0]['timestamp'] for k in tracks)
        raw_end=max(groups[k][-1]['timestamp'] for k in tracks)
        tail=max(0.,raw_end-states[-1]['timestamp'])
        score=(len(states),states[-1]['timestamp']-states[0]['timestamp'],-tail,raw_end-raw_start)
        chains.append(dict(fragment_ids=ids,track_ids=tracks,states=states,score=score))
    valid=[c for c in chains if len(c['states'])>=min_measurements]
    chosen=min(valid,key=lambda c:(tuple(-x for x in c['score']),tuple(c['fragment_ids']))) if valid else None
    selected_ids=set(chosen['fragment_ids']) if chosen else set()
    for d in decisions:d['in_selected_chain']=d['accepted'] and d['source_fragment'] in selected_ids and d['target_fragment'] in selected_ids
    return dict(selected=chosen['states'] if chosen else [],track_statistics=stats,stitching_decisions=decisions,
        chains=chains,selected_fragments=chosen['fragment_ids'] if chosen else [],design_status='RECONSTRUCTED_ROBUSTNESS_DESIGN')


def select_track_v2(rows, maximum_gap=1.0, spatial_gate=3.0):
    """Select the measurement-support-prioritized final stitched chain, without GT."""
    return stitch_tracks(rows,maximum_gap,spatial_gate)['selected']
