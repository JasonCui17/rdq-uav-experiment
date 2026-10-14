#!/usr/bin/env python3
"""Evaluate original and Stone Soup multi-track protocols on one frozen cache."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from rdq_uav.multimodal_v3.tracking import Observations,TrackerConfig,SequenceTrackers
from rdq_uav.multimodal_v3.candidate_adapter import from_cache

ORIGINAL_METHODS=('independent_top1','top1_kf','candidate_kf')
MULTI_METHODS=('multi_candidate_kf','multi_candidate_kf_select')
METHODS=ORIGINAL_METHODS+MULTI_METHODS

def write_csv(path,rows):
    if not rows:Path(path).write_text('');return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(rows,method):
    rows=[r for r in rows if r['has_gt']];es=[r[method+'_error_m'] for r in rows if r[method+'_error_m'] is not None]
    n=len(rows)
    if not n:raise ValueError('No GT for evaluation')
    return dict(queries=n,outputs=len(es),coverage=len(es)/n,success_0_5m=sum(e<=.5 for e in es)/n,
                success_1m=sum(e<=1 for e in es)/n,success_2m=sum(e<=2 for e in es)/n,
                rmse_3d_m=math.sqrt(sum(e*e for e in es)/len(es)) if es else None,
                mean_error_m=float(np.mean(es)) if es else None,median_error_m=float(np.median(es)) if es else None,
                severe_gt10m=sum(e>10 for e in es),missing=n-len(es))


def persistent_wrong_queries(rows,method):
    """Count queries in runs of >=2 consecutive >10m errors on one track."""
    runs={}
    total=0
    for row in rows:
        sequence=row['sequence_id']
        track_id=row.get(method+'_track_id')
        wrong=(row['has_gt'] and row[method+'_status']=='confirmed' and
               row[method+'_error_m'] is not None and row[method+'_error_m']>10 and
               track_id is not None)
        current=runs.get(sequence)
        if wrong and current is not None and current[0]==track_id:
            runs[sequence]=(track_id,current[1]+1)
        else:
            if current is not None and current[1]>=2:
                total+=current[1]
            runs[sequence]=(track_id,1) if wrong else None
    return total+sum(current[1] for current in runs.values()
                     if current is not None and current[1]>=2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tracker-config',type=Path)
    p.add_argument('--multi-config',type=Path,default=Path(__file__).with_name('multi_tracker_defaults.json'))
    p.add_argument('--methods',choices=('original','all'),default='original')
    args=p.parse_args();cache=args.cache.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Use a fresh results directory: {out}')
    manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('Incomplete cache')
    schema_version=int(manifest.get('schema_version',1))
    frames=[json.loads(x) for x in (cache/'index.jsonl').read_text().splitlines()]
    if len(frames)!=manifest['queries'] or len({r['sample_id'] for r in frames})!=len(frames):raise ValueError('Cache index mismatch')
    frames.sort(key=lambda r:(r['sequence_id'],r['query_time']))
    cfg=TrackerConfig(**json.loads(args.tracker_config.read_text())) if args.tracker_config else TrackerConfig()
    top1=SequenceTrackers(cfg,'top1');candidate=SequenceTrackers(cfg,'candidate')
    selected_methods=METHODS if args.methods=='all' else ORIGINAL_METHODS
    multi=None
    if args.methods=='all':
        from rdq_uav.multimodal_v3.stone_tracker import MultiFilterConfig, SequenceMultiTrackers
        from rdq_uav.multimodal_v3.track_selection import SelectionConfig
        multi_config=json.loads(args.multi_config.read_text())
        if set(multi_config)!={'filter','selection'}:
            raise ValueError('Multi config requires filter and selection sections')
        filter_config=MultiFilterConfig(**multi_config['filter'])
        selection_config=SelectionConfig(**multi_config['selection'])
        multi={MULTI_METHODS[0]:SequenceMultiTrackers(filter_config,selection_config,select_evidence=False),
               MULTI_METHODS[1]:SequenceMultiTrackers(filter_config,selection_config,select_evidence=True)}
    out.mkdir(parents=True);rows=[];events=[];started=time.monotonic()
    (out/'tracker_config.json').write_text(json.dumps(asdict(cfg),indent=2))
    if multi is not None:
        (out/'multi_tracker_config.json').write_text(json.dumps(multi_config,indent=2))
    (out/'cache_manifest.json').write_text(json.dumps(manifest,indent=2))
    confirmations_seen=set()
    confirmation_rows=[]
    score_rows=[]
    with (out/'per_query.jsonl').open('w') as qfile,(out/'tracking_events.jsonl').open('w') as efile:
        for idx,frame in enumerate(frames):
            with np.load(cache/frame['file'],allow_pickle=False) as data:
                # Observation objects contain no GT and are the only data passed to trackers.
                candidate_frame=from_cache(data,schema_version)
                dense,glob,baseline=(candidate_frame.dense,candidate_frame.global_pool,
                                     candidate_frame.baseline)
                t=frame['query_time'];seq=frame['sequence_id']
                tracks={'top1_kf':top1.step(seq,t,dense,glob,baseline),'candidate_kf':candidate.step(seq,t,dense,glob,baseline)}
                if multi is not None:
                    tracks.update({name:manager.step(seq,t,candidate_frame) for name,manager in multi.items()})
                # GT is read only after both state updates, exclusively for scoring.
                gt=data['gt_xyz'].tolist();has_gt=bool(data['has_gt'])
            r=dict(sequence_id=seq,sample_id=frame['sample_id'],query_time=t,has_gt=has_gt,gt_xyz=gt)
            xyz={'independent_top1':baseline.xyz[0].tolist() if len(baseline.xyz) else None,
                 **{name:track['xyz'] for name,track in tracks.items()}}
            for name in selected_methods:
                r[name+'_xyz']=xyz[name]
                r[name+'_error_m']=float(np.linalg.norm(np.asarray(xyz[name])-gt)) if has_gt and xyz[name] is not None else None
            for name,track in tracks.items():
                e=dict(sequence_id=seq,sample_id=r['sample_id'],method=name,
                       **{key:value for key,value in track.items() if key not in ('track_scores','track_states')})
                efile.write(json.dumps(e,allow_nan=False)+'\n');events.append(e)
                for key in ('status','event','output_kind','track_id','selected_token_id','innovation_m','measurement_age_s'):
                    r[name+'_'+key]=track.get(key)
                if name in MULTI_METHODS:
                    r[name+'_selection_score']=track['selection_score']
                    r[name+'_switch_reason']=track['switch_reason']
                    for number,score in track['track_scores'].items():
                        score_rows.append(dict(sequence_id=seq,sample_id=r['sample_id'],query_time=t,
                                               method=name,track_id=int(number),**score))
                    for state in track['track_states']:
                        key=(name,seq,state['track_id'])
                        if state['status']=='confirmed' and key not in confirmations_seen:
                            confirmations_seen.add(key)
                            error=(float(np.linalg.norm(np.asarray(state['xyz'])-gt)) if has_gt else None)
                            confirmation_rows.append(dict(method=name,sequence_id=seq,sample_id=r['sample_id'],
                                                          query_time=t,track_id=state['track_id'],
                                                          reacquisition=(state['track_id']==track['track_id']
                                                                         and track['reacquisition']),
                                                          error_m=error,wrong_lock_over_2m=error is not None and error>2))
            qfile.write(json.dumps(r,allow_nan=False)+'\n');qfile.flush();efile.flush();rows.append(r)
            if (idx+1)%100==0 or idx+1==len(frames):print(f'tracking {idx+1}/{len(frames)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    summary={name:metrics(rows,name) for name in selected_methods}
    summary['runtime']={'offline_tracking_wall_s':time.monotonic()-started,'cache_queries':len(rows),
                        'note':'Includes cache loading and event output; excludes frozen network inference'}
    summary['protocol']={'smoke_test':manifest.get('smoke_test',False),'query_schedule':manifest['protocol'],
                         'cache_schema_version':schema_version,'methods':selected_methods,
                         'metric_denominators':{
                             'success_0_5m_success_1m_success_2m_coverage':'all queries with valid 3D GT; missing output is failure',
                             'rmse_3d_m_mean_error_m_median_error_m':'only finite outputs on queries with valid 3D GT',
                             'updated_outputs_prediction_only_outputs_warmup_lost_switches':'all cached queries',
                             'updated_success_1m_count_predicted_success_1m_count':'queries with valid 3D GT and corresponding output kind'},
                         'first_confirmed_output_after_hits':cfg.confirm_hits,
                         'warmup_tentative_outputs':'suppressed, counted missing',
                         'prediction_only_outputs':'separately reported; included in fused tracking metric',
                         'GT_used_for_association':False,'parameter_status':'initial defaults, not calibrated on this validation set'}
    starts=[];byid={r['sample_id']:r for r in rows}
    for name in selected_methods[1:]:
        es=[e for e in events if e['method']==name]
        updated=[byid[e['sample_id']] for e in es if e['output_kind']=='updated']
        predicted=[byid[e['sample_id']] for e in es if e['output_kind']=='predicted']
        summary[name]['updated_outputs']=len(updated);summary[name]['prediction_only_outputs']=len(predicted)
        summary[name]['updated_outputs_labeled_queries']=sum(r['has_gt'] for r in updated)
        summary[name]['prediction_only_outputs_labeled_queries']=sum(r['has_gt'] for r in predicted)
        summary[name]['updated_success_1m_count']=sum(r[name+'_error_m'] is not None and r[name+'_error_m']<=1 for r in updated)
        summary[name]['predicted_success_1m_count']=sum(r[name+'_error_m'] is not None and r[name+'_error_m']<=1 for r in predicted)
        summary[name]['events']=dict(Counter(e['event'] for e in es))
        summary[name]['reset_reasons']=dict(Counter(e.get('reset_reason') for e in es if e.get('reset_reason')))
        summary[name]['reacquisition_confirmations']=sum(e.get('reacquisition',False) for e in es)
        summary[name]['confirmed_wrong_gt10m_queries']=sum(r[name+'_status']=='confirmed' and r[name+'_error_m'] is not None and r[name+'_error_m']>10 for r in rows)
        summary[name]['persistent_wrong_lock_queries']=persistent_wrong_queries(rows,name)
        summary[name]['warmup_missing_queries']=sum(e['output_kind']=='none' and
            (e.get('status')=='tentative' or e.get('tentative_track_count',0)>0) for e in es)
        summary[name]['lost_missing_queries']=sum(e['output_kind']=='none' and
            e.get('status')=='lost' for e in es)
        summary[name]['untrusted_missing_queries']=sum(e['output_kind']=='none' and
            e.get('status')=='no_trusted_track' for e in es)
        summary[name]['no_detection_missing_queries']=sum(e['output_kind']=='none' and
            e.get('status')=='uninitialized' and e.get('tentative_track_count',0)==0 for e in es)
        summary[name]['track_switches']=sum(e['event']=='switch_sustained_advantage' for e in es)
        if name in MULTI_METHODS:
            summary[name]['wrong_confirmations_over_2m']=sum(
                row['wrong_lock_over_2m'] for row in confirmation_rows if row['method']==name)
            continue
        for e in es:
            if e['event']=='confirmed':
                r=byid[e['sample_id']];error=r[name+'_error_m']
                starts.append(dict(method=name,sequence_id=e['sequence_id'],sample_id=e['sample_id'],query_time=e['timestamp'],
                                   track_id=e['track_id'],reacquisition=e['reacquisition'],error_m=error,
                                   wrong_lock_over_2m=error is not None and error>2))
        summary[name]['wrong_confirmations_over_2m']=sum(s['wrong_lock_over_2m'] for s in starts if s['method']==name)
    starts.extend(confirmation_rows)
    for name in selected_methods[1:]:
        counts=Counter()
        for r in rows:
            if not r['has_gt']:
                continue
            base=r['independent_top1_error_m'];e=r[name+'_error_m']
            b=base is not None and base<=1;c=e is not None and e<=1
            counts['both_success' if b and c else 'fixed' if c else 'broken' if b else 'both_failed']+=1
        summary[name]['paired_vs_independent']=dict(counts)
    write_csv(out/'track_confirmations.csv',starts)
    write_csv(out/'reacquisition_events.csv',[
        dict(method=e['method'],sequence_id=e['sequence_id'],sample_id=e['sample_id'],
             query_time=e['timestamp'],track_id=e['track_id'],
             error_m=byid[e['sample_id']][e['method']+'_error_m'])
        for e in events if e.get('reacquisition') and e['method'] in selected_methods])
    # Recovery relative to baseline severe/missing episodes; evaluation only.
    recovery=[]
    for seq in sorted({r['sequence_id'] for r in rows}):
        rs=[r for r in rows if r['sequence_id']==seq];episodes=[];current=[]
        for i,r in enumerate(rs):
            err=r['independent_top1_error_m']
            if err is None or err>10:current.append(i)
            elif current:episodes.append(current);current=[]
        if current:episodes.append(current)
        for ep in episodes:
            last=ep[-1];endtime=rs[last]['query_time']
            for name in selected_methods[1:]:
                first=next((i for i in range(last+1,len(rs)) if rs[i][name+'_error_m'] is not None and rs[i][name+'_error_m']<=1 and rs[i][name+'_output_kind']=='updated'),None)
                recovery.append(dict(sequence_id=seq,method=name,episode_start=rs[ep[0]]['sample_id'],episode_end=rs[last]['sample_id'],
                                     episode_queries=len(ep),episode_success_1m=sum(rs[i][name+'_error_m'] is not None and rs[i][name+'_error_m']<=1 for i in ep),
                                     next_good_updated_sample=rs[first]['sample_id'] if first is not None else None,
                                     recovery_after_episode_s=rs[first]['query_time']-endtime if first is not None else None))
    write_csv(out/'baseline_failure_episode_recovery.csv',recovery)
    write_csv(out/'per_query.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows])
    write_csv(out/'failure_comparison.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows
                                          if any(r[n+'_error_m'] is None or r[n+'_error_m']>1 for n in selected_methods)])
    sequence_metrics=[dict(sequence_id=seq,method=name,**metrics([r for r in rows if r['sequence_id']==seq],name))
                      for seq in sorted({r['sequence_id'] for r in rows}) for name in selected_methods]
    write_csv(out/'sequence_metrics.csv',sequence_metrics)
    if multi is not None:
        with (out/'all_track_history.jsonl').open('w') as handle:
            for name,manager in multi.items():
                for seq,state in sorted(manager.states.items()):
                    for number,history in sorted(state.history.items()):
                        for row in history:
                            handle.write(json.dumps(dict(method=name,sequence_id=seq,**row),allow_nan=False)+'\n')
        with (out/'track_scores.jsonl').open('w') as handle:
            for row in score_rows:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    text=['# Frozen-front-end temporal comparison','','|Method|1m success|Coverage|RMSE(m)|>10m|', '|---|---:|---:|---:|---:|']
    for name in selected_methods:
        m=summary[name];text.append(f'|{name}|{m["success_1m"]:.4%}|{m["coverage"]:.4%}|{m["rmse_3d_m"] if m["rmse_3d_m"] is not None else "NA"}|{m["severe_gt10m"]}|')
    text+=['','No GT is passed into tracking. All methods use the same frozen frontend pass.',
           'Tentative warmup is counted missing. Updated and prediction-only outputs are separated in summary.json.',
           'A tracker can lock onto a persistent background candidate. Inspect track_confirmations.csv and wrong-lock counts.',
           'This uses the existing GT-timestamp query schedule, not a claim of full-rate sensor tracking.',
           'Recovery lag is measured after baseline severe/missing episodes and requires a good updated output; it is descriptive, not a track-ID identity metric.']
    (out/'REPORT.md').write_text('\n'.join(text))
    target=out.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(out))
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'RESULTS ZIP: {target}',flush=True)

if __name__=='__main__':main()
