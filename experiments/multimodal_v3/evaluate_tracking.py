#!/usr/bin/env python3
"""Evaluate three protocols on a reusable frozen cache. CPU only; no GT in trackers."""
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

METHODS=('independent_top1','top1_kf','candidate_kf')

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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tracker-config',type=Path)
    args=p.parse_args();cache=args.cache.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Use a fresh results directory: {out}')
    manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('Incomplete cache')
    frames=[json.loads(x) for x in (cache/'index.jsonl').read_text().splitlines()]
    if len(frames)!=manifest['queries'] or len({r['sample_id'] for r in frames})!=len(frames):raise ValueError('Cache index mismatch')
    frames.sort(key=lambda r:(r['sequence_id'],r['query_time']))
    cfg=TrackerConfig(**json.loads(args.tracker_config.read_text())) if args.tracker_config else TrackerConfig()
    top1=SequenceTrackers(cfg,'top1');candidate=SequenceTrackers(cfg,'candidate')
    out.mkdir(parents=True);rows=[];events=[];started=time.monotonic()
    (out/'tracker_config.json').write_text(json.dumps(asdict(cfg),indent=2))
    (out/'cache_manifest.json').write_text(json.dumps(manifest,indent=2))
    with (out/'per_query.jsonl').open('w') as qfile,(out/'tracking_events.jsonl').open('w') as efile:
        for idx,frame in enumerate(frames):
            with np.load(cache/frame['file'],allow_pickle=False) as data:
                # Observation objects contain no GT and are the only data passed to trackers.
                dense=Observations(data['dense_xyz'],data['dense_score'],data['dense_rank'],data['dense_token_id'])
                glob=Observations(data['global_xyz'],data['global_score'],data['global_score'],data['global_token_id'])
                baseline=Observations(data['baseline_xyz'],data['baseline_score'],data['baseline_score'],data['baseline_token_id'])
                t=frame['query_time'];seq=frame['sequence_id']
                tracks={'top1_kf':top1.step(seq,t,dense,glob,baseline),'candidate_kf':candidate.step(seq,t,dense,glob,baseline)}
                # GT is read only after both state updates, exclusively for scoring.
                gt=data['gt_xyz'].tolist();has_gt=bool(data['has_gt'])
            r=dict(sequence_id=seq,sample_id=frame['sample_id'],query_time=t,has_gt=has_gt,gt_xyz=gt)
            xyz={'independent_top1':baseline.xyz[0].tolist() if len(baseline.xyz) else None,
                 **{name:track['xyz'] for name,track in tracks.items()}}
            for name in METHODS:
                r[name+'_xyz']=xyz[name]
                r[name+'_error_m']=float(np.linalg.norm(np.asarray(xyz[name])-gt)) if has_gt and xyz[name] is not None else None
            for name,track in tracks.items():
                e=dict(sequence_id=seq,sample_id=r['sample_id'],method=name,**track)
                efile.write(json.dumps(e,allow_nan=False)+'\n');events.append(e)
                for key in ('status','event','output_kind','track_id','selected_token_id','innovation_m','measurement_age_s'):
                    r[name+'_'+key]=track[key]
            qfile.write(json.dumps(r,allow_nan=False)+'\n');qfile.flush();efile.flush();rows.append(r)
            if (idx+1)%100==0 or idx+1==len(frames):print(f'tracking {idx+1}/{len(frames)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    summary={name:metrics(rows,name) for name in METHODS}
    summary['runtime']={'offline_tracking_wall_s':time.monotonic()-started,'cache_queries':len(rows),
                        'note':'Includes cache loading and event output; excludes frozen network inference'}
    summary['protocol']={'smoke_test':manifest.get('smoke_test',False),'query_schedule':manifest['protocol'],
                         'first_confirmed_output_after_hits':cfg.confirm_hits,
                         'warmup_tentative_outputs':'suppressed, counted missing',
                         'prediction_only_outputs':'separately reported; included in fused tracking metric',
                         'GT_used_for_association':False,'parameter_status':'initial defaults, not calibrated on this validation set'}
    starts=[];byid={r['sample_id']:r for r in rows}
    for name in ('top1_kf','candidate_kf'):
        es=[e for e in events if e['method']==name]
        updated=[byid[e['sample_id']] for e in es if e['output_kind']=='updated']
        predicted=[byid[e['sample_id']] for e in es if e['output_kind']=='predicted']
        summary[name]['updated_outputs']=len(updated);summary[name]['prediction_only_outputs']=len(predicted)
        summary[name]['updated_success_1m_count']=sum(r[name+'_error_m'] is not None and r[name+'_error_m']<=1 for r in updated)
        summary[name]['predicted_success_1m_count']=sum(r[name+'_error_m'] is not None and r[name+'_error_m']<=1 for r in predicted)
        summary[name]['events']=dict(Counter(e['event'] for e in es))
        summary[name]['reset_reasons']=dict(Counter(e['reset_reason'] for e in es if e['reset_reason']))
        summary[name]['reacquisition_confirmations']=sum(e['reacquisition'] for e in es)
        summary[name]['confirmed_wrong_gt10m_queries']=sum(r[name+'_status']=='confirmed' and r[name+'_error_m'] is not None and r[name+'_error_m']>10 for r in rows)
        for e in es:
            if e['event']=='confirmed':
                r=byid[e['sample_id']];error=r[name+'_error_m']
                starts.append(dict(method=name,sequence_id=e['sequence_id'],sample_id=e['sample_id'],query_time=e['timestamp'],
                                   track_id=e['track_id'],reacquisition=e['reacquisition'],error_m=error,
                                   wrong_lock_over_2m=error is not None and error>2))
        summary[name]['wrong_confirmations_over_2m']=sum(s['wrong_lock_over_2m'] for s in starts if s['method']==name)
    transitions=[]
    for name in ('top1_kf','candidate_kf'):
        counts=Counter()
        for r in rows:
            base=r['independent_top1_error_m'];e=r[name+'_error_m']
            b=base is not None and base<=1;c=e is not None and e<=1
            counts['both_success' if b and c else 'fixed' if c else 'broken' if b else 'both_failed']+=1
        summary[name]['paired_vs_independent']=dict(counts)
    write_csv(out/'track_confirmations.csv',starts)
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
            for name in ('top1_kf','candidate_kf'):
                first=next((i for i in range(last+1,len(rs)) if rs[i][name+'_error_m'] is not None and rs[i][name+'_error_m']<=1 and rs[i][name+'_output_kind']=='updated'),None)
                recovery.append(dict(sequence_id=seq,method=name,episode_start=rs[ep[0]]['sample_id'],episode_end=rs[last]['sample_id'],
                                     episode_queries=len(ep),episode_success_1m=sum(rs[i][name+'_error_m'] is not None and rs[i][name+'_error_m']<=1 for i in ep),
                                     next_good_updated_sample=rs[first]['sample_id'] if first is not None else None,
                                     recovery_after_episode_s=rs[first]['query_time']-endtime if first is not None else None))
    write_csv(out/'baseline_failure_episode_recovery.csv',recovery)
    write_csv(out/'per_query.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows])
    write_csv(out/'failure_comparison.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows
                                          if any(r[n+'_error_m'] is None or r[n+'_error_m']>1 for n in METHODS)])
    sequence_metrics=[dict(sequence_id=seq,method=name,**metrics([r for r in rows if r['sequence_id']==seq],name))
                      for seq in sorted({r['sequence_id'] for r in rows}) for name in METHODS]
    write_csv(out/'sequence_metrics.csv',sequence_metrics)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    text=['# Frozen-front-end temporal comparison','','|Method|1m success|Coverage|RMSE(m)|>10m|', '|---|---:|---:|---:|---:|']
    for name in METHODS:
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
