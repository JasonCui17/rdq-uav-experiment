#!/usr/bin/env python3
"""Report paired accuracy and measured epoch time; never assume LoRA is faster."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from compare import read_rows, metrics, success


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--phase',choices=('benchmark','train'),default='train')
    args=p.parse_args()
    records=[];results=[];missing=[]
    common=None
    for regime in ('full','heads','lora'):
        manifest_path=args.pilot/regime/'manifest.json'
        if not manifest_path.exists(): missing.append(str(manifest_path));continue
        manifest=json.loads(manifest_path.read_text())
        protocol={k:manifest[k] for k in ('warmstart_sha256','split_sha256','epochs')}
        if common is None: common=protocol
        if common!=protocol: raise ValueError('different warmstart, split, or training budgets')
        for run in manifest['runs']:
            folder=Path(run['folder']);timing=folder/args.phase/'efficiency/timing.jsonl'
            params=folder/args.phase/'efficiency/parameters.json'
            predictions=folder/'validation_sub/per_query.jsonl'
            if not all(p.exists() for p in ((timing,params,predictions) if args.phase=='train' else (timing,params))):
                missing.append(f"{regime}/{run['name']}_seed{run['seed']}");continue
            times=[json.loads(l) for l in timing.read_text().splitlines()]
            # Resumed partial epochs remain visible; don't average duplicates silently.
            if len({t['epoch'] for t in times})!=len(times): raise ValueError('duplicate/partial resumed epoch timing; review logs')
            stable=[t for t in times if t['epoch']>0 and t['optimizer_updates']>0]
            measured=stable or times
            par=json.loads(params.read_text())
            if args.phase=='benchmark' and sum(x['optimizer_updates'] for x in times)<par.get('requested_max_steps',-1):
                raise ValueError('benchmark stopped before requested updates; increase prepared epochs or inspect skipped batches')
            rows=read_rows(predictions) if args.phase=='train' else {}
            records.append((regime,run,par,rows,measured,bool(stable)))
    full={(r['name'],r['seed']):(par,rows,t) for regime,r,par,rows,t,_ in records if regime=='full'}
    for regime,run,par,rows,t,stable in records:
        ref=full.get((run['name'],run['seed']))
        if ref is None: missing.append('full reference for '+run['name']);continue
        rp,rr,rt=ref
        if rows.keys()!=rr.keys() or any(rows[k]['gt_xyz']!=rr[k]['gt_xyz'] for k in rows):
            raise ValueError('query/GT mismatch')
        if any(par.get(k)!=rp.get(k) for k in ('torch_version','gpu_name','precision','batch_size','accumulate')):
            raise ValueError('hardware/software/batch/precision mismatch')
        if [(x['samples'],x['optimizer_updates']) for x in t]!=[(x['samples'],x['optimizer_updates']) for x in rt]:
            raise ValueError('measured epoch samples/update counts differ')
        train=statistics.median(x['train_wall_s'] for x in t)
        wall=statistics.median(x['epoch_wall_s'] for x in t)
        peak=[x['epoch_peak_allocated_mib'] for x in t if x['epoch_peak_allocated_mib'] is not None]
        results.append(dict(regime=regime,variant=run['name'],seed=run['seed'],
                            trainable_parameters=par['trainable_parameters'],adapter_parameters=par['adapter_parameters'],
                            trainable_fraction=par['trainable_fraction'],measured_updates=sum(x['optimizer_updates'] for x in t),
                            measured_samples=sum(x['samples'] for x in t),**metrics(rows.values()),
                            median_train_epoch_s=train,median_validation_s=statistics.median(x['validation_wall_s'] for x in t),
                            median_epoch_wall_s=wall,
                            train_speedup_vs_full=statistics.median(x['train_wall_s'] for x in rt)/train if train else None,
                            epoch_speedup_vs_full=statistics.median(x['epoch_wall_s'] for x in rt)/wall if wall else None,
                            peak_allocated_mib=max(peak) if peak else None,excluded_first_epoch=stable,
                            wrong_to_correct=sum(success(v) and not success(rr[k]) for k,v in rows.items()),
                            correct_to_wrong=sum(not success(v) and success(rr[k]) for k,v in rows.items())))
    report=dict(phase=args.phase,missing=missing,results=results,note='3-epoch pilot is an efficiency/convergence screen, not final best-method evidence. Timing excludes independent eval and may omit checkpoint overhead.')
    (args.pilot/f'efficiency_{args.phase}_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if results:
        with (args.pilot/f'efficiency_{args.phase}_comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
