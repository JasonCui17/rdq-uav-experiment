#!/usr/bin/env python3
"""Paired query comparisons; validation selects, heldout only reports."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read_rows(path):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not row['has_gt3d']: continue
        key = (row['sequence_id'], row['sample_id'], row['query_time'])
        if key in result: raise ValueError(f'duplicate query {key}')
        result[key] = row
    if not result: raise ValueError(f'no labeled queries in {path}')
    return result


def success(row):
    return row['final_error'] is not None and row['final_error'] <= 1.


def metrics(rows):
    rows = list(rows)
    errors = sorted(float(r['final_error']) for r in rows if r['final_error'] is not None)
    if any(not math.isfinite(x) for x in errors): raise ValueError('non-finite predictions')
    n = len(rows)
    return {'queries': n, 'coverage': len(errors)/n if n else None,
            'success_1m': sum(success(r) for r in rows)/n if n else None,
            'oracle10_1m': sum(bool(r['final_distances'][:10]) and min(r['final_distances'][:10]) <= 1 for r in rows)/n if n else None,
            'rmse_3d_m': math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            'median_m': statistics.median(errors) if errors else None,
            'p95_m': errors[min(len(errors)-1, math.ceil(.95*len(errors))-1)] if errors else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--split', choices=('validation_sub','heldout_test_sub'), default='validation_sub')
    p.add_argument('--reference', default='control')
    args = p.parse_args()
    manifest = json.loads((args.suite/'manifest.json').read_text())
    loaded, missing = {}, []
    for run in manifest['runs']:
        path = Path(run['folder'])/args.split/'per_query.jsonl'
        if not path.exists():
            missing.append(f"{run['name']}_seed{run['seed']}"); continue
        loaded[(run['name'],run['seed'])] = read_rows(path)
    results, bins = [], []
    for (name, seed), rows in loaded.items():
        ref = loaded.get((args.reference, seed))
        if ref is None: raise ValueError(f'missing reference {args.reference}, seed={seed}')
        if rows.keys() != ref.keys(): raise ValueError('query sets differ; never compare only their intersection')
        for key in rows:
            if rows[key]['gt_xyz'] != ref[key]['gt_xyz']: raise ValueError('GT differs across runs')
        result = {'name':name,'seed':seed, **metrics(rows.values()),
                  'wrong_to_correct':sum(success(r) and not success(ref[k]) for k,r in rows.items()),
                  'correct_to_wrong':sum(not success(r) and success(ref[k]) for k,r in rows.items())}
        results.append(result)
        for lo,hi in ((0,30),(30,60),(60,math.inf)):
            selected = [r for r in rows.values() if lo <= math.sqrt(sum(x*x for x in r['gt_xyz'])) < hi]
            bins.append({'name':name,'seed':seed,'range_m':f'[{lo},{hi})',**metrics(selected)})
    grouped = []
    names = sorted({r['name'] for r in results})
    for name in names:
        group = [r for r in results if r['name']==name]
        entry = {'name':name,'seeds':[r['seed'] for r in group]}
        for key in ('success_1m','rmse_3d_m','coverage','oracle10_1m'):
            vals = [r[key] for r in group if r[key] is not None]
            entry[key+'_mean'] = statistics.mean(vals) if vals else None
            entry[key+'_std'] = statistics.stdev(vals) if len(vals)>1 else None
        grouped.append(entry)
    winner = None
    if args.split == 'validation_sub' and not missing and grouped:
        reference_coverage = next(g['coverage_mean'] for g in grouped if g['name']==args.reference)
        eligible = [g for g in grouped if g['coverage_mean'] >= reference_coverage and g['rmse_3d_m_mean'] is not None]
        if eligible:
            winner = min(eligible,key=lambda g:(-g['success_1m_mean'],g['rmse_3d_m_mean'],g['name']))['name']
    report = {'split':args.split,'reference':args.reference,'missing_runs':missing,
              'selection_rule':'validation success@1m descending, RMSE ascending; coverage must not drop',
              'provisional_validation_winner':winner,'seed_summary':grouped,'paired':results,'range_bins':bins,
              'note':'Single-seed winner is provisional. No background-only GT: these metrics are localization errors, not false-alarm rate.'}
    out = args.suite/f'comparison_{args.split}.json'; out.write_text(json.dumps(report,indent=2,allow_nan=False))
    if results:
        with (args.suite/f'comparison_{args.split}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
