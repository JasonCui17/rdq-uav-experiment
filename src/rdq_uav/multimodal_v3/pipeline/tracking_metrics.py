"""Tracking metric and CSV helpers used after inference."""
import csv,math
from pathlib import Path
import numpy as np

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
