#!/usr/bin/env python3
"""Read frozen M2; write diagnostics elsewhere. Never train or rewrite M2."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_mmuav_center_regressor import ClusterDataset, ROOT
from evaluate_mmuav_center_regression import accepted_rows, arrays
from build_mmuav_center_regression_dataset import write_csv, MODE
from rdq_uav.mmuav.center_regressor import CenterRegressor, regression_metrics, verify_evaluation_ids


def main():
    base=ROOT/'outputs/mmuav_paper_reproduction'
    source=base/'center_regression/pointnet_m2'
    out=base/'center_regression/m2_diagnostics'
    out.mkdir(exist_ok=True)
    names=['training_history.csv','best_val_loss.pth','last.pth','center_regression_comparison.csv',
           'evaluation_summary.json','evaluation_sample_ids.csv']
    hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    summary=json.loads((source/'evaluation_summary.json').read_text())
    history=list(csv.DictReader((source/'training_history.csv').open()))
    assert summary['best_epoch']==3 and summary['actual_epochs']==18
    assert int(min(history,key=lambda r:float(r['val_loss']))['epoch'])==3
    assert int(history[-1]['epoch'])==18
    rows=[r for r in accepted_rows(base/'datasets/center_regression') if r['split']=='validation_sub']
    frozen=[r['sample_id'] for r in csv.DictReader((source/'evaluation_sample_ids.csv').open())]
    verify_evaluation_ids([r['sample_id'] for r in rows],frozen)
    assert len(frozen)==2359
    torch.set_num_threads(1)
    model=CenterRegressor()
    model.load_state_dict(torch.load(source/'best_val_loss.pth',map_location='cpu',weights_only=True),strict=True)
    model.eval(); pred=[]
    with torch.no_grad():
        for local,center,_ in torch.utils.data.DataLoader(ClusterDataset(rows,64,False),batch_size=64):
            pred.append((model(local,center)+center).numpy())
    pred=np.concatenate(pred); geom,gt=arrays(rows)
    metrics=regression_metrics(pred,gt)
    np.testing.assert_allclose(metrics['MSE_coord'],summary['comparison'][1]['MSE_coord'],rtol=1e-5)
    ge=np.linalg.norm(geom-gt,axis=1); pe=np.linalg.norm(pred-gt,axis=1)
    write_csv(out/'per_sample_center_regression.csv',[dict(sample_id=r['sample_id'],sequence_id=r['sequence_id'],
        point_count=r['point_count'],geometric_error_3d=g,pointnet_error_3d=p,delta_error=p-g)
        for r,g,p in zip(rows,ge,pe)])
    axes=[]
    for i,a in enumerate('XYZ'):
        g=float(((geom-gt)[:,i]**2).mean()); p=float(((pred-gt)[:,i]**2).mean())
        axes.append(dict(axis=a,geometric_MSE=g,pointnet_MSE=p,absolute_reduction=g-p,relative_reduction=1-p/g))
    write_csv(out/'center_regression_axis_comparison.csv',axes)
    def group(indices,**fields):
        g=regression_metrics(geom[indices],gt[indices]); p=regression_metrics(pred[indices],gt[indices])
        return dict(**fields,samples=len(indices),geometric_MSE_coord=g['MSE_coord'],pointnet_MSE_coord=p['MSE_coord'],
            geometric_MSE_3D=g['MSE_3D'],pointnet_MSE_3D=p['MSE_3D'],
            geometric_mean_3d_error=g['mean_3d_error'],pointnet_mean_3d_error=p['mean_3d_error'],
            relative_reduction=1-p['MSE_coord']/g['MSE_coord'])
    counts=[group([i for i,r in enumerate(rows) if low<=int(r['point_count'])<=high],point_count_bin=f'{low}-{high}')
            for low,high in [(1,5),(6,10),(11,20),(21,50),(51,100)]]
    sequences=[group([i for i,r in enumerate(rows) if r['sequence_id']==s],sequence_id=s)
               for s in sorted({r['sequence_id'] for r in rows})]
    write_csv(out/'center_regression_by_point_count.csv',counts)
    write_csv(out/'center_regression_by_sequence.csv',sequences)
    ranked=sorted(sequences,key=lambda r:r['relative_reduction'],reverse=True)
    report=dict(evaluation_mode=MODE,best_epoch=3,stopped_epoch=18,samples=2359,
        checkpoint='best_val_loss.pth',frozen_files_sha256=hashes,
        improved_samples=int((pe<ge-1e-8).sum()),degraded_samples=int((pe>ge+1e-8).sum()),
        unchanged_samples=int((np.abs(pe-ge)<=1e-8).sum()),improvement_rate=float((pe<ge-1e-8).mean()),
        axes=axes,point_counts=counts,sequences=sequences,top_10_improvement=ranked[:10],
        top_10_degradation=sorted([r for r in sequences if r['relative_reduction']<0],key=lambda r:r['relative_reduction'])[:10],
        metric_alignment='unresolved; numerically close is not strict reproduction')
    (out/'m2_frozen_summary.json').write_text(json.dumps(report,indent=2))
    small=ROOT/'results/mmuav_reproduction'
    (small/'m2_frozen_summary.json').write_text(json.dumps(report,indent=2))
    comparison=[dict(Model='Geometric baseline',**regression_metrics(geom,gt),relative_reduction_vs_geometric=0),
                dict(Model='M2_FULL',**metrics,relative_reduction_vs_geometric=1-metrics['MSE_coord']/regression_metrics(geom,gt)['MSE_coord'])]
    for name in ('M2_POINTS_ONLY','M2_CENTER_ONLY'):
        comparison.append(dict(Model=name,**{k:'' for k in comparison[0] if k!='Model'}))
    write_csv(out/'m2_5_comparison.csv',comparison)
    assert hashes=={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
