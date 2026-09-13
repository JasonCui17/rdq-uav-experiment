#!/usr/bin/env python3
"""Manual system-level validation: all candidates, no oracle selections."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from build_mmuav_center_regression_dataset import preprocess
from build_mmuav_cluster_dataset import timestamp_files,load_xyz,load_gt
from rdq_uav.baselines.mmuav_preprocess import _dbscan_labels
from rdq_uav.mmuav.attention_lstm import AttentionLSTMClassifier
from rdq_uav.mmuav.center_regressor import CenterRegressor,sample_local,regression_metrics
from rdq_uav.mmuav.pose_trajectory import track_candidates,select_track,ar_complete,resample

BASE=ROOT/'outputs/mmuav_paper_reproduction'


def write(path,rows,fields=None):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0] if rows else ['timestamp','x','y','z']))
        writer.writeheader();writer.writerows(rows)


def trajectory(path,t,p):
    write(path,[dict(timestamp=float(ti),**{a:float(pi[j]) for j,a in enumerate('xyz')}) for ti,pi in zip(t,p)])


def infer_candidates(folder,model):
    """Only fusion NPY, no GT metadata, no accepted IDs or spatial gate."""
    frames=[];audit=[]
    for file in timestamp_files(folder/'lidar_fusion'):
        data=np.load(file,allow_pickle=False)
        # Original fusion writes successful empty frames as shape (0,).
        points=np.empty((0,3)) if data.shape in ((0,),(0,3)) else load_xyz(file)
        centers=[]
        labels=_dbscan_labels(points,1,1) if len(points) else np.empty(0)
        for cid in sorted(set(labels)-{-1}):
            cluster=points[labels==cid];center=cluster.mean(0)
            pred=center.copy()
            if model is not None:
                local=sample_local(cluster,center,64,42+int(cid))
                with torch.no_grad():
                    delta=model(torch.tensor(local[None]),torch.tensor(center[None],dtype=torch.float32)).numpy()[0]
                pred=center+delta
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite corrected candidate')
            centers.append(pred)
            audit.append(dict(timestamp=float(file.stem),cluster_id=int(cid),point_count=len(cluster),
                **{f'geometric_{a}':float(center[j]) for j,a in enumerate('xyz')},
                **{a:float(pred[j]) for j,a in enumerate('xyz')}))
        frames.append((float(file.stem),np.array(centers).reshape(-1,3)))
    return frames,audit


def score(pred,gt):
    matched=np.isfinite(pred).all(1)
    return dict(num_gt=len(gt),matched_timestamp_count=int(matched.sum()),
        missing_prediction_count=int((~matched).sum()),coverage=float(matched.mean()) if len(gt) else 0.,
        error_scope='matched timestamps only; missing predictions separately counted',
        **regression_metrics(pred[matched],gt[matched]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['geometric','full','full_temporal'],required=True)
    p.add_argument('--split',choices=['validation_sub'],default='validation_sub')
    p.add_argument('--data-root',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/train'))
    p.add_argument('--splits',type=Path,default=BASE/'splits/splits.json')
    p.add_argument('--classifier-checkpoint',type=Path,default=BASE/'classification/attention_9d/best_val_loss.pth')
    p.add_argument('--center-checkpoint',type=Path,default=BASE/'center_regression/pointnet_m2/best_val_loss.pth')
    p.add_argument('--cache-dir',type=Path,default=BASE/'datasets/center_regression/sequences')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sequence',help='Optional single validation sequence smoke')
    p.add_argument('--smoke-no-ar-fit',action='store_true',help='Single sequence only; fixed AR coefficients, not formal results')
    args=p.parse_args()
    split=json.loads(args.splits.read_text());sequences=split[args.split]
    if args.sequence:
        if args.sequence not in sequences: p.error('Sequence must belong to validation_sub')
        sequences=[args.sequence]
    if args.smoke_no_ar_fit and not args.sequence: p.error('Smoke flag requires one sequence')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError('Refusing to overwrite output')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    m1=AttentionLSTMClassifier();m1.load_state_dict(torch.load(args.classifier_checkpoint,map_location='cpu',weights_only=True));m1.eval()
    m2=None
    if args.mode!='geometric':
        m2=CenterRegressor();m2.load_state_dict(torch.load(args.center_checkpoint,map_location='cpu',weights_only=True));m2.eval()
    config=dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        evaluation_mode='system_level_no_gt_selection',evaluation_timestamp_only=True,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        classifier_sha256=sha(args.classifier_checkpoint),center_sha256=sha(args.center_checkpoint),
        splits_sha256=sha(args.splits),trajectory_selection='longest_lived_track',
        design_status='RECONSTRUCTED_DESIGN',M3='BYPASSED',AR_order=3,AR_grid_seconds=.1,
        max_gap_seconds=1.,spline_s=.5,nearest_evaluation_tolerance_seconds=.05,
        tracker=dict(process_noise=.15,measurement_noise=.001,missed_distance=3.,covar_trace_thresh=30.,min_points=1))
    (args.output_dir/'run_config.json').write_text(json.dumps(config,indent=2))
    metrics=[];failures=[];all_pred=[];all_gt=[]
    for seq in sequences:
        print(f'{args.mode} | {seq}',flush=True)
        out=args.output_dir/seq;out.mkdir()
        try:
            raw=args.data_root/seq
            if not timestamp_files(raw/'lidar_360') or not timestamp_files(raw/'livox_avia'):
                raise FileNotFoundError('Missing raw input frames')
            cached=args.cache_dir/seq
            # Aggregate provenance only. Never open sequence_records.json,
            # metadata.csv, all_candidates.csv or GT-conditioned candidate shards.
            if cached.exists():
                provenance=json.loads((args.cache_dir.parent/'dataset_summary.json').read_text())
                if (provenance['checkpoint_sha256']!=config['classifier_sha256'] or
                    provenance['splits_sha256']!=config['splits_sha256'] or
                    Path(provenance['source_root']).resolve()!=args.data_root.resolve()):
                    raise ValueError('Cache classifier/split/data-root mismatch')
                if not (cached/'lidar_fusion').is_dir(): raise FileNotFoundError('Incomplete fusion cache')
                folder=cached
            else:
                folder=out/'preprocessed';folder.mkdir();preprocess(raw,folder,m1)
            frames,candidates=infer_candidates(folder,m2)
            write(out/'all_candidate_measurements.csv',candidates)
            tracks=track_candidates(frames)
            write(out/'raw_tracker_trajectory.csv',tracks)
            selected=select_track(tracks);write(out/'selected_trajectory.csv',selected)
            t=np.array([r['timestamp'] for r in selected]);xyz=np.array([[r[a] for a in 'xyz'] for r in selected]).reshape(-1,3)
            status='OK' if len(selected) else 'VALID_EMPTY_RESULT'
            completed_t,completed=t,xyz
            if len(t)>=2 and args.mode=='full_temporal':
                grid=np.arange(t[0],t[-1]+1e-7,.1)
                completed_t,completed,coeff=ar_complete(t,xyz,grid,fit=not args.smoke_no_ar_fit)
                (out/'ar_coefficients.json').write_text(json.dumps(dict(coefficients=coeff.tolist(),
                    fit_input='selected predicted trajectory only; NO GT',smoke=args.smoke_no_ar_fit),indent=2))
            trajectory(out/'completed_trajectory.csv',completed_t,completed)
            # Evaluation time coordinates only; XYZ remains unread until predictions are saved.
            query=np.array([float(f.stem) for f in timestamp_files(raw/'ground_truth')])
            prediction=np.full((len(query),3),np.nan)
            if args.mode=='full_temporal':
                interpolated=resample(completed_t,completed,query)
                prediction=resample(completed_t,completed,query,smooth=True)
            else:
                if len(t):
                    for i,q in enumerate(query):
                        j=np.argmin(abs(t-q))
                        if abs(t[j]-q)<=.05: prediction[i]=xyz[j]
                interpolated=prediction.copy()
            trajectory(out/'interpolated_trajectory.csv',query,interpolated)
            trajectory(out/'smoothed_final_trajectory.csv',query,prediction)
            gt_t,gt=load_gt(raw)  # ONLY AFTER final prediction generation.
            np.testing.assert_array_equal(gt_t,query)
            result=dict(sequence_id=seq,status=status,**score(prediction,gt))
            (out/'metrics.json').write_text(json.dumps(result,indent=2));metrics.append(result)
            all_pred.extend(prediction);all_gt.extend(gt)
        except Exception as error:
            failures.append(dict(sequence_id=seq,status='PROCESSING_FAILURE',error=f'{type(error).__name__}: {error}'))
            print(f'PROCESSING_FAILURE {seq}: {error}',flush=True)
    write(args.output_dir/'per_sequence_metrics.csv',metrics)
    write(args.output_dir/'processing_failures.csv',failures,['sequence_id','status','error'])
    overall=score(np.array(all_pred).reshape(-1,3),np.array(all_gt).reshape(-1,3))
    summary=dict(mode=args.mode,planned_sequences=len(sequences),processed_sequences=len(metrics),
        processing_failure_sequences=failures,valid_empty_sequences=[r['sequence_id'] for r in metrics if r['status']=='VALID_EMPTY_RESULT'],
        overall=overall,overall_scope='successful processing sequences; failures listed separately',
        sequence_mean_coverage=float(np.mean([r['coverage'] for r in metrics])) if metrics else None,
        smoke=bool(args.sequence))
    (args.output_dir/'overall_metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if failures: raise RuntimeError('PROCESSING_FAILURE: inspect processing_failures.csv')


if __name__=='__main__':main()
