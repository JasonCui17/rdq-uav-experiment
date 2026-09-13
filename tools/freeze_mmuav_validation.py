#!/usr/bin/env python3
"""Audit saved validation artifacts and freeze; never open heldout sensor/GT data."""
import csv
import hashlib
import json
import math
import subprocess
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/mmuav_paper_reproduction'
FINAL=BASE/'final_reproduction'


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def matched(path):
    rows=read_csv(path)
    times=[Decimal(r['timestamp']) for r in rows]
    if len(set(times))!=len(times): raise ValueError(f'Duplicate evaluation timestamps: {path}')
    return set(times),{t for t,r in zip(times,rows) if all(math.isfinite(float(r[a])) for a in 'xyz')}


def main():
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    split_path=BASE/'splits/splits.json'
    sequences=json.loads(split_path.read_text())['validation_sub']
    summaries={};configs={};sets={};artifact_hashes={}
    for mode in ['geometric','full','full_temporal']:
        folder=FINAL/'validation'/mode
        summary=json.loads((folder/'overall_metrics.json').read_text())
        config=json.loads((folder/'run_config.json').read_text())
        metrics=read_csv(folder/'per_sequence_metrics.csv')
        failures=read_csv(folder/'processing_failures.csv')
        assert summary['planned_sequences']==summary['processed_sequences']==len(sequences)==15
        assert not failures and not summary['processing_failure_sequences'] and not summary['valid_empty_sequences']
        assert len(metrics)==15 and {r['sequence_id'] for r in metrics}==set(sequences)
        assert config['split']=='validation_sub' and config['sequence'] is None and not config['smoke_no_ar_fit']
        assert config['splits_sha256']==sha(split_path)
        assert config['classifier_sha256']==sha(config['classifier_checkpoint'])
        assert config['center_sha256']==sha(config['center_checkpoint'])
        sets[mode]={}
        for r in metrics:
            seq=r['sequence_id']
            all_times,valid=matched(folder/seq/'smoothed_final_trajectory.csv')
            assert len(all_times)==int(r['num_gt'])
            assert len(valid)==int(r['matched_timestamp_count'])
            assert len(all_times)-len(valid)==int(r['missing_prediction_count'])
            assert math.isclose(len(valid)/len(all_times),float(r['coverage']),abs_tol=1e-12)
            assert r['status']=='OK'
            sets[mode][seq]=(all_times,valid)
        for key in ['num_gt','matched_timestamp_count','missing_prediction_count']:
            assert sum(int(r[key]) for r in metrics)==summary['overall'][key]
        assert math.isclose(summary['overall']['coverage'],summary['overall']['matched_timestamp_count']/summary['overall']['num_gt'])
        for key in ['MSE_coord','MSE_3D','mean_3d_error']:
            pooled=sum(float(r[key])*int(r['matched_timestamp_count']) for r in metrics)/summary['overall']['matched_timestamp_count']
            assert math.isclose(pooled,summary['overall'][key],rel_tol=1e-10)
        summaries[mode]=summary['overall'];configs[mode]=config
        artifact_hashes[mode]={name:sha(folder/name) for name in ['overall_metrics.json','per_sequence_metrics.csv','processing_failures.csv','run_config.json']}
    ignored={'mode','output_dir'}
    base=configs['full']
    for mode,config in configs.items():
        assert {k:v for k,v in config.items() if k not in ignored}=={k:v for k,v in base.items() if k not in ignored}
    differences=[]
    for seq in sequences:
        assert all(sets[mode][seq][0]==sets['full'][seq][0] for mode in sets)
        a=sets['geometric'][seq][1];b=sets['full'][seq][1]
        if a!=b: differences.append(dict(sequence_id=seq,geometric_only=len(a-b),full_only=len(b-a)))
    results=dict(validation_audit_passed=True,same_matched_timestamp_set=not differences,
        matched_timestamp_differences=differences,GEOMETRIC=summaries['geometric'],FULL=summaries['full'],
        FULL_TEMPORAL=summaries['full_temporal'],
        full_vs_geometric=dict(MSE_coord_relative_reduction=1-summaries['full']['MSE_coord']/summaries['geometric']['MSE_coord'],
            mean_3d_error_relative_reduction=1-summaries['full']['mean_3d_error']/summaries['geometric']['mean_3d_error']),
        temporal_vs_full=dict(coverage_absolute_improvement=summaries['full_temporal']['coverage']-summaries['full']['coverage'],
            additional_matched_timestamps=summaries['full_temporal']['matched_timestamp_count']-summaries['full']['matched_timestamp_count']),
        findings=['Learned center correction remains valuable in the full trajectory pipeline.',
            'Temporal processing primarily improves trajectory coverage.',
            'M2 module-level improvement survives integration into the tracker.',
            'Paper Pose MSE 2.21375 cannot be directly compared to local metrics.'],
        metric_alignment='unresolved',artifact_sha256=artifact_hashes,heldout_used_for_selection=False)
    code_files=['tools/run_mmuav_pose_pipeline.py','src/rdq_uav/mmuav/pose_trajectory.py',
        'tools/build_mmuav_center_regression_dataset.py','tools/build_mmuav_cluster_dataset.py',
        'src/rdq_uav/baselines/mmuav_preprocess.py','src/rdq_uav/mmuav/attention_lstm.py',
        'src/rdq_uav/mmuav/center_regressor.py']
    frozen=dict(status='FROZEN_AFTER_VALIDATION',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        validation_git_commit=base['git_commit'],split_definition_hash=sha(split_path),data_root=base['data_root'],
        M1_checkpoint=base['classifier_checkpoint'],M1_checkpoint_sha256=base['classifier_sha256'],
        M2_checkpoint=base['center_checkpoint'],M2_checkpoint_sha256=base['center_sha256'],
        preprocessing=dict(mid360_eps=2,mid360_min_samples=10,window_size=20,window_stride=20,
            final_window='extra final20; source overwrite',accumulation='nonoverlap',livox_FPS_max_points=100,
            fusion_eps=1,fusion_min_samples=10),candidate_DBSCAN=dict(eps=1,min_samples=1),
        M2=dict(variant='full',num_points=64,sampling_seed='42 + per-frame cluster_id',geometric_center='full cluster mean before sampling'),
        tracker=base['tracker'],trajectory_selection=base['trajectory_selection'],
        AR_order=3,AR_grid_seconds=.1,max_gap_seconds=1.,
        AR_fit='per selected predicted trajectory, per-axis OLS; no GT; <8 states fallback [2,-1,0,0]',
        interpolation_method='linear; support and maximum1s gap mask',spline_s=.5,spline_degree=3,
        spline_short_track_fallback='linear if fewer than4 states',
        nearest_evaluation_tolerance_seconds=.05,evaluation_timestamp_only=True,
        evaluation_definition='frame-pooled matched-only MSE_coord; MSE_3D=3*MSE_coord; missing and coverage explicit',
        M3='BYPASSED',seed=42,heldout_used_for_selection=False,
        pipeline_code_sha256={file:sha(ROOT/file) for file in code_files},validation_results=results)
    content=json.dumps(frozen,indent=2)+'\n'
    path=FINAL/'frozen_reproduction_config.json'
    if path.exists() and path.read_text()!=content: raise FileExistsError('Refusing to change existing frozen config')
    path.write_text(content)
    small=ROOT/'results/mmuav_reproduction';small.mkdir(exist_ok=True)
    (small/'frozen_reproduction_config.json').write_text(content)
    (small/'validation_pose_pipeline_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(dict(validation_audit_passed=True,same_matched_timestamp_set=not differences,
        differences=differences,frozen_config=str(path),frozen_config_sha256=sha(path)),indent=2))


if __name__=='__main__':main()
