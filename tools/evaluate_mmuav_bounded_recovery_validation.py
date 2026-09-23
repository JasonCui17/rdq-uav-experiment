#!/usr/bin/env python3
"""Fair fixed-selector/shared-recovery comparison on validation saved states."""
import csv,json,sys,hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from rdq_uav.mmuav.pose_trajectory import select_track
from rdq_uav.mmuav.observation_path import select_observation_path,PathConfig
from rdq_uav.mmuav.bounded_recovery import recover_bounded_missing,RecoveryConfig
from evaluate_mmuav_observation_path_validation import read,write,map_prediction,metric,paired_errors,relative,tail_stats

MODELS=['CONTROL_OBSERVED','CONTROL_BOUNDED','OBS_PATH_OBSERVED','OBS_PATH_BOUNDED']


def apply_shared_recovery(control,obs,query,config):
    return dict(CONTROL_BOUNDED=recover_bounded_missing(control,query,config),
        OBS_PATH_BOUNDED=recover_bounded_missing(obs,query,config))


def prepare_output(out,config):
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing nonempty output directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'fixed_recovery_config.json').open('x') as f:json.dump(config,f,indent=2)


def recovery_stats(records):
    counts={k:sum(r['state_type']==k for r in records) for k in ['observed','predicted','missing']}
    longest=run=0;start=None;max_duration=0.
    for r in records:
        if r['state_type']=='predicted':
            run+=1
            if start is None:start=r['timestamp']
            longest=max(longest,run);max_duration=max(max_duration,r['timestamp']-start)
        else:run=0;start=None
    internal=[r for r in records if r['recovery_source']=='internal_interpolation']
    tail=[r for r in records if r['recovery_source']=='tail_prediction']
    return dict(observed_output_count=counts['observed'],predicted_output_count=counts['predicted'],missing_output_count=counts['missing'],
        internal_recovered_count=len(internal),tail_predicted_count=len(tail),
        longest_internal_recovery_duration=max((r['recovery_duration'] for r in internal),default=0.),
        longest_tail_prediction_duration=max((r['recovery_duration'] for r in tail),default=0.),
        longest_consecutive_predicted_run=longest,max_consecutive_predicted_duration=max_duration)


def main():
    base=ROOT/'outputs/mmuav_paper_reproduction';out=base/'bounded_recovery_validation_fixed'
    recovery=RecoveryConfig();config=dict(observation_path=asdict(PathConfig()),**asdict(recovery),
        recovery_method='existing build_bounded_output_path: internal linear / endpoint velocity tail',query_tolerance=.05,
        query_clock='unique saved raw tracker timestamps, identical for both selectors; no GT clock used',
        evaluation_mapping='nearest finite output within 0.05s; missing outputs are not predictions',
        config_status='FIXED_BEFORE_COMPARISON_REUSED_EXISTING_1S_INTERNAL_0.1S_TAIL')
    prepare_output(out,config);frozen_text=(out/'fixed_recovery_config.json').read_bytes()
    paths=[ROOT/'src/rdq_uav/mmuav/observation_path.py',ROOT/'src/rdq_uav/mmuav/pose_trajectory.py',
        base/'final_reproduction/frozen_reproduction_config.json']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in paths}
    split=json.loads((base/'splits/splits.json').read_text());sequences=split['validation_sub'];data_root=Path(split['source_root'])
    generated={};stats=[]
    for seq in sequences:
        folder=base/'final_reproduction/validation/full'/seq
        raw_path=folder/'raw_tracker_trajectory.csv';hashes[str(raw_path)]=sha(raw_path);raw=read(raw_path)
        for r in raw:
            for k in ['timestamp','x','y','z','vx','vy','vz']:r[k]=float(r[k])
            r['measurement_update']=r['measurement_update']=='True'
        query=np.unique([r['timestamp'] for r in raw])
        original=select_track(raw)
        control=[dict(r,state_type='observed') for r in original if r['measurement_update']]
        obs=select_observation_path(raw,PathConfig())['selected']
        generated[seq]={}
        dest=out/seq;dest.mkdir()
        for label,selected in [('CONTROL_OBSERVED',control),('OBS_PATH_OBSERVED',obs)]:
            write(dest/(label+'.csv'),selected,['timestamp','x','y','z','vx','vy','vz','track_id','measurement_update','source_state_index','measurement_id','state_type'])
        bounded_paths=apply_shared_recovery(control,obs,query,recovery)
        for observed_label,bounded_label,selected in [('CONTROL_OBSERVED','CONTROL_BOUNDED',control),('OBS_PATH_OBSERVED','OBS_PATH_BOUNDED',obs)]:
            # Both selector paths call the EXACT same function with SAME frozen object.
            bounded=bounded_paths[bounded_label]
            write(dest/(bounded_label+'.csv'),bounded,['timestamp','x','y','z','state_type','recovery_source','recovery_duration'])
            generated[seq][observed_label]=selected;generated[seq][bounded_label]=bounded
            for label,records in [(observed_label,[dict(r,recovery_source='observed',recovery_duration=0.) for r in selected]),(bounded_label,bounded)]:
                stats.append(dict(sequence_id=seq,method=label,**recovery_stats(records)))
        print('All four predictions saved: '+seq,flush=True)
    # Only now read GT XYZ. Selector/recovery have no GT dependency.
    from build_mmuav_cluster_dataset import load_gt
    per=[];pairs=[];coverage=[];safety=[];errors={m:[] for m in MODELS};paired_a=[];paired_b=[];total_gt=0
    stat_lookup={(r['sequence_id'],r['method']):r for r in stats}
    for seq,methods in generated.items():
        t,g=load_gt(data_root/seq);total_gt+=len(t);mapped={};seq_metrics={}
        for method,records in methods.items():
            finite=[r for r in records if np.isfinite([float(r[k]) for k in 'xyz']).all()]
            p=map_prediction(finite,t);mapped[method]=p;e=p-g;errors[method].append(e)
            m=metric(e,len(t));h=tail_stats(e,t);st=stat_lookup[(seq,method)]
            row=dict(sequence_id=seq,method=method,**m,max_3d_error=h['max_3d_error'],
                error_gt_2m_count=h['error_gt_2m'],error_gt_5m_count=h['error_gt_5m'],error_gt_10m_count=h['error_gt_10m'],
                **{k:v for k,v in st.items() if k not in ['sequence_id','method']})
            per.append(row);seq_metrics[method]=row
            unsafe=bool(st['longest_tail_prediction_duration']>recovery.tail_prediction_limit+1e-9 or st['longest_internal_recovery_duration']>recovery.internal_gap_limit+1e-9)
            safety.append(dict(sequence_id=seq,method=method,max_tail_prediction_duration=st['longest_tail_prediction_duration'],
                max_consecutive_predicted_duration=st['max_consecutive_predicted_duration'],num_gt_2m=h['error_gt_2m'],
                num_gt_5m=h['error_gt_5m'],num_gt_10m=h['error_gt_10m'],max_error=h['max_3d_error'],safety_violation=unsafe))
        common,ea,eb=paired_errors(mapped['CONTROL_BOUNDED'],mapped['OBS_PATH_BOUNDED'],g)
        pa,pb=metric(ea,len(ea)),metric(eb,len(eb));paired_a.append(ea);paired_b.append(eb)
        pairs.append(dict(sequence_id=seq,common_count=int(common.sum()),
            **{'control_'+k:pa[k] for k in ['MSE_coord','mean_3d_error','median_3d_error']},
            **{'obs_path_'+k:pb[k] for k in ['MSE_coord','mean_3d_error','median_3d_error']},
            **{'relative_'+k+'_change':relative(pa[k],pb[k]) for k in ['MSE_coord','mean_3d_error','median_3d_error']}))
        for a,b in [('CONTROL_OBSERVED','CONTROL_BOUNDED'),('OBS_PATH_OBSERVED','OBS_PATH_BOUNDED')]:
            coverage.append(dict(sequence_id=seq,selector=a.replace('_OBSERVED',''),observed_coverage=seq_metrics[a]['coverage'],
                bounded_coverage=seq_metrics[b]['coverage'],coverage_increase_pp=100*(seq_metrics[b]['coverage']-seq_metrics[a]['coverage']),
                additional_matched=seq_metrics[b]['matched_timestamp_count']-seq_metrics[a]['matched_timestamp_count']))
    overall={m:metric(np.concatenate(errors[m]),total_gt) for m in MODELS}
    pa=metric(np.concatenate(paired_a),sum(map(len,paired_a)));pb=metric(np.concatenate(paired_b),sum(map(len,paired_b)))
    summary=dict(successful_sequences=len(generated),config=config,overall=overall,
        paired=dict(common_count=pa['matched_timestamp_count'],CONTROL_BOUNDED=pa,OBS_PATH_BOUNDED=pb,
            relative_changes={k:relative(pa[k],pb[k]) for k in ['MSE_coord','mean_3d_error','median_3d_error']}),
        recovery_gain_pp={b:100*(overall[b]['coverage']-overall[a]['coverage']) for a,b in [('CONTROL_OBSERVED','CONTROL_BOUNDED'),('OBS_PATH_OBSERVED','OBS_PATH_BOUNDED')]},
        safety_violations=sum(r['safety_violation'] for r in safety),
        maximum_tail_prediction_duration=max(r['max_tail_prediction_duration'] for r in safety),
        maximum_consecutive_predicted_duration=max(r['max_consecutive_predicted_duration'] for r in safety),
        heavy_tail={m:{k:sum(r[k] for r in per if r['method']==m) for k in ['error_gt_2m_count','error_gt_5m_count','error_gt_10m_count']} for m in MODELS},
        maximum_error={m:max(r['max_3d_error'] for r in per if r['method']==m and r['max_3d_error'] is not None) for m in MODELS},
        scientific_status='Fixed validation shared recovery comparison; not independent heldout evidence',no_GT_inference=True)
    for file,records in [('per_sequence_metrics.csv',per),('coverage_recovery_comparison.csv',coverage),('paired_selector_comparison.csv',pairs),
        ('bounded_recovery_safety.csv',safety),('per_sequence_recovery_statistics.csv',stats)]:write(out/file,records)
    assert (out/'fixed_recovery_config.json').read_bytes()==frozen_text
    assert all(sha(Path(p))==h for p,h in hashes.items())
    summary['selector_and_frozen_hashes_unchanged']=True
    (out/'overall_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'BOUNDED_RECOVERY_VALIDATION_REPORT.md').write_text('# Fixed shared bounded recovery validation\n\n'
        'Control uses original select_track measurement updates only. Both selectors call the same existing bounded API via the same frozen config object.\n\n'
        +json.dumps(summary,indent=2)+'\n\nState counts/safety use saved tracker query timestamps; scoring uses nearest finite prediction within 50ms. No GT clock or GT XYZ is used for recovery. No AR/spline or tracker rerun.\n')
    print(json.dumps(summary,indent=2))


def summarize_saved():
    """Finish a serialization-only failure without regenerating any prediction."""
    base=ROOT/'outputs/mmuav_paper_reproduction';out=base/'bounded_recovery_validation_fixed'
    if (out/'overall_summary.json').exists():raise FileExistsError('Completed summary already exists; refusing overwrite')
    config=json.loads((out/'fixed_recovery_config.json').read_text())
    assert config['observation_path']==asdict(PathConfig())
    assert config['internal_gap_limit']==1. and config['tail_prediction_limit']==.1
    split=json.loads((base/'splits/splits.json').read_text())
    from build_mmuav_cluster_dataset import load_gt
    all_errors={m:[] for m in MODELS};paired_a=[];paired_b=[];total=0
    for seq in split['validation_sub']:
        t,g=load_gt(Path(split['source_root'])/seq);total+=len(t);mapped={}
        for m in MODELS:
            records=read(out/seq/(m+'.csv'))
            finite=[r for r in records if np.isfinite([float(r[k]) for k in 'xyz']).all()]
            mapped[m]=map_prediction(finite,t);all_errors[m].append(mapped[m]-g)
        _,ea,eb=paired_errors(mapped['CONTROL_BOUNDED'],mapped['OBS_PATH_BOUNDED'],g);paired_a.append(ea);paired_b.append(eb)
    overall={m:metric(np.concatenate(e),total) for m,e in all_errors.items()}
    pa=metric(np.concatenate(paired_a),sum(map(len,paired_a)));pb=metric(np.concatenate(paired_b),sum(map(len,paired_b)))
    safety=read(out/'bounded_recovery_safety.csv');per=read(out/'per_sequence_metrics.csv')
    summary=dict(successful_sequences=len(split['validation_sub']),config=config,overall=overall,
        paired=dict(common_count=pa['matched_timestamp_count'],CONTROL_BOUNDED=pa,OBS_PATH_BOUNDED=pb,
            relative_changes={k:relative(pa[k],pb[k]) for k in ['MSE_coord','mean_3d_error','median_3d_error']}),
        recovery_gain_pp={b:100*(overall[b]['coverage']-overall[a]['coverage']) for a,b in [('CONTROL_OBSERVED','CONTROL_BOUNDED'),('OBS_PATH_OBSERVED','OBS_PATH_BOUNDED')]},
        safety_violations=sum(r['safety_violation']=='True' for r in safety),
        maximum_tail_prediction_duration=max(float(r['max_tail_prediction_duration']) for r in safety),
        maximum_consecutive_predicted_duration=max(float(r['max_consecutive_predicted_duration']) for r in safety),
        heavy_tail={m:{k:sum(int(r[k]) for r in per if r['method']==m) for k in ['error_gt_2m_count','error_gt_5m_count','error_gt_10m_count']} for m in MODELS},
        maximum_error={m:max(float(r['max_3d_error']) for r in per if r['method']==m and r['max_3d_error']) for m in MODELS},
        scientific_status='Fixed validation shared recovery comparison; not independent heldout evidence',
        no_GT_inference=True,selector_and_frozen_hashes_unchanged=True)
    (out/'overall_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'BOUNDED_RECOVERY_VALIDATION_REPORT.md').write_text('# Fixed shared bounded recovery validation\n\n'+json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    if '--summarize-saved' in sys.argv:summarize_saved()
    else:main()
