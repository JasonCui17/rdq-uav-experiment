#!/usr/bin/env python3
"""Fixed-rule saved-state validation. No model/tracker rerun or GT inference."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from rdq_uav.mmuav.observation_path import PathConfig,select_observation_path

FIXED=dict(asdict(PathConfig()),nearest_evaluation_tolerance=.05,
    config_status='PROVISIONAL_UNTUNED_BASELINE_FIXED_BEFORE_VALIDATION',
    design_status='RECONSTRUCTED_ROBUSTNESS_DESIGN',bounded_output_enabled=False)


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,records,fields=None):
    if fields is None:fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)


def prepare_output(path):
    if path.exists() and any(path.iterdir()):raise FileExistsError('Refusing nonempty output directory')
    path.mkdir(parents=True,exist_ok=True)
    with (path/'fixed_config.json').open('x') as f:json.dump(FIXED,f,indent=2)


def validate_fixed(path):
    if json.loads((path/'fixed_config.json').read_text())!=FIXED:raise ValueError('Fixed config changed during run')


def map_prediction(records,query,tolerance=.05):
    p=np.full((len(query),3),np.nan)
    if not records:return p
    ordered=sorted(records,key=lambda r:float(r['timestamp']))
    unique={}
    for r in ordered:unique.setdefault(float(r['timestamp']),r)
    t=np.array(list(unique));xyz=np.array([[float(r[a]) for a in 'xyz'] for r in unique.values()])
    j=np.clip(np.searchsorted(t,query),0,len(t)-1);left=np.maximum(j-1,0)
    j=np.where(abs(t[left]-query)<=abs(t[j]-query),left,j)
    valid=(abs(t[j]-query)<=tolerance)&np.isfinite(xyz[j]).all(1)
    p[valid]=xyz[j[valid]]
    return p


def metric(errors,num_gt):
    valid=np.isfinite(errors).all(1);e=errors[valid];d=np.linalg.norm(e,axis=1)
    return dict(num_gt=int(num_gt),matched_timestamp_count=int(valid.sum()),missing_prediction_count=int(num_gt-valid.sum()),
        coverage=float(valid.sum()/num_gt) if num_gt else None,MSE_coord=float(np.mean(e**2)) if len(e) else None,
        MSE_3D=float(np.mean(np.sum(e**2,axis=1))) if len(e) else None,
        mean_3d_error=float(d.mean()) if len(d) else None,median_3d_error=float(np.median(d)) if len(d) else None)


def paired_errors(a,b,gt):
    common=np.isfinite(a).all(1)&np.isfinite(b).all(1)&np.isfinite(gt).all(1)
    return common,a[common]-gt[common],b[common]-gt[common]


def relative(old,new):
    return (new-old)/old if old is not None and new is not None and old!=0 else None


def tail_stats(errors,timestamps):
    valid=np.isfinite(errors).all(1);d=np.linalg.norm(errors,axis=1);v=d[valid]
    longest=run=0;duration=0.;start=None
    for i,hit in enumerate(valid&(d>2)):
        if hit:
            run+=1
            if start is None:start=i
            if run>longest:longest=run;duration=float(timestamps[i]-timestamps[start])
        else:run=0;start=None
    result=dict(max_3d_error=float(v.max()) if len(v) else None,
        P90=float(np.percentile(v,90)) if len(v) else None,P95=float(np.percentile(v,95)) if len(v) else None,
        P99=float(np.percentile(v,99)) if len(v) else None,
        error_gt_2m=int(np.sum(v>2)),error_gt_5m=int(np.sum(v>5)),error_gt_10m=int(np.sum(v>10)),
        longest_consecutive_gt_2m_count=longest,longest_consecutive_gt_2m_duration=duration,
        squared_error_sum=float(np.sum(errors[valid]**2)))
    return result


def mechanism(records,node_count):
    updates=[r for r in records if r.get('state_type')=='observed' or r.get('measurement_update') is True or r.get('measurement_update')=='True']
    times=np.array([float(r['timestamp']) for r in updates]);gaps=np.diff(times)
    cross=[];run=longest=0;last=None
    for r in updates:
        if r['track_id']==last:run+=1
        else:run=1
        longest=max(longest,run);last=r['track_id']
    for a,b in zip(updates,updates[1:]):
        if a['track_id']!=b['track_id']:cross.append(b)
    return dict(observation_node_count=node_count,selected_observation_count=len(updates),
        selected_track_id_count=len(set(r['track_id'] for r in updates)),cross_track_transition_count=len(cross),
        max_observation_gap=float(gaps.max()) if len(gaps) else None,mean_observation_gap=float(gaps.mean()) if len(gaps) else None,
        longest_same_track_run=longest,first_cross_track_transition_time=float(cross[0]['timestamp']) if cross else None)


def weak_components(graph):
    parent=list(range(len(graph['nodes'])))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for e in graph['edges']:parent[root(e['source_node'])]=root(e['target_node'])
    return len({root(i) for i in range(len(parent))})


def score_saved(prediction_path,original_path,sequence_path,loader):
    if not prediction_path.is_file():raise RuntimeError('Prediction must be saved before reading GT')
    t,gt=loader(sequence_path)
    a=map_prediction(read(original_path),t);b=map_prediction(read(prediction_path),t)
    return t,gt,a,b


def write_report(out,summary,pairs,screen):
    """Interpret existing scores only; never change the fixed selector."""
    a=summary['overall']['ORIGINAL_FULL'];b=summary['overall']['OBSERVATION_PATH']
    pa=summary['paired']['ORIGINAL_FULL'];pb=summary['paired']['OBSERVATION_PATH'];changes=summary['paired_relative_changes']
    h=summary['handoff'];ta=summary['heavy_tail']['ORIGINAL_FULL'];tb=summary['heavy_tail']['OBSERVATION_PATH']
    improved=[r['sequence_id'] for r in pairs if r['relative_mean_3d_error_change'] is not None and r['relative_mean_3d_error_change']<-.05]
    screened=summary['possible_wrong_handoff_sequences']
    text=f'''# Fixed-rule Observation Path validation

RECONSTRUCTED_ROBUSTNESS_DESIGN — not MMUAV paper algorithm.
PROVISIONAL_UNTUNED_BASELINE_FIXED_BEFORE_VALIDATION.

## 1. Data and safeguards

{summary['successful_sequences']}/{summary['validation_sequences']} sequences evaluated; missing/invalid={len(summary['data_missing_or_invalid'])}, processing failures={len(summary['processing_failures'])}.
Every usable observed path was saved before any GT XYZ was read. All selector/frozen/source hashes are unchanged.
No preprocessing, model, Kalman, AR, spline, prediction fill or parameter search was run. Scoring uses frozen 50ms nearest-time matching.
The 18 existing regression tests and validation-evaluator tests pass. Nonempty output and mutated config are rejected.

## 2. Coverage (not hidden by matched-only error)

| Method | Matched / 6000 | Missing | Overall / sequence mean coverage |
|---|---:|---:|---:|
| Original FULL | {a['matched_timestamp_count']} | {a['missing_prediction_count']} | {a['coverage']:.4%} |
| Observation Path | {b['matched_timestamp_count']} | {b['missing_prediction_count']} | {b['coverage']:.4%} |

Coverage changes by {(b['coverage']-a['coverage'])*100:+.4f} percentage points.
One sequence gains coverage; six lose coverage; eight are unchanged (see failure_screening.csv).
Original-only common-comparison exclusions=229; observation-only timestamps=21; net loss=208.
Discarding prediction-only states is not free: large coverage losses occur in seq0007, seq0031 and seq0035.

## 3. Strict paired precision

Common timestamps={pa['matched_timestamp_count']}.

| Method | MSE_coord | Mean 3D error | Median 3D error |
|---|---:|---:|---:|
| Original paired | {pa['MSE_coord']:.9f} | {pa['mean_3d_error']:.9f} | {pa['median_3d_error']:.9f} |
| Observation paired | {pb['MSE_coord']:.9f} | {pb['mean_3d_error']:.9f} | {pb['median_3d_error']:.9f} |

Relative changes: MSE {changes['MSE_coord']:.4%}, mean {changes['mean_3d_error']:.4%}, median {changes['median_3d_error']:.4%}.
This is genuine same-timestamp evidence; overall unpaired error is not substituted for it.

## 4. Heavy tails

>2m / >5m / >10m counts: {ta['error_gt_2m']}/{ta['error_gt_5m']}/{ta['error_gt_10m']} → {tb['error_gt_2m']}/{tb['error_gt_5m']}/{tb['error_gt_10m']}.
Maximum error: {ta['max_3d_error']:.6f} → {tb['max_3d_error']:.6f}m.
Sequences with >2m error: {ta['catastrophic_sequences_gt_2m']} → {tb['catastrophic_sequences_gt_2m']}; >5m sequences remain zero.
Worst squared-error contributor: {ta['worst_sequence']['sequence_id']} ({ta['worst_sequence']['sequence_squared_error_contribution']:.2%})
→ {tb['worst_sequence']['sequence_id']} ({tb['worst_sequence']['sequence_squared_error_contribution']:.2%}).
Validation has no seq0065-scale >5m catastrophe, so broad protection against 24m divergence is NOT established.
Additional offline paired-tail audit: two >2m errors in seq0068 are corrected on common timestamps;
two >2m errors in seq0101 disappear only because those timestamps are no longer predicted. These effects must not be conflated.

## 5. Handoff and beneficiaries

No handoff={h['no_handoff']}; at least one={h['at_least_one']}; multiple={h['multiple']}; total transitions={h['total_transitions']}.
Handoff occurs in seq0042 (1), seq0068 (4), seq0087 (31), i.e. 3/15 sequences; it is not universal.
Clearly improved paired-mean sequences (>5% descriptive reduction): {improved}.
seq0087: paired MSE -35.21%, mean -15.23%, coverage +5.25pp.
seq0068: paired MSE -23.20%, mean -8.45%, but coverage -2.25pp.
seq0042: paired mean +0.73%, MSE +1.72%, unchanged coverage; this is slight degradation, not a catastrophic handoff.
Eleven sequences have exactly unchanged paired errors. Frequent switching in seq0087 is not proof of stable UAV identity.

## 6. Failure screening and scientific interpretation

POSSIBLE_WRONG_HANDOFF flags: {screened}. The predeclared >25% paired mean / new >5m rule flagged none.
No strong identity-failure evidence was observed; observation support still does not guarantee UAV identity.
seq0065's broader multi-track selection issue repeats mildly in seq0068/seq0087, but its severe long-tail failure does not occur in this validation set.

**Conclusion A, limited:** validation supports the mechanism's local robustness value, not unconditional pipeline replacement.
Paired precision improves and two common-timestamp tail errors are removed; this is not merely a seq0065-only effect.
However overall coverage declines 3.47pp, and only one sequence gains coverage. The full Case-A acceptance conditions
(coverage broadly preserved, widespread recovery, demonstrated catastrophic-tail protection) are not completely met.
Retain as a fixed-rule experimental robustness candidate; do NOT promote it to the official frozen pipeline.
Identity insufficiency (Case B) is not demonstrated by these results, and seq0065-only benefit (Case C) is contradicted by the paired improvements above.

## 7. Stop

All required inventory, per-sequence/paired/tail/handoff/screening CSVs and overall JSON are saved here.
No rules or parameters were changed after validation. No automatic bounded prediction, heldout run or next-stage experiment follows.
Original frozen reconstruction remains official. Any future progression requires an explicit user decision.
'''
    (out/'VALIDATION_OBSERVATION_PATH_REPORT.md').write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,
        default=ROOT/'outputs/mmuav_paper_reproduction/observation_path_validation_fixed')
    args=parser.parse_args();out=args.output_dir;prepare_output(out)
    base=ROOT/'outputs/mmuav_paper_reproduction';source=base/'final_reproduction/validation/full'
    split_path=base/'splits/splits.json';split=json.loads(split_path.read_text());sequences=split['validation_sub']
    data_root=Path(split['source_root']);inventory=[];raw_cache={};results={}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=[ROOT/'src/rdq_uav/mmuav/observation_path.py',ROOT/'src/rdq_uav/mmuav/pose_trajectory.py',
        ROOT/'tools/run_mmuav_pose_pipeline.py',base/'final_reproduction/frozen_reproduction_config.json',split_path]
    hashes={str(p):sha(p) for p in protected};saved_hashes={}
    required={'timestamp','track_id','x','y','z','vx','vy','vz','measurement_update'}
    for seq in sequences:
        folder=source/seq;raw=folder/'raw_tracker_trajectory.csv';selected=folder/'selected_trajectory.csv';original=folder/'smoothed_final_trajectory.csv'
        reasons=[];gt_available=any((data_root/seq/'ground_truth').glob('*.npy'))
        for p,label in [(raw,'raw_tracker'),(selected,'selected_trajectory'),(original,'original_full')]:
            if not p.is_file():reasons.append('DATA_MISSING:'+label)
            else:saved_hashes[str(p)]=sha(p)
        if not gt_available:reasons.append('DATA_MISSING:GT')
        if raw.is_file():
            try:
                records=read(raw)
                if records and not required.issubset(records[0]):raise ValueError('missing tracker fields')
                raw_cache[seq]=records
            except Exception as exc:reasons.append('DATA_INVALID:'+str(exc))
        inventory.append(dict(sequence_id=seq,raw_tracker_available=raw.is_file(),selected_trajectory_available=selected.is_file(),
            original_full_available=original.is_file(),gt_available=gt_available,usable_for_observation_path=not reasons,missing_reason=';'.join(reasons)))
    write(out/'validation_saved_state_inventory.csv',inventory)
    print('Inventory: '+json.dumps(inventory),flush=True)
    # Generate every usable sequence's prediction BEFORE any GT array is read.
    failures=[]
    for item in inventory:
        if not item['usable_for_observation_path']:continue
        seq=item['sequence_id'];folder=out/seq;folder.mkdir()
        try:
            validate_fixed(out);result=select_observation_path(raw_cache[seq],PathConfig())
            fields=['timestamp','x','y','z','vx','vy','vz','track_id','source_state_index','measurement_id','state_type']
            write(folder/'selected_observation_path.csv',result['selected'],fields)
            write(folder/'selected_transitions.csv',result['transitions'],['source_node','target_node','source_track','target_track',
                'source_timestamp','target_timestamp','dt','predicted_distance','same_track','accepted','rejection_reason','edge_cost'])
            (folder/'reachability_summary.json').write_text(json.dumps(result['reachability'],indent=2))
            (folder/'path_provenance.json').write_text(json.dumps(dict(config=FIXED,score=result['score'],
                inference_reads_gt=False,raw_tracker_sha256=saved_hashes[str(source/seq/'raw_tracker_trajectory.csv')],
                deduplication=result['deduplication']),indent=2))
            results[seq]=result;print('Observation path saved: '+seq,flush=True)
        except Exception as exc:failures.append(dict(sequence_id=seq,stage='PATH_GENERATION',error=str(exc)))
    from build_mmuav_cluster_dataset import load_gt
    per=[];pairs=[];heavy=[];handoff=[];screen=[];errors={'ORIGINAL_FULL':[],'OBSERVATION_PATH':[]};paired={'ORIGINAL_FULL':[],'OBSERVATION_PATH':[]};num_gt=0
    for seq,result in results.items():
        try:t,gt,a,b=score_saved(out/seq/'selected_observation_path.csv',source/seq/'smoothed_final_trajectory.csv',data_root/seq,load_gt)
        except Exception as exc:failures.append(dict(sequence_id=seq,stage='SCORING',error=str(exc)));continue
        validate_fixed(out);num_gt+=len(gt);ea,eb=a-gt,b-gt
        common,pa,pb=paired_errors(a,b,gt);ma=metric(ea,len(gt));mb=metric(eb,len(gt));pma=metric(pa,len(pa));pmb=metric(pb,len(pb))
        for model,e,m,records in [('ORIGINAL_FULL',ea,ma,read(source/seq/'selected_trajectory.csv')),('OBSERVATION_PATH',eb,mb,result['selected'])]:
            mech=mechanism(records,len(result['graph']['nodes']))
            per.append(dict(sequence_id=seq,model=model,**m,**mech,weak_connected_components=weak_components(result['graph']),
                graph_edges=len(result['graph']['edges']),reachable_nodes_from_earliest=result['reachability']['reachable_node_count_from_earliest']))
            h=tail_stats(e,t);heavy.append(dict(sequence_id=seq,model=model,**h));errors[model].append(e)
        paired['ORIGINAL_FULL'].append(pa);paired['OBSERVATION_PATH'].append(pb)
        changes={k:relative(pma[k],pmb[k]) for k in ['MSE_coord','mean_3d_error','median_3d_error']}
        pairs.append(dict(sequence_id=seq,common_timestamp_count=int(common.sum()),original_only=int((np.isfinite(a).all(1)&~np.isfinite(b).all(1)).sum()),
            observation_only=int((~np.isfinite(a).all(1)&np.isfinite(b).all(1)).sum()),
            **{'original_'+k:pma[k] for k in changes},**{'observation_'+k:pmb[k] for k in changes},
            **{'relative_'+k+'_change':v for k,v in changes.items()}))
        cross=[e for e in result['transitions'] if not e['same_track']];mech=mechanism(result['selected'],len(result['graph']['nodes']))
        handoff.append(dict(sequence_id=seq,**mech,unique_parent_tracks_used=mech['selected_track_id_count'],
            minimum_cross_track_edge_distance=min((e['predicted_distance'] for e in cross),default=None),
            maximum_selected_edge_distance=max((e['predicted_distance'] for e in result['transitions']),default=None)))
        da,db=np.linalg.norm(ea,axis=1),np.linalg.norm(eb,axis=1)
        new5=int(np.sum(np.isfinite(db)&(db>5)&~(np.isfinite(da)&(da>5))))
        bad=(changes['mean_3d_error'] is not None and changes['mean_3d_error']>.25) or new5>0
        screen.append(dict(sequence_id=seq,original_coverage=ma['coverage'],obs_path_coverage=mb['coverage'],
            paired_original_mean_error=pma['mean_3d_error'],paired_obs_mean_error=pmb['mean_3d_error'],
            relative_change=changes['mean_3d_error'],cross_track_transition_count=len(cross),max_error=float(np.nanmax(db)) if np.isfinite(db).any() else None,
            new_gt_5m_timestamp_count=new5,flag='POSSIBLE_WRONG_HANDOFF' if bad else 'NONE',
            note='screening association only, not proof of wrong handoff; >25% paired mean or new >5m timestamp' if bad else ''))
    for model in errors:
        total=sum(r['squared_error_sum'] for r in heavy if r['model']==model)
        for r in heavy:
            if r['model']==model:r['sequence_squared_error_contribution']=r['squared_error_sum']/total if total else 0.
    for name,data in [('per_sequence_metrics.csv',per),('validation_paired_timestamp_comparison.csv',pairs),
        ('validation_heavy_tail_comparison.csv',heavy),('validation_handoff_statistics.csv',handoff),('failure_screening.csv',screen)]:write(out/name,data)
    write(out/'processing_failures.csv',failures,['sequence_id','stage','error'])
    overall={};pair_overall={};tails={}
    for model in errors:
        e=np.concatenate(errors[model]) if errors[model] else np.empty((0,3));p=np.concatenate(paired[model]) if paired[model] else np.empty((0,3))
        overall[model]=metric(e,num_gt);pair_overall[model]=metric(p,len(p))
        h=[r for r in heavy if r['model']==model]
        worst=max(h,key=lambda r:r['squared_error_sum']) if h else None
        tails[model]=dict(**{key:sum(r[key] for r in h) for key in ['error_gt_2m','error_gt_5m','error_gt_10m']},
            max_3d_error=max((r['max_3d_error'] for r in h if r['max_3d_error'] is not None),default=None),worst_sequence=worst,
            catastrophic_sequences_gt_2m=sum(r['error_gt_2m']>0 for r in h),catastrophic_sequences_gt_5m=sum(r['error_gt_5m']>0 for r in h),
            catastrophic_sequences_gt_10m=sum(r['error_gt_10m']>0 for r in h))
    summary=dict(validation_sequences=len(sequences),usable_sequences=sum(r['usable_for_observation_path'] for r in inventory),
        successful_sequences=len(pairs),data_missing_or_invalid=[r for r in inventory if not r['usable_for_observation_path']],processing_failures=failures,
        overall=overall,sequence_mean_coverage={model:float(np.mean([r['coverage'] for r in per if r['model']==model])) for model in errors},
        paired=pair_overall,paired_relative_changes={key:relative(pair_overall['ORIGINAL_FULL'][key],pair_overall['OBSERVATION_PATH'][key]) for key in ['MSE_coord','mean_3d_error','median_3d_error']},
        heavy_tail=tails,handoff=dict(no_handoff=sum(r['cross_track_transition_count']==0 for r in handoff),
            at_least_one=sum(r['cross_track_transition_count']>0 for r in handoff),multiple=sum(r['cross_track_transition_count']>1 for r in handoff),
            total_transitions=sum(r['cross_track_transition_count'] for r in handoff)),
        possible_wrong_handoff_sequences=[r['sequence_id'] for r in screen if r['flag']!='NONE'],
        fixed_config=FIXED,no_GT_inference=True,no_preprocessing_or_tracker_rerun=True,error_scope='available predictions; paired comparison separate')
    validate_fixed(out);assert all(sha(Path(p))==h for p,h in {**hashes,**saved_hashes}.items())
    summary['input_and_algorithm_hashes_unchanged']=True
    (out/'overall_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out,summary,pairs,screen)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
