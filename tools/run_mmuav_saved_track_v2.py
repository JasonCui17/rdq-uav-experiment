#!/usr/bin/env python3
"""Saved tracker-only V2 inference, followed by isolated post-hoc GT evaluation."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from rdq_uav.mmuav.track_robustness import stitch_tracks,FIELDS


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,data,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)


def evaluate(data,gt_t,gt):
    from document_mmuav_archive import nearest
    times=np.array([float(r['timestamp']) for r in data]);p=np.array([[float(r[a]) for a in 'xyz'] for r in data]).reshape(-1,3)
    gap,j=nearest(gt_t,times)
    valid=gap<=.05
    if len(times):valid &= np.isfinite(p[j]).all(1)
    if not valid.any():return dict(coverage=0.,matched=0,MSE_coord=None,mean_3d_error=None,median_3d_error=None)
    e=p[j[valid]]-gt[valid];d=np.linalg.norm(e,axis=1)
    return dict(coverage=float(valid.mean()),matched=int(valid.sum()),MSE_coord=float(np.mean(e**2)),
        mean_3d_error=float(d.mean()),median_3d_error=float(np.median(d)))


def main():
    base=ROOT/'outputs/mmuav_paper_reproduction'
    source=base/'final_reproduction/heldout/full/seq0065'
    output=base/'posthoc_robustness/seq0065_v2'
    if output.exists() and any(output.iterdir()):raise FileExistsError('Refusing to overwrite existing V2 case study')
    output.mkdir(parents=True,exist_ok=True)
    frozen=base/'final_reproduction/frozen_reproduction_config.json'
    source_files=list(source.glob('*.csv'))+[frozen]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in source_files}
    tracker_path=source/'raw_tracker_trajectory.csv'
    raw=read(tracker_path)
    result=stitch_tracks(raw)
    write(output/'track_statistics.csv',result['track_statistics'],list(result['track_statistics'][0]))
    fields=['source_track','target_track','source_fragment','target_fragment','dt','predicted_distance','accepted','reason','in_selected_chain']
    write(output/'stitching_decisions.csv',result['stitching_decisions'],fields)
    write(output/'robust_selected_trajectory.csv',result['selected'],FIELDS)
    chains=[dict(fragment_ids=c['fragment_ids'],track_ids=c['track_ids'],score=c['score']) for c in result['chains']]
    (output/'selection_provenance.json').write_text(json.dumps(dict(design_status=result['design_status'],raw_tracker_sha256=sha(tracker_path),
        inference_reads_gt=False,maximum_gap_s=1.,spatial_gate_m=3.,min_track_measurements=2,
        fragment_definition='maximal consecutive update runs; split on prediction or >1s gap',
        edge_order='distance ascending, target support count/duration descending, dt then fragment IDs',
        selected_fragments=result['selected_fragments'],candidate_chains=chains),indent=2)+'\n')
    # GT is imported and read ONLY after V2 predictions and selection provenance exist.
    from build_mmuav_cluster_dataset import load_gt
    gt_t,gt=load_gt(Path('/home/jasoncui/datasets/MMAUD/official/train/seq0065'))
    original=read(source/'smoothed_final_trajectory.csv')
    v1_path=base/'posthoc_robustness/seq0065/robustness_v1_selected_trajectory.csv'
    models={'ORIGINAL':evaluate(original,gt_t,gt)}
    if v1_path.exists():models['ROBUSTNESS_V1']=evaluate(read(v1_path),gt_t,gt)
    models['ROBUSTNESS_V2']=evaluate(result['selected'],gt_t,gt)
    comparison=dict(scientific_status='POST-HOC CASE STUDY; not independent heldout evaluation',
        original_frozen_results_preserved=True,evaluation_timestamp_tolerance_s=.05,errors_scope='available matched predictions; different coverage sets',
        no_AR_interpolation_spline=True,models=models)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    accepted=[d for d in result['stitching_decisions'] if d['in_selected_chain']]
    tracks=list(dict.fromkeys(r['track_id'] for r in result['selected']))
    report=f'''# seq0065 V2 saved-data post-hoc case study

RECONSTRUCTED_ROBUSTNESS_DESIGN — not MMUAV paper algorithm.

## Mechanism and change

Original select_track() prefers raw lifespan; a 5.501s prediction-only tail made the old divergent track win.
Corrected candidates and newer supported tracks persist. Exact association gate/competition cause remains unproven (hypothesis/detection IDs absent).

V1 selected a seed before stitching and preferred future support count over endpoint distance.
V2 first extracts supported runs (break at prediction-only state or >1s gap), then forms disjoint candidate chains.
Original tracks require >=2 updates; single-state runs can be linked, and final chains require >=2 states.
Edges require strict 0<dt<=1s and ||pA+vA*dt-pB||<=3m. Eligible edges are processed by ascending prediction distance,
then target support count/duration, dt and fragment IDs. Each fragment has at most one predecessor/successor.
This is deterministic greedy linking, not a globally optimal association solver. Overlapping fragments cannot be truncated to force a connection.
Final chain score: (measurement_count, supported_duration, -prediction_only_tail, raw_duration).
Raw duration/tail refer to parent-track support statistics; output contains only observed measurement-update states.
All extra GT/oracle columns are excluded from algorithm inputs. Original select_track and all Kalman/model/temporal parameters are unchanged.

## Stitching and result

Selected tracks: {tracks}

Accepted selected-chain links: {len(accepted)}; fragments: {len(result['selected_fragments'])}.
Cross-track selected links: {sum(d['source_track']!=d['target_track'] for d in accepted)}.
Exact links, dt, predicted distance and rejected alternatives are in stitching_decisions.csv.
No prediction-only state appears in V2 output; no duplicate timestamp or reversed time is permitted.

```json
{json.dumps(models,indent=2)}
```

V2 cannot produce a long prediction-only catastrophic tail by construction. It can still stitch a wrong observed fragment;
support count is not a UAV identity classifier. Median need not improve, and coverage sets differ, so this table is not a fully paired precision claim.
If V2 selects only old-track fragments and loses coverage, removal of the divergent tail does NOT constitute
complete recovery. Fixed non-overlapping endpoints and greedy allocation do not guarantee reconnecting later correct tracks.
No GT-dependent selection, threshold search, model rerun, AR, interpolation or spline was used.

## Scientific boundary

This is POST-HOC analysis on previously evaluated heldout seq0065.
Original frozen results remain the official reconstruction result. No heldout aggregate is recomputed.
Further validation_sub saved-track evaluation could check generalization at fixed thresholds, but is not performed in this task.
'''
    (output/'seq0065_v2_analysis.md').write_text(report)
    assert all(sha(Path(p))==h for p,h in before.items())
    print(json.dumps(dict(models=models,selected_tracks=tracks,selected_fragments=len(result['selected_fragments']),selected_links=len(accepted)),indent=2))


if __name__=='__main__':main()
