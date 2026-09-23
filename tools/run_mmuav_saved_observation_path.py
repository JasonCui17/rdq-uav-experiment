#!/usr/bin/env python3
"""Saved-state mechanism demonstration. No GT before selected path is saved."""
import argparse,csv,json,sys,hashlib
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from rdq_uav.mmuav.observation_path import (PathConfig,extract_observation_nodes,
    deduplicate_observation_nodes,build_observation_dag,audit_reachability,solve_best_observation_path)


def write(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--lambda-d',type=float,default=.1);parser.add_argument('--lambda-g',type=float,default=.1)
    parser.add_argument('--node-reward',type=float,default=1.)
    args=parser.parse_args();config=PathConfig(node_reward=args.node_reward,lambda_d=args.lambda_d,lambda_g=args.lambda_g)
    base=ROOT/'outputs/mmuav_paper_reproduction';source=base/'final_reproduction/heldout/full/seq0065'
    out=base/'posthoc_robustness/seq0065_observation_path';out.mkdir(parents=True,exist_ok=True)
    if (out/'selected_observation_path.csv').exists():raise FileExistsError('Refusing case-study overwrite')
    raw_path=source/'raw_tracker_trajectory.csv';digest=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw=read(raw_path);nodes,dedup=deduplicate_observation_nodes(extract_observation_nodes(raw))
    graph=build_observation_dag(nodes,config);reach=audit_reachability(graph)
    reach.update(deduplication=dedup,rejection_counts=graph['rejection_counts'],posterior_state_not_raw_measurement=True,
        status='LEGAL_CROSS_TRACK_HANDOFF_EXISTS' if reach['has_cross_track_path_from_earliest'] else 'NO_REACHABLE_HANDOFF_FROM_EARLIEST')
    fields=['source_node','target_node','source_track','target_track','source_timestamp','target_timestamp','dt','predicted_distance','same_track','accepted','rejection_reason','edge_cost']
    write(out/'observation_path_reachability.csv',graph['edge_audit'],fields)
    (out/'reachability_summary.json').write_text(json.dumps(reach,indent=2)+'\n')
    (out/'path_config.json').write_text(json.dumps(dict(asdict(config),parameter_status='PROVISIONAL_UNTUNED_BASELINE',
        design_status='RECONSTRUCTED_ROBUSTNESS_DESIGN',bounded_output_enabled=False,raw_tracker_sha256=digest),indent=2)+'\n')
    node_rows=[dict(node_id=i,**n.row()) for i,n in enumerate(nodes)]
    row_fields=['timestamp','x','y','z','vx','vy','vz','track_id','source_state_index','measurement_id','state_type']
    write(out/'observation_nodes.csv',node_rows,['node_id']+row_fields)
    if args.audit_only:print(json.dumps(reach,indent=2));return
    result=solve_best_observation_path(graph,config)
    write(out/'selected_observation_path.csv',result['selected'],row_fields)
    write(out/'selected_transitions.csv',result['transitions'],fields)
    sources=set(result['selected_node_ids']);chosen={(e['source_node'],e['target_node']) for e in result['transitions']}
    alternatives=[dict(e,on_selected_path=(e['source_node'],e['target_node']) in chosen) for e in graph['edges'] if e['source_node'] in sources]
    write(out/'alternative_edge_audit.csv',alternatives,fields+['on_selected_path'])
    # GT-only evaluation begins AFTER observed path and all selection outputs exist.
    from build_mmuav_cluster_dataset import load_gt
    from run_mmuav_saved_track_v2 import evaluate
    gt_t,gt=load_gt(Path('/home/jasoncui/datasets/MMAUD/official/train/seq0065'))
    models={'ORIGINAL':evaluate(read(source/'smoothed_final_trajectory.csv'),gt_t,gt)}
    for label,folder,file in [('V1','seq0065','robustness_v1_selected_trajectory.csv'),('V2','seq0065_v2','robust_selected_trajectory.csv')]:
        p=base/'posthoc_robustness'/folder/file
        if p.exists():models[label]=evaluate(read(p),gt_t,gt)
    models['OBSERVATION_PATH']=evaluate(result['selected'],gt_t,gt)
    selected=result['selected'];transitions=result['transitions'];cross=[e for e in transitions if not e['same_track']]
    summary=dict(models=models,selected_observation_count=len(selected),cross_track_transitions=len(cross),
        score=result['score'],cumulative_motion_cost=result['cumulative_motion_cost'],selected_node_ids=result['selected_node_ids'],
        maximum_selected_gap=max((e['dt'] for e in transitions),default=None),
        selected_first_timestamp=selected[0]['timestamp'] if selected else None,
        selected_last_timestamp=selected[-1]['timestamp'] if selected else None,
        scientific_status='POST-HOC mechanism demonstration; NOT independent heldout improvement',
        different_coverage_sets_not_paired_precision=True,prediction_only_nodes=0,bounded_output_enabled=False)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# Observation-supported state path selection

RECONSTRUCTED_ROBUSTNESS_DESIGN; PROVISIONAL_UNTUNED_BASELINE.

## Reachability first

{json.dumps(reach,indent=2)}

The audit ran before DP selection. It uses posterior Kalman update states, not purported raw detections.
Nodes from singleton parent tracks are retained. Exact measurement deduplication is unavailable in these CSVs;
same timestamp + XYZ max absolute tolerance 1e-8 is deterministic approximate dedup only.
The reachability CSV lists evaluated pairs within the time window; >1s and non-increasing-time rejection counts are summarized.
Earliest-node reachability is a graph fact, not a GT-based claim about target identity.

## Frozen mechanism baseline

Node reward=1, lambda_d=0.1, lambda_g=0.1, d0=3m, tau0=1s, max_dt=1s, max_distance=3m.
These transparent provisional costs were specified before case evaluation, with no search or revision using seq0065 metrics.
Every legal edge is retained. DP maximizes sum(node reward)-sum(edge cost), with deterministic
count/span/cost/node-order ties. Track IDs are provenance, not an identity constraint.
Prediction-only nodes never enter the graph; optional bounded-output interface is tested but not enabled.

## Saved-data mechanism result

{json.dumps({k:v for k,v in summary.items() if k!='selected_node_ids'},indent=2)}

Different models have different coverage sets; lower available-prediction error alone is not a paired precision gain.
Strictly increasing timestamps prohibit duplicate same-time scoring. Full audit and alternatives remain saved.
All original frozen and V1/V2 artifacts remain unchanged. No model, DBSCAN, Kalman, AR or spline was run.
The path is observation-supported but is not guaranteed to retain UAV identity: clutter can also receive reward.
Short gaps/missing endpoints remain; no GT-based correction or bounded prediction was used to hide them.

## Scientific status

Seq0065 participated in design; this is post-hoc reachability and representation evidence, NOT generalization.
Fixed-rule validation_sub saved-state testing may be warranted if lawful early-to-late handoff exists,
but no formal validation or aggregate heldout rescoring is performed here.
'''
    (out/'observation_path_analysis.md').write_text(text+'\n')
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==digest
    print(json.dumps({k:v for k,v in summary.items() if k!='selected_node_ids'},indent=2))


if __name__=='__main__':main()
