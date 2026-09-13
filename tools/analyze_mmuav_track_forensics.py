#!/usr/bin/env python3
"""Existing seq0065 CSV forensics and one fixed POST-HOC selection case study."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from build_mmuav_cluster_dataset import load_gt
from rdq_uav.mmuav.track_robustness import select_track_v2,supported
from document_mmuav_archive import nearest,series,rows
BASE=ROOT/'outputs/mmuav_paper_reproduction'
OUT=BASE/'posthoc_robustness/seq0065'
SOURCE=BASE/'final_reproduction/heldout/full/seq0065'


def write(name,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)


def metrics(p,g):
    valid=np.isfinite(p).all(1);e=p[valid]-g[valid];d=np.linalg.norm(e,axis=1)
    return dict(coverage=float(valid.mean()),matched=int(valid.sum()),MSE_coord=float(np.mean(e**2)),
        mean_3d_error=float(d.mean()),median_3d_error=float(np.median(d)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=list((BASE/'final_reproduction/heldout').rglob('*.csv'))+list((BASE/'final_reproduction/validation').rglob('*.json'))+[BASE/'final_reproduction/frozen_reproduction_config.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    gt_t,gt=load_gt(Path('/home/jasoncui/datasets/MMAUD/official/train/seq0065'));origin=gt_t[0]
    raw=rows(SOURCE/'raw_tracker_trajectory.csv');selected=rows(SOURCE/'selected_trajectory.csv')
    groups=defaultdict(list)
    for r in raw:
        for a in ['timestamp','x','y','z','vx','vy','vz']:r[a]=float(r[a])
        r['measurement_update']=supported(r);groups[r['track_id']].append(r)
    audit=[];track_stats=[]
    for tid,group in groups.items():
        last=None;run=0;longest=0;errors=[]
        for r in group:
            if supported(r):last=r['timestamp'];run=0
            else:run+=1
            longest=max(longest,run);gap,j=nearest(np.array([r['timestamp']]),gt_t)
            e=float(np.linalg.norm(np.array([r[a] for a in 'xyz'])-gt[j[0]]))
            if supported(r):errors.append(e)
            audit.append(dict(r,time_since_last_measurement=None if last is None else r['timestamp']-last,
                consecutive_prediction_only_states=run,posthoc_gt_error_3d=e))
        updates=[r for r in group if supported(r)]
        track_stats.append(dict(track_id=tid,start_time=group[0]['timestamp'],end_time=group[-1]['timestamp'],
          duration=group[-1]['timestamp']-group[0]['timestamp'],measurement_update_count=len(updates),
          measurement_update_ratio=len(updates)/len(group),longest_consecutive_prediction_only=longest,
          median_supported_gt_error=float(np.median(errors)) if errors else None))
    write('seq0065_all_tracks_audit.csv',sorted(audit,key=lambda r:(r['timestamp'],r['track_id'])))
    write('track_summary.csv',track_stats)
    ft,fp=series(SOURCE/'smoothed_final_trajectory.csv');gap,j=nearest(gt_t,ft)
    original=fp[j].copy();original[gap>.05]=np.nan;err=np.linalg.norm(original-gt,axis=1)
    diverge=None
    for i in range(len(err)-2):
        if np.all(err[i:i+3]>2):diverge=float(gt_t[i]);break
    assert diverge is not None
    tid=selected[0]['track_id'];sg=groups[tid];updates=[r for r in sg if supported(r)];last=updates[-1]['timestamp']
    after=[r for r in sg if r['timestamp']>=diverge]
    cand=rows(SOURCE/'all_candidate_measurements.csv');ct=defaultdict(list)
    for r in cand:ct[float(r['timestamp'])].append(r)
    cand_audit=[]
    for t,group in sorted(ct.items()):
        _,j=nearest(np.array([t]),gt_t);g=gt[j[0]]
        geom=np.array([[float(r['geometric_'+a]) for a in 'xyz'] for r in group]);corr=np.array([[float(r[a]) for a in 'xyz'] for r in group])
        cand_audit.append(dict(timestamp=t,candidate_count=len(group),nearest_geometric_error=float(np.linalg.norm(geom-g,axis=1).min()),
          nearest_corrected_error=float(np.linalg.norm(corr-g,axis=1).min())))
    write('candidate_timestamp_audit.csv',cand_audit)
    state_evidence=[]
    ctimes=np.array(sorted(ct))
    for r in audit:
        if r['track_id']!=tid:continue
        _,j=nearest(np.array([r['timestamp']]),ctimes);c=ct[ctimes[j[0]]]
        p=np.array([r[a] for a in 'xyz']);distance=min(np.linalg.norm(p-np.array([float(x[a]) for a in 'xyz'])) for x in c)
        state_evidence.append(dict(r,nearest_corrected_candidate_distance=float(distance),candidate_time_gap_ms=float(abs(ctimes[j[0]]-r['timestamp'])*1000)))
    write('selected_track_evidence.csv',state_evidence)
    # Establish the allowed-fix precondition using GT strictly as post-hoc evidence.
    alternatives=[r for r in track_stats if r['track_id']!=tid and r['measurement_update_count']>=3 and
        r['median_supported_gt_error']<1 and r['end_time']>diverge and r['start_time']<diverge+3]
    assert alternatives and all(not supported(r) for r in after), 'Fix precondition not established'
    v1=select_track_v2(raw);write('robustness_v1_selected_trajectory.csv',v1)
    vt=np.array([r['timestamp'] for r in v1]);vp=np.array([[r[a] for a in 'xyz'] for r in v1])
    vg,vj=nearest(gt_t,vt);prediction=vp[vj].copy();prediction[vg>.05]=np.nan
    comparison=[dict(model='ORIGINAL',**metrics(original,gt)),dict(model='ROBUSTNESS_V1',**metrics(prediction,gt))]
    write('original_vs_v1_metrics.csv',comparison)
    tt,tp=series(BASE/'final_reproduction/heldout/full_temporal/seq0065/smoothed_final_trajectory.csv')
    _,tj=nearest(gt_t,tt);temporal=tp[tj];temp_err=np.linalg.norm(temporal-gt,axis=1)
    summary=dict(first_divergence_timestamp=diverge,elapsed_s=diverge-origin,selected_track_id=tid,
        last_valid_measurement_timestamp=last,prediction_only_duration=sg[-1]['timestamp']-last,
        after_divergence_states=len(after),after_divergence_prediction_only=sum(not supported(r) for r in after),
        alternative_track_ids=[r['track_id'] for r in alternatives],classification='MULTIPLE',
        primary_failure='TRACK_SELECTION_FAILURE',secondary_failure='TRACK_ASSOCIATION_FAILURE',confidence='HIGH',
        center_evidence='CENTER REGRESSION NOT PRIMARY FAILURE',temporal_evidence='TEMPORAL MODULE INHERITS TRACKING FAILURE',
        comparison=comparison,scientific_status='POST-HOC CASE STUDY; original frozen results remain official',
        v1_design='measurement-count selection + forward supported stitching; gap<=1s; velocity endpoint gate<=3m; no temporal refit')
    (OUT/'forensics_summary.json').write_text(json.dumps(summary,indent=2))
    fig=plt.figure(figsize=(15,9));ax=fig.add_subplot(231,projection='3d');ax.plot(*gt.T,c='red',label='GT')
    for group in groups.values():p=np.array([[r[a] for a in 'xyz'] for r in group]);ax.plot(*p.T,alpha=.5,lw=.8)
    ax.set(title='A All tracks (unmodified)',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.legend()
    ax=fig.add_subplot(232);ax.plot(gt_t-origin,gt[:,1],c='red',label='GT');ax.plot(gt_t-origin,original[:,1],c='black',label='Original selected')
    ax.scatter([float(r['timestamp'])-origin for r in cand],[float(r['y']) for r in cand],s=3,c='gray',alpha=.5,label='Corrected candidates')
    for group in groups.values():ax.plot([r['timestamp']-origin for r in group],[r['y'] for r in group],lw=.7,alpha=.5)
    ax.set(title='B Time-Y: candidates and all tracks',ylabel='Y (m)');ax.legend(fontsize=7)
    ids=list(groups);ax=fig.add_subplot(233)
    for n,k in enumerate(ids):
        group=groups[k];ax.scatter([r['timestamp']-origin for r in group],[n]*len(group),s=8,c=['blue' if supported(r) else '#dddddd' for r in group])
    ax.set(title='C Blue=measurement; gray=prediction',ylabel='Track index')
    ax=fig.add_subplot(234);ax.plot([r['timestamp']-origin for r in cand_audit],[r['nearest_corrected_error'] for r in cand_audit]);ax.set(title='D Nearest corrected candidate error',ylabel='3D error (m)')
    ax=fig.add_subplot(235);ax.plot(gt_t-origin,err,label='Original');ax.plot(gt_t-origin,temp_err,label='Temporal');ax.set(title='E Selected/final error',ylabel='3D error (m)');ax.legend()
    ax=fig.add_subplot(236)
    for n,k in enumerate(ids):g=groups[k];ax.plot([r['timestamp']-origin for r in g],[n]*len(g),lw=2)
    ax.set(title='F Track ID timeline (index in track_summary)',ylabel='Track index')
    for ax in fig.axes[1:]:ax.set_xlabel('Elapsed time (s)');ax.axvline(diverge-origin,c='orange',ls='--');ax.axvspan(8,13,color='orange',alpha=.06);ax.grid(alpha=.2)
    fig.suptitle('seq0065 Tracker Forensics — POST-HOC diagnosis');fig.tight_layout();fig.savefig(OUT/'seq0065_tracker_forensics.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for a,ax in enumerate(axes.flat[:3]):
        ax.plot(gt_t-origin,gt[:,a],c='red',label='GT');ax.plot(gt_t-origin,original[:,a],c='gray',label='Original');ax.plot(gt_t-origin,prediction[:,a],c='blue',label='V1')
        ax.set(xlabel='Elapsed time (s)',ylabel='XYZ'[a]+' (m)');ax.legend()
    axes.flat[3].plot(gt_t-origin,err,c='gray',label='Original');axes.flat[3].plot(gt_t-origin,np.linalg.norm(prediction-gt,axis=1),c='blue',label='V1');axes.flat[3].set(xlabel='Elapsed time (s)',ylabel='3D error (m)');axes.flat[3].legend()
    fig.suptitle('POST-HOC CASE STUDY — not independent heldout improvement');fig.tight_layout();fig.savefig(OUT/'seq0065_original_vs_v1.png',dpi=300);plt.close(fig)
    sections=[('Symptom','旧selected轨迹在转向附近与候选云分叉，Y预测持续离开目标。'),
      ('First divergence time',f'{diverge:.6f}，相对首GT {diverge-origin:.3f}s；规则为连续3个GT评价点error>2m，仅用于post-hoc diagnosis。'),
      ('Candidate evidence','完整候选和逐timestamp最近GT误差见candidate_timestamp_audit.csv；不以GT删除任何候选。'),
      ('Center-regression evidence','CENTER REGRESSION NOT PRIMARY FAILURE；存在近GT的corrected候选和measurement-supported新track，而旧轨迹prediction-only发散。'),
      ('Tracker evidence',f'旧track最后measurement={last:.6f}；prediction-only tail={sg[-1]["timestamp"]-last:.3f}s；divergence之后{len(after)}个状态全部prediction-only。track_candidates()的CV预测无法追随该转向，而旧track仍存活。'),
      ('Track-selection evidence',f'select_track()以raw lifespan优先，选中了旧track。符合后续正确track证据的IDs：{[r["track_id"] for r in alternatives]}。GT支持“正确”的判断仅用于诊断；v1不读取这些oracle IDs。'),
      ('Temporal evidence','TEMPORAL MODULE INHERITS TRACKING FAILURE；旧FULL已严重发散，ar_complete/resample沿用其selected输入。未调整AR或spline，也没有重新拟合。'),
      ('Root cause','MULTIPLE。PRIMARY FAILURE = TRACK_SELECTION_FAILURE；secondary = TRACK_ASSOCIATION_FAILURE（旧track失去measurement）。Confidence = HIGH。新track仍跟随候选，所以不是整个tracker完全不能重建后续目标。'),
      ('Minimal robustness fix','独立track_robustness.py/select_track_v2：measurement update count优先，supported duration其次，prediction-only tail惩罚；仅measurement-supported状态；按未来端点速度预测连续性拼接，1s/3m沿用现有尺度。RECONSTRUCTED_ROBUSTNESS_DESIGN；不是论文算法。现有模型、Kalman、select_track、AR、spline都未修改。'+json.dumps(comparison)),
      ('Scientific status','This is POST-HOC analysis on previously evaluated heldout data. Original frozen results remain the official reconstruction result. V1仅seq0065一次离线case study，没有正式heldout重评分；NaN缺测保留。')]
    report='# seq0065 Root Cause Analysis\n\n'+'\n\n'.join(f'## {i+1}. {name}\n\n{body}' for i,(name,body) in enumerate(sections))+'\n'
    (ROOT/'results/mmuav_reproduction/seq0065_root_cause_analysis.md').write_text(report)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
