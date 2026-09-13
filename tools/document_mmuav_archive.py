#!/usr/bin/env python3
"""Read-only evidence interpretation/plotting. Never call DBSCAN, models or fitting."""
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_mmuav_cluster_dataset import load_gt,timestamp_files,load_xyz
BASE=ROOT/'outputs/mmuav_paper_reproduction'
HELD=BASE/'final_reproduction/heldout'
OUT=ROOT/'results/mmuav_reproduction'
VIS=OUT/'final_visuals'
COLORS={'gt':'#c62828','geometric':'#888888','full':'#1565c0','full_temporal':'#2e7d32'}
REASONS=['NO_CANDIDATE','NO_VALID_TRACK','OUTSIDE_SELECTED_TRACK','GAP_TOO_LARGE','OUTSIDE_INTERPOLATION_SUPPORT','OTHER']


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def series(path):
    r=rows(path)
    return np.array([float(x['timestamp']) for x in r]),np.array([[float(x[a]) for a in 'xyz'] for x in r]).reshape(-1,3)


def write(name,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader();w.writerows(data)


def nearest(query,times):
    if not len(times):return np.full(len(query),np.inf),np.zeros(len(query),dtype=int)
    j=np.clip(np.searchsorted(times,query),0,len(times)-1);left=np.maximum(j-1,0)
    j=np.where(abs(query-times[left])<=abs(query-times[j]),left,j)
    return abs(query-times[j]),j


def read_seq(seq,data_root):
    gt_t,gt=load_gt(data_root/seq)
    data={'gt':(gt_t,gt)}
    for mode in ['geometric','full','full_temporal']:
        data[mode]=series(HELD/mode/seq/'smoothed_final_trajectory.csv')
    return data


def diagnose(data_root):
    by_seq=[];details=[]
    for metric in rows(HELD/'full_temporal/per_sequence_metrics.csv'):
        seq=metric['sequence_id'];folder=HELD/'full_temporal'/seq
        query,pred=series(folder/'smoothed_final_trajectory.csv')
        st,sp=series(folder/'selected_trajectory.csv');ct,cp=series(folder/'completed_trajectory.csv')
        candidates=rows(folder/'all_candidate_measurements.csv')
        candidate_times=np.unique([float(r['timestamp']) for r in candidates])
        candidate_gap,_=nearest(query,candidate_times)
        counts=Counter()
        for i in np.flatnonzero(~np.isfinite(pred).all(1)):
            q=query[i]
            if len(st)<2:reason='NO_VALID_TRACK'
            elif q<st[0] or q>st[-1]:reason='OUTSIDE_SELECTED_TRACK'
            elif len(ct)<2 or q<ct[0] or q>ct[-1]:reason='OUTSIDE_INTERPOLATION_SUPPORT'
            else:
                j=np.clip(np.searchsorted(ct,q),1,len(ct)-1)
                if ct[j]-ct[j-1]>1 and min(abs(q-ct[j]),abs(q-ct[j-1]))>.05:reason='GAP_TOO_LARGE'
                elif candidate_gap[i]>.05:reason='NO_CANDIDATE'
                else:reason='OTHER'
            counts[reason]+=1
            details.append(dict(sequence_id=seq,timestamp=float(q),primary_reason=reason,
                candidate_available_within_50ms=bool(candidate_gap[i]<=.05)))
        assert sum(counts.values())==int(metric['missing_prediction_count'])
        by_seq.append(dict(sequence_id=seq,missing=sum(counts.values()),
            **{r:counts[r] for r in REASONS}))
    total=Counter({r:sum(s[r] for s in by_seq) for r in REASONS});assert sum(total.values())==528
    write('missing_reason_summary.csv',[dict(reason=r,count=total[r],percentage=100*total[r]/528) for r in REASONS])
    write('missing_reason_by_sequence.csv',by_seq)
    # Small flags separate stream observability from the actual final-mask rule.
    missing_notes=dict(attribution='Exclusive proximate support-rule attribution, NOT unique sensor-level root cause',
        precedence=REASONS[1:2]+REASONS[2:3]+REASONS[4:5]+REASONS[3:4]+REASONS[0:1]+REASONS[5:6],
        missing_without_candidate_within_50ms=sum(not r['candidate_available_within_50ms'] for r in details),
        missing_with_candidate_within_50ms=sum(r['candidate_available_within_50ms'] for r in details),counts=dict(total))
    data=read_seq('seq0065',data_root);qt,gt=data['gt'];ft,fp=data['full'];tt,tp=data['full_temporal']
    np.testing.assert_array_equal(qt,ft);np.testing.assert_array_equal(qt,tt)
    fe=np.linalg.norm(fp-gt,axis=1);te=np.linalg.norm(tp-gt,axis=1)
    bad=np.flatnonzero(np.isfinite(fe)&(fe>2));peak=int(np.nanargmax(fe))
    # Contiguous >2m interval containing the maximum; missing points break intervals.
    left=peak;right=peak
    while left>0 and np.isfinite(fe[left-1]) and fe[left-1]>2:left-=1
    while right+1<len(fe) and np.isfinite(fe[right+1]) and fe[right+1]>2:right+=1
    candidates=rows(HELD/'full/seq0065/all_candidate_measurements.csv');evidence=[]
    grouped={}
    for r in candidates:grouped.setdefault(float(r['timestamp']),[]).append(r)
    for t,group in sorted(grouped.items()):
        gap,j=nearest(np.array([t]),qt);target=gt[j[0]]
        geom=np.array([[float(r['geometric_'+a]) for a in 'xyz'] for r in group])
        corr=np.array([[float(r[a]) for a in 'xyz'] for r in group])
        evidence.append(dict(timestamp=t,min_geometric_error=float(np.linalg.norm(geom-target,axis=1).min()),
            min_corrected_error=float(np.linalg.norm(corr-target,axis=1).min()),
            max_correction_norm=float(np.linalg.norm(corr-geom,axis=1).max()),gt_time_gap_ms=float(gap[0]*1000)))
    interval=[r for r in evidence if qt[left]<=r['timestamp']<=qt[right]]
    selected_rows=rows(HELD/'full/seq0065/selected_trajectory.csv')
    selected_interval=[r for r in selected_rows if qt[left]<=float(r['timestamp'])<=qt[right]]
    # Do not guess association IDs: hypothesis/detection IDs are not cached.
    label='UNKNOWN';confidence='low'
    if interval:
        raw_bad=np.mean([r['min_geometric_error']>2 for r in interval])
        corr_bad=np.mean([r['min_corrected_error']>2 for r in interval])
        if raw_bad>=.8 and corr_bad>=.8:label='A. 输入候选本身严重错误';confidence='medium'
        elif np.mean([r['min_corrected_error']<=1 for r in interval])>=.8:
            label='B. tracker / trajectory selection 与可用目标候选不一致';confidence='medium'
        elif np.mean([r['min_geometric_error']<=1 and r['min_corrected_error']>2 for r in interval])>=.8:
            label='C. M2 center correction 异常';confidence='medium'
    diagnostic=dict(seq='seq0065',diagnostic_threshold_m=2,threshold_status='post-hoc interpretation only, not pipeline parameter',
        first_full_error_above2_timestamp=float(qt[bad[0]]) if len(bad) else None,
        peak_timestamp=float(qt[peak]),peak_full_error=float(fe[peak]),
        peak_interval=[float(qt[left]),float(qt[right])],
        interval_candidate_timestamps=len(interval),interval_selected_states=len(selected_interval),
        interval_selected_prediction_only_states=sum(r['measurement_update']=='False' for r in selected_interval),
        interval_min_geometric_error_median=float(np.median([r['min_geometric_error'] for r in interval])) if interval else None,
        interval_min_corrected_error_median=float(np.median([r['min_corrected_error'] for r in interval])) if interval else None,
        interval_max_correction_norm=max([r['max_correction_norm'] for r in interval],default=None),
        full_peak=float(np.nanmax(fe)),temporal_peak=float(np.nanmax(te)),
        most_likely_failure_stage=label,confidence=confidence)
    write('seq0065_stage_evidence.csv',evidence)
    (OUT/'interpretation_summary.json').write_text(json.dumps(dict(failure=diagnostic,missing=missing_notes),indent=2))
    text=f'''# seq0065 finite failure diagnosis

只读 saved candidate / tracker / selected / completed / final CSV 和 GT；无模型推理、拟合或重跑。
诊断阈值2m仅用于标记异常，不参与任何算法选择。

- 首次 FULL error >2m：{diagnostic['first_full_error_above2_timestamp']}。
- 峰值时间：{qt[peak]:.6f}；FULL error={fe[peak]:.3f}m。
- 包含峰值的连续>2m区间：{qt[left]:.6f}—{qt[right]:.6f}（缺测会切断区间）。
- 区间内 candidate timestamps={len(interval)}；selected states={len(selected_interval)}；
  prediction-only states={diagnostic['interval_selected_prediction_only_states']}。
- 最近原始候选到GT的误差中位数：{diagnostic['interval_min_geometric_error_median']}m；
  最近corrected candidate误差中位数：{diagnostic['interval_min_corrected_error_median']}m。
- 最大M2 correction norm：{diagnostic['interval_max_correction_norm']}m。
- FULL / temporal最大误差：{diagnostic['full_peak']:.3f} / {diagnostic['temporal_peak']:.3f}m。

GT用于事后比较所有候选，不参与生成或选择。最接近GT的候选只是oracle诊断，
不能证明tracker实际关联了哪个点；原记录没有保存hypothesis/detection IDs。
Raw tracker与selected的对应可由track_id核对；selected仅取固定最长存活track，
不把多个track连成一条，也不因结果差重选。Temporal是否放大只能比较已有FULL与final，
不从整体平均误差强行推断因果。见seq0065_stage_evidence.csv和fig4全时轴。

Most likely failure stage = {label}

Confidence = {confidence}
'''
    (OUT/'seq0065_failure_diagnosis.md').write_text(text)
    print(json.dumps(dict(failure=diagnostic,missing=missing_notes),indent=2))
    return diagnostic,evidence


def plot_line(ax,t,p,color,label):
    # Existing NaNs remain gaps; no interpolation or replacement.
    ax.plot(t,p,color=color,label=label,lw=1.05)


def xyz3(ax,data,modes):
    for mode in modes:
        t,p=data[mode];ax.plot(*p.T,color=COLORS[mode],label=mode.upper(),lw=1.)
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax.tick_params(labelsize=7);ax.view_init(22,-55)


def times3(axes,data,modes,origin=None):
    origin=data['gt'][0][0] if origin is None else origin
    for a,ax in enumerate(axes):
        for mode in modes:
            t,p=data[mode];plot_line(ax,t-origin,p[:,a],COLORS[mode],mode.upper())
        ax.set_ylabel('XYZ'[a]+' (m)');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time since sequence start (s)')


def save(fig,name):
    fig.savefig(VIS/name,dpi=300,bbox_inches='tight');plt.close(fig)


def selections():
    metrics=[r for r in rows(HELD/'full_temporal/per_sequence_metrics.csv') if r['sequence_id']!='seq0065' and float(r['coverage'])>0]
    metrics.sort(key=lambda r:(float(r['mean_3d_error']),r['sequence_id']))
    median=float(np.median([float(r['mean_3d_error']) for r in metrics]))
    ranked=sorted(metrics,key=lambda r:(round(abs(float(r['mean_3d_error'])-median),12),r['sequence_id']))
    rep=ranked[0]['sequence_id']
    result=dict(representative=rep,normal_mean_error_median=median,
        representative_rule='Exclude seq0065 from presentation only; coverage>0; closest mean error to median; ties within1e-12 use sequence_id',
        good=metrics[0]['sequence_id'],hard=metrics[-1]['sequence_id'],
        six=[metrics[0]['sequence_id'],metrics[1]['sequence_id'],rep,ranked[1]['sequence_id'],metrics[-1]['sequence_id'],'seq0065'])
    (OUT/'visualization_selection.json').write_text(json.dumps(result,indent=2));return result


def visualizations(data_root,selection,diagnostic,evidence):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
    seq=selection['representative'];data=read_seq(seq,data_root)
    pre=HELD/'full'/seq/'preprocessed'
    if not pre.exists():pre=BASE/'datasets/center_regression/sequences'/seq
    files=timestamp_files(data_root/seq/'lidar_360')
    # Earliest nonoverlap20 block with a saved retained Mid360 point; otherwise first20.
    start=0
    for i in range(0,len(files)-19,20):
        if any(np.load(pre/'lidar_360_processed'/p.name,allow_pickle=False).size for p in files[i:i+20]):start=i;break
    chosen=files[start:start+20];raw=[load_xyz(p) for p in chosen]
    retained=[]
    for p in chosen:
        x=np.load(pre/'lidar_360_processed'/p.name,allow_pickle=False)
        retained.append(x.reshape(-1,3))
    points=np.concatenate(raw);kept=np.concatenate(retained)
    # Cache only: exact surviving-point membership, no new cluster fit or classifier.
    if len(kept):is_kept=cKDTree(kept).query(points,k=1)[0]<1e-5
    else:is_kept=np.zeros(len(points),dtype=bool)
    cand=rows(HELD/'full'/seq/'all_candidate_measurements.csv');grouped={}
    for r in cand:grouped.setdefault(float(r['timestamp']),[]).append(r)
    sample=None
    for file in timestamp_files(pre/'lidar_fusion'):
        group=grouped.get(float(file.stem),[])
        if len(group)!=1:continue
        x=np.load(file,allow_pickle=False)
        if x.size==0:continue
        x=load_xyz(file);r=group[0];center=np.array([float(r['geometric_'+a]) for a in 'xyz'])
        if len(x)==int(r['point_count']) and np.allclose(x.mean(0),center,atol=1e-5):sample=(x,r);break
    if sample is None:raise ValueError('No cache frame has unambiguous saved one-candidate membership')
    cluster,r=sample;center=np.array([float(r['geometric_'+a]) for a in 'xyz']);pred=np.array([float(r[a]) for a in 'xyz'])
    gap,j=nearest(np.array([float(r['timestamp'])]),data['gt'][0]);gt=data['gt'][1][j[0]]
    fig=plt.figure(figsize=(27,7));outer=fig.add_gridspec(1,6,wspace=.3)
    rng=np.random.default_rng(42)
    def scatter(ax,p,color,label=None):
        if len(p)>6000:p=p[rng.choice(len(p),6000,replace=False)]
        ax.scatter(*p.T,s=1,color=color,label=label,alpha=.6)
    ax=fig.add_subplot(outer[0],projection='3d')
    scatter(ax,raw[0],'gray');ax.set_title('1 Point Cloud Frames\nReal single Mid360 frame')
    ax=fig.add_subplot(outer[1],projection='3d')
    for i,p in enumerate(raw):scatter(ax,p,plt.cm.viridis(i/19))
    ax.set_title('2 Accumulation & Clustering\n20 real frames; colors = frame time')
    ax.text2D(0,-.12,'Original DBSCAN labels were not cached.\nNo new clustering was performed.',transform=ax.transAxes,fontsize=8)
    ax=fig.add_subplot(outer[2],projection='3d')
    scatter(ax,points[~is_kept],'#bdbdbd','Not retained');scatter(ax,kept,'#1565c0','M1-retained cache')
    ax.set_title('3 Dynamic Cluster Classification\nSaved retained vs other input points');ax.legend(loc='upper left')
    ax=fig.add_subplot(outer[3],projection='3d');scatter(ax,cluster,'#888888','Raw candidate')
    for p,c,mark,label in [(center,'gray','x','Geometric'),(pred,'#1565c0','*','M2'),(gt,'#c62828','o','GT')]:
        ax.scatter(*p,s=65,c=c,marker=mark,label=label)
    ax.set_title(f'4 Center Regression\nGeometric {np.linalg.norm(center-gt):.3f}m / M2 {np.linalg.norm(pred-gt):.3f}m');ax.legend(loc='upper left')
    sub=outer[4].subgridspec(4,1,hspace=.48)
    ax=fig.add_subplot(sub[0],projection='3d');xyz3(ax,data,['gt','full']);ax.set_title('5 UAV Tracking')
    axes=[fig.add_subplot(sub[i]) for i in range(1,4)];times3(axes,data,['gt','full'])
    sub=outer[5].subgridspec(3,1,hspace=.3);axes=[fig.add_subplot(sub[i]) for i in range(3)]
    times3(axes,data,['gt','full','full_temporal'])
    metrics={m:json.loads((HELD/m/seq/'metrics.json').read_text()) for m in ['full','full_temporal']}
    axes[0].set_title(f"6 Trajectory Completion\nFULL {metrics['full']['coverage']:.1%} / TEMP {metrics['full_temporal']['coverage']:.1%}")
    axes[0].legend(loc='upper left')
    for ax in fig.axes:
        if getattr(ax,'name','')=='3d':ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.tick_params(labelsize=7)
    fig.suptitle(f'Representative {seq} | Visualization reconstructed from frozen pipeline/cache; no model inference',y=1.02)
    fig.text(.01,-.08,'Classification membership is reconstructed from saved processed points; original pre-LSTM cluster IDs/noise labels were not saved.\nM2 panel uses an exact one-candidate fusion frame; GT is post-hoc illustration only. Missing predictions remain gaps.',fontsize=9)
    save(fig,'fig1_reproduction_pipeline.png')
    fig=plt.figure(figsize=(15,5))
    for i,(tag,s) in enumerate([('GOOD',selection['good']),('MEDIAN',seq),('HARD',selection['hard'])],1):
        ax=fig.add_subplot(1,3,i,projection='3d');xyz3(ax,read_seq(s,data_root),['gt','geometric','full','full_temporal']);ax.set_title(f'{tag}: {s}')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02))
    save(fig,'fig2_trajectory_examples.png')
    fig=plt.figure(figsize=(17,17));grid=fig.add_gridspec(6,4,hspace=.5,wspace=.4)
    for i,s in enumerate(selection['six']):
        d=read_seq(s,data_root);ax=fig.add_subplot(grid[i,0],projection='3d');xyz3(ax,d,['gt','full_temporal'])
        ax.set_title(s+ ('\nHEAVY-TAIL FAILURE CASE' if s=='seq0065' else ''))
        axes=[fig.add_subplot(grid[i,j]) for j in range(1,4)]
        for j,a in enumerate(axes):
            for mode in ['gt','full_temporal']:
                t,p=d[mode];plot_line(a,t-d['gt'][0][0],p[:,j],COLORS[mode],mode.upper())
            a.set(xlabel='Time (s)',ylabel='XYZ'[j]+' (m)');a.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2)
    save(fig,'fig3_tracking_and_completion.png')
    d=read_seq('seq0065',data_root);fig=plt.figure(figsize=(15,9));grid=fig.add_gridspec(2,3,hspace=.4,wspace=.3)
    ax=fig.add_subplot(grid[0,0],projection='3d');xyz3(ax,d,['gt','full','full_temporal']);ax.set_title('A seq0065 / real failure')
    for j in range(3):
        ax=fig.add_subplot(grid[(j+1)//3,(j+1)%3])
        for mode in ['gt','full','full_temporal']:
            t,p=d[mode];plot_line(ax,t-d['gt'][0][0],p[:,j],COLORS[mode],mode.upper())
        ax.set(xlabel='Time (s)',ylabel='XYZ'[j]+' (m)',title='BCD'[j]+' Time-'+ 'XYZ'[j]);ax.grid(alpha=.2)
    ax=fig.add_subplot(grid[1,1]);origin=d['gt'][0][0]
    for mode in ['full','full_temporal']:
        t,p=d[mode];plot_line(ax,t-origin,np.linalg.norm(p-d['gt'][1],axis=1),COLORS[mode],mode.upper())
    ax.set(xlabel='Time (s)',ylabel='3D error (m)',title='E Error timeline');ax.legend()
    ax=fig.add_subplot(grid[1,2]);cand=rows(HELD/'full/seq0065/all_candidate_measurements.csv')
    ax.scatter([float(r['timestamp'])-origin for r in cand],[float(r['geometric_y']) for r in cand],s=5,c='gray',label='All geometric candidates')
    ax.scatter([float(r['timestamp'])-origin for r in cand],[float(r['y']) for r in cand],s=5,c=COLORS['full'],label='All corrected candidates')
    st,sp=series(HELD/'full/seq0065/selected_trajectory.csv');plot_line(ax,st-origin,sp[:,1],'black','Selected track Y')
    plot_line(ax,d['gt'][0]-origin,d['gt'][1][:,1],COLORS['gt'],'GT Y')
    ax.axvspan(diagnostic['peak_interval'][0]-origin,diagnostic['peak_interval'][1]-origin,color='orange',alpha=.12)
    ax.set(xlabel='Time (s)',ylabel='Y (m)',title='F Candidates / selected state');ax.legend(fontsize=7)
    save(fig,'fig4_seq0065_failure.png')
    summary=json.loads((OUT/'final_reproduction_summary.json').read_text());fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    x=np.arange(2);axes[0].bar(x-.18,[.27,.05],.36,label='Paper');axes[0].bar(x+.18,[summary['module']['before'],summary['module']['after']],.36,label='Ours')
    axes[0].set(xticks=x,xticklabels=['Before','After'],ylabel='Reported / local MSE',title='Center regression\n81.48% vs 77.32% reduction');axes[0].legend()
    vals=[r['mean_3d_error'] for r in summary['heldout_paired_comparison']]
    axes[1].bar(['Geometric','FULL'],vals,color=['gray',COLORS['full']]);axes[1].set(ylabel='Mean 3D error (m)',title='Heldout paired (4499 timestamps)\nMean error down 50.47%')
    for i,v in enumerate(vals):axes[1].text(i,v+.015,f'{v:.3f}m',ha='center')
    vals=[summary['heldout'][m]['coverage']*100 for m in ['geometric','full','full_temporal']]
    axes[2].bar(['Geometric','FULL','FULL_TEMPORAL'],vals,color=['gray',COLORS['full'],COLORS['full_temporal']]);axes[2].set(ylabel='Coverage (%)',ylim=(0,105),title='Trajectory completeness\nFULL_TEMPORAL +16.17 pp')
    for i,v in enumerate(vals):axes[2].text(i,v+1,f'{v:.2f}%',ha='center')
    fig.tight_layout();fig.text(.02,-.03,'Paper comparison = numerical reference only; metric alignment unresolved.',fontsize=9)
    save(fig,'fig5_final_results_summary.png')
    return dict(raw_window=[p.stem for p in chosen],raw_points=len(points),retained_points=len(kept),
        example_candidate_timestamp=r['timestamp'],classification_reconstruction='saved processed membership; DBSCAN labels unavailable',
        example_gt_time_gap_ms=float(gap[0]*1000))


def documentation():
    """Render explanatory materials only; never execute the documented algorithms."""
    stages=[
      ('Mid360 preprocess','20 raw frames','去零、累计、DBSCAN eps=2/min=10、9D feature、M1分类','按帧保留动态簇','稀疏点的时间结构与背景剔除','tools/build_mmuav_center_regression_dataset.py','preprocess; logits_only','PUBLIC operations + reconstructed M1 adapter'),
      ('Livox preprocess','raw Avia [N,>=3]','去零；作者FPS最多100点','processed points','限制点数且保留空间覆盖','src/rdq_uav/baselines/mmuav_preprocess.py','process_lidar_livox; farthest_point_sample','PUBLIC'),
      ('Fusion','两传感器processed点及时间戳','作者时间组织和fusion DBSCAN','lidar_fusion NPY','合并两传感器观测','src/rdq_uav/baselines/mmuav_preprocess.py','process_fusion','PUBLIC'),
      ('Candidate clustering','每帧fusion点','DBSCAN eps=1/min=1；全部有效簇均保留；完整簇求mean','[Ni,3]点簇及[3]中心','形成每时刻全部测量候选','tools/run_mmuav_pose_pipeline.py','infer_candidates','PUBLIC clustering + reconstructed callable adapter'),
      ('M1 classifier','[B,20,9]','LSTM(9,64,1)，所有hidden states标量attention，Linear(64,2)','logits [B,2]; attention [B,20]','时间动态簇分类；仅logits决定类别','src/rdq_uav/mmuav/attention_lstm.py','AttentionLSTMClassifier.forward','RECONSTRUCTED DESIGN'),
      ('M2 center regression','local points [B,64,3] + observed center [B,3]','shared MLP 3/64/128/256、max pool、concat center、259/128/64/3','delta [B,3]; center+delta','从观测形状与绝对位置学习中心偏移','src/rdq_uav/mmuav/center_regressor.py','CenterRegressor.forward; sample_local; predict_delta','RECONSTRUCTED DESIGN'),
      ('Kalman tracking','每timestamp全部[K,3]中心','StoneSoup CV Kalman + nearest-neighbour；归档多track','timestamp/track_id/XYZ/velocity/update','关联和滤波；不使用GT','src/rdq_uav/mmuav/pose_trajectory.py','track_candidates','PUBLIC parameters + reconstructed output/archive wrapper'),
      ('Track selection','multi-track states','最长duration；tie按measurement count再初始XYZ','selected [T,3]','固定、无GT的唯一轨迹选择','src/rdq_uav/mmuav/pose_trajectory.py','select_track','RECONSTRUCTED DESIGN'),
      ('AR(3)','selected track','各轴三阶OLS，0.1s grid，最多补1s内部gap；短序列fallback','completed trajectory','补短期失测；不是GT拟合','src/rdq_uav/mmuav/pose_trajectory.py','ar_complete','RECONSTRUCTED DESIGN'),
      ('Interpolation','completed track及查询时间','线性插值；禁止support外预测；大gap保持missing','interpolated [Tprime,3]','统一时间坐标；只允许GT timestamp、不允许GT XYZ','src/rdq_uav/mmuav/pose_trajectory.py','resample(smooth=False)','PUBLIC mathematical form + reconstructed support rules'),
      ('B-spline','completed states','splrep/splev，s=0.5，k=3；有效点不足时线性；同support mask','smoothed final XYZ','固定时间平滑；本轮不重新拟合','src/rdq_uav/mmuav/pose_trajectory.py','resample(smooth=True)','PUBLIC mathematical form + reconstructed support rules'),
      ('Evaluation','已生成预测 + GT','冻结timestamp tolerance=0.05s；matched-only error + missing/coverage','overall/per-sequence metrics','区分定位误差与轨迹完整性','tools/run_mmuav_pose_pipeline.py','score; main','RECONSTRUCTED local evaluator'),
      ('Finalization','现有CSV/metrics/cache','共同timestamp比较、长尾统计、离线诊断及绘图','报告、CSV、PNG、manifest','规范解释并封存既有实验','tools/finalize_mmuav_reproduction.py; tools/document_mmuav_archive.py','main; diagnose; visualizations; documentation','RECONSTRUCTED offline reporting')]
    lines=['# MMUAV Reproduction Pipeline','', '## 总体流程','', '![Pipeline](final_visuals/reproduction_pipeline_flowchart.png)','',
      'M3 polynomial bias = **BYPASSED**。这是公开算法与明确重建组件组成的定位分支，不是已确认的论文完整原实现。', '',
      '## 每一步输入、操作、输出与目的','']
    for name,inp,op,out,purpose,file,fn,status in stages:
        lines += [f'### {name}','',f'- 输入：{inp}',f'- 操作：{op}',f'- 输出：{out}',f'- 解决问题：{purpose}','']
    lines += ['## 算法与实际代码映射','', '| Stage | Input | Core operation | Output | Our code | Function/Class | Public/Reconstructed |','|---|---|---|---|---|---|---|']
    for name,inp,op,out,purpose,file,fn,status in stages:
        lines.append(f'| {name} | {inp} | {op} | {out} | `{file}` | `{fn}` | {status} |')
    lines += ['', '## Data Flow / Tensor Shape','', '```text','Raw [N,>=3] → XYZ [N,3]','Mid360 20-frame → feature [B,20,9] → M1 logits [B,2]','Fusion → candidates [Ni,3] → full-cluster geometric center [3]','sample_local [64,3] + observed center [3] → M2 delta [3] → corrected center [3]','per timestamp [K,3] → Kalman [x,vx,y,vy,z,vz]','selected [T,3] → AR/interpolation/spline [Tprime,3]','final CSV: timestamp,x,y,z (missing retained as NaN)','```','',
      '几何中心在采样前由完整簇求mean；GT从不进入M2 forward。M2模块级2359个GT-conditioned validation样本不充当系统输入。系统中全部合法候选进入tracker，GT XYZ只在预测生成后评价。', '',
      '## Public / Reconstructed边界','', 'PUBLIC：Mid360累计/去零/DBSCAN及9D提取、Livox FPS、fusion、候选DBSCAN、作者StoneSoup基础参数、线性插值和B-spline数学形式。RECONSTRUCTED：M1 attention、M2 PointNet-like残差回归、无GT最长track选择、AR3具体OLS与fallback、support/gap规则和本地评价器。M3旁路。训练DBSCAN eps=1与系统Mid360 eps=2存在公开代码差异，M1分类validation F1不可解释成系统候选检测100%。', '',
      '## 三条冻结系统路径','', 'GEOMETRIC与FULL使用同tracker/selection；FULL对所有候选应用M2。FULL_TEMPORAL再用AR3、interpolation和spline。heldout使用冻结配置一次独立评价，没有根据heldout调参。误差只在matched timestamps上计算，coverage和missing同时保留。', '',
      '## 图表与有限诊断','', '代表选择见visualization_selection.json：排除已知heavy-tail seq0065仅用于主展示，在其余有效sequence中选mean error最接近中位数者；等距按sequence ID打破tie。seq0065仍计入所有正式结果并独立展示。图1原始DBSCAN标签未缓存：Panel 2按实际frame timestamp着色而不是伪造簇编号；Panel 3由保存的processed点匹配重建。未重新运行DBSCAN或模型。所有绘图采样仅用于降低显示密度，不改变任何中心、预测或评分。', '',
      'missing分类是互斥的近端support规则归因，不是唯一传感器根因。511点超出selected track范围，17点为gap拒绝；337个missing时间附近无保存候选，该标记与support原因重叠。', '',
      'MMUAV REPRODUCTION ARCHIVED AND CLOSED']
    (OUT/'MMUAV_REPRODUCTION_PIPELINE.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(12,14));ax.set(xlim=(0,12),ylim=(0,18));ax.axis('off')
    def box(x,y,label,reconstructed=False):
        ax.add_patch(FancyBboxPatch((x-1.65,y-.32),3.3,.64,boxstyle='round,pad=0.05',facecolor='#e3f2fd' if reconstructed else '#eeeeee',edgecolor='#555'))
        ax.text(x,y,label,ha='center',va='center',fontsize=9)
    def arrow(x,y,u,v):ax.annotate('',xy=(u,v+.34),xytext=(x,y-.34),arrowprops=dict(arrowstyle='->',color='#555'))
    box(6,17.4,'Raw MMAUD LiDAR')
    left=['Mid360: zero removal','20-frame accumulation','DBSCAN eps=2 / min=10','9D + M1 Attention-LSTM','Dynamic cluster filtering']
    right=['Livox Avia','Zero removal','FPS <=100']
    for x,labels in [(3,left),(9,right)]:
        for i,label in enumerate(labels):
            y=16.2-i*.9;box(x,y,label,'M1' in label)
            if i:arrow(x,y+.9,x,y)
        arrow(6,17.4,x,16.2);arrow(x,16.2-(len(labels)-1)*.9,6,11.5)
    chain=['LiDAR fusion','Candidate DBSCAN / full mean','M2 PointNet-like residual center','All corrected centers -> Kalman','Multi tracks -> longest-lived track','AR(3): grid 0.1s / gap <=1s','Linear timestamp interpolation','B-spline s=0.5','Final XYZ trajectory','GT evaluation (post-hoc only)']
    for i,label in enumerate(chain):
        y=11.5-i*1.0;box(6,y,label,i in [2,4,5,9])
        if i:arrow(6,y+1,6,y)
    ax.text(.4,1,'M3 polynomial bias: BYPASSED\nBlue: reconstructed component / design\nGray: public operations / mathematical forms',fontsize=9)
    save(fig,'reproduction_pipeline_flowchart.png')


def manifest():
    frozen_path=BASE/'final_reproduction/frozen_reproduction_config.json'
    frozen=json.loads(frozen_path.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    commands={}
    for split,folder in [('validation_sub','validation'),('heldout_test_sub','heldout')]:
        for mode in ['geometric','full','full_temporal']:
            commands[f'{split}/{mode}']=f'conda run --no-capture-output -n mmuav python tools/run_mmuav_pose_pipeline.py --split {split} --mode {mode} --frozen-config outputs/mmuav_paper_reproduction/final_reproduction/frozen_reproduction_config.json --output-dir outputs/mmuav_paper_reproduction/final_reproduction/{folder}/{mode}'
    data=dict(repo_branch=git('branch','--show-current'),final_commit=git('rev-parse','HEAD'),
      final_commit_scope='Archive code/documentation revision; manifest is committed subsequently without algorithm changes',
      python_version=platform.python_version(),versions={k:version(k) for k in ['torch','numpy','scipy','stonesoup']},
      checkpoints={k:dict(path=frozen[k+'_checkpoint'],sha256=sha(frozen[k+'_checkpoint'])) for k in ['M1','M2']},
      split=dict(path=str(BASE/'splits/splits.json'),sha256=sha(BASE/'splits/splits.json')),
      frozen_config=dict(path=str(frozen_path),sha256=sha(frozen_path)),commands=commands,
      final_report='results/mmuav_reproduction/MMUAV_3D_REPRODUCTION_FINAL.md',
      figures=[str(p.relative_to(ROOT)) for p in sorted(VIS.glob('*.png'))],
      no_experiment_rerun=True,no_model_training_or_fitting=True,status='MMUAV REPRODUCTION ARCHIVED AND CLOSED')
    (OUT/'reproducibility_manifest.json').write_text(json.dumps(data,indent=2)+'\n')


def main():
    VIS.mkdir(exist_ok=True)
    frozen=json.loads((OUT/'frozen_reproduction_config.json').read_text());data_root=Path(frozen['data_root'])
    # Evidence guard: the algorithm and checkpoints remain unchanged.
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for file,digest in frozen['pipeline_code_sha256'].items():assert sha(ROOT/file)==digest
    for k in ['M1','M2']:assert sha(Path(frozen[k+'_checkpoint']))==frozen[k+'_checkpoint_sha256']
    diagnostic,evidence=diagnose(data_root);selection=selections();visual=visualizations(data_root,selection,diagnostic,evidence)
    (OUT/'visualization_provenance.json').write_text(json.dumps(dict(selection=selection,cache=visual,
        no_model_inference=True,no_clustering_rerun=True,no_parameter_fitting=True,dpi=300,colors=COLORS),indent=2))


if __name__=='__main__':
    if '--documentation-only' in sys.argv:documentation()
    elif '--manifest-only' in sys.argv:manifest()
    else:
        main();documentation()
