#!/usr/bin/env python3
"""Offline final wrap-up only: saved predictions + GT; no model/pipeline execution."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from build_mmuav_cluster_dataset import load_gt
from rdq_uav.mmuav.center_regressor import regression_metrics
BASE=ROOT/'outputs/mmuav_paper_reproduction'
FINAL=BASE/'final_reproduction'
OUT=ROOT/'results/mmuav_reproduction'
MODES=['geometric','full','full_temporal']


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def table(results):
    lines=['| Mode | Coverage | Matched / Missing | MSE_coord | MSE_3D | Mean / Median 3D error (m) |',
           '|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        m=results[mode]
        lines.append(f"| {mode.upper()} | {m['coverage']:.2%} | {m['matched_timestamp_count']} / {m['missing_prediction_count']} | {m['MSE_coord']:.6f} | {m['MSE_3D']:.6f} | {m['mean_3d_error']:.6f} / {m['median_3d_error']:.6f} |")
    return '\n'.join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    frozen_path=FINAL/'frozen_reproduction_config.json';frozen=json.loads(frozen_path.read_text())
    assert sha(frozen_path)==sha(OUT/'frozen_reproduction_config.json')
    frozen_hash=sha(frozen_path)
    assert sha(Path(frozen['M1_checkpoint']))==frozen['M1_checkpoint_sha256']
    assert sha(Path(frozen['M2_checkpoint']))==frozen['M2_checkpoint_sha256']
    for file,digest in frozen['pipeline_code_sha256'].items(): assert sha(ROOT/file)==digest
    validation={};heldout={};per_seq={};configs={};input_hashes={}
    for split,destination in [('validation',validation),('heldout',heldout)]:
        for mode in MODES:
            folder=FINAL/split/mode
            summary=json.loads((folder/'overall_metrics.json').read_text())
            config=json.loads((folder/'run_config.json').read_text())
            rows=read(folder/'per_sequence_metrics.csv')
            assert summary['planned_sequences']==summary['processed_sequences']==len(rows)==15
            assert not summary['processing_failure_sequences'] and not summary['valid_empty_sequences']
            assert not read(folder/'processing_failures.csv')
            assert config['classifier_sha256']==frozen['M1_checkpoint_sha256']
            assert config['center_sha256']==frozen['M2_checkpoint_sha256']
            assert config['splits_sha256']==frozen['split_definition_hash']
            assert not config['smoke_no_ar_fit'] and config['sequence'] is None
            if split=='heldout': assert config['frozen_config_sha256']==frozen_hash
            destination[mode]=summary['overall'];configs[f'{split}/{mode}']=config
            per_seq[f'{split}/{mode}']=rows
            input_hashes[f'{split}/{mode}']={n:sha(folder/n) for n in ['overall_metrics.json','run_config.json','per_sequence_metrics.csv','processing_failures.csv']}
            if split=='validation':
                assert input_hashes[f'{split}/{mode}']==frozen['validation_results']['artifact_sha256'][mode]
            for key in ['tracker','trajectory_selection','AR_order','AR_grid_seconds','max_gap_seconds',
                        'spline_s','nearest_evaluation_tolerance_seconds','M3']:
                assert config[key]==frozen[key]
            for key in ['num_gt','matched_timestamp_count','missing_prediction_count']:
                assert sum(int(r[key]) for r in rows)==summary['overall'][key]
    splits=json.loads(Path(configs['heldout/full']['splits']).read_text())
    sequences=splits['heldout_test_sub']
    assert {r['sequence_id'] for r in per_seq['heldout/full']}==set(sequences)
    predictions={mode:{} for mode in MODES};gt_map={}
    for seq in sequences:
        gt_t,gt=load_gt(Path(frozen['data_root'])/seq)
        for t,p in zip(gt_t,gt):
            key=(seq,float(t));assert key not in gt_map;gt_map[key]=p
        for mode in MODES:
            rows=read(FINAL/'heldout'/mode/seq/'smoothed_final_trajectory.csv')
            assert len(rows)==len(gt_t)
            seen=set()
            for r in rows:
                key=(seq,float(r['timestamp']));assert key not in seen and key in gt_map;seen.add(key)
                p=np.array([float(r[a]) for a in 'xyz'])
                if np.isfinite(p).all(): predictions[mode][key]=p
            assert seen=={(seq,float(t)) for t in gt_t}
    for mode in MODES:
        keys=sorted(predictions[mode])
        recomputed=regression_metrics(np.array([predictions[mode][k] for k in keys]),np.array([gt_map[k] for k in keys]))
        assert len(keys)==heldout[mode]['matched_timestamp_count']
        for key in ['MSE_coord','MSE_3D','mean_3d_error','median_3d_error']:
            assert math.isclose(recomputed[key],heldout[mode][key],rel_tol=1e-10)
    def paired(a,b,name):
        ka,kb=set(predictions[a]),set(predictions[b]);keys=sorted(ka&kb)
        rows=[]
        for mode in [a,b]:
            result=regression_metrics(np.array([predictions[mode][k] for k in keys]),np.array([gt_map[k] for k in keys]))
            row=dict(model=mode.upper(),common_timestamp_count=len(keys),
                left_only_timestamp_count=len(ka-kb),right_only_timestamp_count=len(kb-ka),
                comparison_left=a,comparison_right=b,**result)
            if a=='geometric' and b=='full':
                row.update(geometric_only_timestamp_count=len(ka-kb),full_only_timestamp_count=len(kb-ka))
            rows.append(row)
        write(name,rows)
        return rows
    primary=paired('geometric','full','heldout_paired_timestamp_comparison.csv')
    auxiliary=paired('full','full_temporal','heldout_temporal_paired_timestamp_comparison.csv')
    primary_reductions={k:1-primary[1][k]/primary[0][k] for k in ['MSE_coord','mean_3d_error','median_3d_error']}
    tail=[]
    for mode in ['full','full_temporal']:
        rows=per_seq[f'heldout/{mode}']
        total=sum(float(r['MSE_3D'])*int(r['matched_timestamp_count']) for r in rows)
        for rank,r in enumerate(sorted(rows,key=lambda r:float(r['MSE_3D']),reverse=True)[:3],1):
            squared=float(r['MSE_3D'])*int(r['matched_timestamp_count'])
            tail.append(dict(mode=mode,rank=rank,sequence_id=r['sequence_id'],
                matched_timestamp_count=int(r['matched_timestamp_count']),MSE_3D=float(r['MSE_3D']),
                squared_error_sum=squared,overall_squared_error_contribution=squared/total))
    write('heavy_tail_sequence_contributions.csv',tail)
    module=json.loads((BASE/'center_regression/pointnet_m2/evaluation_summary.json').read_text())
    before,after=[m['MSE_coord'] for m in module['comparison']]
    ours_reduction=1-after/before;paper_reduction=1-.05/.27
    ablations=read(BASE/'center_regression/m2_5_comparison.csv')
    comparison=[
        dict(Section='Center regression before',Metric='MSE',Paper=.27,Our_Reproduction=before,Comparison_Status='numerically_close',Notes='metric/sample alignment unresolved'),
        dict(Section='Center regression after',Metric='MSE',Paper=.05,Our_Reproduction=after,Comparison_Status='numerically_close_not_exact',Notes='architecture alignment partial; not strict reproduction'),
        dict(Section='Center regression reduction',Metric='%',Paper=f'{paper_reduction:.2%}',Our_Reproduction=f'{ours_reduction:.2%}',Comparison_Status='comparable_numerically',Notes='NUMERICAL COMPARISON only'),
        dict(Section='Final pose',Metric='Paper Pose MSE',Paper=2.21375,Our_Reproduction='N/A',Comparison_Status='metric_unresolved',Notes='Official challenge test, not local heldout; NUMERICAL REFERENCE ONLY')]
    for key,label in [('MSE_coord','MSE_coord'),('MSE_3D','MSE_3D'),('mean_3d_error','mean 3D error (m)'),('coverage','coverage')]:
        comparison.append(dict(Section='Final heldout',Metric=label,Paper='N/A',Our_Reproduction=heldout['full_temporal'][key],Comparison_Status='local_metric',Notes='FULL_TEMPORAL; not aligned to paper Pose MSE'))
    write('final_paper_comparison.csv',comparison)
    summary=dict(status='MMUAV 3D LOCALIZATION RECONSTRUCTION COMPLETE',frozen_config_sha256=frozen_hash,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        validation=validation,heldout=heldout,heldout_paired_comparison=primary,paired_relative_reductions=primary_reductions,
        auxiliary_temporal_paired_comparison=auxiliary,heavy_tail_top3=tail,
        module=dict(before=before,after=after,reduction=ours_reduction,paper_before=.27,paper_after=.05,
            paper_reduction=paper_reduction,before_relative_numerical_difference=before/.27-1,
            after_relative_numerical_difference=after/.05-1,metric_alignment='unresolved',sample_alignment='unresolved',architecture_alignment='partial'),
        temporal=dict(coverage_absolute_improvement=heldout['full_temporal']['coverage']-heldout['full']['coverage'],
            additional_matched_timestamps=heldout['full_temporal']['matched_timestamp_count']-heldout['full']['matched_timestamp_count'],
            full_only_timestamps=auxiliary[0]['left_only_timestamp_count'],
            temporal_only_timestamps=auxiliary[0]['right_only_timestamp_count']),
        paper_pose_mse=2.21375,paper_pose_comparison='NUMERICAL REFERENCE ONLY',
        artifacts_sha256=input_hashes,no_pipeline_rerun=True,no_training_or_fitting=True)
    (OUT/'final_reproduction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    limitations='''1. 论文 7D dynamic feature 定义未完全恢复；本轮使用 9D。
2. 论文 center regression 的 PointNet/MLP 具体结构有歧义，本轮为明确标记的重建。
3. 论文 24D third-order polynomial 定义未恢复，M3 在本轮旁路。
4. Paper Pose MSE 与本地 MSE_coord / MSE_3D 口径未对齐，不能据此判定优劣。
5. 本地 heldout 是官方 train 的 sequence-level 固定留出集，不是官方 challenge test。
'''
    (OUT/'known_limitations.md').write_text('# Known limitations\n\n'+limitations)
    takeaways='''# Method takeaways

1. **Geometric center 不是可靠的最终目标位置。** 稀疏/残缺 cluster 的均值不能直接视作 UAV center；本轮系统轨迹也含明显长尾失败。
2. **学习式 center correction 在三个层次有效。** M2 模块级、validation 系统级，以及 heldout 共同时间戳对照均显示收益；不是仅凭数量近似进行比较。
3. **Observed absolute position 具有很强的校正价值。** M2.5 中 CENTER_ONLY 接近 FULL，POINTS_ONLY 也改善，FULL 最佳；支持位置先验和局部结构互补，不证明已经学到 UAV 特有形状或全部因果来源。
4. **Temporal processing 主要改善完整性。** Coverage 必须与 available-prediction error 分开报告；共同时间戳误差只是辅助，不能替代 coverage，更不能将组合收益单独归因于 AR。
5. **向 Radar+RGB 迁移的是研究假设，不是已证实的雷达结论。** 不直接使用雷达/LiDAR cluster centroid 作为最终 3D 表示：Radar cluster geometry + observed spatial position prior + RGB target feature → learned target-center correction → corrected 3D position。Temporal module 独立负责连续性。Radar target association、坐标关系与可观测性仍需在自己的任务中验证。

本轮仅完成重建和固定留出评价；不做新实验、不调参，不以未知口径匹配论文数值。
'''
    (OUT/'method_takeaways.md').write_text(takeaways)
    module_table=['| Model | MSE_coord | Mean 3D error (m) |','|---|---:|---:|']
    for r in ablations: module_table.append(f"| {r['Model']} | {float(r['MSE_coord']):.6f} | {float(r['mean_3d_error']):.6f} |")
    paired_table=['| Model | Same timestamps | MSE_coord | MSE_3D | Mean / Median 3D error (m) |','|---|---:|---:|---:|---:|']
    for r in primary: paired_table.append(f"| {r['model']} | {r['common_timestamp_count']} | {r['MSE_coord']:.6f} | {r['MSE_3D']:.6f} | {r['mean_3d_error']:.6f} / {r['median_3d_error']:.6f} |")
    tail_table=['| Mode | Sequence | MSE_3D | Contribution to overall squared error |','|---|---|---:|---:|']
    for r in tail: tail_table.append(f"| {r['mode']} | {r['sequence_id']} | {r['MSE_3D']:.6f} | {r['overall_squared_error_contribution']:.2%} |")
    report=f'''# MMUAV 3D Reproduction Final

## 1. Objective

重建一版可运行、可评价的 MMUAV LiDAR 3D 定位/轨迹分支，并规范比较论文数值。
固定 sequence-level split：72 train / 15 validation / 15 heldout，seed=42。
本次收尾只读既有预测和 GT，不训练、不拟合、不重跑。Heldout 未参与模型或配置选择。

## 2. Reconstructed Pipeline

Raw Mid360 → 20-frame accumulation / DBSCAN / M1 → processed Mid360；
Raw Livox → zero removal / FPS → processed Livox；
fusion → candidate DBSCAN → M2 FULL → Kalman → longest-lived valid track
→ AR(3) → interpolation → B-spline → final XYZ → GT evaluation。M3 BYPASSED。
全部合法 candidate 进入系统推理；GT 不参与 candidate/track 选择、补全或平滑。
GT timestamp 只作评价时间坐标，GT XYZ 仅在预测生成后参与评分。

## 3. Public Code vs Reconstructed Components

| Component | Status |
|---|---|
| Mid360 / Livox / fusion / candidate clustering | Public-code logic, unchanged parameters; M1 tuple logits adapter |
| M0 classifier | Public 9D ordinary LSTM; not paper 7D Attention LSTM |
| M1 | Reconstructed 9D scalar attention over all hidden states |
| M2 | Reconstructed PointNet-like residual regressor, local points + observed center |
| Kalman / association | Public StoneSoup settings and update logic |
| Track selection / deleted-track archive | Reconstructed selection and engineering sidecar |
| AR3 completion | Reconstructed per-track OLS, no GT; fixed limits |
| Linear / B-spline | Public maths; bounded support / maximum1s gap reconstruction |
| M3 polynomial bias | Bypassed; unknown24D definition |

原仓库 dtc111111/Multi-Modal-UAV commit f11b57390effbe9623ee2c7d561afddc8d0cdfa7。
Frozen config SHA256: `{frozen_hash}`。本地重建不等于公开代码 baseline，也不等于论文严格实现。

## 4. Module-level Results

Classifier dataset: train58,701 clusters/140 positives；validation12,036/46 positives。
Official public checkpoint F1=.9318；M0 retrained best F1=.9890；M1 best F1=1.0000。
这些是固定构建条件下的 classifier 指标，不是系统 candidate detection recall。
M2/M2.5 使用 GT-conditioned module-level 的2359个冻结 validation IDs：

{chr(10).join(module_table)}

FULL 最佳epoch3/停止epoch18。绝对观测位置提供大部分校正价值，局部结构提供额外收益；
不把 FULL 的改善全部归因于目标形状。

## 5. Validation Results

{table(validation)}

15/15 三模式成功，无 failure/empty。GEOMETRIC 与 FULL matched timestamp sets 完全一致。
FULL 的中心校正收益进入轨迹后仍存在；temporal主要增加coverage。

## 6. Frozen Heldout Results

{table(heldout)}

15/15 三模式成功，无 failure/empty。GEOMETRIC 与 FULL 原始匹配集合不完全一致。
原始汇总降幅只作描述；严格中心校正证据以如下完全配对集合为准：

{chr(10).join(paired_table)}

共同时间戳={primary[0]['common_timestamp_count']}；geometric-only={primary[0]['left_only_timestamp_count']}；full-only={primary[0]['right_only_timestamp_count']}。
配对 MSE_coord下降{primary_reductions['MSE_coord']:.2%}，mean3D下降{primary_reductions['mean_3d_error']:.2%}，median3D下降{primary_reductions['median_3d_error']:.2%}。
原始汇总 MSE_coord/mean/median 降幅分别为
{1-heldout['full']['MSE_coord']/heldout['geometric']['MSE_coord']:.2%} /
{1-heldout['full']['mean_3d_error']/heldout['geometric']['mean_3d_error']:.2%} /
{1-heldout['full']['median_3d_error']/heldout['geometric']['median_3d_error']:.2%}，不是严格配对估计。
Temporal coverage提升{summary['temporal']['coverage_absolute_improvement']*100:.2f}pp，新增970个匹配时间点。
其 available-prediction error 来自不同集合，不能将总体 MSE 变化解释成同样本精度提升。
辅助同时间戳比较见 heldout_temporal_paired_timestamp_comparison.csv，不替代coverage。
FULL∩FULL_TEMPORAL={auxiliary[0]['common_timestamp_count']}；temporal新增{auxiliary[0]['right_only_timestamp_count']}个，
另有{auxiliary[0]['left_only_timestamp_count']}个FULL可用时间戳不在temporal可用集合内，净增加970个。
辅助配对 MSE_coord：{auxiliary[0]['MSE_coord']:.6f} → {auxiliary[1]['MSE_coord']:.6f}；
mean3D：{auxiliary[0]['mean_3d_error']:.6f} → {auxiliary[1]['mean_3d_error']:.6f}m。
不以总体MSE的下降证明同样本精度大幅提升；本轮不改变这些支持区间规则。
Error仅对matched计算，missing/coverage显式报告；不能隐藏无预测时间点，也不对missing伪造有限误差。

Performance contains heavy-tail failures。Top3按per-sequence MSE_3D排序，贡献为
MSE_3D × matched_count / overall squared-error sum，不作因果定位：

{chr(10).join(tail_table)}

## 7. Comparison with Paper

论文 §4.2 / Results table，参考既有 paper-code audit；原文见
[MMUAV technical report](https://arxiv.org/abs/2405.16464)。

| Center regression | Before | After | Reduction |
|---|---:|---:|---:|
| Paper | .270000 | .050000 | {paper_reduction:.2%} |
| Our reconstruction (local MSE_coord) | {before:.6f} | {after:.6f} | {ours_reduction:.2%} |

Before高于论文数值{(before/.27-1)*100:.2f}%；after高于{(after/.05-1)*100:.2f}%。
**NUMERICAL COMPARISON, not strict reproduction**：metric_alignment=unresolved；
sample_alignment=unresolved；architecture_alignment=partial。

| Final metric reference | Value | Status |
|---|---:|---|
| Paper official test Pose MSE | 2.21375 | NUMERICAL REFERENCE ONLY |
| Our FULL_TEMPORAL heldout MSE_coord | {heldout['full_temporal']['MSE_coord']:.6f} | Local metric |
| Our FULL_TEMPORAL heldout MSE_3D | {heldout['full_temporal']['MSE_3D']:.6f} | Local metric |

没有证据证明论文PoseMSE等于任一当地定义，而且challenge test与official-train留出集不同。
因此不计算与论文最终PoseMSE的百分比优劣，不声称达到或超过论文。

## 8. Key Findings

几何均值不是可靠的最终目标位置。学习式中心校正在module、validation和paired-heldout有效。
Observed absolute position 与局部结构具有互补价值；temporal主要验证coverage收益。
Heldout有明显长尾失败：典型点准确不代表所有sequence可靠；本轮只归因统计，不解决或调参。

## 9. Transferable Ideas for Radar+RGB

不要直接将 Radar/LiDAR cluster centroid 当最终3D表示。
Radar cluster geometry + observed spatial position prior + RGB target feature
→ learned target-center correction → corrected3D position；temporal独立负责连续性。
这是待验证研究假设，不是本次LiDAR结果已经证明Radar有效。

## 10. Known Limitations

{limitations}

## 11. Final Status

Frozen configuration和模型保持不变；未进行收尾训练、拟合、重跑或heldout结果驱动的方法选择。
This is a completed reconstruction, not strict paper-level numerical replication.

**MMUAV 3D LOCALIZATION RECONSTRUCTION COMPLETE**
'''
    (OUT/'MMUAV_3D_REPRODUCTION_FINAL.md').write_text(report)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    x=np.arange(2);width=.35
    axes[0].bar(x-width/2,[.27,.05],width,label='Paper')
    axes[0].bar(x+width/2,[before,after],width,label='Our local MSE_coord')
    axes[0].set_xticks(x,['Before','After']);axes[0].set_ylabel('Reported MSE / local MSE_coord')
    axes[0].set_title('Center regression: NUMERICAL REFERENCE\nmetric / sample alignment unresolved');axes[0].legend()
    bars=axes[1].bar(np.arange(3),[heldout[m]['mean_3d_error'] for m in MODES],color=['gray','steelblue','seagreen'])
    axes[1].set_xticks(np.arange(3),['Geometric','FULL','FULL_TEMPORAL']);axes[1].set_ylabel('Mean 3D error (m)')
    axes[1].set_title('Heldout: available-prediction error\nmatched sets differ; coverage shown separately')
    for bar,mode in zip(bars,MODES):
        axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.035,
            f"{heldout[mode]['mean_3d_error']:.3f}m\ncoverage {heldout[mode]['coverage']:.2%}",ha='center',fontsize=9)
    axes[1].set_ylim(0,max(heldout[m]['mean_3d_error'] for m in MODES)*1.3)
    fig.tight_layout();fig.savefig(OUT/'final_reproduction_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(paired=primary,primary_reductions=primary_reductions,heavy_tail=tail,
        report=str(OUT/'MMUAV_3D_REPRODUCTION_FINAL.md')),indent=2))


if __name__=='__main__':main()
