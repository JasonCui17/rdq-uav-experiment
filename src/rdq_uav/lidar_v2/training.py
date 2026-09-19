"""Query-causal training/evaluation helpers. No target metadata enters the model."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .isolation import assert_temporal_clip_integrity
from .runtime import move_batch,evaluate_batch,summarize_metrics
from .contracts import resolve_precision,require_occurrence_aligned_evaluation

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def finite_or_raise(name,tensor,context=''):
    if not torch.isfinite(tensor).all():raise FloatingPointError(f'Non-finite {name}: {context}')

def inspect_dataset_timing(dataset,indices):
    rows=[]
    for i in indices:
        q=dataset[int(i)];times=q['event_timestamps'];t=q['query_time']
        assert_temporal_clip_integrity([q],clip_index=int(i),require_events=True)
        if any(x>t for x in times) or bool((q['delta_t']>0).any()):raise AssertionError('Future event violation')
        if q['target_valid']:
            if not math.isfinite(q['target_timestamp']) or not torch.isfinite(q['target_xyz']).all():raise AssertionError('Invalid target')
        rows.append(dict(sample_id=q['sample_id'],sequence_id=q['sequence_id'],query_time=t,
                         target_timestamp=q.get('target_timestamp'),target_valid=q['target_valid'],
                         query_equals_target=q.get('target_timestamp')==t,event_count=len(times),
                         oldest_event=min(times) if times else None,newest_event=max(times) if times else None,
                         delta_t_max=float(q['delta_t'].max()) if len(q['delta_t']) else None))
    return rows

def evaluation_loss_batch(batch):
    """Score endpoint only in validation rolling clips, without changing context."""
    result=dict(batch)
    mask=batch['target_valid_clip']&batch['score_mask']&batch['query_valid_mask']
    result['target_valid_clip']=mask;result['target_valid']=mask.flatten()
    result['spatial_supervise_mask_occurrence']=mask.flatten()
    return result

@torch.no_grad()
def validate(model,loader,criterion,selector,device,precision='fp32',export_dir=None,checkpoint_path=None):
    """Score every validation endpoint exactly once, including empty observations.

    Retains V1 raw/NMS metrics and adds temporal XYZ metrics in all support groups.
    Export is optional; training validation avoids feature I/O.
    """
    policy=precision if hasattr(precision,'context') else resolve_precision(precision,device)
    model.eval();rows=[];candidates=[];features=[];spatial_sum=cls_sum=reg_sum=temp_sum=0.;ns=nt=no_support=endpoints=0
    checkpoint_id=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() if checkpoint_path else ''
    for raw in tqdm(loader,desc='VAL',dynamic_ncols=True):
        require_occurrence_aligned_evaluation(raw)
        batch=move_batch(raw,device)
        with policy.context(device):out=model(batch)
        for key in ('logits','pred_xyz','temporal_pred_xyz'):finite_or_raise(key,out[key])
        labels_batch=evaluation_loss_batch(batch);loss=criterion(out,labels_batch)
        n=loss['num_supervised_samples'];t=loss['num_temporal_supervised'];ns+=n;nt+=t
        no_support+=loss['num_no_current_support'];endpoints+=int(labels_batch['spatial_supervise_mask_occurrence'].sum())
        spatial_sum+=float(loss['spatial_loss'])*n;cls_sum+=float(loss['loss_cls'])*n;reg_sum+=float(loss['loss_reg'])*n;temp_sum+=float(loss['temporal_loss'])*t
        rows.extend(evaluate_batch(out,batch,selector,criterion))
        if export_dir is not None:
            chosen=selector(out);pos,_,_,_=criterion.labels(out,batch)
            for b,item in enumerate(chosen):
                if not bool(batch['score_mask'].flatten()[b]):continue
                for kind in ('raw','nms'):
                    c=item[kind]
                    for rank in range(len(c['score'])):
                        feature_index=len(features);features.append(c['feature'][rank].float().cpu().numpy())
                        xyz=c['xyz'][rank].float();valid=bool(batch['target_valid'][b])
                        candidates.append(dict(sequence_id=batch['sequence_id'][b],sample_id=batch['sample_id'][b],
                            query_time=float(batch['query_time'][b]),candidate_set=kind,rank=rank+1,score=float(c['score'][rank]),
                            pred_x=float(xyz[0]),pred_y=float(xyz[1]),pred_z=float(xyz[2]),
                            distance_to_gt=float(torch.linalg.vector_norm(xyz-batch['target_xyz'][b])) if valid else None,
                            source_token_id=int(c['source_token_id'][rank]),feature_index=feature_index,
                            checkpoint_id=checkpoint_id,model_version=model.version,selector_version=selector.VERSION,
                            coordinate_config_id='EXISTING_MMUAV_COORDINATE_ASSUMPTION'))
    if len({r['sample_id'] for r in rows})!=len(rows):raise AssertionError('Duplicate validation endpoint')
    metrics=summarize_metrics(rows)
    health=dict(spatial_loss=spatial_sum/max(1,ns),loss_cls=cls_sum/max(1,ns),loss_reg=reg_sum/max(1,ns),
                temporal_loss=temp_sum/max(1,nt),num_supervised_samples=ns,num_temporal_supervised=nt,
                evaluated_endpoint_occurrences=endpoints,endpoint_spatial_mask_occurrences=endpoints,
                spatial_supervised_occurrences=ns,
                current_support_occurrences=ns,no_current_support_occurrences=no_support,
                evaluation_precision=policy.effective)
    health['loss']=health['spatial_loss']+criterion.weight*health['temporal_loss']
    if export_dir is not None:
        output=Path(export_dir);output.mkdir(parents=True,exist_ok=False)
        write_csv(output/'candidates.csv',candidates);write_csv(output/'query_predictions.csv',rows)
        np.savez_compressed(output/'candidate_features.npz',features=np.stack(features) if features else np.empty((0,128),np.float32))
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics,rows,health

def spatial_selection_metrics(metrics,epoch):
    all_metrics=metrics['all']
    return dict(epoch=int(epoch),nms_recall_at_10_1m=float(all_metrics['nms_recall_at_10_1m']),
        nms_top1_success_1m=float(all_metrics['nms_top1_success_1m']),
        nms_top1_error_median=float(all_metrics['nms_top1_error_median']))


def temporal_selection_metrics(metrics,epoch):
    all_metrics=metrics['all']
    return dict(epoch=int(epoch),temporal_success_1m=float(all_metrics['temporal_success_1m']),
        temporal_error_median=float(all_metrics['temporal_error_median']),
        temporal_error_p90=float(all_metrics['temporal_error_p90']))


def better_spatial(candidate,best):
    if best is None:return True
    return (candidate['nms_recall_at_10_1m'],candidate['nms_top1_success_1m'],-candidate['nms_top1_error_median'],-candidate['epoch'])>(
        best['nms_recall_at_10_1m'],best['nms_top1_success_1m'],-best['nms_top1_error_median'],-best['epoch'])


def better_temporal(candidate,best):
    if best is None:return True
    return (candidate['temporal_success_1m'],-candidate['temporal_error_median'],-candidate['temporal_error_p90'],-candidate['epoch'])>(
        best['temporal_success_1m'],-best['temporal_error_median'],-best['temporal_error_p90'],-best['epoch'])


def save_checkpoint(path,model,optimizer,scheduler,epoch,global_step,selection,effective_cfg,metadata):
    torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
        epoch=epoch,global_step=global_step,global_optimizer_step=global_step,checkpoint_selection=selection,
        effective_config=effective_cfg,resolved_model_config=effective_cfg,data_config=effective_cfg['data'],
        loss_config=effective_cfg['loss'],selector_config=effective_cfg['selector'],
        training_precision=effective_cfg['effective_runtime']['training_precision'],
        evaluation_precision=effective_cfg['effective_runtime']['evaluation_precision'],
        spatial_selection_metrics=selection.get('spatial'),temporal_selection_metrics=selection.get('temporal'),
        git_commit=metadata.get('git_commit'),run_id=metadata.get('run_id'),eqs=metadata.get('eqs'),denoise=False),path)
