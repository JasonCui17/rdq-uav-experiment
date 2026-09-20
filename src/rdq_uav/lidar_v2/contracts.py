"""Fail-fast frozen V2 configuration, precision, and evaluation contracts."""
from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass

import torch

ERROR='Unsupported configuration for frozen LiDAR V2 architecture.'

# Fields that were previously declarative/stale rather than behavioral.
PREVIOUSLY_SILENT_FIELDS=(
    'data.query_mode','data.denoise','model.transformer.l2_global',
    'model.transformer.attention_dropout','model.transformer.ffn_dropout','model.transformer.drop_path',
    'model.merge.use_child_position','model.merge.use_log_point_count','model.merge.explicit_octant_occupancy',
    'model.merge.occupancy_slots','model.merge.attention_pool','model.up.adjacent_only',
    'evaluation.k','evaluation.radii_m','selector.version','train.checkpoint_metric',
)


def _get(cfg,path):
    value=cfg
    for part in path.split('.'):value=value[part]
    return value


def validate_frozen_v2_config(cfg):
    """Reject public options that contradict the frozen pre-training architecture."""
    if 'temporal' in cfg:
        raise ValueError(f'{ERROR} temporal architecture was removed; use data.query_clip_length for spatial sampling')
    stale_loss={'temporal_weight','temporal_smooth_l1_beta'}&set(cfg.get('loss',{}))
    if stale_loss:
        raise ValueError(f'{ERROR} removed temporal loss fields: {sorted(stale_loss)}')
    if 'checkpoint_metric' in cfg.get('train',{}):
        raise ValueError(f'{ERROR} train.checkpoint_metric is deprecated; checkpoint_policy is fixed and explicit')
    required={
        'model.name':'lidar_uav_v2','model.dim':128,'model.voxel.embedding':'sbe_lite',
        'model.voxel.scales':[.5,1.,2.],'model.voxel.sbe.vqsa.enabled':True,
        'model.voxel.sbe.vqsa.embed_dim':16,'model.voxel.sbe.vqsa.heads':2,'model.voxel.sbe.vqsa.dropout':0.,
        'model.transformer.l2_global':True,'model.transformer.attention_dropout':0.,
        'model.transformer.ffn_dropout':0.,'model.transformer.drop_path':0.,
        'model.merge.use_child_position':True,'model.merge.use_log_point_count':True,
        'model.merge.explicit_octant_occupancy':True,'model.merge.occupancy_slots':8,
        'model.merge.attention_pool':False,'model.up.adjacent_only':True,
        'model.head.residual_scale_m':1.,'data.query_mode':'gt_timestamp','data.denoise':False,
        'data.query_clip_length':8,'data.query_clip_stride':1,
        'evaluation.precision':'fp32','evaluation.k':[1,5,10,20],'evaluation.radii_m':[.5,1.,2.],
        'selector.version':'stable_topk100_radius1m_v1',
    }
    for path,expected in required.items():
        try:actual=_get(cfg,path)
        except KeyError as exc:raise ValueError(f'{ERROR} missing {path}') from exc
        if actual!=expected:raise ValueError(f'{ERROR} {path}={actual!r}; required {expected!r}')
    policy=cfg.get('checkpoint_policy',{})
    expected_policy={
        'last':'last.pt',
        'spatial':{'filename':'best_spatial.pt','order':['nms_recall_at_10_1m','nms_top1_success_1m','negative_nms_top1_error_median','earlier_epoch']},
    }
    if policy!=expected_policy:
        raise ValueError(f'{ERROR} checkpoint_policy={policy!r}; required {expected_policy!r}')
    return cfg


@dataclass(frozen=True)
class PrecisionPolicy:
    configured: str
    effective: str
    enabled: bool
    dtype: torch.dtype

    def context(self,device):
        return torch.autocast(device_type=device.type,dtype=self.dtype,enabled=self.enabled) if self.enabled else nullcontext()


def resolve_precision(configured,device):
    configured=str(configured).lower();configured={'bfloat16':'bf16','float16':'fp16','float32':'fp32'}.get(configured,configured);device=torch.device(device)
    if configured=='fp32':return PrecisionPolicy(configured,'fp32',False,torch.float32)
    if configured not in ('bf16','fp16'):raise ValueError(f'Unsupported precision: {configured}')
    if device.type!='cuda':return PrecisionPolicy(configured,'fp32',False,torch.float32)
    if configured=='bf16' and not torch.cuda.is_bf16_supported():
        return PrecisionPolicy(configured,'fp32',False,torch.float32)
    dtype=torch.bfloat16 if configured=='bf16' else torch.float16
    return PrecisionPolicy(configured,configured,True,dtype)


def effective_config(cfg,device):
    validate_frozen_v2_config(cfg);resolved=copy.deepcopy(cfg)
    train=resolve_precision(cfg['train']['amp_dtype'],device);evaluation=resolve_precision(cfg['evaluation']['precision'],device)
    resolved['effective_runtime']=dict(training_precision=train.effective,evaluation_precision=evaluation.effective,
        residual_scale_m=1.0,validation_unique_query_packing=False,checkpoint_policy=copy.deepcopy(cfg['checkpoint_policy']))
    return resolved,train,evaluation


def require_occurrence_aligned_evaluation(batch):
    valid=batch['query_valid_mask'].flatten();mapping=batch['occurrence_to_unique'];expected=torch.arange(len(mapping),device=mapping.device)
    aligned=(not bool(batch.get('unique_query_packing',False)) and int(batch['spatial_num_samples'])==len(mapping)
        and torch.equal(mapping[valid],expected[valid]))
    if not aligned:
        raise RuntimeError('Evaluation currently requires occurrence-aligned spatial queries. Disable UQP for validation/evaluation.')
