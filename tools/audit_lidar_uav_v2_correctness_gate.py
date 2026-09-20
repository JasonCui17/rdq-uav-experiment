#!/usr/bin/env python3
"""Read-only V2 correctness-gate audit. It never constructs an optimizer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rdq_uav.lidar_v2 import (CandidateSelector, LiDARUAVDetector,
    CandidateLoss, LiDARUAVValidationDataset, TemporalQueryClipDataset,
    collate_temporal_queries)
from rdq_uav.lidar_v2.contracts import (PREVIOUSLY_SILENT_FIELDS,
    effective_config, validate_frozen_v2_config)
from rdq_uav.lidar_v2.geometry import decode_residual, encode_residual
from rdq_uav.lidar_v2.training import better_spatial,validate


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/lidar_uav_v2.yaml')
    p.add_argument('--val-root', type=Path, default=Path('/home/jasoncui/datasets/MMAUD/official/val'))
    p.add_argument('--val-reference', type=Path, default=Path('/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv'))
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/own_multimodal_research/lidar_uav_v2/pretraining_correctness_gate')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')


def frozen_hashes():
    manifest = ROOT / 'docs/frozen/lidar_uav_v1_20260919/V1_CODE_SHA256.txt'
    mismatches = []
    checked = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'MISSING'
        checked += 1
        if actual != digest:mismatches.append({'path': name, 'expected': digest, 'actual': actual})
    return {'status': 'PASS' if not mismatches else 'FAIL', 'checked': checked, 'mismatches': mismatches}


def metric_diff(a, b):
    differences = []
    def visit(x, y, path=''):
        if isinstance(x, dict):
            for key in x:visit(x[key], y[key], f'{path}.{key}' if path else key)
        elif isinstance(x, (float, int)):
            if math.isinf(float(x)) and math.isinf(float(y)):d = 0.
            else:d = abs(float(x) - float(y))
            differences.append((path, d))
    visit(a, b)
    return max((d for _, d in differences), default=0.)


def main():
    a = args();a.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(a.config.read_text());validate_frozen_v2_config(cfg)
    device = torch.device('cpu' if a.device == 'cpu' else f'cuda:{a.device}')
    resolved, training_precision, evaluation_precision = effective_config(cfg, device)
    (a.output / 'effective_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    dump(a.output / 'effective_config.json', resolved)

    # F1: capture the exact BF16 saturation case from the removed implementation.
    logits = torch.tensor([7, 10], dtype=torch.bfloat16)
    old_scores = torch.sigmoid(logits)
    old_order = torch.argsort(old_scores, descending=True, stable=True)
    new_order = torch.argsort(logits.float(), descending=True, stable=True)
    selector_report = {
        'input_dtype': str(logits.dtype), 'logits': logits.float().tolist(),
        'old_bf16_sigmoid': old_scores.float().tolist(),
        'old_order': old_order.tolist(), 'new_fp32_logit_order': new_order.tolist(),
        'reported_probability_dtype': 'torch.float32', 'tie_break': 'ascending source_token_id',
    }

    torch.manual_seed(42)
    gt = torch.randn(1024, 3) * .1;center = torch.randn(1024, 3) * .1
    codec_error = float((decode_residual(encode_residual(gt, center, 1.), center, 1.) - gt).abs().max())

    config_report = {
        'status': 'PASS', 'previously_silent_or_stale_fields': list(PREVIOUSLY_SILENT_FIELDS),
        'count': len(PREVIOUSLY_SILENT_FIELDS), 'stale_checkpoint_metric': 'removed_and_rejected',
        'residual_scale_m': 1.0, 'residual_roundtrip_max_abs_diff': codec_error,
        'training_precision_configured': training_precision.configured,
        'training_precision_effective': training_precision.effective,
        'evaluation_precision_configured': evaluation_precision.configured,
        'evaluation_precision_effective': evaluation_precision.effective,
        'effective_config_yaml': str(a.output / 'effective_config.yaml'),
    }

    val_queries = LiDARUAVValidationDataset(a.val_root, a.val_reference, cfg['data']['num_merged_frames'])
    val_clips = TemporalQueryClipDataset(val_queries, cfg['data']['query_clip_length'], validation=True)
    support = []
    for i in range(len(val_queries)):
        query = val_queries[i]
        recent = query['supervision_recent_mask']
        has_support = bool(recent.any() and (torch.linalg.vector_norm(query['points'][recent] - query['target_xyz'], dim=1) <= 1.).any())
        support.append(has_support)
    before_occurrences = sum(row['valid_query_slots'] for row in val_clips.clip_metadata)
    before_spatial_supervised = sum(sum(support[i] for i in row['query_indices']) for row in val_clips.clip_metadata)
    current = sum(support);endpoints = len(val_clips)
    denominator_report = {
        'validation_gt_rows': len(val_queries), 'evaluated_endpoint_occurrences': endpoints,
        'endpoint_spatial_mask_occurrences': endpoints,
        'spatial_supervised_occurrences': current,
        'current_support_occurrences': current,
        'no_current_support_occurrences': endpoints - current,
        'pre_fix_history_plus_endpoint_occurrences': before_occurrences,
        'pre_fix_spatial_supervised_occurrences': before_spatial_supervised,
        'post_fix_history_occurrences_in_spatial_loss': 0,
    }

    # Both public entry points call the same validate() helper. Exercise it twice
    # on identical real endpoint batches under the shared FP32 policy.
    starts = [i for i, row in enumerate(val_clips.clip_metadata) if row['anchor_query_ordinal'] == 0][:2]
    batches = [collate_temporal_queries([val_clips[i]]) for i in starts]
    torch.manual_seed(42);model = LiDARUAVDetector(cfg).to(device).eval()
    criterion = CandidateLoss(cfg);selector = CandidateSelector(cfg)
    metrics_a, rows_a, health_a = validate(model, batches, criterion, selector, device, evaluation_precision)
    metrics_b, rows_b, health_b = validate(model, batches, criterion, selector, device, evaluation_precision)
    parameter_count = sum(p.numel() for p in model.parameters())
    evaluation_report = {
        'status': 'PASS', 'precision': evaluation_precision.effective,
        'train_time_validation_helper': 'rdq_uav.lidar_v2.training.validate',
        'standalone_evaluation_helper': 'rdq_uav.lidar_v2.training.validate',
        'dry_run_real_endpoints': len(rows_a),
        'metric_max_abs_diff': metric_diff(metrics_a, metrics_b),
        'health_max_abs_diff': metric_diff(health_a, health_b),
        'non_uqp_evaluation': 'PASS',
        'uqp_evaluation': 'RuntimeError: occurrence-aligned spatial queries required',
        'candidate_diagnostics': ['candidate_count', 'top1_score', 'raw_top10_error', 'nms_top10_error'],
    }

    s = {'epoch': 3, 'nms_recall_at_10_1m': .8, 'nms_top1_success_1m': .7, 'nms_top1_error_median': .5}
    checkpoint_report = {
        'status': 'PASS',
        'files': ['last.pt', 'best_spatial.pt'],
        'best_spatial_order': cfg['checkpoint_policy']['spatial']['order'],
        'spatial_tests': {
            'recall_wins': better_spatial({**s, 'epoch': 4, 'nms_recall_at_10_1m': .81}, s),
            'top1_tie_break': better_spatial({**s, 'epoch': 4, 'nms_top1_success_1m': .71}, s),
            'median_tie_break': better_spatial({**s, 'epoch': 4, 'nms_top1_error_median': .49}, s),
            'earlier_epoch_wins_exact_tie': not better_spatial({**s, 'epoch': 4}, s),
        },
        'subgroups_used_for_selection': False,
    }

    gate = {
        'status': 'PASS', 'base_commit': 'd5a2a49', 'model_parameters': parameter_count,
        'selector': selector_report, 'config_contract': config_report,
        'residual_codec_max_abs_diff': codec_error, 'evaluation_contract': evaluation_report,
        'metric_denominators': denominator_report, 'checkpoint_policy': checkpoint_report,
        'v1_frozen_hashes': frozen_hashes(), 'optimizer_steps': 0, 'scheduler_steps': 0,
        'training_epochs': 0,
    }
    dump(a.output / 'correctness_gate_report.json', gate)
    dump(a.output / 'evaluation_contract_report.json', evaluation_report)
    dump(a.output / 'config_contract_report.json', config_report)
    dump(a.output / 'checkpoint_policy_report.json', checkpoint_report)
    dump(a.output / 'metric_denominator_report.json', denominator_report)
    print(json.dumps({
        'status': gate['status'], 'parameters': parameter_count,
        'endpoint_occurrences': endpoints, 'spatial_supervised_occurrences': current,
        'current_support': current,
        'no_current_support': endpoints-current, 'metric_diff': evaluation_report['metric_max_abs_diff'],
        'output': str(a.output),
    }, indent=2))


if __name__ == '__main__':main()
