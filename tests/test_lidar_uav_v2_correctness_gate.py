"""Regression tests for the V2 pre-training correctness contracts.

No optimizer or scheduler step is constructed in this module.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
import tempfile
import unittest

import torch
import yaml

from rdq_uav.lidar_v2 import CandidateLoss, CandidateSelector
from rdq_uav.lidar_v2.contracts import (
    ERROR,
    effective_config,
    require_occurrence_aligned_evaluation,
    resolve_precision,
    validate_frozen_v2_config,
)
from rdq_uav.lidar_v2.geometry import decode_residual, encode_residual
from rdq_uav.lidar_v2.runtime import evaluate_batch
from rdq_uav.lidar_v2.training import better_spatial, save_checkpoint


CFG = yaml.safe_load(open('configs/lidar_uav_v2.yaml'))


def _selector_outputs(logits: torch.Tensor):
    n = len(logits)
    return {
        'logits': logits,
        'pred_xyz': torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3) * 3,
        'fine_features': torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 128),
        'source_token_id': torch.arange(n),
        'batch_index': torch.zeros(n, dtype=torch.long),
        'aux_stats': {'num_samples': 1},
    }


def _loss_fixture(endpoint_has_support=True, temporal_length=3):
    n = temporal_length
    target = torch.zeros((n, 3))
    points = target.clone()
    if not endpoint_has_support:
        points[-1] = 20
    outputs = {
        'logits': torch.zeros(n, requires_grad=True),
        'residual_xyz': torch.zeros((n, 3), requires_grad=True),
        'pred_xyz': torch.zeros((n, 3)),
        'voxel_centers': points.clone(),
        'batch_index': torch.arange(n),
        'layouts': SimpleNamespace(point_to_l0=torch.arange(n)),
    }
    batch = {
        'points': points,
        'point_batch_index': torch.arange(n),
        'supervision_recent_mask': torch.ones(n, dtype=torch.bool),
        'spatial_target_xyz': target,
        'spatial_target_valid': torch.ones(n, dtype=torch.bool),
        'target_xyz': target,
        'target_xyz_clip': target.reshape(1, n, 3),
        'target_valid': torch.ones(n, dtype=torch.bool),
        'target_valid_clip': torch.tensor([[False] * (n - 1) + [True]]),
        'query_valid_mask': torch.ones((1, n), dtype=torch.bool),
        'spatial_supervise_mask_occurrence': torch.tensor([False] * (n - 1) + [True]),
        'occurrence_to_unique': torch.arange(n),
        'spatial_num_samples': n,
        'num_samples': n,
    }
    return outputs, batch


class CorrectnessGateTests(unittest.TestCase):
  def test_f1_ranking_uses_fp32_raw_logits_before_sigmoid(self):
    selector = CandidateSelector(CFG)
    for logits in (torch.tensor([7, 10], dtype=torch.bfloat16),
                   torch.tensor([7, 10], dtype=torch.float32),
                   torch.tensor([-10, -8], dtype=torch.bfloat16)):
        selected = selector(_selector_outputs(logits))[0]['raw']
        self.assertEqual(selected['source_token_id'].tolist(), [1, 0])
        self.assertEqual(selected['score'].dtype, torch.float32)
    tied = selector(_selector_outputs(torch.tensor([2, 2], dtype=torch.bfloat16)))[0]['raw']
    self.assertEqual(tied['source_token_id'].tolist(), [0, 1])

  def test_f2_endpoint_only_spatial_supervision(self):
    criterion = CandidateLoss(CFG)
    out, batch = _loss_fixture(True, 3)
    loss = criterion(out, batch)
    self.assertEqual(loss['num_supervised_samples'], 1)
    self.assertEqual(loss['num_no_current_support'], 0)

    out, batch = _loss_fixture(False, 3)
    loss = criterion(out, batch)
    self.assertEqual(loss['num_supervised_samples'], 0)
    self.assertEqual(loss['num_no_current_support'], 1)
    self.assertEqual(float(loss['loss']), 0)

  def test_f2_endpoint_mask_supports_t1_t8_and_padding(self):
    for length in (1, 8):
      with self.subTest(length=length):
        out, batch = _loss_fixture(True, length)
        self.assertEqual(int(batch['spatial_supervise_mask_occurrence'].sum()), 1)
        self.assertEqual(CandidateLoss(CFG)(out, batch)['num_supervised_samples'], 1)

  def test_f2_uqp_counts_selected_occurrences_not_unique_or(self):
    out, batch = _loss_fixture(True, 2)
    # Spatial query 0 appears twice; only its second occurrence is selected.
    batch['query_valid_mask'] = torch.ones((1, 3), dtype=torch.bool)
    batch['target_valid_clip'] = torch.tensor([[False, True, True]])
    batch['target_xyz_clip'] = torch.zeros((1, 3, 3))
    batch['target_valid'] = torch.ones(3, dtype=torch.bool)
    batch['occurrence_to_unique'] = torch.tensor([0, 0, 1])
    batch['spatial_supervise_mask_occurrence'] = torch.tensor([False, True, True])
    loss = CandidateLoss(CFG)(out, batch)
    self.assertEqual(loss['num_supervised_samples'], 2)

  def test_f3_frozen_architecture_fields_fail_fast(self):
    validate_frozen_v2_config(CFG)
    mutations = [
        ('model.transformer', 'l2_global', False),
        ('model.merge', 'explicit_octant_occupancy', False),
        ('model.voxel.sbe.vqsa', 'enabled', False),
        ('model', 'voxel.embedding', 'legacy'),
        ('evaluation', 'precision', 'bf16'),
        ('train.unique_query_packing', 'enabled', True),
    ]
    for parent, key, value in mutations:
        cfg = copy.deepcopy(CFG)
        node = cfg
        for part in parent.split('.'):
            node = node[part]
        if '.' in key:
            first, second = key.split('.')
            node[first][second] = value
        else:
            node[key] = value
        with self.assertRaisesRegex(ValueError, ERROR):validate_frozen_v2_config(cfg)
    stale = copy.deepcopy(CFG)
    stale['train']['checkpoint_metric'] = 'anything'
    with self.assertRaisesRegex(ValueError, 'deprecated'):validate_frozen_v2_config(stale)
    stale = copy.deepcopy(CFG);stale['temporal'] = {'enabled': False}
    with self.assertRaisesRegex(ValueError, 'temporal architecture was removed'):validate_frozen_v2_config(stale)
    stale = copy.deepcopy(CFG);stale['loss']['temporal_weight'] = 1.0
    with self.assertRaisesRegex(ValueError, 'removed temporal loss fields'):validate_frozen_v2_config(stale)

  def test_f3_effective_config_and_precision_contract(self):
    effective, training, evaluation = effective_config(CFG, torch.device('cpu'))
    self.assertEqual((training.configured, training.effective), ('bf16', 'fp32'))
    self.assertEqual((evaluation.configured, evaluation.effective), ('fp32', 'fp32'))
    self.assertIs(effective['effective_runtime']['validation_unique_query_packing'], False)
    self.assertIs(resolve_precision('fp32', torch.device('cpu')).enabled, False)

  def test_f4_residual_codec_roundtrip_and_fixed_scale(self):
    torch.manual_seed(7)
    gt = torch.randn(128, 3) * .1
    center = torch.randn(128, 3) * .1
    decoded = decode_residual(encode_residual(gt, center, 1.0), center, 1.0)
    self.assertLessEqual(float((decoded - gt).abs().max()), 1e-7)
    with self.assertRaisesRegex(ValueError, 'must equal 1.0'):encode_residual(gt, center, 0.5)


def _evaluation_fixture(packed=False):
    # Two rolling clips q0..q7 and q4..q11; only q7 and q11 are endpoints.
    n = 16
    score = torch.zeros((2, 8), dtype=torch.bool)
    score[:, -1] = True
    target = torch.zeros((n, 3))
    batch = {
        'query_valid_mask': torch.ones((2, 8), dtype=torch.bool),
        'score_mask': score,
        'target_valid': torch.ones(n, dtype=torch.bool),
        'target_xyz': target,
        'target_timestamp': torch.arange(n, dtype=torch.float64),
        'query_time': torch.arange(n, dtype=torch.float64),
        'sample_id': [f'q{i}' for i in list(range(8)) + list(range(4, 12))],
        'sequence_id': ['seq'] * n,
        'supervision_recent_mask': torch.ones(n, dtype=torch.bool),
        'points': target.clone(),
        'point_batch_index': torch.arange(n),
        'spatial_target_xyz': target,
        'spatial_target_valid': torch.ones(n, dtype=torch.bool),
        'occurrence_to_unique': torch.arange(n),
        'spatial_num_samples': n,
        'num_samples': n,
        'unique_query_packing': packed,
    }
    outputs = {
        'logits': torch.zeros(n),
        'residual_xyz': torch.zeros((n, 3)),
        'pred_xyz': target.clone(),
        'fine_features': torch.zeros((n, 128)),
        'voxel_centers': target.clone(),
        'source_token_id': torch.arange(n),
        'batch_index': torch.arange(n),
        'layouts': SimpleNamespace(point_to_l0=torch.arange(n)),
        'aux_stats': {'num_samples': n},
    }
    return outputs, batch


class EvaluationAndCheckpointTests(unittest.TestCase):
  def test_f5_non_uqp_evaluates_both_endpoints_and_uqp_fails(self):
    outputs, batch = _evaluation_fixture(False)
    rows = evaluate_batch(outputs, batch, CandidateSelector(CFG), CandidateLoss(CFG))
    self.assertEqual([row['sample_id'] for row in rows], ['q7', 'q11'])
    batch['unique_query_packing'] = True
    batch['spatial_num_samples'] = 12
    batch['occurrence_to_unique'][8:12] = torch.arange(4, 8)
    with self.assertRaisesRegex(RuntimeError, 'Disable UQP'):require_occurrence_aligned_evaluation(batch)

  def test_checkpoint_comparators_are_lexicographic_and_deterministic(self):
    s = dict(epoch=3, nms_recall_at_10_1m=.8, nms_top1_success_1m=.7, nms_top1_error_median=.5)
    self.assertTrue(better_spatial({**s, 'epoch': 4, 'nms_recall_at_10_1m': .81}, s))
    self.assertTrue(better_spatial({**s, 'epoch': 4, 'nms_top1_success_1m': .71}, s))
    self.assertTrue(better_spatial({**s, 'epoch': 4, 'nms_top1_error_median': .49}, s))
    self.assertFalse(better_spatial({**s, 'epoch': 4}, s))

  def test_checkpoint_payload_contains_reproduction_contract(self):
    class State:
      def state_dict(self):return {'state': 1}
    effective, _, _ = effective_config(CFG, torch.device('cpu'))
    selection = {'spatial': {'epoch': 1}}
    metadata = {'git_commit': 'abc', 'run_id': 'run', 'eqs': {'stride': 4, 'current_offset': 0}}
    with tempfile.TemporaryDirectory() as directory:
      path = f'{directory}/last.pt'
      save_checkpoint(path, torch.nn.Linear(1, 1), State(), State(), 1, 2, selection, effective, metadata)
      state = torch.load(path, map_location='cpu')
    for key in ('model_state', 'optimizer_state', 'scheduler_state', 'epoch',
                'global_optimizer_step', 'effective_config', 'training_precision',
                'evaluation_precision', 'spatial_selection_metrics',
                'git_commit', 'run_id', 'eqs'):
      self.assertIn(key, state)
    self.assertNotIn('temporal_selection_metrics', state)


if __name__ == '__main__':unittest.main()
