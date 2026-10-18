"""Detached training diagnostics and cumulative validation progress."""
from __future__ import annotations
import math
import numpy as np
import torch


class RunningLocalization:
    """O(1) state; accumulate squared errors, never average batch RMSEs."""
    def __init__(self):
        self.queries = self.outputs = 0
        self.squared = np.zeros(3, dtype=np.float64)
        self.success = np.zeros(3, dtype=np.int64)

    def update(self, rows):
        for row in rows:
            if not row['has_gt3d']:
                continue
            self.queries += 1
            if row['pred_xyz'] is None:
                continue
            error = np.asarray(row['pred_xyz'], dtype=np.float64)-np.asarray(row['gt_xyz'], dtype=np.float64)
            if error.shape != (3,) or not np.isfinite(error).all():
                raise ValueError('live localization requires finite XYZ')
            self.outputs += 1
            self.squared += error**2
            self.success += np.linalg.norm(error) <= np.asarray([.5, 1., 2.])

    def compute(self):
        axis = np.sqrt(self.squared/self.outputs) if self.outputs else [None]*3
        return {'rmse_x_m': axis[0], 'rmse_y_m': axis[1], 'rmse_z_m': axis[2],
                'rmse_3d_m': math.sqrt(float(self.squared.sum())/self.outputs) if self.outputs else None,
                'coverage': self.outputs/self.queries if self.queries else None,
                **{f'success_{r:g}m': int(n)/self.queries if self.queries else None
                   for r, n in zip((.5, 1., 2.), self.success)},
                'gt_queries': self.queries, 'predicted_queries': self.outputs}


def dino_loss_components(weighted):
    """Disjoint weighted components sum to the unchanged native DINO loss."""
    result = {}
    for name, value in weighted.items():
        if '_dn' in name:
            group = 'dn'
        elif name.endswith('_enc'):
            group = 'enc'
        elif name.rsplit('_', 1)[-1].isdigit():
            group = 'aux'
        elif name in ('loss_class', 'loss_ce'):
            group = 'cls'
        elif name == 'loss_bbox':
            group = 'bbox'
        elif name == 'loss_giou':
            group = 'giou'
        else:
            group = 'other'
        scalar = value.detach().float().sum()
        result[group] = result.get(group, scalar.new_zeros(()))+scalar
    return result


@torch.no_grad()
def visual_metric_rows(output, targets, batch):
    """Source-pixel records in original batch order; support compact vision indices."""
    count = len(targets.has_box)
    ids = output.top2d_indices(count)
    compact = {int(original): local for local, original in enumerate(batch['vision_batch_index'])}
    rows = []
    for index in range(count):
        if not bool(targets.has_box[index]):
            continue
        if index not in compact:
            raise ValueError('2D supervision requires a valid visual input')
        selected = ids[index]
        rows.append({'sample_id': batch['sample_id'][index], 'sequence_id': batch['sequence_id'][index],
                     'query_time': float(batch['query_time'][index]),
                     'image_time': float(batch['image_time'][index]),
                     'image_path': batch['left_image_path'][index],
                     'image_source_wh': batch['image_source_wh'][compact[index]].tolist(),
                     'has_gt2d': True, 'gt_box_xyxy_px': targets.box_xyxy_px[index].float().cpu().tolist(),
                     'pred_boxes_xyxy_px': output.box_xyxy_px[selected].float().cpu().tolist(),
                     'pred_box_scores': output.score_2d_after[selected].float().cpu().tolist()})
    return rows


def localization_display(metrics, prefix):
    fields = {'rx': 'rmse_x_m', 'ry': 'rmse_y_m', 'rz': 'rmse_z_m', 'r3': 'rmse_3d_m',
              'Cov': 'coverage', 'S1': 'success_1m', 'N': 'gt_queries'}
    return {f'{prefix}/{name}': '--' if metrics[key] is None else metrics[key] for name, key in fields.items()}


try:
    from lightning.pytorch.callbacks import TQDMProgressBar
except ImportError:
    TQDMProgressBar = object


class StageProgressBar(TQDMProgressBar):
    """Avoid showing B1's undefined 3D metrics or stale validation as current training."""
    def get_metrics(self, trainer, pl_module):
        return dict(pl_module.live_progress)
