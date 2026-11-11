"""High-score LiDAR continuation and optional visual correction of the output."""

from bisect import bisect_right
from dataclasses import replace
import math

import numpy as np
import torch

from .modeling.geometry import project_omni_radtan
from .tracking import stable_order
from .visual_birth import VisualBirthTracker


TRACKING_MODES = ('legacy', 'high_score', 'bearing50')
BIRTH_AND_UPDATE_TOPK = 10
BEARING_MAX_IMAGE_AGE_S = .06
BEARING_MAX_MEASUREMENT_AGE_S = .15
BEARING_BOX_MARGIN_PX = 4.
BEARING_GAIN = .5
BEARING_CAP_M = .5
JACOBIAN_STEP_M = .05


def build_lidar_pairs(frames, data_root):
    """Latest scan indices per sequence and query, including older cache formats."""
    streams = {}
    pairs = {}
    for frame in frames:
        sequence = frame['sequence_id']
        if sequence not in streams:
            sequence_dir = data_root / sequence
            if not sequence_dir.is_dir():
                raise FileNotFoundError(sequence_dir)
            streams[sequence] = [
                sorted(float(path.stem) for path in (sequence_dir / sensor).glob('*.npy'))
                for sensor in ('livox_avia', 'lidar_360')
            ]
        # Query records preserve full-precision timestamps in their sample IDs.
        # query_time may have been rounded when the cache was exported.
        sample_id = frame['sample_id']
        prefix, _, exact_time = sample_id.rpartition('_')
        time = (float(exact_time) if prefix in (f'{sequence}_sensor', f'{sequence}_query')
                else frame['query_time'])
        pairs[frame['query_time']] = tuple(
            bisect_right(times, time) - 1 for times in streams[sequence]
        )
    return pairs


class CausalVisualTracker(VisualBirthTracker):
    def __init__(self, config, projection, *, visual_lookup, lidar_pair_lookup, bearing=False):
        super().__init__(replace(config, high_rank_first=True), projection, visual_lookup=visual_lookup)
        self.lidar_pair_lookup = lidar_pair_lookup
        self.last_lidar_pair = None
        self.segment = 0
        self.camera_origin = None
        if bearing:
            rotation = projection.rotation_camera_from_radar[0].numpy()
            translation = projection.translation_camera_from_radar_m[0].numpy()
            self.camera_origin = np.linalg.solve(rotation, -translation).astype(float)

    def step(self, timestamp, frame):
        t = float(timestamp)
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Timestamps must be finite and strictly increasing')
        dt = None if self.last_time is None else t - self.last_time
        pair = self.lidar_pair_lookup(t)
        fresh = pair != self.last_lidar_pair and any(index >= 0 for index in pair)
        self.last_lidar_pair = pair

        selected_score = None
        if self.kf is not None and (dt > self.config.filter.reset_gap_s
                                    or t - self.last_update > self.config.filter.max_prediction_age_s):
            reason = 'input_gap' if dt > self.config.filter.reset_gap_s else 'strong_timeout'
            self.kf = None
            self.last_time = t
            row = self._record(t, 'lost', reason=reason)
        else:
            dense = frame.dense
            obs = dense.take(stable_order(dense)[:BIRTH_AND_UPDATE_TOPK])
            if self.kf is None:
                row = super().step(t, replace(frame, dense=obs))
                if row['event'] == 'birth':
                    self.segment += 1
                    index = np.flatnonzero(obs.token_id == row['selected_token_id'])[0]
                    selected_score = float(obs.score[index])
            else:
                self.last_time = t
                self.kf.predict(dt)
                selected = self._select_measurement(obs) if fresh else None
                if selected is None:
                    row = self._record(t, 'predict')
                else:
                    self.kf.update(obs.xyz[selected])
                    self.last_update = t
                    selected_score = float(obs.score[selected])
                    row = self._record(t, 'high_update', obs, selected)
        # The evaluator uses a fixed CSV schema taken from the first row.
        row.setdefault('reason', None)
        row.update(segment_id=self.segment, fresh_lidar=fresh, selected_score=selected_score, readout=None,
                   position_covariance=None if self.kf is None else self.kf.P[:3, :3].tolist())
        if self.camera_origin is not None and row['event'] in ('predict', 'high_update'):
            self._correct_bearing(t, row, self.visual_lookup(t))
        return row

    def _record(self, t, event, obs=None, selected=None, reason=None):
        return {
            'xyz': None if self.kf is None else self.kf.x[:3].tolist(),
            'velocity_mps': None if self.kf is None else self.kf.x[3:].tolist(),
            'event': event,
            'reason': reason,
            'selected_token_id': None if selected is None else int(obs.token_id[selected]),
            'observation_xyz': None if selected is None else obs.xyz[selected].tolist(),
            'measurement_age_s': None if self.kf is None else t - self.last_update,
            'birth_requires_vision': False,
            'birth_visual_available': False,
            'birth_visual_candidates': None,
            'birth_visual_verified': False,
        }

    def _select_measurement(self, obs):
        cfg = self.config
        delta, mahalanobis_sq = self.kf.innovations(obs.xyz)
        accepted = ((np.linalg.norm(delta, axis=1) <= cfg.filter.max_innovation_m)
                    & (mahalanobis_sq <= cfg.filter.mahalanobis_gate_sq)
                    & (obs.score >= cfg.high_score))
        # The input is already ordered by original logit, then by token ID.
        for i in np.flatnonzero(accepted):
            trial, _ = self.kf.posterior(obs.xyz[i])
            if np.linalg.norm(trial[3:]) <= cfg.filter.max_speed_mps:
                return i
        return None

    def _correct_bearing(self, t, row, hint):
        if hint is None or hint.score < self.config.birth_score:
            return
        age = t - hint.image_time
        if not 0 <= age <= BEARING_MAX_IMAGE_AGE_S or row['measurement_age_s'] > BEARING_MAX_MEASUREMENT_AGE_S:
            return
        box = np.asarray(hint.box)
        wh = self.projection.image_size_wh[0].numpy()
        if np.any(box[:2] < 0.) or np.any(box[2:] >= wh) or np.any(box[2:] <= box[:2]):
            return
        x = np.asarray(row['xyz'])
        image_state = x - age * np.asarray(row['velocity_mps'])
        basis = np.eye(3) * JACOBIAN_STEP_M
        samples = np.vstack((image_state, image_state + basis, image_state - basis))
        pixels, valid = project_omni_radtan(torch.as_tensor(samples, dtype=torch.float32),
                                           torch.zeros(len(samples), dtype=torch.long), self.projection)
        uv = pixels.numpy()
        outside = np.maximum(np.maximum(box[:2] - uv[0], uv[0] - box[2:]), 0.)
        if not bool(valid.all()) or np.linalg.norm(outside) > BEARING_BOX_MARGIN_PX:
            return
        jacobian = (uv[1:4] - uv[4:7]).T.astype(float) / (2 * JACOBIAN_STEP_M)
        residual = .5 * (box[:2] + box[2:]) - uv[0]
        delta = BEARING_GAIN * (jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + np.eye(2) * .01, residual))
        shift = float(np.linalg.norm(delta))
        if shift > BEARING_CAP_M:
            delta *= BEARING_CAP_M / shift
        ray = x - self.camera_origin
        new_ray = ray + delta
        if np.linalg.norm(new_ray) <= 1e-8:
            return
        revised = self.camera_origin + new_ray * (np.linalg.norm(ray) / np.linalg.norm(new_ray))
        row['readout'] = {
            'base_xyz': row['xyz'],
            'image_time': hint.image_time,
            'image_score': hint.score,
            'pixel_error': residual.tolist(),
            'shift_m': float(np.linalg.norm(revised - x)),
            'range_difference_m': float(np.linalg.norm(revised - self.camera_origin) - np.linalg.norm(ray)),
        }
        row['xyz'] = revised.tolist()
