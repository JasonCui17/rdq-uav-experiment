"""GT-free time alignment and source-image crop geometry for track guidance."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import math

import numpy as np


class ImageTrackPrior:
    """Whitelist inference fields from the frozen target evaluator's records.

    Causal mode extrapolates only from a query no later than image_time.
    lag05 mode interpolates delayed positions; reports actual observation
    availability, which can exceed image_time + 0.5 due to query alignment.
    """
    def __init__(self, rows, max_gap_s=.25):
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('Invalid prior time gap')
        self.max_gap_s = max_gap_s
        self.rows = defaultdict(list)
        for row in rows:
            item = {k: row.get(k) for k in ('query_time', 'xyz', 'velocity_mps',
                                           'lagged_xyz', 'lookahead_s', 'event')}
            if not math.isfinite(item['query_time']):
                raise ValueError('Invalid track time')
            for key in ('xyz', 'velocity_mps', 'lagged_xyz'):
                value = item[key]
                if value is not None and (np.shape(value) != (3,) or not np.isfinite(value).all()):
                    raise ValueError('Invalid track vector: ' + key)
            self.rows[row['sequence_id']].append(item)
        self.times = {}
        for seq, records in self.rows.items():
            records.sort(key=lambda r: r['query_time'])
            times = [r['query_time'] for r in records]
            if np.any(np.diff(times) <= 0):
                raise ValueError('Duplicate track query times')
            self.times[seq] = times

    def estimate(self, sequence_id, image_time, mode):
        if mode not in ('causal', 'lag05') or not math.isfinite(image_time):
            raise ValueError('Invalid time alignment request')
        records = self.rows.get(sequence_id, [])
        if not records:
            return None
        times = self.times[sequence_id]
        if mode == 'causal':
            index = bisect_right(times, image_time) - 1
            if index < 0:
                return None
            row = records[index]
            age = image_time - times[index]
            if age > self.max_gap_s or row['xyz'] is None or row['velocity_mps'] is None:
                return None
            xyz = np.asarray(row['xyz']) + age * np.asarray(row['velocity_mps'])
            latest = times[index]
        else:
            right = bisect_left(times, image_time)
            if right < len(times) and times[right] == image_time:
                left = right
            else:
                left = right - 1
            if left < 0 or right >= len(times) or times[right] - times[left] > self.max_gap_s:
                return None
            a, b = records[left], records[right]
            if a['lagged_xyz'] is None or b['lagged_xyz'] is None:
                return None
            # Never interpolate a timeout gap or unrelated new target birth.
            if left != right and b['event'] == 'birth' and a['xyz'] is not None:
                return None
            if any(r['lookahead_s'] is None or not math.isfinite(r['lookahead_s'])
                   or not 0 <= r['lookahead_s'] <= .5 for r in (a, b)):
                raise ValueError('lag05 requires recorded lookahead in [0, 0.5]')
            alpha = 0. if left == right else (image_time - times[left]) / (times[right] - times[left])
            xyz = (1 - alpha) * np.asarray(a['lagged_xyz']) + alpha * np.asarray(b['lagged_xyz'])
            latest = max(r['query_time'] + r['lookahead_s'] for r in (a, b))
        return {'xyz': xyz.tolist(), 'latest_observation_time': latest,
                'image_lookahead_s': max(0., latest - image_time), 'mode': mode}


def centered_crop(center_xy, source_wh, side_px):
    """Shift a fixed square into the source image; no GT-dependent sizing."""
    w, h = map(int, source_wh)
    if min(w, h) <= 0 or type(side_px) is not int or side_px <= 0:
        raise ValueError('Invalid crop dimensions')
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Invalid crop center')
    cw, ch = min(w, side_px), min(h, side_px)
    x = int(np.clip(round(center[0] - cw / 2), 0, w - cw))
    y = int(np.clip(round(center[1] - ch / 2), 0, h - ch))
    return [x, y, x + cw, y + ch]


def restore_crop_boxes(boxes, crop_xyxy):
    """Local crop xyxy pixels -> calibrated source left-image pixels."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        raise ValueError('Nonfinite crop boxes')
    return boxes + np.tile(np.asarray(crop_xyxy, dtype=float)[:2], 2)
