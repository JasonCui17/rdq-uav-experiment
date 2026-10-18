"""Single-UAV box refinement from visual motion, with causal radar support.

This is a bounded observation-space adapter, not OC-SORT/YOLOV reproduction.
No detector weights, class scores or 3D track positions are changed.
"""
from collections import defaultdict
import math

import numpy as np


def refine_visual_tracks(frames, *, lookahead_s=0., history_s=.5, min_support=3,
                         confidence_threshold=.25):
    """Return at most one box per frame, plus actual sensor lookahead.

    frames contain identity/time, predicted boxes/scores and a CAUSAL radar
    prior. No annotation is accepted as a motion/selection input. Low-score or
    unsupported neighbors are excluded; insufficient support uses raw Top1.
    """
    if (not all(math.isfinite(v) for v in (lookahead_s, history_s, confidence_threshold)) or
        not 0 <= lookahead_s <= .5 or history_s <= 0 or min_support < 3 or
        not 0 <= confidence_threshold <= 1):
        raise ValueError('Invalid fixed visual refinement policy')
    groups = defaultdict(list)
    identities = set()
    for index, frame in enumerate(frames):
        identity = (frame['sequence_id'], frame['image_path'])
        t = float(frame['image_time'])
        boxes = np.asarray(frame['boxes'], dtype=float).reshape(-1, 4)
        scores = np.asarray(frame['scores'], dtype=float)
        if identity in identities or not math.isfinite(t) or scores.shape != (len(boxes),):
            raise ValueError('Duplicate identity or invalid prediction/time')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Invalid predicted boxes/scores')
        identities.add(identity)
        best = int(np.argmax(scores)) if len(scores) else None
        box, score = (None, None) if best is None else (boxes[best], float(scores[best]))
        prior = frame['prior']
        supported = False
        available = t
        if prior is not None:
            xyz = np.asarray(prior['xyz'], dtype=float)
            uv = np.asarray(prior['projected_xy_px'], dtype=float)
            available = float(prior['latest_observation_time'])
            if (xyz.shape != (3,) or uv.shape != (2,) or not np.isfinite(xyz).all() or
                not np.isfinite(uv).all() or not math.isfinite(available) or available > t):
                raise ValueError('Refinement requires finite causal radar priors')
            available = t  # image is available later than its causal radar evidence
            if box is not None and score >= confidence_threshold:
                outside = np.maximum(np.maximum(box[:2]-uv, uv-box[2:]), 0)
                supported = np.linalg.norm(outside) <= np.clip(16*20/max(np.linalg.norm(xyz), 1e-6), 8, 48)
        groups[frame['sequence_id']].append((t, index, box, score, prior is not None, supported, available))
    results = [None]*len(frames)
    for group in groups.values():
        group.sort(key=lambda r: r[0])
        times = np.asarray([r[0] for r in group])
        if np.any(np.diff(times) <= 0):
            raise ValueError('Image times must be unique within a sequence')
        for t, index, box, score, has_prior, _, _ in group:
            chosen, support, latest = box, 0, t
            if has_prior and box is not None:
                lower = np.searchsorted(times, t-history_s, side='left')
                upper = np.searchsorted(times, t+lookahead_s, side='right')
                neighbors = [r for r in group[lower:upper] if r[5]]
                support = len(neighbors)
                if support >= min_support:
                    dt = np.asarray([r[0]-t for r in neighbors])
                    b = np.asarray([r[2] for r in neighbors])
                    centers = (b[:, :2]+b[:, 2:])/2
                    ii, jj = np.triu_indices(support, 1)
                    delta = dt[jj]-dt[ii]
                    valid = np.abs(delta) > 1e-6
                    if valid.any():
                        velocity = np.median((centers[jj[valid]]-centers[ii[valid]])/delta[valid, None], axis=0)
                        center = np.median(centers-dt[:, None]*velocity, axis=0)
                        size = np.median(b[:, 2:]-b[:, :2], axis=0)
                        chosen = np.concatenate((center-size/2, center+size/2))
                        latest = max(t, max(r[6] for r in neighbors))
            results[index] = {'boxes': [] if chosen is None else [chosen.tolist()],
                              'scores': [] if score is None else [score],
                              'support': support, 'image_lookahead_s': max(0., latest-t)}
    return results


def refine_visual_detections(frames, *, nms_iou=.5, topk=100, **policy):
    """Refine Top1 but retain every other detection before the original NMS.

    This preserves the original candidate budget; it does not force a single
    output. The matched max-one controls remain available separately.
    """
    from .image_detection import merge_detections
    estimates = refine_visual_tracks(frames, **policy)
    result = []
    for frame, estimate in zip(frames, estimates):
        boxes = np.asarray(frame['boxes'], dtype=float).reshape(-1, 4).copy()
        scores = np.asarray(frame['scores'], dtype=float)
        if len(boxes):
            boxes[int(np.argmax(scores))] = estimate['boxes'][0]
        merged = merge_detections([{'boxes': boxes.tolist(), 'scores': scores.tolist()}], nms_iou, topk)
        result.append({**estimate, **merged})
    return result
