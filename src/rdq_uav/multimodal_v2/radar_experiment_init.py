"""Strict A2 warmstart: permit only explicitly configured new modules."""
from __future__ import annotations
import torch


def load_experiment_warmstart(detector, checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    prefix = 'network.lidar.detector.'
    selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not selected:
        raise ValueError('ABC warmstart requires network.lidar.detector.* in an A2 Lightning checkpoint')
    own = detector.state_dict()
    allowed = {k for k in own if k.startswith(('quality_head.', 'query_decoder.', 'metadata_proj.', 'temporal_motion.')) or k.endswith(('.lora_A', '.lora_B'))}
    missing, extra = set(own) - set(selected), set(selected) - set(own)
    mismatch = [k for k in set(own) & set(selected) if own[k].shape != selected[k].shape]
    if missing != allowed or extra or mismatch:
        raise ValueError(f'A2 warmstart mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shape={mismatch}')
    detector.load_state_dict(selected, strict=False)
    return {'checkpoint': str(checkpoint), 'loaded_keys': len(selected), 'new_keys': sorted(missing)}
