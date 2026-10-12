"""Opt-in LoRA on explicit A2 attention Linear modules; no PEFT dependency."""
from __future__ import annotations

import math
import re
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_TARGETS = (r'encoder[12]\.blocks\.\d+\.(qkv|proj)',)


class LoRALinear(nn.Linear):
    """Preserve original weight/bias keys for strict A2 warmstart."""
    def __init__(self, source: nn.Linear, rank=8, alpha=16., dropout=0.):
        if rank < 1 or rank > min(source.in_features, source.out_features):
            raise ValueError('LoRA rank must be in [1, min(in_features,out_features)]')
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('LoRA requires finite positive alpha and dropout in [0,1)')
        super().__init__(source.in_features, source.out_features, bias=source.bias is not None,
                         device=source.weight.device, dtype=source.weight.dtype)
        self.weight = source.weight
        self.bias = source.bias
        self.rank, self.scaling = rank, float(alpha)/rank
        self.lora_A = nn.Parameter(source.weight.new_empty((rank, source.in_features)))
        self.lora_B = nn.Parameter(source.weight.new_zeros((source.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + self.scaling * delta


def inject_lora(detector, config):
    patterns = config.get('targets', DEFAULT_TARGETS)
    if not patterns or isinstance(patterns, str):
        raise ValueError('lora.targets must be a nonempty list of full-match regexes')
    found = []
    for pattern in patterns:
        matches = [(name, module) for name, module in detector.named_modules() if re.fullmatch(pattern, name)]
        if not matches:
            raise ValueError(f'LoRA target matched no module: {pattern}')
        for name, module in matches:
            if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
                raise ValueError(f'LoRA target must be an original Linear: {name}')
            if name in [n for n, _ in found]:
                raise ValueError(f'overlapping LoRA target: {name}')
            found.append((name, module))
    # Do not perturb initialization of any other new head or future module.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.get('seed', 42)))
        for name, module in found:
            parent_name, attr = name.rsplit('.', 1)
            parent = detector.get_submodule(parent_name)
            setattr(parent, attr, LoRALinear(module, int(config.get('rank', 8)),
                                            float(config.get('alpha', 16.)), float(config.get('dropout', 0.))))
    return [name for name, _ in found]


def set_efficient_train_mode(detector, mode):
    if mode not in ('heads', 'lora'):
        raise ValueError(f'unknown efficient mode {mode}')
    adapters = [(n,p) for n,p in detector.named_parameters() if n.endswith(('.lora_A','.lora_B'))]
    if (mode == 'lora') != bool(adapters):
        raise ValueError('lora mode requires adapters; heads mode must not contain adapters')
    detector.requires_grad_(False)
    if detector.query_decoder is None:
        detector.head.requires_grad_(True)
    else:
        detector.query_decoder.requires_grad_(True)
    for name in ('quality_head', 'metadata_proj'):
        module = getattr(detector, name, None)
        if module is not None:
            module.requires_grad_(True)
    if mode == 'lora':
        for _, parameter in adapters:
            parameter.requires_grad_(True)
    # Do not wrap the full backbone in no_grad(): frozen downstream weights
    # still propagate gradients to earlier adapters. Frozen prefixes naturally
    # avoid autograd when neither inputs nor parameters require gradients.
    return parameter_report(detector)


def parameter_report(detector):
    total = sum(p.numel() for p in detector.parameters())
    trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    adapter = sum(p.numel() for n,p in detector.named_parameters() if p.requires_grad and n.endswith(('.lora_A','.lora_B')))
    return dict(total_parameters=total, trainable_parameters=trainable, adapter_parameters=adapter,
                trainable_fraction=trainable/max(total,1), lora_targets=getattr(detector,'lora_targets', []),
                trainable_names=[n for n,p in detector.named_parameters() if p.requires_grad])
