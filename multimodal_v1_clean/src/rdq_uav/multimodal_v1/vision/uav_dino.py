from __future__ import annotations
import math
import torch
from torch import nn


def adapt_dino_class_head_to_single_uav(detector: nn.Module, *, prior_prob: float=.01) -> nn.Module:
    """Replace detrex DINO classification heads with one UAV logit after COCO checkpoint load.

    Backbone/transformer/bbox weights remain untouched. The original decoder
    head sharing pattern is preserved. Call this only after loading the reference
    checkpoint, otherwise checkpoint shape mismatch is expected.
    """
    heads=getattr(detector,'class_embed',None)
    if not isinstance(heads,nn.ModuleList) or len(heads)==0:
        raise TypeError('detector.class_embed must be a non-empty ModuleList')
    in_features=heads[0].in_features
    shared=all(head is heads[0] for head in heads)
    reference=heads[0]
    def make_head():
        h=nn.Linear(in_features,1).to(
            device=reference.weight.device,
            dtype=reference.weight.dtype,
        )
        nn.init.trunc_normal_(h.weight,std=.02,a=-.04,b=.04)
        nn.init.constant_(h.bias,math.log(prior_prob/(1-prior_prob)))
        return h
    if shared:
        h=make_head(); new=nn.ModuleList([h for _ in heads])
    else:
        new=nn.ModuleList([make_head() for _ in heads])
    detector.class_embed=new
    if hasattr(detector,'transformer') and hasattr(detector.transformer,'decoder'):
        detector.transformer.decoder.class_embed=new
    if hasattr(detector,'num_classes'): detector.num_classes=1
    criterion=getattr(detector,'criterion',None)
    if criterion is not None and hasattr(criterion,'num_classes'): criterion.num_classes=1
    return detector
