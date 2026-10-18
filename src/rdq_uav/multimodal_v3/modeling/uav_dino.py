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


def load_coco_pretrained_dino(detector: nn.Module, checkpoint) -> dict:
    """Load a matching full COCO DINO state before replacing its class heads.

    Accept the detrex .pth model container or a raw state dict. Strict loading
    rejects partial/backbone-only files and incompatible model variants.
    Only trusted project checkpoints should be supplied to torch.load.
    """
    from pathlib import Path
    from collections.abc import Mapping

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"COCO DINO checkpoint not found: {path}")
    heads = getattr(detector, "class_embed", None)
    if not isinstance(heads, nn.ModuleList) or not heads or heads[0].out_features != 80:
        raise ValueError("Load COCO weights into the original 80-class DINO before UAV adaptation")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a detrex COCO checkpoint containing a model state dict")
    state = payload.get("model", payload)
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected tensor model weights; do not pass a V2 Lightning checkpoint")
    detector.load_state_dict(state, strict=True)
    report = {"checkpoint": str(path.resolve()), "loaded_state_keys": len(state),
              "source_num_classes": 80, "strict": True}
    print(f"Loaded full COCO DINO checkpoint: {report}")
    return report
