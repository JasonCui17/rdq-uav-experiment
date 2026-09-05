from __future__ import annotations

import copy
from typing import Any

from torch import nn

from rdq_uav.models.model import MultiModalClassifier


def build_model(
    config: dict[str, Any], *, load_backbone_pretrained: bool = True
) -> MultiModalClassifier:
    """Build a model, optionally skipping external backbone initialization.

    A checkpoint already contains every backbone parameter. Evaluation,
    visualization and resumed training therefore must not contact a model hub
    before loading that state dictionary.
    """
    model_config = copy.deepcopy(config)
    if not load_backbone_pretrained:
        model_config["backbone"]["pretrained"] = False
    return MultiModalClassifier(model_config)


def build_parameter_groups(
    model: nn.Module, backbone_lr: float, new_modules_lr: float
) -> list[dict[str, Any]]:
    backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_tokenizer.backbone"):
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"})
    if other_parameters:
        groups.append({"params": other_parameters, "lr": new_modules_lr, "name": "new_modules"})
    if not groups:
        raise RuntimeError("Model has no trainable parameters")
    return groups
