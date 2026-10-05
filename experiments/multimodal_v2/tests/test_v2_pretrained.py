"""COCO loading must precede UAV adaptation and preserve localization weights."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from rdq_uav.multimodal_v2.uav_dino import (
    adapt_dino_class_head_to_single_uav, load_coco_pretrained_dino,
)

class Detector(nn.Module):
    def __init__(self, shared=False):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.bbox_embed = nn.Linear(4, 4)
        head = nn.Linear(4, 80)
        self.class_embed = nn.ModuleList([head, head] if shared else [head, nn.Linear(4, 80)])
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.class_embed = self.class_embed
        self.num_classes = 80
        self.criterion = SimpleNamespace(num_classes=80)

@pytest.mark.parametrize("shared", [False, True])
def test_coco_then_uav_preserves_backbone_and_bbox(tmp_path, shared):
    source, target = Detector(shared), Detector(shared)
    with torch.no_grad():
        for p in source.parameters(): p.fill_(0.125)
    checkpoint = tmp_path / "coco.pth"
    torch.save({"model": source.state_dict()}, checkpoint)
    report = load_coco_pretrained_dino(target, checkpoint)
    assert report["strict"] and report["source_num_classes"] == 80
    adapt_dino_class_head_to_single_uav(target)
    assert torch.equal(target.backbone.weight, source.backbone.weight)
    assert torch.equal(target.bbox_embed.weight, source.bbox_embed.weight)
    assert all(head.out_features == 1 for head in target.class_embed)
    assert target.transformer.decoder.class_embed is target.class_embed
    assert (target.class_embed[0] is target.class_embed[1]) is shared
    assert target.num_classes == target.criterion.num_classes == 1
    target.backbone(torch.ones(1, 3)).sum().backward()
    assert target.backbone.weight.grad is not None

def test_missing_or_partial_checkpoint_fails(tmp_path):
    detector = Detector()
    with pytest.raises(FileNotFoundError):
        load_coco_pretrained_dino(detector, tmp_path / "missing.pth")
    checkpoint = tmp_path / "partial.pth"
    torch.save({"model": {"backbone.weight": detector.backbone.weight}}, checkpoint)
    with pytest.raises(RuntimeError): load_coco_pretrained_dino(detector, checkpoint)
    adapt_dino_class_head_to_single_uav(detector)
    with pytest.raises(ValueError, match="80-class"): load_coco_pretrained_dino(detector, checkpoint)
