"""CDN must reach the criterion without leaking GT queries into candidates."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rdq_uav.multimodal_v2.data import ViewTransform
from rdq_uav.multimodal_v2.dino_adapter import DINOAdapter
from rdq_uav.multimodal_v2.dino_supervision import build_dino_targets, supervised_dino_loss


class Transformer(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.decoder = SimpleNamespace(class_embed=heads)
        self.last_mask = None

    def forward(self, features, masks, positions, queries, *, attn_masks):
        self.last_mask = attn_masks[0]
        batch = len(features[0])
        padding = 0 if queries[0] is None else queries[0].shape[1]
        count = padding + 3
        tokens = torch.arange(count, dtype=torch.float32).view(1, 1, count, 1)
        states = tokens.repeat(2, batch, 1, 4) + features[0].mean((1, 2, 3))[None, :, None, None]
        refs = torch.full((batch, count, 4), .5)
        return states, refs, refs[None], states[-1, :, -3:], refs[:, -3:]


class Detector(nn.Module):
    """CPU test double for detrex's CUDA-only CDN utility."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Identity()
        self.backbone.pos_drop = nn.Identity()
        self.backbone.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.backbone.out_indices = (1, 2, 3)
        self.backbone.num_features = (4, 4, 4, 4)
        self.neck = lambda features: list(features.values())
        self.position_embedding = lambda mask: torch.zeros(len(mask), 4, *mask.shape[1:])
        self.class_embed = nn.ModuleList([nn.Linear(4, 1) for _ in range(3)])
        self.bbox_embed = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.label_enc = nn.Embedding(1, 4)
        self.transformer = Transformer(self.class_embed)
        self.dn_number, self.label_noise_ratio, self.box_noise_scale = 1, .5, 1.
        self.num_queries, self.num_classes, self.embed_dim = 3, 1, 4
        self.cdn_calls = 0

    def prepare_for_cdn(self, targets, **kwargs):
        self.cdn_calls += 1
        batch = len(targets)
        return torch.zeros(batch, 2, 4), torch.zeros(batch, 2, 4), torch.ones(5, 5).bool(), {
            "single_padding": 2, "dn_num": 1,
        }

    @staticmethod
    def _set_aux_loss(logits, boxes):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(logits[:-1], boxes[:-1])]

    def dn_post_process(self, logits, boxes, meta):
        meta["output_known_lbs_bboxes"] = {
            "pred_logits": logits[-1, :, :2], "pred_boxes": boxes[-1, :, :2],
            "aux_outputs": self._set_aux_loss(logits[:, :, :2], boxes[:, :, :2]),
        }
        return logits[:, :, 2:], boxes[:, :, 2:]


def inputs():
    feature = torch.ones(2, 4, 2, 2, requires_grad=True)
    pyramid = SimpleNamespace(dino_features={"p1": feature})
    boxes = torch.tensor([[10., 10., 20., 20.], [0., 0., 0., 0.]])
    valid = torch.tensor([True, False])
    transforms = [ViewTransform((100, 100), (50, 50))] * 2
    targets = build_dino_targets(boxes, valid, transforms)
    return feature, pyramid, boxes, valid, transforms, targets


def test_cdn_mask_loss_and_compact_unlabeled_selection():
    feature, pyramid, boxes, valid, transforms, targets = inputs()
    detector = Detector()
    adapter = DINOAdapter(detector)
    raw = adapter.forward_from_pyramid(pyramid, torch.zeros(2, 2, 2).bool(),
                                      allow_training_candidate_path=True, targets=targets)
    assert detector.cdn_calls == 1 and detector.transformer.last_mask is not None
    assert raw["pred_logits"].shape == (2, 3, 1)
    assert raw["decoder_query_features"].shape == (2, 3, 4)
    assert raw["decoder_features_all_layers"].shape == (2, 2, 3, 4)
    assert raw["decoder_query_features"][0, :, 0].tolist() == [3., 4., 5.]
    assert len(targets[1]["boxes"]) == 0
    assert torch.allclose(targets[0]["boxes"], torch.tensor([[.15, .15, .1, .1]]))

    def criterion(output, selected_targets, meta):
        assert len(selected_targets) == output["pred_boxes"].shape[0] == 1
        known = meta["output_known_lbs_bboxes"]
        assert known["pred_logits"].shape == (1, 2, 1)
        assert known["aux_outputs"][0]["pred_logits"].shape == (1, 2, 1)
        return {"loss_class": output["pred_logits"].square().mean(),
                "loss_class_dn": known["pred_logits"].square().mean()}
    criterion.weight_dict = {"loss_class": 1., "loss_class_dn": 1.}
    detector.criterion = criterion
    loss, terms, count = supervised_dino_loss(detector, raw, gt_box_xyxy_source=boxes,
                                             gt_2d_valid=valid, transforms=transforms)
    assert count == 1 and "loss_class_dn" in terms
    loss.backward()
    assert feature.grad[0].abs().sum() > 0
    assert feature.grad[1].abs().sum() == 0
    assert raw["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"].shape[0] == 2


def test_evaluation_has_no_gt_queries_and_rejects_targets():
    _, pyramid, _, _, _, targets = inputs()
    detector = Detector().eval()
    adapter = DINOAdapter(detector)
    raw = adapter.forward_from_pyramid(pyramid, torch.zeros(2, 2, 2).bool())
    assert detector.cdn_calls == 0 and detector.transformer.last_mask is None
    assert raw["dn_meta"] is None and raw["pred_boxes"].shape == (2, 3, 4)
    with pytest.raises(ValueError, match="evaluation"):
        adapter.forward_from_pyramid(pyramid, torch.zeros(2, 2, 2).bool(), targets=targets)


def test_all_unlabeled_skips_cdn_and_criterion():
    _, pyramid, boxes, valid, transforms, _ = inputs()
    valid[:] = False
    detector = Detector()
    targets = build_dino_targets(boxes, valid, transforms)
    raw = DINOAdapter(detector).forward_from_pyramid(
        pyramid, torch.zeros(2, 2, 2).bool(), allow_training_candidate_path=True, targets=targets)
    assert detector.cdn_calls == 0 and raw["dn_meta"] is None
    loss, terms, count = supervised_dino_loss(detector, raw, gt_box_xyxy_source=boxes,
                                             gt_2d_valid=valid, transforms=transforms)
    assert count == 0 and terms == {} and loss.item() == 0.
