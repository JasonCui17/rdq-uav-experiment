"""DINO continuation from a writable, shared Swin visual pyramid."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from .swin_adapter import SwinPyramidAdapter, SwinPyramidOutput


def _inverse_sigmoid(value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    value=value.clamp(min=0,max=1); return torch.log(value.clamp(min=eps)/(1-value).clamp(min=eps))


class DINOAdapter(nn.Module):
    """Run one shared Swin and continue through the unchanged detrex DINO head."""
    def __init__(self, detector: nn.Module) -> None:
        super().__init__(); self.detector=detector; self.swin=SwinPyramidAdapter(detector.backbone,register_backbone=False)

    def forward_from_pyramid(self,pyramid:SwinPyramidOutput,image_masks:torch.Tensor,*,allow_training_candidate_path:bool=False)->dict[str,Any]:
        if self.detector.training and not allow_training_candidate_path:
            raise RuntimeError("P2 identity path is inference-only. FullMultimodalV1 may explicitly enable the differentiable candidate path; detector-native DINO training remains the SSOD/P8 responsibility.")
        multi_level_features=self.detector.neck(pyramid.dino_features)
        if image_masks.ndim != 3 or image_masks.shape[0] != multi_level_features[0].shape[0]:
            raise ValueError('image_masks must be [B,H,W]')
        image_masks = image_masks.to(torch.bool)
        masks=[F.interpolate(image_masks[:,None].float(),size=feature.shape[-2:],mode='nearest')[:,0].to(torch.bool) for feature in multi_level_features]
        positions=[self.detector.position_embedding(mask) for mask in masks]
        # detrex CUDA deformable attention does not support BF16.
        # Keep the rest of E5 under AMP; run this transformer in FP32.
        with torch.autocast(device_type=multi_level_features[0].device.type, enabled=False):
            decoder_states,initial_reference,intermediate_references,encoder_state,encoder_reference=self.detector.transformer(
                [feature.float() for feature in multi_level_features],
                masks,
                [position.float() for position in positions],
                (None,None),
                attn_masks=[None,None])
        decoder_states[0]+=self.detector.label_enc.weight[0,0]*0.0
        classes=[]; boxes=[]
        for level in range(decoder_states.shape[0]):
            reference=initial_reference if level==0 else intermediate_references[level-1]; reference=_inverse_sigmoid(reference)
            logits=self.detector.class_embed[level](decoder_states[level]); residual=self.detector.bbox_embed[level](decoder_states[level])
            if reference.shape[-1]==4: residual=residual+reference
            else:
                if reference.shape[-1]!=2: raise RuntimeError("DINO reference points must have dimension 2 or 4")
                residual=torch.cat((residual[...,:2]+reference,residual[...,2:]),dim=-1)
            classes.append(logits); boxes.append(residual.sigmoid())
        stacked_classes=torch.stack(classes); stacked_boxes=torch.stack(boxes)
        encoder_logits=self.detector.transformer.decoder.class_embed[-1](encoder_state)
        return {"pred_logits":stacked_classes[-1],"pred_boxes":stacked_boxes[-1],"decoder_query_features":decoder_states[-1],
                "decoder_features_all_layers":decoder_states,"aux_outputs":self.detector._set_aux_loss(stacked_classes,stacked_boxes),
                "enc_outputs":{"pred_logits":encoder_logits,"pred_boxes":encoder_reference},"pyramid":pyramid,
                "multi_level_features":tuple(multi_level_features),"multi_level_masks":tuple(masks)}

